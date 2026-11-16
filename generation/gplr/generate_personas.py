import argparse
import hashlib
import json
import os
import pickle
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gplr import GPLR

API_KEY = ""
MODEL = "claude-haiku-5-5"
REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BATCH_LIMIT = 50000
UNREPRESENTABLE = 'Unrepresentable'
SHORT_FIELDS = ('title', 'name', 'summarization', 'summary', 'profile', 'description', 'text')
FULL_FIELDS = ('title', 'name', 'categories', 'description', 'profile', 'summarization', 'summary', 'reasoning', 'text')

STEP1_USERS, STEP1_SETS, STEP2_PICK, STEP2_SETS = 100, 40, 5, 8

STEP1_SYSTEM = ("You are an assistant skilled at summarizing, capable of deducing high-level consumer keywords "
                "based on a user's purchases.")
STEP1_PROMPT = """Take a deep breath and work according to the instructions step by step. Now you will conduct a \
series of analyses on a dataset of {domain}. This dataset contains each user's purchasing transactions and the \
bought items (the items the user bought, rated or reviewed). From this dataset, I will provide you the purchase \
information from about {user_count} ({percent:.1f} percent) users, per-user's purchasing data has been grouped by \
their ID and transferred to a natural language description for your better understanding of their purchasing \
behaviors.

Your task is to generate representative and accurate {persona_num} user personas according to these users' \
purchasing patterns. Please notice that we give you 2 important targets you should consider and optimize:
- High Coverage: We hope that your generated persona set can cover as many users as possible. We define the \
'coverage' as the total number of the users which can be labeled with at least one of your generated persona set.
- High Accuracy: We hope that each of your generated persona has a precise definition. An ambiguous or sweeping \
persona definition should be avoided.

Repeat your task one more time, the goal is to generate a proper set of {persona_num} representative and accurate \
user personas existing in the data subset and explain them quantitatively.

For each persona, you should write a short name and a corresponding definition.

Now considering the user purchasing data given below:

{users}"""

MERGE_SYSTEM = ("You are an assistant skilled at reading, observing and summarizing, capable of finding similar or "
                "repeated descriptions of user personas, and good at finding the most representative ones.")
STEP2_PROMPT = """Take a deep breath and work according to the instructions step by step. Now we have {set_num} \
persona_sets, each containing {persona_num} personas, and I will randomly select {pick} persona_sets, each \
containing {persona_num} personas, for a total of {total} personas. Your task is to select the {persona_num} most \
representative personas from these {total} personas and output the results. If you find that the content of a \
certain group is not {persona_num} personas but less than {persona_num}, or even irrelevant information, you \
should ignore this group of information and only refer to personas in other groups. Note that you may find that \
the personas you read have some similarities or even some duplications. You need to find these similar or \
duplicate personas and select the {persona_num} most representative personas accordingly. You should not refer \
to any information related to the number of occurrences in these personas, as this information is very likely to \
be unreasonable. You should ignore the occurrence times information and select the {persona_num} most \
representative personas based only on their descriptions.

Give each selected persona with its name and its definition; the definition must describe the persona only, \
without remarks about the selection.

Here are the {pick} sets of results that make up the {total} personas you need to choose from:

{sets}"""
STEP3_PROMPT = """Take a deep breath and work according to the instructions step by step. Now we have {pick} \
persona sets, each containing {persona_num} personas, for a total of {total} personas. Your task is to select the \
{persona_num} most representative personas from these {total} personas and output the results. Note that you may \
find that the personas you read have some similarities or even some duplications. You need to find these similar \
or duplicate personas and select the {persona_num} most representative personas accordingly, that is to say, \
these {persona_num} personas occurs most times and can cover most of them.

Give each selected persona with its name and its definition; the definition must describe the persona only, \
without remarks about the selection.

Here are the {pick} sets of personas that make up the {total} personas you need to choose from:

{sets}"""

USER_SYSTEM = ("Now you are an intelligent e-commerce domain assistant. You are skilled at summarizing, and capable "
               "of assigning high-level consumer personas based on a user's purchase behavior.")
USER_PROMPT = """Take a deep breath and work according to the instructions step by step.

Your goal is to identify users' shopping behaviors based on products they have bought and label them with a given \
set of personas. You need to select at least one persona, at most {max_personas} personas from our given persona \
list. But make sure that for each assignment you should find strong evidence in their purchase transactions. \
Please keep the procedure as accurate as possible.

Please provide the output in json format. Prefer to return arrays instead of comma separated strings. The \
following is an explanation of your return format:
{{"personas": ["Persona1", "Persona2", "Persona3"]}}

In the case that you feel there does not exist any suitable persona from the given list that can properly describe \
a user's purchasing behavior, you can label the user as an 'unrepresentable' user as the following example:
{{"personas": ["Unrepresentable"]}}

The products are {domain}.

Here is the persona list you should choose from:
{personas}

Here is the data of user {user}'s transaction data for you to analyze:

{transactions}

Remind one more time that you can only select from the given {persona_num} personas' list and only use the exactly \
given persona, you cannot use other words to describe. You do not need to explain how you get the result, so \
please respond no more than the required format."""

ITEM_SYSTEM = ("Now you are an intelligent e-commerce domain assistant. You have a good understanding of various "
               "customer personas, and is skilled at connecting products with personas mostly likely to be "
               "interested in them.")
ITEM_PROMPT = """Take a deep breath and work according to the instructions step by step.

I will provide you the information of a product, and a set of customer personas summarized from an e-commerce \
dataset. The persona set includes {persona_num} persona names and their corresponding definitions.

Your task is to connect the product with a subset of personas in the persona set. Each of your selected persona \
should be likely to show a purchasing interest in this specific product, based on the product information and the \
persona definition.

Please adhere to the following principles when performing this task:
1. You should understand each persona strictly according to its definition, avoiding over-imaginations based on \
the persona's name or any hallucinations.
2. You can utilize your real-world knowledge to understand the attributes of the product as suggested by its \
information.
3. Ensure the connection made between the product and the persona must be well-substantiated. In other words, the \
product's expected attributes should be highly relevant to the persona's definition.

The products are {domain}.

Here are the {persona_num} most representative personas:
{personas}

To standardize the output, please follow the following rules:
1. You may connect a product to one to {max_personas} of the most relevant personas. In fact, you should only \
list personas that are strongly related, even if there're only two or three.
2. Respond in json format, {{"personas": ["Persona1", "Persona2"]}}, and avoid any unnecessary explanations.

The information of the product I want you to analyze is:
{item}"""

PERSONA_SET_SCHEMA = {
    "type": "object",
    "properties": {
        "personas": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "definition": {"type": "string"}},
                "required": ["name", "definition"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["personas"],
    "additionalProperties": False,
}


def label_schema(names):
    return {
        "type": "object",
        "properties": {"personas": {"type": "array", "items": {"type": "string", "enum": list(names)}}},
        "required": ["personas"],
        "additionalProperties": False,
    }


def _clean(text):
    return re.sub(r'\s+', ' ', str(text)).strip()


def _entry_texts(entry):
    if isinstance(entry, dict) and isinstance(entry.get('response'), str):
        try:
            inner = json.loads(entry['response'])
        except ValueError:
            inner = re.sub(r'"(summarization|reasoning)"\s*:', ' ', entry['response'])
            inner = re.sub(r'[{}"]', ' ', inner).replace("\\'", "'").replace('\\n', ' ').strip(' ,')
        entry = inner if isinstance(inner, (dict, str)) else ''
    if isinstance(entry, str):
        return _clean(entry), _clean(entry)
    if not isinstance(entry, dict):
        return '', ''

    def field(name):
        value = entry.get(name)
        if isinstance(value, (list, tuple)):
            value = ', '.join(str(v) for v in value)
        return _clean(value) if value else ''

    full = ' | '.join(text for text in (field(name) for name in FULL_FIELDS) if text)
    short = next((text for text in (field(name) for name in SHORT_FIELDS) if text), full)
    return short, full


def load_item_texts(path, item_num):
    if path.endswith('.pkl'):
        with open(path, 'rb') as f:
            raw = pickle.load(f)
    elif path.endswith('.jsonl'):
        raw = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f):
                if line.strip():
                    row = json.loads(line)
                    raw[row.get('item_id', row.get('id', line_no))] = row
    else:
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
    if isinstance(raw, (list, tuple)):
        raw = dict(enumerate(raw))
    short, full = [''] * item_num, [''] * item_num
    for key, entry in raw.items():
        item = int(key)
        if not 0 <= item < item_num:
            raise ValueError("{}: item id {} is outside the {} items of the dataset".format(path, item, item_num))
        short[item], full[item] = _entry_texts(entry)
    return short, full


def describe_user(user, items, short):
    listed = [short[item] for item in items if short[item]]
    if not listed:
        return ''
    return ("The user {} has totally purchased {} unique products, we show the description of each product: he "
            "bought: {}").format(user, len(listed), '; '.join("({}) {}".format(n + 1, text) for n, text in enumerate(listed)))


class LLM:
    def __init__(self, args, raw_dir):
        self.args, self.raw_dir = args, raw_dir
        self.state_path = os.path.join(raw_dir, 'batches.json')
        self.offline = args.build_only
        self._sdk = None
        self.tokens = {}

    def _path(self, stage):
        return os.path.join(self.raw_dir, stage + '.jsonl')

    def answers(self, stage):
        answers = {}
        if os.path.exists(self._path(stage)):
            with open(self._path(stage), 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        answers[row['id']] = row['answer']
        return answers

    def _pending(self):
        if os.path.exists(self.state_path):
            with open(self.state_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _set_pending(self, stage, batch_ids):
        pending = self._pending()
        if batch_ids:
            pending[stage] = batch_ids
        else:
            pending.pop(stage, None)
        with open(self.state_path, 'w', encoding='utf-8') as f:
            json.dump(pending, f)

    def _store(self, stage, out, request_id, message):
        used = self.tokens.setdefault(stage, [0, 0])
        used[0] += message.usage.input_tokens
        used[1] += message.usage.output_tokens
        text = next((block.text for block in message.content if block.type == 'text'), '')
        if message.stop_reason != 'end_turn' or not text:
            print("  {} {}: stop_reason {}".format(stage, request_id, message.stop_reason))
            return False
        try:
            answer = json.loads(text)
        except ValueError:
            print("  {} {}: the answer is not JSON".format(stage, request_id))
            return False
        out.write(json.dumps({'id': request_id, 'answer': answer}, ensure_ascii=False) + '\n')
        return True

    def _client(self):
        if self._sdk is None:
            import anthropic
            self._sdk = (anthropic, anthropic.Anthropic(api_key=API_KEY or None, max_retries=5))
        return self._sdk

    def _params(self, request):
        system, prompt, schema = request
        return dict(
            model=MODEL,
            max_tokens=16000,
            system=system,
            output_config={"effort": self.args.effort, "format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": prompt}],
        )

    def _send_direct(self, stage, requests):
        anthropic, client = self._client()

        def one(item):
            request_id, request = item
            try:
                return request_id, client.messages.create(**self._params(request)), None
            except anthropic.AuthenticationError:
                raise
            except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
                return request_id, None, e

        try:
            with ThreadPoolExecutor(max_workers=self.args.direct) as pool, open(self._path(stage), 'a', encoding='utf-8') as out:
                for request_id, message, error in pool.map(one, requests.items()):
                    if error is not None:
                        print("  {} {}: {}".format(stage, request_id, error))
                    else:
                        self._store(stage, out, request_id, message)
                        out.flush()
        except anthropic.AuthenticationError:
            sys.exit("Anthropic credentials are missing or invalid: set API_KEY at the top of this file.")

    def _send_batches(self, stage, requests):
        anthropic, client = self._client()
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        ids, batch_ids = list(requests), []
        for start in range(0, len(ids), BATCH_LIMIT):
            batch_requests = [
                Request(custom_id="{}-{}".format(stage, request_id),
                        params=MessageCreateParamsNonStreaming(**self._params(requests[request_id])))
                for request_id in ids[start:start + BATCH_LIMIT]
            ]
            try:
                batch = client.messages.batches.create(requests=batch_requests)
            except anthropic.AuthenticationError:
                sys.exit("Anthropic credentials are missing or invalid: set API_KEY at the top of this file.")
            except anthropic.RateLimitError as e:
                sys.exit("Rate limited while creating the batch; run the same command again later. ({})".format(e.message))
            except anthropic.APIStatusError as e:
                sys.exit("The batch was not created ({}): {}".format(e.status_code, e.message))
            except anthropic.APIConnectionError as e:
                sys.exit("Could not reach the API: {}".format(e))
            batch_ids.append(batch.id)
            self._set_pending(stage, batch_ids)
            print("  {}: submitted batch {} ({} requests)".format(stage, batch.id, len(batch_requests)))

    def submit(self, stage, requests):
        if self.offline or stage in self._pending():
            return
        answered = self.answers(stage)
        requests = {request_id: request for request_id, request in requests.items() if request_id not in answered}
        if not requests:
            return
        if self.args.direct:
            self._send_direct(stage, requests)
        else:
            self._send_batches(stage, requests)

    def collect(self, stage):
        batch_ids = self._pending().get(stage)
        if self.offline or not batch_ids:
            return
        anthropic, client = self._client()
        for batch_id in batch_ids:
            while True:
                try:
                    batch = client.messages.batches.retrieve(batch_id)
                except anthropic.APIConnectionError:
                    time.sleep(self.args.poll)
                    continue
                if batch.processing_status == 'ended':
                    break
                counts = batch.request_counts
                print("  {} {}: {} processing, {} succeeded, {} errored".format(
                    stage, batch_id, counts.processing, counts.succeeded, counts.errored))
                time.sleep(self.args.poll)
            with open(self._path(stage), 'a', encoding='utf-8') as out:
                for result in client.messages.batches.results(batch_id):
                    request_id = int(result.custom_id.rsplit('-', 1)[1])
                    if result.result.type != 'succeeded':
                        print("  {} {}: {}".format(stage, request_id, result.result.type))
                        continue
                    self._store(stage, out, request_id, result.result.message)
        self._set_pending(stage, None)

    def ask(self, stage, requests):
        for _ in range(2):
            self.submit(stage, requests)
            self.collect(stage)
            answers = self.answers(stage)
            if self.offline or all(request_id in answers for request_id in requests):
                break
        return {request_id: answers[request_id] for request_id in requests if request_id in answers}

    def report(self):
        for stage, (tokens_in, tokens_out) in self.tokens.items():
            print("  {}: {} input tokens, {} output tokens".format(stage, tokens_in, tokens_out))


def _persona_list(answer, limit=None):
    personas, seen = [], {UNREPRESENTABLE.lower()}
    for row in (answer or {}).get('personas', []):
        name, definition = _clean(row.get('name', '')), _clean(row.get('definition', ''))
        if name and name.lower() not in seen:
            seen.add(name.lower())
            personas.append({'name': name, 'definition': definition})
    return personas[:limit] if limit else personas


def _numbered(personas):
    return '\n'.join("{}. {} - {}".format(n + 1, p['name'], p['definition']) for n, p in enumerate(personas))


def _sets_text(sets):
    return '\n\n'.join("Persona set {}:\n{}".format(n + 1, _numbered(personas)) for n, personas in enumerate(sets))


def step1_requests(trn, short, args):
    rng = np.random.RandomState(args.seed)
    user_num = trn.shape[0]
    count = min(STEP1_USERS, user_num)
    requests = {}
    for n in range(STEP1_SETS):
        users = np.sort(rng.choice(user_num, count, replace=False))
        texts = [describe_user(user, trn.indices[trn.indptr[user]:trn.indptr[user + 1]], short) for user in users]
        prompt = STEP1_PROMPT.format(domain=args.domain, user_count=count, percent=100.0 * count / user_num,
                                     persona_num=args.persona_num, users='\n\n'.join(text for text in texts if text))
        requests[n] = (STEP1_SYSTEM, prompt, PERSONA_SET_SCHEMA)
    return requests


def generate_persona_set(llm, trn, short, args):
    answers = llm.ask('persona_step1', step1_requests(trn, short, args))
    candidates = [personas for personas in (_persona_list(answers[n]) for n in sorted(answers)) if personas]
    if len(candidates) < STEP2_PICK:
        sys.exit("persona set: only {} of {} candidate sets were answered; run the same command again".format(
            len(candidates), STEP1_SETS))

    rng = np.random.RandomState(args.seed + 1)
    requests = {}
    for n in range(STEP2_SETS):
        picked = [candidates[i] for i in rng.choice(len(candidates), STEP2_PICK, replace=False)]
        prompt = STEP2_PROMPT.format(set_num=len(candidates), pick=STEP2_PICK, persona_num=args.persona_num,
                                     total=sum(len(personas) for personas in picked), sets=_sets_text(picked))
        requests[n] = (MERGE_SYSTEM, prompt, PERSONA_SET_SCHEMA)
    answers = llm.ask('persona_step2', requests)
    merged = [personas for personas in (_persona_list(answers[n]) for n in sorted(answers)) if personas]
    if not merged:
        sys.exit("persona set: no merged set was answered; run the same command again")

    prompt = STEP3_PROMPT.format(pick=len(merged), persona_num=args.persona_num,
                                 total=sum(len(personas) for personas in merged), sets=_sets_text(merged))
    answers = llm.ask('persona_step3', {0: (MERGE_SYSTEM, prompt, PERSONA_SET_SCHEMA)})
    personas = _persona_list(answers.get(0), args.persona_num)
    if not personas:
        sys.exit("persona set: the final set was not answered; run the same command again")
    if len(personas) != args.persona_num:
        print("  persona set: the LLM returned {} distinct personas instead of {}".format(len(personas), args.persona_num))
    return personas


def _label_ids(answer, persona2idx, limit):
    names = list(dict.fromkeys((answer or {}).get('personas', [])))
    if UNREPRESENTABLE in names:
        return None
    return [persona2idx[name] for name in names if name in persona2idx][:limit]


def item_requests(full, personas, args):
    schema = label_schema([p['name'] for p in personas])
    listing = _numbered(personas)
    return {item: (ITEM_SYSTEM, ITEM_PROMPT.format(persona_num=len(personas), domain=args.domain, personas=listing,
                                                   max_personas=args.max_personas, item=text), schema)
            for item, text in enumerate(full) if text}


def user_request(user, trn, short, personas, args):
    transactions = describe_user(user, trn.indices[trn.indptr[user]:trn.indptr[user + 1]], short)
    if not transactions:
        return None
    prompt = USER_PROMPT.format(max_personas=args.max_personas, domain=args.domain, personas=_numbered(personas),
                                user=user, transactions=transactions, persona_num=len(personas))
    return USER_SYSTEM, prompt, label_schema([p['name'] for p in personas] + [UNREPRESENTABLE])


def load_rounds(args, raw_dir):
    settings = {'sample_rate': args.sample_rate, 'rounds': args.rounds, 'epsilon': args.epsilon, 'seed': args.seed}
    rounds_path = os.path.join(raw_dir, 'rounds.json')
    drawn = []
    if os.path.exists(rounds_path):
        with open(rounds_path, 'r', encoding='utf-8') as f:
            stored = json.load(f)
        if args.build_only:
            settings = stored['settings']
        elif stored['settings'] != settings:
            sys.exit("{} was drawn with {}, not with {}.\nKeep those settings, remove the file to draw the rounds "
                     "again (stored LLM answers are reused), or write to another --out_dir.".format(
                         rounds_path, stored['settings'], settings))
        drawn = stored['rounds']
    return settings, drawn, rounds_path


def label_users(llm, trn, short, personas, persona2idx, args, rounds):
    user_num = trn.shape[0]
    settings, drawn, rounds_path = rounds
    budget = min(int(round(settings['sample_rate'] * user_num)), user_num)
    round_num = 1 if budget >= user_num else max(1, min(settings['rounds'], budget))
    sizes = [budget // round_num] * round_num
    sizes[-1] += budget - sum(sizes)

    gplr = GPLR(trn, len(personas), epsilon=args.epsilon, seed=settings['seed'])
    failed = set()
    for r in range(len(drawn) if llm.offline else round_num):
        if r < len(drawn):
            users = drawn[r]
        else:
            users = gplr.next_users(sizes[r])
            drawn.append(users)
            with open(rounds_path, 'w', encoding='utf-8') as f:
                json.dump({'settings': settings, 'rounds': drawn}, f)
        gplr.asked.update(users)
        requests = {user: user_request(user, trn, short, personas, args) for user in users}
        answers = llm.ask('users', {user: request for user, request in requests.items() if request})
        labels = {user: (_label_ids(answers[user], persona2idx, args.max_personas) or []) for user in users if user in answers}
        gplr.add_labels(labels)
        failed.update(user for user in users if user not in answers)
        print("  users, round {}/{}: {} asked, {} labeled, {} unrepresentable, {} without answer".format(
            r + 1, round_num, len(users), sum(1 for v in labels.values() if v), sum(1 for v in labels.values() if not v),
            sum(1 for user in users if user not in answers)))
    return gplr, sorted(failed - set(gplr.labels) - gplr.unrepresentable), settings


def _write(path, content):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(content, f, ensure_ascii=False)


def _dry_run(trn, short, full, personas, args):
    def show(title, request):
        system, prompt, _ = request
        clipped = prompt if len(prompt) <= 6000 else prompt[:4000] + "\n[... {} characters ...]\n".format(len(prompt) - 5000) + prompt[-1000:]
        print("\n==================== {}\n[system]\n{}\n[user]\n{}".format(title, system, clipped))

    print("model {}, effort {}, {}".format(MODEL, args.effort, "direct ({} threads)".format(args.direct) if args.direct else "Message Batches"))
    if personas is None:
        requests = step1_requests(trn, short, args)
        print("persona set: {} + {} + 1 requests; the first stage sends {} characters".format(
            STEP1_SETS, STEP2_SETS, sum(len(request[1]) for request in requests.values())))
        show("persona set, step 1 (request 1 of {})".format(STEP1_SETS), requests[0])
        print("\nThe item and user prompts need the persona set: nothing more to show before it exists.")
        return
    requests = item_requests(full, personas, args)
    print("items: {} requests, {} characters".format(len(requests), sum(len(request[1]) for request in requests.values())))
    if requests:
        show("item personas (first item)", requests[min(requests)])
    user_num = trn.shape[0]
    budget = min(int(round(args.sample_rate * user_num)), user_num)
    sample = [request for request in (user_request(user, trn, short, personas, args)
                                      for user in range(0, user_num, max(1, user_num // 200))) if request]
    print("\nusers: {} of {} users are labeled by the LLM, about {:.0f} characters per request".format(
        budget, user_num, np.mean([len(request[1]) for request in sample]) if sample else 0))
    if sample:
        show("user personas (first user)", sample[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--item_text', type=str, default=None)
    parser.add_argument('--domain', type=str, default='a recommendation dataset')
    parser.add_argument('--out_dir', type=str, default='gplr')
    parser.add_argument('--persona_num', type=int, default=20)
    parser.add_argument('--max_personas', type=int, default=5)
    parser.add_argument('--sample_rate', type=float, default=0.2)
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--epsilon', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--effort', type=str, default='low', choices=['low', 'medium', 'high', 'xhigh', 'max'])
    parser.add_argument('--direct', type=int, default=0, metavar='N')
    parser.add_argument('--poll', type=int, default=60)
    parser.add_argument('--build_only', action='store_true')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    data_dir = os.path.join(REPO_DIR, 'data', args.dataset)
    with open(os.path.join(data_dir, 'trn_mat.pkl'), 'rb') as f:
        trn = sp.csr_matrix(pickle.load(f) != 0).astype(np.float64)
    trn.sort_indices()
    user_num, item_num = trn.shape
    out_dir = os.path.join(data_dir, args.out_dir)
    raw_dir = os.path.join(out_dir, 'raw')
    set_path = os.path.join(out_dir, 'persona_set.json')

    if args.item_text is None:
        args.item_text = os.path.join(data_dir, '{}_item_profile.jsonl'.format(args.dataset))
        if not os.path.exists(args.item_text):
            parser.error('--item_text is required ({} does not exist)'.format(args.item_text))
    short, full = load_item_texts(args.item_text, item_num)
    print("{}: {} users, {} items ({} without text), {} training interactions".format(
        args.dataset, user_num, item_num, sum(1 for text in full if not text), trn.nnz))

    personas = None
    if os.path.exists(set_path):
        with open(set_path, 'r', encoding='utf-8') as f:
            personas = _persona_list({'personas': json.load(f)})
    if args.dry_run:
        _dry_run(trn, short, full, personas, args)
        return
    os.makedirs(raw_dir, exist_ok=True)
    llm = LLM(args, raw_dir)

    if personas is None:
        if args.build_only:
            sys.exit("{} does not exist: nothing to build from".format(set_path))
        personas = generate_persona_set(llm, trn, short, args)
        _write(set_path, personas)
    names = [p['name'] for p in personas]
    persona2idx = {name: idx for idx, name in enumerate(names)}
    _write(os.path.join(out_dir, 'persona2idx.json'), persona2idx)
    print("persona set ({}): {}".format(len(names), ', '.join(names)))

    digest = hashlib.sha1(json.dumps(personas, sort_keys=True).encode()).hexdigest()[:12]
    digest_path = os.path.join(raw_dir, 'persona_set.sha1')
    if os.path.exists(digest_path):
        with open(digest_path, 'r', encoding='utf-8') as f:
            if f.read().strip() != digest:
                sys.exit("{} changed after items or users were labeled with it.\nRemove items.jsonl, users.jsonl, "
                         "rounds.json and persona_set.sha1 in {} (or write to another --out_dir) and run again.".format(set_path, raw_dir))
    else:
        with open(digest_path, 'w', encoding='utf-8') as f:
            f.write(digest)

    rounds = load_rounds(args, raw_dir)

    requests = item_requests(full, personas, args)
    llm.submit('items', requests)

    gplr, failed_users, sampling = label_users(llm, trn, short, personas, persona2idx, args, rounds)
    if not gplr.labels:
        sys.exit("no user has a persona label yet; run the same command again")
    predicted, isolated = gplr.predict(args.top_k, args.beta)
    user_personas = dict(gplr.labels)
    user_personas.update(predicted)

    answers = llm.ask('items', requests)
    item_personas = {item: (_label_ids(answers.get(item), persona2idx, args.max_personas) or []) for item in range(item_num)}
    llm.report()

    _write(os.path.join(out_dir, 'tri_graph_tidx2pidx.json'), {str(item): item_personas[item] for item in range(item_num)})
    _write(os.path.join(out_dir, 'tri_graph_uidx2pidx_llm.json'), {str(user): gplr.labels[user] for user in sorted(gplr.labels)})
    _write(os.path.join(out_dir, 'tri_graph_uidx2pidx.json'), {str(user): user_personas[user] for user in sorted(user_personas)})
    used = np.bincount([p for ps in user_personas.values() for p in ps], minlength=len(names))
    meta = {
        'dataset': args.dataset, 'domain': args.domain, 'model': MODEL, 'effort': args.effort,
        'persona_num': len(names), 'max_personas': args.max_personas,
        'sampling': sampling,
        'prediction': {'top_k': args.top_k, 'beta': args.beta, 'epsilon': args.epsilon},
        'users': user_num, 'users_llm_labeled': len(gplr.labels), 'users_unrepresentable': len(gplr.unrepresentable),
        'users_without_answer': len(failed_users), 'users_predicted': len(predicted),
        'users_predicted_without_labeled_neighbour': isolated,
        'items': item_num, 'items_with_persona': sum(1 for ps in item_personas.values() if ps),
        'items_without_answer': sum(1 for item in requests if item not in answers),
        'user_persona_links': int(used.sum()), 'item_persona_links': sum(len(ps) for ps in item_personas.values()),
        'users_per_persona': {name: int(used[idx]) for name, idx in persona2idx.items()},
    }
    _write(os.path.join(out_dir, 'gplr_meta.json'), meta)
    print("{}: {} personas; users: {} labeled by the LLM, {} unrepresentable, {} predicted ({} of them reach no labeled "
          "user), {} without answer; items: {} of {} linked to a persona".format(
              out_dir, len(names), len(gplr.labels), len(gplr.unrepresentable), len(predicted), isolated,
              len(failed_users), meta['items_with_persona'], item_num))
    if failed_users or meta['items_without_answer']:
        print("some requests got no answer; run the same command again to ask for them once more")


if __name__ == '__main__':
    main()
