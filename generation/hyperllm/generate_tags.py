import argparse
import json
import os
import pickle
import re
import sys
import time

API_KEY = ""
MODEL = "claude-haiku-5-5"
REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEXT_FIELDS = ('title', 'categories', 'description', 'profile', 'summarization', 'reasoning', 'text')
BATCH_LIMIT = 50000

SYSTEM_PROMPT = """You will be given the information of an item from {domain}.

Please generate a concise summary highlighting the main features and appeal of the item.

Please generate 3 level tags for the item. The tags must be different and not duplicated at different levels. \
Each level can have multiple tags. Higher-level tags should represent broad, general categories that define the \
item's overall domain or purpose. Lower-level tags should represent increasingly detailed classifications.

Please generate edges between different level tags. Each edge should represent a parent-child relationship, where \
a higher-level tag is the parent and a lower-level tag is the child. The tags on each edge must be within the \
generated set of tags.

The same tags will be shared by many items, so write each tag as a short, common category name in Title Case \
(one to three words), not as a phrase specific to this single item."""

TAG_LIST = {"type": "array", "items": {"type": "string"}}
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "level_1": TAG_LIST,
        "level_2": TAG_LIST,
        "level_3": TAG_LIST,
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"parent": {"type": "string"}, "child": {"type": "string"}},
                "required": ["parent", "child"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "level_1", "level_2", "level_3", "edges"],
    "additionalProperties": False,
}


def _unpack_response(response):
    try:
        inner = json.loads(response)
        if isinstance(inner, (dict, str)):
            return inner
    except ValueError:
        pass
    text = re.sub(r'"(summarization|reasoning)"\s*:', ' ', response)
    text = re.sub(r'[{}"]', ' ', text).replace("\\'", "'").replace('\\n', ' ')
    return re.sub(r'\s+', ' ', text).strip(' ,')


def _entry_text(entry):
    if isinstance(entry, str):
        return entry.strip()
    if isinstance(entry, dict):
        if isinstance(entry.get('response'), str):
            return _entry_text(_unpack_response(entry['response']))
        parts = []
        for field in TEXT_FIELDS:
            value = entry.get(field)
            if isinstance(value, (list, tuple)):
                value = ', '.join(str(v) for v in value)
            if value:
                parts.append(str(value).strip())
        return ' | '.join(parts)
    return ''


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
    texts = [''] * item_num
    for key, entry in raw.items():
        item = int(key)
        if not 0 <= item < item_num:
            raise ValueError("{}: item id {} is outside the {} items of the dataset".format(path, item, item_num))
        texts[item] = _entry_text(entry)
    return texts


def _clean(tag):
    return re.sub(r'\s+', ' ', str(tag)).strip()


def build_tag_files(answers, item_num):
    tag_ids, names = {}, []
    tag_to_item, tag_to_tag = {}, {}
    for item in range(item_num):
        answer = answers.get(item)
        if not answer:
            continue
        own = {}
        for level in ('level_1', 'level_2', 'level_3'):
            for tag in answer.get(level, []):
                tag = _clean(tag)
                if not tag:
                    continue
                key = tag.lower()
                if key not in tag_ids:
                    tag_ids[key] = len(names)
                    names.append(tag)
                own[key] = tag_ids[key]
        for tag_id in own.values():
            tag_to_item.setdefault(tag_id, []).append(item)
        for edge in answer.get('edges', []):
            parent, child = own.get(_clean(edge.get('parent', '')).lower()), own.get(_clean(edge.get('child', '')).lower())
            if parent is None or child is None or parent == child:
                continue
            tag_to_tag.setdefault(parent, set()).add(child)
            tag_to_tag.setdefault(child, set()).add(parent)
    return {
        'id_to_tag': {str(i): name for i, name in enumerate(names)},
        'tag_to_id': {name: i for i, name in enumerate(names)},
        'tag_to_item': {str(i): tag_to_item[i] for i in sorted(tag_to_item)},
        'tag_to_tag': {str(i): sorted(tag_to_tag[i]) for i in sorted(tag_to_tag)},
    }


def write_tag_files(files, out_dir, item_num):
    for name, content in files.items():
        with open(os.path.join(out_dir, name + '.json'), 'w', encoding='utf-8') as f:
            json.dump(content, f, ensure_ascii=False)
    links = sum(len(items) for items in files['tag_to_item'].values())
    tagged = len({item for items in files['tag_to_item'].values() for item in items})
    edges = sum(len(tags) for tags in files['tag_to_tag'].values()) // 2
    everywhere = [files['id_to_tag'][tag] for tag, items in files['tag_to_item'].items() if len(items) == item_num]
    print("{}: {} tags, {} tag-item links ({:.1f} per tagged item), {} tag-tag edges, {} of {} items tagged".format(
        out_dir, len(files['id_to_tag']), links, links / max(tagged, 1), edges, tagged, item_num))
    if everywhere:
        print("  tags given to every item (HyperLLM cannot draw a negative item for them and skips their item links): {}".format(everywhere))


def _read_answers(path):
    answers = {}
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    answers[row['item']] = row['answer']
    return answers


def _collect(client, batch_id, answers_path):
    failed, tokens_in, tokens_out = 0, 0, 0
    with open(answers_path, 'a', encoding='utf-8') as f:
        for result in client.messages.batches.results(batch_id):
            item = int(result.custom_id.split('-')[1])
            if result.result.type != 'succeeded':
                failed += 1
                print("  item {}: {}".format(item, result.result.type))
                continue
            message = result.result.message
            tokens_in += message.usage.input_tokens
            tokens_out += message.usage.output_tokens
            text = next((block.text for block in message.content if block.type == 'text'), '')
            if message.stop_reason != 'end_turn' or not text:
                failed += 1
                print("  item {}: stop_reason {}".format(item, message.stop_reason))
                continue
            f.write(json.dumps({'item': item, 'answer': json.loads(text)}, ensure_ascii=False) + '\n')
    print("  {}: {} input tokens, {} output tokens".format(batch_id, tokens_in, tokens_out))
    return failed


def request_tags(texts, items, args, raw_dir):
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = anthropic.Anthropic(api_key=API_KEY or None)
    answers_path = os.path.join(raw_dir, 'answers.jsonl')
    state_path = os.path.join(raw_dir, 'batches.json')
    if os.path.exists(state_path):
        with open(state_path, 'r', encoding='utf-8') as f:
            batch_ids = json.load(f)
        print("resuming the submitted batches {}".format(batch_ids))
    else:
        system = SYSTEM_PROMPT.format(domain=args.domain)
        batch_ids = []
        for start in range(0, len(items), BATCH_LIMIT):
            requests = [
                Request(
                    custom_id="item-{}".format(item),
                    params=MessageCreateParamsNonStreaming(
                        model=MODEL,
                        max_tokens=16000,
                        system=system,
                        output_config={"effort": args.effort, "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
                        messages=[{"role": "user", "content": "Item Information: {}".format(texts[item])}],
                    ),
                )
                for item in items[start:start + BATCH_LIMIT]
            ]
            try:
                batch = client.messages.batches.create(requests=requests)
            except anthropic.AuthenticationError:
                sys.exit("Anthropic credentials are missing or invalid: set API_KEY at the top of this file.")
            except anthropic.RateLimitError as e:
                sys.exit("Rate limited while creating the batch; run the same command again later. ({})".format(e.message))
            except anthropic.APIStatusError as e:
                sys.exit("The batch was not created ({}): {}".format(e.status_code, e.message))
            except anthropic.APIConnectionError as e:
                sys.exit("Could not reach the API: {}".format(e))
            batch_ids.append(batch.id)
            with open(state_path, 'w', encoding='utf-8') as f:
                json.dump(batch_ids, f)
            print("submitted batch {} ({} items)".format(batch.id, len(requests)))

    failed = 0
    for batch_id in batch_ids:
        while True:
            try:
                batch = client.messages.batches.retrieve(batch_id)
            except anthropic.APIConnectionError:
                time.sleep(args.poll)
                continue
            if batch.processing_status == 'ended':
                break
            counts = batch.request_counts
            print("  {}: {} processing, {} succeeded, {} errored".format(batch_id, counts.processing, counts.succeeded, counts.errored))
            time.sleep(args.poll)
        failed += _collect(client, batch_id, answers_path)
    os.remove(state_path)
    return failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--item_text', type=str, default=None)
    parser.add_argument('--domain', type=str, default='a recommendation dataset')
    parser.add_argument('--effort', type=str, default='low', choices=['low', 'medium', 'high', 'xhigh', 'max'])
    parser.add_argument('--out_dir', type=str, default='hyperllm')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--poll', type=int, default=60)
    parser.add_argument('--build_only', action='store_true')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    data_dir = os.path.join(REPO_DIR, 'data', args.dataset)
    with open(os.path.join(data_dir, 'trn_mat.pkl'), 'rb') as f:
        item_num = pickle.load(f).shape[1]
    out_dir = os.path.join(data_dir, args.out_dir)
    raw_dir = os.path.join(out_dir, 'raw_trial' if args.limit else 'raw')
    answers_path = os.path.join(raw_dir, 'answers.jsonl')

    if not args.build_only:
        if args.item_text is None:
            args.item_text = os.path.join(data_dir, '{}_item_profile.jsonl'.format(args.dataset))
            if not os.path.exists(args.item_text):
                parser.error('--item_text is required ({} does not exist)'.format(args.item_text))
        texts = load_item_texts(args.item_text, item_num)
        answered = {item for item, answer in _read_answers(answers_path).items()
                    if any(answer.get(level) for level in ('level_1', 'level_2', 'level_3'))}
        wanted = range(args.limit if args.limit else item_num)
        empty = [item for item in wanted if not texts[item]]
        items = [item for item in wanted if texts[item] and item not in answered]
        print("{}: {} items, {} already answered, {} without text, {} to ask".format(
            args.dataset, item_num, len(answered), len(empty), len(items)))
        if args.dry_run:
            print("model {}, effort {}, {} characters of item text in total\n\n[system]\n{}".format(
                MODEL, args.effort, sum(len(texts[item]) for item in items), SYSTEM_PROMPT.format(domain=args.domain)))
            for item in items[:3]:
                print("\n[user, item {}]\nItem Information: {}".format(item, texts[item]))
            return
        os.makedirs(raw_dir, exist_ok=True)
        if items or os.path.exists(os.path.join(raw_dir, 'batches.json')):
            failed = request_tags(texts, items, args, raw_dir)
            if failed:
                print("{} items got no answer; run the same command again to ask for them once more".format(failed))

    answers = _read_answers(answers_path)
    if args.limit:
        for item in sorted(answers)[:args.limit]:
            print(item, json.dumps(answers[item], ensure_ascii=False))
        return
    os.makedirs(out_dir, exist_ok=True)
    write_tag_files(build_tag_files(answers, item_num), out_dir, item_num)


if __name__ == '__main__':
    main()
