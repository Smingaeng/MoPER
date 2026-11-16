import argparse
import json
import math
import os
import pickle
import re
import sys
import time

import numpy as np
import scipy.sparse as sp

API_KEY = ""
MODEL = "claude-haiku-5-5"
REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BATCH_LIMIT = 50000
TEXT_FIELDS = ("profile", "summarization", "summary", "description", "title", "text")
DOMAINS = {"amazon_book": ("book", "books"), "amazon_movie": ("movie", "movies"), "yelp": ("business", "businesses")}

SYSTEM_PROMPT = """You are a collaborative filtering-based recommender system to help me determine which {items} a specific user might choose. I will provide you with:
- profile: a single sentence describing the user's {item} preferences (no more than 200 words);
- candidate set: a list of {items}, one per line in the format [ID] {item} description.
Based on this information, please recommend {count} new {items} for the user from the candidate set.
- Your final answer must be a list of {item} IDs, for example: [{example}].
- Please do not output intermediate content or other irrelevant information."""

USER_PROMPT = """profile: {profile}
candidate set:
{candidates}"""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"ids": {"type": "array", "items": {"type": "integer"}}},
    "required": ["ids"],
    "additionalProperties": False,
}


def unpack(entry):
    if isinstance(entry, dict) and isinstance(entry.get("response"), str):
        try:
            entry = json.loads(entry["response"])
        except ValueError:
            entry = re.sub(r'"(summarization|reasoning)"\s*:', " ", entry["response"])
            entry = re.sub(r'[{}"]', " ", entry).replace("\\'", "'").replace("\\n", " ")
    if isinstance(entry, str):
        return re.sub(r"\s+", " ", entry).strip(" ,")
    if isinstance(entry, dict):
        for field in TEXT_FIELDS:
            if entry.get(field):
                return re.sub(r"\s+", " ", str(entry[field])).strip()
    return ""


def load_texts(path, num):
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            raw = pickle.load(f)
    elif path.endswith(".jsonl"):
        raw = {}
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                if line.strip():
                    row = json.loads(line)
                    raw[row.get("user_id", row.get("item_id", row.get("id", line_no)))] = row
    else:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    if isinstance(raw, (list, tuple)):
        raw = dict(enumerate(raw))
    texts = [""] * num
    for key, entry in raw.items():
        idx = int(key)
        if not 0 <= idx < num:
            sys.exit(f"{path}: id {idx} is outside 0..{num - 1}")
        texts[idx] = unpack(entry)
    return texts


def default_path(data_dir, names):
    for name in names:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            return path
    return None


def pca(emb, dim):
    emb = np.asarray(emb, dtype=np.float64)
    centered = emb - emb.mean(axis=0)
    eigvec, _, _ = np.linalg.svd(np.cov(centered, rowvar=False), full_matrices=False)
    return centered @ eigvec[:, :dim]


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def similar_users(user_vec, k, chunk=2048):
    u = unit(user_vec)
    out = np.empty((len(u), k), dtype=np.int64)
    for start in range(0, len(u), chunk):
        sim = u[start:start + chunk] @ u.T
        rows = np.arange(sim.shape[0])
        sim[rows, start + rows] = -np.inf
        out[start:start + chunk] = np.argsort(-sim, axis=1, kind="stable")[:, :k]
    return out


def rank_candidates(trn, neighbors, item_vec, num):
    item_unit = unit(item_vec)
    ranked = []
    for u in range(trn.shape[0]):
        own = trn.indices[trn.indptr[u]:trn.indptr[u + 1]]
        counts = np.asarray(trn[neighbors[u]].sum(axis=0)).ravel()
        counts[own] = 0
        items = np.flatnonzero(counts)
        if len(items) == 0 or len(own) == 0:
            ranked.append(np.empty(0, dtype=np.int64))
            continue
        center = item_vec[own].mean(axis=0)
        center = center / max(np.linalg.norm(center), 1e-12)
        score = (item_unit[items] @ center) / np.sqrt(counts[items])
        ranked.append(items[np.argsort(-score, kind="stable")[:num]])
    return ranked


def budgets(trn):
    n = np.diff(trn.indptr)
    quartiles = np.quantile(n, [0.25, 0.5, 0.75])
    k = np.zeros(len(n), dtype=np.int64)
    for u, c in enumerate(n):
        above = [q for q in quartiles if c < q]
        if above:
            k[u] = max(0, math.floor(above[0]) - c + 1)
    return k, quartiles


def build_requests(users, ranked, counts, user_text, item_text, item, items):
    requests = {}
    for u in users:
        candidates = "\n".join(f"[{i}] {item_text[i]}" for i in ranked[u])
        example = ", ".join(str(i) for i in range(1, counts[u] + 1))
        system = SYSTEM_PROMPT.format(item=item, items=items, count=counts[u], example=example)
        requests[u] = (system, USER_PROMPT.format(profile=user_text[u], candidates=candidates))
    return requests


def select(ids, candidates, count):
    pool = [int(i) for i in candidates]
    allowed = set(pool)
    chosen = []
    for i in ids or []:
        if isinstance(i, int) and i in allowed and i not in chosen:
            chosen.append(i)
        if len(chosen) == count:
            break
    filled = 0
    for i in pool:
        if len(chosen) == count:
            break
        if i not in chosen:
            chosen.append(i)
            filled += 1
    return chosen, filled


def read_answers(path):
    answers = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    answers[row["user"]] = row["ids"]
    return answers


def request_ids(requests, args, raw_dir):
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = anthropic.Anthropic(api_key=API_KEY or None)
    answers_path = os.path.join(raw_dir, "answers.jsonl")
    state_path = os.path.join(raw_dir, "batches.json")
    if os.path.exists(state_path):
        with open(state_path, encoding="utf-8") as f:
            batch_ids = json.load(f)
        print(f"resuming the submitted batches {batch_ids}")
    else:
        users, batch_ids = list(requests), []
        for start in range(0, len(users), BATCH_LIMIT):
            batch_requests = [
                Request(
                    custom_id=f"user-{u}",
                    params=MessageCreateParamsNonStreaming(
                        model=MODEL,
                        max_tokens=4096,
                        system=requests[u][0],
                        output_config={"effort": args.effort, "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
                        messages=[{"role": "user", "content": requests[u][1]}],
                    ),
                )
                for u in users[start:start + BATCH_LIMIT]
            ]
            try:
                batch = client.messages.batches.create(requests=batch_requests)
            except anthropic.AuthenticationError:
                sys.exit("Anthropic credentials are missing or invalid: set API_KEY at the top of this file.")
            except anthropic.RateLimitError as e:
                sys.exit(f"Rate limited while creating the batch; run the same command again later. ({e.message})")
            except anthropic.APIStatusError as e:
                sys.exit(f"The batch was not created ({e.status_code}): {e.message}")
            except anthropic.APIConnectionError as e:
                sys.exit(f"Could not reach the API: {e}")
            batch_ids.append(batch.id)
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(batch_ids, f)
            print(f"submitted batch {batch.id} ({len(batch_requests)} users)")

    failed, tokens_in, tokens_out = 0, 0, 0
    for batch_id in batch_ids:
        while True:
            try:
                batch = client.messages.batches.retrieve(batch_id)
            except anthropic.APIConnectionError:
                time.sleep(args.poll)
                continue
            if batch.processing_status == "ended":
                break
            c = batch.request_counts
            print(f"  {batch_id}: {c.processing} processing, {c.succeeded} succeeded, {c.errored} errored")
            time.sleep(args.poll)
        with open(answers_path, "a", encoding="utf-8") as f:
            for result in client.messages.batches.results(batch_id):
                user = int(result.custom_id.split("-")[1])
                if result.result.type != "succeeded":
                    failed += 1
                    print(f"  user {user}: {result.result.type}")
                    continue
                message = result.result.message
                tokens_in += message.usage.input_tokens
                tokens_out += message.usage.output_tokens
                text = next((block.text for block in message.content if block.type == "text"), "")
                try:
                    ids = json.loads(text)["ids"]
                except (ValueError, KeyError, TypeError):
                    ids = None
                if message.stop_reason != "end_turn" or not isinstance(ids, list):
                    failed += 1
                    print(f"  user {user}: stop_reason {message.stop_reason}")
                    continue
                f.write(json.dumps({"user": user, "ids": ids}) + "\n")
    print(f"  {tokens_in} input tokens, {tokens_out} output tokens")
    os.remove(state_path)
    return failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--user_profile", type=str, default=None)
    parser.add_argument("--item_profile", type=str, default=None)
    parser.add_argument("--domain", type=str, nargs=2, default=None, metavar=("ITEM", "ITEMS"))
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--neighbors", type=int, default=100)
    parser.add_argument("--num_candidates", type=int, default=20)
    parser.add_argument("--effort", type=str, default="low", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--work_dir", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--poll", type=int, default=60)
    parser.add_argument("--build_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    data_dir = os.path.join(REPO_DIR, "data", args.dataset)
    with open(os.path.join(data_dir, "trn_mat.pkl"), "rb") as f:
        trn = sp.csr_matrix((pickle.load(f) != 0).astype(np.float32))
    trn.sort_indices()
    user_num, item_num = trn.shape
    with open(os.path.join(data_dir, "usr_emb_np.pkl"), "rb") as f:
        user_emb = pickle.load(f)
    with open(os.path.join(data_dir, "itm_emb_np.pkl"), "rb") as f:
        item_emb = pickle.load(f)
    if len(user_emb) != user_num or len(item_emb) != item_num:
        sys.exit("usr_emb_np.pkl / itm_emb_np.pkl do not match trn_mat.pkl")
    item, items = tuple(args.domain) if args.domain else DOMAINS.get(args.dataset, ("item", "items"))
    work_dir = args.work_dir or os.path.join(data_dir, "promax")
    raw_dir = os.path.join(work_dir, "raw_trial" if args.limit else "raw")
    answers_path = os.path.join(raw_dir, "answers.jsonl")
    out_path = args.out or os.path.join(data_dir, "new_trn_rag_mat.pkl")
    if not (args.dry_run or args.limit) and os.path.exists(out_path) and not args.overwrite:
        sys.exit(f"{out_path} exists; pass --overwrite to replace it or --out to write elsewhere")

    k, quartiles = budgets(trn)
    neighbors = similar_users(pca(user_emb, args.dim), args.neighbors)
    ranked = rank_candidates(trn, neighbors, pca(item_emb, args.dim), args.num_candidates)
    counts = np.array([min(int(k[u]), len(ranked[u])) for u in range(user_num)])
    users = [u for u in range(user_num) if counts[u] > 0]
    if args.limit:
        users = users[:args.limit]
    print(f"{args.dataset}: {user_num} users, {item_num} items, {trn.nnz} training interactions; quartiles of the "
          f"interaction counts {quartiles.tolist()}; {len(users)} users get {int(counts[users].sum())} new interactions")

    if not args.build_only:
        user_path = args.user_profile or default_path(data_dir, ("usr_prf.pkl", f"{args.dataset}_user_profile.jsonl"))
        item_path = args.item_profile or default_path(data_dir, (f"{args.dataset}_item_profile.jsonl", "itm_prf.pkl"))
        if user_path is None or item_path is None:
            sys.exit("user / item profile texts not found: pass --user_profile and --item_profile")
        requests = build_requests(users, ranked, counts, load_texts(user_path, user_num), load_texts(item_path, item_num), item, items)
        answered = read_answers(answers_path)
        todo = {u: r for u, r in requests.items() if u not in answered}
        if args.dry_run:
            print(f"model {MODEL}, effort {args.effort}, {len(todo)} requests, "
                  f"{sum(len(s) + len(m) for s, m in todo.values())} characters")
            for u in list(todo)[:2]:
                print(f"\n[system, user {u}]\n{todo[u][0]}\n[user]\n{todo[u][1]}")
            return
        os.makedirs(raw_dir, exist_ok=True)
        if todo or os.path.exists(os.path.join(raw_dir, "batches.json")):
            failed = request_ids(todo, args, raw_dir)
            if failed:
                print(f"{failed} users got no answer; run the same command again to ask for them once more")

    answers = read_answers(answers_path)
    rows, cols, filled, missing = [], [], 0, 0
    for u in users:
        if u not in answers:
            missing += 1
        chosen, n_filled = select(answers.get(u), ranked[u], int(counts[u]))
        filled += n_filled
        rows += [u] * len(chosen)
        cols += chosen
        if args.limit:
            print(u, chosen)
    print(f"{len(users) - missing} users answered, {missing} without answer; {filled} of {len(cols)} new interactions "
          f"taken from the ranking because the answer was missing or short")
    if args.limit:
        return
    added = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=trn.shape)
    mat = ((trn + added) > 0).astype(np.float32).tocoo()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(mat, f)
    print(f"{out_path}: {mat.nnz} interactions ({trn.nnz} training + {mat.nnz - trn.nnz} new)")


if __name__ == "__main__":
    main()
