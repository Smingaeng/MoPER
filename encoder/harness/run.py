#!/home/kyung/miniconda3/envs/mosaic/bin/python
"""PERSONA-MOSAIC experiment harness.

One entry point for launching runs, turning their logs into ledger records, and reporting against the
project goals.  Read harness/CLAUDE.md for the rules; this file only enforces the machine-checkable part.

    harness/run.py launch   --model persona_mosaic --dataset amazon_book --override k=v ... --label "..."
    harness/run.py launch   --queue q003                      # take model/dataset/overrides/hypothesis from queue.yaml
    harness/run.py collect                                     # parse finished runs + import logs not yet in the ledger
    harness/run.py status                                      # running jobs, GPUs, queue
    harness/run.py report   [--dataset amazon_book] [--all]    # table vs references, goal checks, noise floor
    harness/run.py show ID | diag ID | note ID --file f.md | kill ID [--delete-log] | queue | validate ...

Ledger ids look like persona_mosaic-amazon_book-1030-031033 (model-dataset-MMDD-HHMMSS, launch or log time).
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

HARNESS_DIR = Path(__file__).resolve().parent
ENCODER_DIR = HARNESS_DIR.parent
LOG_DIR = ENCODER_DIR / "save" / "log"
CONF_DIR = ENCODER_DIR / "config" / "modelconf"
LEDGER = HARNESS_DIR / "ledger.jsonl"
QUEUE = HARNESS_DIR / "queue.yaml"
CONSTRAINTS = HARNESS_DIR / "constraints.yaml"
REFERENCES = HARNESS_DIR / "references.yaml"
RUNS_DIR = HARNESS_DIR / "runs"
STATE = HARNESS_DIR / "state.json"
PYTHON = sys.executable
GOAL_METRICS = ["recall@5", "recall@10", "ndcg@5", "ndcg@10"]
ALL_METRICS = ["recall@5", "recall@10", "recall@20", "ndcg@5", "ndcg@10", "ndcg@20"]
SELECTION_METRIC = "recall@20"  # the shared trainer early-stops / selects on validation recall@20 (unchanged)
LAUNCH_GAP_S = 3.0  # two runs started in the same second share one log file name

TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - (.*)$")
LOGNAME_RE = re.compile(r"^(?P<dataset>.+)_(?P<stamp>[A-Z][a-z]{2}-\d{2}-\d{4}_\d{2}-\d{2}-\d{2})\.log$")
VAL_RE = re.compile(r"Epoch\s+(\d+) Validation set (.*)")
METRIC_RE = re.compile(r"(\w+)@(\d+): ([\d.]+)")
FINAL_RE = re.compile(r"Best Epoch (\d+)\. Final test result: (\{.*\})\.")
ARRAY_RE = re.compile(r"'(\w+)': array\(\[([^\]]*)\]\)")
SAVE_RE = re.compile(r"Save model parameters to (.*)$")


# ----------------------------------------------------------------------------- small utils
def now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def load_yaml(path: Path, default=None):
    if not path.exists():
        return default
    with open(path) as f:
        return yaml.safe_load(f) or default


def load_ledger() -> list[dict]:
    if not LEDGER.exists():
        return []
    out = []
    with open(LEDGER) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def save_ledger(records: list[dict]) -> None:
    tmp = LEDGER.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(LEDGER)


def save_queue(q: dict) -> None:
    """Rewrite queue.yaml keeping its leading comment block."""
    header = []
    if QUEUE.exists():
        for line in QUEUE.read_text().splitlines():
            if line.startswith("#"):
                header.append(line)
            else:
                break
    body = yaml.safe_dump(q, allow_unicode=True, sort_keys=False, width=110)
    QUEUE.write_text("\n".join(header) + ("\n" if header else "") + body)


def find_record(records: list[dict], ident: str) -> dict:
    hits = [r for r in records if r["id"] == ident or r["id"].endswith(ident)]
    if not hits:
        die(f"no ledger record matches {ident!r}")
    if len(hits) > 1:
        die(f"{ident!r} is ambiguous: " + ", ".join(r["id"] for r in hits))
    return hits[0]


def fmt_value(v) -> str:
    """Override value as the configurator's yaml.safe_load will read it back (PyYAML needs 1.0e-3, not 1e-3)."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, float):
        s = f"{v:.12g}"
        if "e" in s or "E" in s:
            s = f"{v:.15f}".rstrip("0").rstrip(".")
        if "." not in s:
            s += ".0"
        return s
    return str(v)


def parse_overrides(items: list[str]) -> dict:
    out = {}
    for item in items or []:
        if "=" not in item:
            die(f"override {item!r} is not key=value")
        k, v = item.split("=", 1)
        out[k.strip()] = yaml.safe_load(v)
    return out


def overrides_text(ov: dict) -> str:
    return " ".join(f"{k}={fmt_value(v)}" for k, v in ov.items())


def dup_key(model: str, dataset: str, overrides: dict, embedding_size=None, summary: str | None = None) -> str:
    """Identity of a configuration: overrides + embedding width + the model's own build summary (persona_mosaic
    prints the clustering cache, item text source, gate mode ...), so runs made from different data files or code
    versions are not mistaken for repeats."""
    fp = re.sub(r";\s*CONTROL:.*$", "", summary or "")
    return json.dumps([model, dataset, overrides, embedding_size, fp], sort_keys=True)


def stamp_to_id(stamp: str) -> str:
    d = dt.datetime.strptime(stamp, "%b-%d-%Y_%H-%M-%S")
    return d.strftime("%m%d-%H%M%S")


def model_defaults(model: str, dataset: str) -> tuple[dict, dict]:
    conf = load_yaml(CONF_DIR / f"{model}.yml", {}) or {}
    m = conf.get("model", {}) or {}
    top = {k: v for k, v in m.items() if not isinstance(v, dict)}
    block = dict(m.get(dataset, {}) or {})
    return top, block


# ----------------------------------------------------------------------------- log parsing
def parse_model_line(text: str, dataset: str) -> tuple[dict, dict, str | None]:
    head = text.split("{", 1)[0]
    top = {}
    for m in re.finditer(r"(\w+):([^,]*)", head):
        try:
            top[m.group(1)] = yaml.safe_load(m.group(2).strip())
        except Exception:
            top[m.group(1)] = m.group(2).strip()
    block = {}
    m = re.search(re.escape(dataset) + r":(\{.*?\})", text)
    if m:
        try:
            block = ast.literal_eval(m.group(1))
        except Exception:
            block = {}
    summary = None
    for key in ("persona_clusters", "llm_prior_note"):
        i = text.find(key + ":")
        if i >= 0:
            summary = text[i + len(key) + 1 :].strip()
            break
    return top, block, summary


def parse_diag_final(lines: list[str]) -> dict:
    """Numbers from the end-of-run [diag] block (persona_mosaic)."""
    out: dict = {}
    for line in lines:
        if "[diag] gate" in line:
            node = "user" if "gate user" in line else "item"
            g: dict = {}
            m = re.search(r"top-1 share=\[([^\]]*)\]", line)
            if m:
                g["top1_share"] = [float(x) for x in m.group(1).split(",")]
            m = re.search(r"norm\. entropy=([\d.]+)", line)
            if m:
                g["entropy"] = float(m.group(1))
            m = re.search(r"balance=([\d.]+)", line)
            if m:
                g["balance"] = float(m.group(1))
            m = re.search(r"cluster-sim=([\d.]+), topology=([\d.]+)", line)
            if m:
                g["tv_zero_cluster_sim"], g["tv_zero_topology"] = float(m.group(1)), float(m.group(2))
            m = re.search(r"gate \w+ \[(\w+)\]", line)
            if m:
                g["mode"] = m.group(1)
            g["warmup_forced"] = "forced uniform" in line
            out.setdefault("gate", {})[node] = g
        elif "[diag] expert" in line:
            m = re.search(r"expert (\d+): injection=(\w+); \|E_free\| user row-norm=([\d.]+), \|prior\| user row-norm=([\d.]+) \(item ([\d.]+) / ([\d.]+)\)", line)
            if m:
                e = {
                    "injection": m.group(2),
                    "e_free_user": float(m.group(3)),
                    "prior_user": float(m.group(4)),
                    "e_free_item": float(m.group(5)),
                    "prior_item": float(m.group(6)),
                }
                mm = re.search(r"layer alpha=\[([^\]]*)\]", line)
                if mm:
                    e["layer_alpha"] = [float(x) for x in mm.group(1).split(",")]
                mm = re.search(r"std over nodes=\[([^\]]*)\]", line)
                if mm:
                    e["mod_std_over_nodes"] = [float(x) for x in mm.group(1).split(",")]
                out.setdefault("experts", {})[m.group(1)] = e
        elif "[diag] persona swap" in line:
            node = "users" if "(users" in line else "items"
            m = re.search(r"top-20 overlap=([\d.]+), \w+ embedding cos=([\d.]+)", line)
            if m:
                out.setdefault("swap", {})[node] = {"top20_overlap": float(m.group(1)), "emb_cos": float(m.group(2))}
        elif "[diag] text score" in line:
            m = re.search(r"beta=([-\d.]+); mean\|beta\*T_u\.T_i\|=([\d.]+) vs mean\|e_u\.e_i\|=([\d.]+).*corr\(text, graph\)=([-\d.]+)", line)
            if m:
                out["text_score"] = {"beta": float(m.group(1)), "text_mag": float(m.group(2)), "graph_mag": float(m.group(3)), "corr": float(m.group(4))}
    gates = out.get("gate", {})
    if gates:
        out["gate_collapsed"] = all(max(g.get("top1_share", [0])) >= 0.99 for g in gates.values() if g.get("top1_share"))
    return out


def parse_log(path: Path) -> dict:
    model = path.parent.name
    m = LOGNAME_RE.match(path.name)
    if not m:
        raise ValueError(f"unexpected log name {path.name}")
    dataset, stamp = m.group("dataset"), m.group("stamp")
    rec: dict = {
        "model": model,
        "dataset": dataset,
        "log_path": str(path.relative_to(ENCODER_DIR)),
        "log_stamp": stamp,
    }
    first_ts = last_ts = None
    curve: dict = {"epoch": []}
    diag_lines: list[dict] = []
    final_block: list[str] = []
    in_final = False
    last_epoch = None
    with open(path, errors="replace") as f:
        for raw in f:
            mm = TS_RE.match(raw.rstrip("\n"))
            if not mm:
                continue
            ts, text = mm.group(1), mm.group(2)
            first_ts = first_ts or ts
            last_ts = ts
            if text.startswith("Train|"):
                t = {}
                for kv in re.finditer(r"(\w+):([^,]*)", text[len("Train|") :]):
                    t[kv.group(1)] = kv.group(2).strip()
                tagm = re.search(r"tag:(.*)$", text)
                rec["tag"] = tagm.group(1).strip() if tagm else ""
                rec["seed"] = yaml.safe_load(t.get("seed", "null"))
                rec["patience"] = yaml.safe_load(t.get("patience", "null"))
                rec["test_step"] = yaml.safe_load(t.get("test_step", "null"))
            elif text.startswith("Model|"):
                top, block, summary = parse_model_line(text[len("Model|") :].strip(), dataset)
                rec["embedding_size"] = block.get("embedding_size", top.get("embedding_size"))
                rec["config"] = block
                if summary:
                    rec["summary"] = summary
            elif text.startswith("Test--"):
                in_final = True
            elif "[diag]" in text:
                entry = {"epoch": "final" if in_final else last_epoch, "text": text}
                diag_lines.append(entry)
                if in_final:
                    final_block.append(text)
            else:
                vm = VAL_RE.search(text)
                if vm:
                    ep = int(vm.group(1))
                    last_epoch = ep
                    curve["epoch"].append(ep)
                    for met, k, val in METRIC_RE.findall(vm.group(2)):
                        curve.setdefault(f"{met}@{k}", []).append(float(val))
                    continue
                fm = FINAL_RE.search(text)
                if fm:
                    rec["best_epoch"] = int(fm.group(1))
                    test = {}
                    for met, arr in ARRAY_RE.findall(fm.group(2)):
                        vals = [float(x) for x in arr.replace("\n", " ").split(",") if x.strip()]
                        ks = [5, 10, 20][: len(vals)]
                        for k, v in zip(ks, vals):
                            test[f"{met}@{k}"] = v
                    rec["test"] = test
                sm = SAVE_RE.search(text)
                if sm:
                    rec["checkpoint"] = sm.group(1).strip()
    rec["started_at"] = first_ts
    rec["last_log_at"] = last_ts
    if first_ts and last_ts:
        d = dt.datetime.strptime(last_ts, "%Y-%m-%d %H:%M:%S") - dt.datetime.strptime(first_ts, "%Y-%m-%d %H:%M:%S")
        rec["minutes"] = round(d.total_seconds() / 60, 1)
    rec["epochs_run"] = (last_epoch + 1) if last_epoch is not None else 0
    rec["val_curve"] = curve
    if curve["epoch"] and SELECTION_METRIC in curve:
        i = max(range(len(curve["epoch"])), key=lambda j: curve[SELECTION_METRIC][j])
        rec["val_best"] = {k: v[i] for k, v in curve.items() if k != "epoch"}
        rec["val_best"]["epoch"] = curve["epoch"][i]
    rec["diag_lines"] = diag_lines
    rec["diag_final"] = parse_diag_final(final_block) if final_block else {}
    # overrides = dataset block minus the current yml defaults (top-level merged with the dataset block)
    top_def, block_def = model_defaults(model, dataset)
    defaults = {**top_def, **block_def}
    rec["overrides"] = {k: v for k, v in rec.get("config", {}).items() if k not in defaults or defaults[k] != v}
    cons = load_yaml(CONSTRAINTS, {}) or {}
    ctrl_keys = set(cons.get("control_keys", []))
    rec["control"] = any(k in ctrl_keys and v not in (None, False, "none") for k, v in rec["overrides"].items())
    rec["dup_key"] = dup_key(model, dataset, rec["overrides"], rec.get("embedding_size"), rec.get("summary"))
    rec["status"] = "done" if "test" in rec else "incomplete"
    return rec


# ----------------------------------------------------------------------------- constraints
def validate(model: str, dataset: str, overrides: dict, allow: list[str]) -> list[str]:
    """Return a list of violations (empty = ok). `allow` names rule keys the caller consciously overrides."""
    cons = load_yaml(CONSTRAINTS, {}) or {}
    allow = set(allow or [])
    problems = []
    if model not in cons.get("launchable_models", []) and "baseline" not in allow:
        problems.append(f"model {model!r} is not launchable (baselines are analysis-only; use --allow baseline for a reference run)")
    top_def, block_def = model_defaults(model, dataset)
    known = set(top_def) | set(block_def) | set(cons.get("extra_known_keys", {}).get(model, []))
    for k, v in overrides.items():
        if k in cons.get("forbidden_override_keys", {}) and k not in allow:
            problems.append(f"{k}: {cons['forbidden_override_keys'][k]} (override with --allow {k} only with a written reason)")
        if known and k not in known and "new-key" not in allow:
            problems.append(f"{k}: not a key of config/modelconf/{model}.yml (typo? --allow new-key if the knob is new)")
        rule = (cons.get("value_rules") or {}).get(k)
        if rule and k not in allow:
            if "enum" in rule and v not in rule["enum"]:
                problems.append(f"{k}={v!r}: must be one of {rule['enum']}")
            if "min" in rule and isinstance(v, (int, float)) and not isinstance(v, bool) and v < rule["min"]:
                problems.append(f"{k}={v}: below minimum {rule['min']} ({rule.get('why', '')})")
            if "max" in rule and isinstance(v, (int, float)) and not isinstance(v, bool) and v > rule["max"]:
                problems.append(f"{k}={v}: above maximum {rule['max']} ({rule.get('why', '')})")
            if "max_exclusive" in rule and isinstance(v, (int, float)) and not isinstance(v, bool) and v >= rule["max_exclusive"]:
                problems.append(f"{k}={v}: must be below {rule['max_exclusive']} ({rule.get('why', '')})")
            if "min_exclusive" in rule and isinstance(v, (int, float)) and not isinstance(v, bool) and v <= rule["min_exclusive"]:
                problems.append(f"{k}={v}: must be above {rule['min_exclusive']} ({rule.get('why', '')})")
    return problems


# ----------------------------------------------------------------------------- processes / GPUs
def pid_alive(pid: int | None, ident: str) -> bool:
    if not pid:
        return False
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes()
    except FileNotFoundError:
        return False
    return b"train_encoder.py" in cmd and ident.encode() in cmd


def gpu_table() -> list[dict]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        ).stdout
    except Exception:
        return [{"index": 0, "mem_used": 0, "mem_total": 0, "util": 0}]
    rows = []
    for line in out.strip().splitlines():
        idx, used, total, util = [x.strip() for x in line.split(",")]
        rows.append({"index": int(idx), "mem_used": int(used), "mem_total": int(total), "util": int(util)})
    return rows or [{"index": 0, "mem_used": 0, "mem_total": 0, "util": 0}]


def running_jobs(records: list[dict]) -> list[dict]:
    return [r for r in records if r.get("status") == "running"]


def pick_gpu(records: list[dict], forced: int | None) -> int:
    if forced is not None:
        return forced
    gpus = gpu_table()
    jobs = running_jobs(records)
    load = {g["index"]: 0 for g in gpus}
    for j in jobs:
        if j.get("gpu") in load:
            load[j["gpu"]] += 1
    # fewest harness jobs, then least memory used
    return min(gpus, key=lambda g: (load[g["index"]], g["mem_used"]))["index"]


def stagger() -> None:
    state = load_yaml(STATE, {}) if STATE.exists() else {}
    try:
        state = json.loads(STATE.read_text()) if STATE.exists() else {}
    except Exception:
        state = {}
    last = float(state.get("last_launch", 0))
    wait = LAUNCH_GAP_S - (time.time() - last)
    if wait > 0:
        time.sleep(wait)
    state["last_launch"] = time.time()
    STATE.write_text(json.dumps(state))


def find_log_for(rec: dict) -> Path | None:
    """The log whose Train| tag carries '@<id>' (harness runs) and that started after the launch."""
    d = LOG_DIR / rec["model"]
    if not d.exists():
        return None
    needle = f"@{rec['id']}"
    launched = dt.datetime.strptime(rec["launched_at"], "%Y-%m-%d %H:%M:%S").timestamp() - 5
    for p in sorted(d.glob(f"{rec['dataset']}_*.log"), key=lambda p: p.stat().st_mtime, reverse=True):
        if p.stat().st_mtime < launched:
            continue
        try:
            with open(p) as f:
                head = f.read(4000)
        except Exception:
            continue
        if needle in head:
            return p
    return None


# ----------------------------------------------------------------------------- commands
def cmd_validate(args) -> None:
    ov = parse_overrides(args.override)
    problems = validate(args.model, args.dataset, ov, args.allow)
    if problems:
        print("REJECTED:")
        for p in problems:
            print("  -", p)
        sys.exit(2)
    print(f"ok: {args.model} {args.dataset} {overrides_text(ov) or '(defaults)'}")


def cmd_launch(args) -> None:
    records = load_ledger()
    queue_entry = None
    if args.queue:
        q = load_yaml(QUEUE, {}) or {}
        entries = q.get("experiments", []) or []
        hits = [e for e in entries if e.get("id") == args.queue]
        if not hits:
            die(f"queue entry {args.queue!r} not found")
        queue_entry = hits[0]
        if queue_entry.get("status") not in ("ready",):
            die(f"queue entry {args.queue} has status {queue_entry.get('status')!r}; only 'ready' entries can be launched")
        tier = str(queue_entry.get("tier", "A")).upper()
        auto = (load_yaml(CONSTRAINTS, {}) or {}).get("auto_approve", {}) or {}
        if tier != "A" and not args.approved and not auto.get(tier, False):
            die(f"queue entry {args.queue} is tier {tier}: needs the user's approval (--approved after they said yes, "
                f"or auto_approve.{tier}: true in constraints.yaml)")
        args.model = queue_entry.get("model", args.model)
        args.dataset = queue_entry.get("dataset", args.dataset)
        if queue_entry.get("repeat"):
            args.repeat = True
        args.allow = list(args.allow or []) + list(queue_entry.get("allow", []) or [])
        args.reason = args.reason or queue_entry.get("reason", "")
        ov = dict(queue_entry.get("overrides", {}) or {})
        label = queue_entry.get("label") or args.queue
        hypothesis = queue_entry.get("hypothesis", "")
        prediction = queue_entry.get("prediction", "")
        decision = queue_entry.get("decision_rule", "")
        contribution = queue_entry.get("contribution", [])
    else:
        ov = parse_overrides(args.override)
        label = args.label or overrides_text(ov) or "default"
        hypothesis, prediction, decision, contribution = args.hypothesis, args.prediction, args.decision, []
    if not args.model or not args.dataset:
        die("--model and --dataset are required (or --queue)")
    if not args.smoke and not hypothesis and not args.no_hypothesis:
        die("a run needs --hypothesis/--prediction/--decision (or --no-hypothesis for a reference / repeat run)")
    cons_lock = load_yaml(CONSTRAINTS, {}) or {}
    if cons_lock.get("phase_lock") and args.model == "persona_mosaic" and not args.smoke and "phase" not in (args.allow or []):
        achieved = cons_lock.get("achieved") or {}
        current = None
        for ph in cons_lock.get("phases") or []:
            if any(g not in (achieved.get(d) or {}) for d in ph.get("datasets", []) for g in ph.get("goals", [])):
                current = ph
                break
        if current is not None and args.dataset not in current.get("datasets", []):
            die(f"phase lock: current phase {current.get('name')} covers {current.get('datasets')}; "
                f"{args.dataset} waits until that phase is achieved (--allow phase only if the user says so)")
    problems = validate(args.model, args.dataset, ov, args.allow)
    if problems:
        print("REJECTED by harness/constraints.yaml:")
        for p in problems:
            print("  -", p)
        sys.exit(2)
    key = json.dumps([args.model, args.dataset, ov], sort_keys=True)
    dup = [r for r in records if json.dumps([r.get("model"), r.get("dataset"), r.get("overrides", {})], sort_keys=True) == key
           and r.get("status") in ("running", "done")]
    if dup and not args.repeat and not args.smoke:
        die(f"same config already in the ledger ({', '.join(r['id'] for r in dup)}); use --repeat to measure the nondeterminism floor")
    cons_all = load_yaml(CONSTRAINTS, {}) or {}
    need = (cons_all.get("gpu_min_free_mib") or {})
    need = int(need.get(args.dataset, need.get("default", 0)) or 0)
    if args.gpu is not None:
        gpu = args.gpu
    else:
        gpus = gpu_table()
        jobs = running_jobs(records)
        load = {g["index"]: len([j for j in jobs if j.get("gpu") == g["index"]]) for g in gpus}
        # jobs launched in the last 5 minutes may not have allocated their memory yet: reserve their expected need
        needs = cons_all.get("gpu_min_free_mib") or {}
        now_ts = time.time()
        for g in gpus:
            reserved = 0
            for j in jobs:
                if j.get("gpu") != g["index"] or not j.get("launched_at"):
                    continue
                age = now_ts - dt.datetime.strptime(j["launched_at"], "%Y-%m-%d %H:%M:%S").timestamp()
                if age < 300:
                    reserved += int(needs.get(j.get("dataset"), needs.get("default", 0)) or 0)
            g["mem_used"] = min(g["mem_total"], g["mem_used"] + reserved)
        ok = [g for g in gpus if g["mem_total"] - g["mem_used"] >= need]
        if not ok and need and not args.ignore_memory:
            free = ", ".join(f"gpu {g['index']}: {g['mem_total'] - g['mem_used']} MiB free" for g in gpus)
            print(f"deferred: {args.model} {args.dataset} needs {need} MiB free on one GPU ({free}); queue entry stays ready")
            sys.exit(3)
        pool = ok or gpus
        gpu = min(pool, key=lambda g: (load[g["index"]], -(g["mem_total"] - g["mem_used"])))["index"]
    stagger()
    ident = f"{args.model}-{args.dataset}-{dt.datetime.now().strftime('%m%d-%H%M%S')}"
    tag = f"{label} @{ident}"
    cmd = [PYTHON, "train_encoder.py", "--model", args.model, "--dataset", args.dataset, "--cuda", str(gpu)]
    if ov:
        cmd += ["--override"] + [f"{k}={fmt_value(v)}" for k, v in ov.items()]
    cmd += ["--tag", tag]
    if args.smoke:
        cmd = ["timeout", f"{int(args.smoke * 60)}s"] + cmd
    RUNS_DIR.mkdir(exist_ok=True)
    out_path = RUNS_DIR / f"{ident}.out"
    with open(out_path, "w") as out:
        out.write("$ " + " ".join(cmd) + "\n")
        out.flush()
        proc = subprocess.Popen(cmd, cwd=ENCODER_DIR, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    dev = structure_deviation({"model": args.model, "dataset": args.dataset, "overrides": ov}, cons_all)
    if dev:
        print(f"note: structural exploration ({overrides_text(dev)}) - never a goal candidate until adopted in shared_structure on all datasets")
    rec = {
        "id": ident,
        "model": args.model,
        "dataset": args.dataset,
        "label": label,
        "overrides": ov,
        "structure_deviation": dev,
        "source": "harness",
        "smoke_minutes": args.smoke or None,
        "hypothesis": hypothesis,
        "prediction": prediction,
        "decision_rule": decision,
        "contribution": contribution,
        "queue_id": args.queue,
        "allow": args.allow or [],
        "reason": args.reason or "",
        "launched_at": now(),
        "gpu": gpu,
        "pid": proc.pid,
        "out_path": str(out_path.relative_to(ENCODER_DIR)),
        "status": "running",
    }
    records.append(rec)
    save_ledger(records)
    if queue_entry is not None:
        queue_entry["status"] = "launched"
        queue_entry["ledger_id"] = ident
        save_queue(q)
    print(f"launched {ident} on GPU {gpu} (pid {proc.pid}): {args.model} {args.dataset} {overrides_text(ov) or '(defaults)'}")
    print(f"  out: {out_path}")
    if args.wait:
        while pid_alive(proc.pid, ident):
            time.sleep(60)
        cmd_collect(argparse.Namespace(quiet=False))


def finalize_running(records: list[dict], quiet: bool) -> list[dict]:
    finished = []
    for rec in running_jobs(records):
        alive = pid_alive(rec.get("pid"), rec["id"])
        log = Path(ENCODER_DIR / rec["log_path"]) if rec.get("log_path") else find_log_for(rec)
        if log is not None and log.exists():
            parsed = parse_log(log)
            keep = {k: rec[k] for k in ("id", "label", "overrides", "source", "smoke_minutes", "hypothesis", "prediction",
                                         "decision_rule", "contribution", "queue_id", "allow", "reason", "launched_at", "gpu", "pid",
                                         "out_path", "analysis", "notes") if k in rec}
            rec.clear()
            rec.update(parsed)
            rec.update(keep)
            rec["label"] = keep.get("label") or rec.get("tag", "")
        if alive:
            rec["status"] = "running"
            continue
        # process gone
        if rec.get("smoke_minutes"):
            rec["status"] = "smoke-finished"
            finished.append(rec)
            continue
        if rec.get("test"):
            rec["status"] = "done"
        else:
            rec["status"] = "failed"
            out = ENCODER_DIR / rec.get("out_path", "")
            if out.exists():
                tail = out.read_text(errors="replace").splitlines()[-8:]
                rec["failure_tail"] = tail
        rec["finished_at"] = now()
        finished.append(rec)
        if not quiet:
            print(f"finished: {rec['id']} -> {rec['status']}")
    return finished


def import_logs(records: list[dict], quiet: bool) -> list[dict]:
    known = {r.get("log_path") for r in records if r.get("log_path")}
    added = []
    for model_dir in sorted(LOG_DIR.glob("*")):
        if not model_dir.is_dir():
            continue
        for log in sorted(model_dir.glob("*.log")):
            rel = str(log.relative_to(ENCODER_DIR))
            if rel in known:
                continue
            try:
                parsed = parse_log(log)
            except Exception as e:  # noqa: BLE001
                print(f"skip {rel}: {e}")
                continue
            tag = parsed.get("tag", "")
            if "@" in tag and re.search(r"@\w+-\w+-\d{4}-\d{6}$", tag):
                # a harness run whose record is still 'running' and not yet linked: leave it to finalize_running
                ident = tag.rsplit("@", 1)[1]
                owner = [r for r in records if r["id"] == ident]
                if owner:
                    owner[0]["log_path"] = rel
                    continue
            parsed["id"] = f"{parsed['model']}-{parsed['dataset']}-{stamp_to_id(parsed['log_stamp'])}"
            parsed["label"] = tag
            parsed["source"] = "manual"
            if parsed["status"] == "incomplete":
                # a log still growing (started outside the harness) is a running manual job
                age = time.time() - log.stat().st_mtime
                parsed["status"] = "running-manual" if age < 900 else "incomplete"
            records.append(parsed)
            added.append(parsed)
            if not quiet:
                print(f"imported: {parsed['id']} [{tag}] -> {parsed['status']}")
    # refresh manual jobs that were still running at the last collect
    for r in records:
        if r.get("status") in ("running-manual",) and r.get("log_path"):
            log = ENCODER_DIR / r["log_path"]
            if log.exists():
                parsed = parse_log(log)
                parsed["id"], parsed["label"], parsed["source"] = r["id"], r["label"], r["source"]
                for k in ("analysis", "notes"):
                    if k in r:
                        parsed[k] = r[k]
                if parsed["status"] == "incomplete" and time.time() - log.stat().st_mtime < 900:
                    parsed["status"] = "running-manual"
                elif parsed["status"] == "done" and not quiet:
                    print(f"finished (manual): {r['id']} [{r['label']}]")
                r.clear()
                r.update(parsed)
    return added


def cmd_collect(args) -> None:
    records = load_ledger()
    finished = finalize_running(records, args.quiet)
    import_logs(records, args.quiet)
    # smoke runs: delete their logs and records (they only validate code)
    for rec in [r for r in records if r.get("status") == "smoke-finished"]:
        if rec.get("log_path"):
            p = ENCODER_DIR / rec["log_path"]
            if p.exists():
                p.unlink()
                print(f"smoke {rec['id']}: log deleted ({rec.get('epochs_run', 0)} epochs ran; see {rec.get('out_path')})")
        records.remove(rec)
    save_ledger(records)
    done_now = [r for r in finished if r.get("status") == "done"]
    if done_now and not args.quiet:
        print("\nnewly finished runs (write their analysis with `run.py note ID --file ...`):")
        for r in done_now:
            print(f"  {r['id']}  " + "  ".join(f"{m}={r['test'].get(m, float('nan')):.4f}" for m in GOAL_METRICS))
    n_run = len(running_jobs(records)) + len([r for r in records if r.get("status") == "running-manual"])
    print(f"ledger: {len(records)} records, {n_run} running")


def cmd_status(args) -> None:
    records = load_ledger()
    finalize_running(records, quiet=True)
    import_logs(records, quiet=True)
    save_ledger(records)
    gpus = gpu_table()
    jobs = running_jobs(records) + [r for r in records if r.get("status") == "running-manual"]
    print("GPUs:")
    for g in gpus:
        n = len([j for j in jobs if j.get("gpu") == g["index"]])
        print(f"  gpu {g['index']}: {g['mem_used']}/{g['mem_total']} MiB, util {g['util']}%, harness jobs {n}")
    print(f"running ({len(jobs)}):")
    for j in jobs:
        prog = ""
        vb = j.get("val_best")
        if vb:
            prog = f"ep {j.get('epochs_run', 0)}, best val {SELECTION_METRIC} {vb.get(SELECTION_METRIC, 0):.4f} @ {vb.get('epoch')}"
        print(f"  {j['id']}  [{j.get('label', '')}]  gpu {j.get('gpu', '?')}  {prog}")
    q = load_yaml(QUEUE, {}) or {}
    entries = q.get("experiments", []) or []
    ready = [e for e in entries if e.get("status") == "ready"]
    proposed = [e for e in entries if e.get("status") == "proposed"]
    print(f"queue: {len(ready)} ready, {len(proposed)} proposed (awaiting the user)")
    for e in ready + proposed:
        print(f"  {e['id']} [{e.get('status')}, tier {e.get('tier', 'A')}] {e.get('label', '')}")


def cmd_queue(args) -> None:
    q = load_yaml(QUEUE, {}) or {}
    for e in q.get("experiments", []) or []:
        ov = overrides_text(e.get("overrides", {}) or {})
        print(f"{e['id']:6s} {e.get('status', '?'):9s} tier {e.get('tier', 'A')}  {e.get('model', '')} {e.get('dataset', '')}  {ov}")
        print(f"       {e.get('label', '')}")
        if args.verbose:
            for k in ("hypothesis", "prediction", "decision_rule", "ledger_id"):
                if e.get(k):
                    print(f"       {k}: {e[k]}")


def reference_rows(records: list[dict], dataset: str) -> dict:
    refs = (load_yaml(REFERENCES, {}) or {}).get(dataset, {}) or {}
    out = {}
    for name, spec in refs.items():
        if isinstance(spec, dict) and spec.get("values"):
            out[name] = {"values": spec["values"], "note": spec.get("note", ""), "id": None}
        elif isinstance(spec, dict) and spec.get("id"):
            hits = [r for r in records if r["id"] == spec["id"]]
            if hits and hits[0].get("test"):
                out[name] = {"values": hits[0]["test"], "note": spec.get("note", ""), "id": spec["id"]}
            else:
                out[name] = {"values": None, "note": f"ledger id {spec['id']} missing or unfinished", "id": spec["id"]}
    return out


def noise_floor(records: list[dict], dataset: str) -> tuple[dict, int]:
    """Max spread between finished runs with identical model/dataset/overrides (fixed seed) -> per-metric floor."""
    groups: dict = {}
    for r in records:
        if r.get("dataset") == dataset and r.get("status") == "done" and r.get("test") and not r.get("control"):
            groups.setdefault(r.get("dup_key"), []).append(r)
    floor = {m: 0.0 for m in ALL_METRICS}
    n = 0
    for rs in groups.values():
        if len(rs) < 2:
            continue
        n += 1
        for m in ALL_METRICS:
            vals = [r["test"].get(m) for r in rs if r["test"].get(m) is not None]
            if len(vals) >= 2:
                floor[m] = max(floor[m], max(vals) - min(vals))
    return floor, n


def cmd_report(args) -> None:
    records = load_ledger()
    dataset = args.dataset
    cons = load_yaml(CONSTRAINTS, {}) or {}
    refs = reference_rows(records, dataset)
    floor_obs, n_groups = noise_floor(records, dataset)
    floor_def = cons.get("noise_floor_default", {"recall": 0.003, "ndcg": 0.002})
    floor = {m: max(floor_obs[m], floor_def.get(m.split("@")[0], 0.0)) for m in ALL_METRICS}
    cols = ALL_METRICS if args.wide else GOAL_METRICS

    def row(vals: dict) -> str:
        return " ".join(f"{vals.get(m, float('nan')):7.4f}" for m in cols)

    print(f"== {dataset}: references ==")
    print(f"{'':34s}" + " ".join(f"{m:>7s}" for m in cols))
    for name, spec in refs.items():
        if spec["values"]:
            print(f"  {name:32s}" + row(spec["values"]) + (f"   ({spec['note']})" if spec["note"] else ""))
        else:
            print(f"  {name:32s}{spec['note']}")
    margin = float(cons.get("goal_margin", 0.0))
    goals: dict = {}
    for gname, gspec in (cons.get("goals") or {}).items():
        names = [n for n in gspec.get("references", []) if refs.get(n, {}).get("values")]
        if not names:
            goals[gname] = None
            print(f"  {gname}: reference(s) {gspec.get('references')} missing from the ledger")
            continue
        target = {m: max(refs[n]["values"][m] for n in names) for m in ALL_METRICS if all(m in refs[n]["values"] for n in names)}
        goals[gname] = {"target": target, "names": names, "emb": gspec.get("candidate_embedding_size")}
        print(f"  {gname + ' target = max(' + ', '.join(names) + ')':32s}" + row(target)
              + f"   candidates: persona_mosaic d={gspec.get('candidate_embedding_size')}")
        print(f"  {gname + f' (mean gain >= +{margin:.0%}) e.g.':32s}" + row({m: v * (1 + margin) for m, v in target.items()}))
    print("  noise floor (fixed seed, same config repeats; max of observed and default): "
          + " ".join(f"{m}={floor[m]:.4f}" for m in cols) + f"  [{n_groups} repeated configs]")
    base = refs.get(cons.get("delta_reference", "MOSAIC"), {}).get("values")

    non_cand = cons.get("non_candidate_overrides") or {}

    def non_candidate(r: dict) -> bool:
        ov = r.get("overrides") or {}
        return any(k in ov and ov[k] in vals for k, vals in non_cand.items())

    def goal_cell(r: dict, gt: dict | None) -> str:
        """Mean relative gain over the goal's target across the four goal metrics; '*' = mean >= goal_margin."""
        if gt is None or r.get("control") or non_candidate(r) or r.get("model") != "persona_mosaic" or structure_deviation(r, cons):
            return "   -   "
        if gt["emb"] and r.get("embedding_size") != gt["emb"]:
            return "   -   "
        gains = [r["test"][m] / gt["target"][m] - 1 for m in GOAL_METRICS if gt["target"].get(m) and r["test"].get(m) is not None]
        if not gains:
            return "   ?   "
        g = sum(gains) / len(gains)
        return f"{'*' if g >= margin else ' '}{100 * g:+5.1f}%"

    rows = [r for r in records if r.get("dataset") == dataset and r.get("status") == "done" and r.get("test")]
    if not args.all:
        rows = [r for r in rows if r.get("model") == "persona_mosaic"]
    rows.sort(key=lambda r: r["test"].get(args.sort, 0), reverse=True)
    gnames = list(goals)
    print(f"\n== {dataset}: runs (sorted by {args.sort}; Gn = mean relative gain over that goal's target across the four metrics, "
          f"* = goal reached; ctrl = control run, not a candidate) ==")
    hdr = (f"{'id':34s} {'d':>2s} {'ep':>4s} " + " ".join(f"{m:>7s}" for m in cols) + f" {'dN@10%':>7s} {'gate':5s} "
           + " ".join(f"{g:>7s}" for g in gnames) + "  label / overrides")
    print(hdr)
    for r in rows:
        t = r["test"]
        d = ""
        if base and base.get("ndcg@10"):
            d = f"{100 * (t.get('ndcg@10', 0) - base['ndcg@10']) / base['ndcg@10']:+6.1f}%"
        gate = "-"
        df = r.get("diag_final") or {}
        if "gate_collapsed" in df:
            gate = "COLL" if df["gate_collapsed"] else "ok"
        dev = structure_deviation(r, cons) if r.get("model") == "persona_mosaic" else {}
        tagc = "ctrl" if r.get("control") else ("n/c" if non_candidate(r) else ("struct!=" + ",".join(f"{k}={v}" for k, v in dev.items()) if dev else ""))
        gcells = " ".join(goal_cell(r, goals[g]) for g in gnames)
        ov = overrides_text(r.get("overrides", {}))
        lab = r.get("label", "")
        desc = (lab if lab and lab != ov else "") + (f"  [{ov}]" if ov else "") + (f"  {tagc}" if tagc else "")
        emb = r.get("embedding_size")
        print(f"{r['id']:34s} {str(emb) if emb else '?':>2s} {r.get('best_epoch', 0):4d} " + row(t)
              + f" {d:>7s} {gate:5s} {gcells}  {desc.strip()}")
    print(f"  Gn = mean over R@5, R@10, N@5, N@10 of (run / target - 1); * = mean >= +{margin:.0%}. "
          "Differences below the noise floor are not meaningful. ctrl = control run, n/c = not a candidate (e.g. parameter-free gate), "
          "struct!= = deviates from constraints.shared_structure (exploration, never a goal candidate). "
          "gate: COLL = top-1 share >= .99 (allowed; performance first), - = no [diag] block in that log")
    inc = [r for r in records if r.get("dataset") == dataset and r.get("status") in ("running", "running-manual", "failed", "incomplete", "invalid")]
    if inc:
        print("\nnot counted: " + ", ".join(f"{r['id']}({r['status']})" for r in inc))


def effective_value(rec: dict, key: str):
    """A run's value of a knob: its override, else the model yml default for that dataset."""
    ov = rec.get("overrides") or {}
    if key in ov:
        return ov[key]
    cfg = rec.get("config") or {}
    if key in cfg:
        return cfg[key]
    top, block = model_defaults(rec.get("model", "persona_mosaic"), rec.get("dataset", ""))
    return block.get(key, top.get(key))


def structure_deviation(rec: dict, cons: dict) -> dict:
    """Structural knobs of the run that differ from constraints.shared_structure (empty = shared methodology)."""
    shared = cons.get("shared_structure") or {}
    if rec.get("model") != "persona_mosaic":
        return {}
    out = {}
    for k, v in shared.items():
        actual = effective_value(rec, k)
        if actual is None:
            continue
        if isinstance(v, bool) or isinstance(actual, bool):
            same = bool(actual) == bool(v)
        elif isinstance(v, (int, float)) and isinstance(actual, (int, float)):
            same = abs(float(actual) - float(v)) < 1e-9
        else:
            same = str(actual).lower() == str(v).lower()
        if not same:
            out[k] = actual
    return out


def goal_targets_for(records: list[dict], dataset: str, cons: dict) -> dict:
    refs = reference_rows(records, dataset)
    out = {}
    for gname, gspec in (cons.get("goals") or {}).items():
        names = [n for n in gspec.get("references", []) if refs.get(n, {}).get("values")]
        if not names:
            out[gname] = None
            continue
        target = {m: max(refs[n]["values"][m] for n in names) for m in GOAL_METRICS if all(m in refs[n]["values"] for n in names)}
        out[gname] = {"target": target, "names": names, "emb": gspec.get("candidate_embedding_size")}
    return out


def mean_gain(test: dict, target: dict) -> float | None:
    gains = [test[m] / target[m] - 1 for m in GOAL_METRICS if target.get(m) and test.get(m) is not None]
    return sum(gains) / len(gains) if gains else None


def cmd_goals(args) -> None:
    """Standing of every (dataset, goal) pair: target, best candidate so far, phase, achieved."""
    records = load_ledger()
    cons = load_yaml(CONSTRAINTS, {}) or {}
    margin = float(cons.get("goal_margin", 0.0))
    nc = cons.get("non_candidate_overrides") or {}
    achieved = cons.get("achieved") or {}
    phases = cons.get("phases") or []
    datasets = args.datasets or sorted({r["dataset"] for r in records if r.get("model") == "persona_mosaic"})
    current = None
    for ph in phases:
        open_pairs = [(d, g) for d in ph.get("datasets", []) for g in ph.get("goals", []) if g not in (achieved.get(d) or {})]
        if open_pairs:
            current = (ph.get("name"), open_pairs)
            break
    if current:
        print(f"current phase: {current[0]} -> open pairs: " + ", ".join(f"{d}:{g}" for d, g in current[1]))
    else:
        print("current phase: all listed phases achieved")
    print(f"{'dataset':13s} {'goal':4s} {'phase':5s} {'status':10s} {'target +' + f'{margin:.0%}':>34s}   best candidate (mean gain)")
    for d in datasets:
        goals = goal_targets_for(records, d, cons)
        for gname, gt in goals.items():
            ph = next((p.get("name") for p in phases if d in p.get("datasets", []) and gname in p.get("goals", [])), "-")
            if gt is None:
                print(f"{d:13s} {gname:4s} {ph:5s} {'no ref':10s}")
                continue
            cands = [r for r in records if r.get("dataset") == d and r.get("model") == "persona_mosaic" and r.get("status") == "done"
                     and r.get("test") and not r.get("control") and not any(k in (r.get("overrides") or {}) and r["overrides"][k] in v for k, v in nc.items())
                     and (not gt["emb"] or r.get("embedding_size") == gt["emb"]) and not structure_deviation(r, cons)]
            scored = sorted(((mean_gain(r["test"], gt["target"]), r) for r in cands), key=lambda x: -(x[0] if x[0] is not None else -9))
            if gname in (achieved.get(d) or {}):
                status = "ACHIEVED"
            elif scored and scored[0][0] is not None and scored[0][0] >= margin:
                status = "REACHED->add"
            else:
                status = "open"
            tgt = " ".join(f"{m.split('@')[0][0].upper()}@{m.split('@')[1]}={gt['target'][m] * (1 + margin):.4f}" for m in GOAL_METRICS if m in gt["target"])
            best = f"{scored[0][1]['id']} ({100 * scored[0][0]:+.1f}%) {scored[0][1].get('label', '')[:50]}" if scored and scored[0][0] is not None else "-"
            print(f"{d:13s} {gname:4s} {ph:5s} {status:10s} {tgt:>34s}   {best}")


def cmd_show(args) -> None:
    records = load_ledger()
    r = find_record(records, args.id)
    out = {k: v for k, v in r.items() if k not in ("val_curve", "diag_lines")}
    print(json.dumps(out, indent=2, ensure_ascii=False))
    curve = r.get("val_curve", {})
    if curve.get("epoch"):
        eps = curve["epoch"]
        step = max(1, len(eps) // 12)
        print("val curve (sampled): " + "; ".join(
            f"{eps[i]}:{curve.get('recall@10', [0] * len(eps))[i]:.4f}/{curve.get('ndcg@10', [0] * len(eps))[i]:.4f}" for i in range(0, len(eps), step)))
        print("  (epoch: recall@10/ndcg@10 on validation)")


def cmd_diag(args) -> None:
    records = load_ledger()
    r = find_record(records, args.id)
    for d in r.get("diag_lines", []):
        print(f"[{d['epoch']}] {d['text']}")
    if not r.get("diag_lines"):
        print("(no [diag] lines in this log)")


def cmd_note(args) -> None:
    records = load_ledger()
    r = find_record(records, args.id)
    text = Path(args.file).read_text() if args.file else args.text
    if not text:
        die("--file or --text required")
    r.setdefault("analysis", [])
    r["analysis"].append({"at": now(), "text": text.strip()})
    save_ledger(records)
    print(f"analysis stored on {r['id']} ({len(r['analysis'])} entries)")


def cmd_mark(args) -> None:
    """Correct a record by hand: status (invalid / done), label, a reason. Invalid runs leave every table."""
    records = load_ledger()
    r = find_record(records, args.id)
    if args.status:
        if args.status not in ("invalid", "done", "incomplete"):
            die("--status must be invalid, done or incomplete")
        r["status"] = args.status
    if args.label:
        r["label"] = args.label
    if args.reason:
        r.setdefault("notes", []).append({"at": now(), "text": args.reason})
    save_ledger(records)
    print(f"{r['id']}: status={r['status']} label={r.get('label', '')!r}")


def cmd_kill(args) -> None:
    records = load_ledger()
    r = find_record(records, args.id)
    if r.get("source") != "harness":
        die("only harness-launched runs can be killed here (manual runs belong to the user)")
    pid = r.get("pid")
    if pid and pid_alive(pid, r["id"]):
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except Exception as e:  # noqa: BLE001
            print(f"kill failed: {e}")
        time.sleep(2)
    r["status"] = "killed"
    r["finished_at"] = now()
    if args.delete_log:
        log = ENCODER_DIR / r["log_path"] if r.get("log_path") else find_log_for(r)
        if log and Path(log).exists():
            Path(log).unlink()
            print(f"deleted {log}")
    save_ledger(records)
    print(f"{r['id']} killed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("launch", help="start a run (validated against constraints.yaml)")
    p.add_argument("--model")
    p.add_argument("--dataset")
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--label")
    p.add_argument("--hypothesis", default="")
    p.add_argument("--prediction", default="")
    p.add_argument("--decision", default="", help="decision rule: what each outcome implies for the next step")
    p.add_argument("--no-hypothesis", action="store_true", help="reference / repeat run without a hypothesis")
    p.add_argument("--queue", help="queue.yaml entry id to launch (copies its config and hypothesis)")
    p.add_argument("--approved", action="store_true", help="the user approved this tier-B queue entry")
    p.add_argument("--gpu", type=int)
    p.add_argument("--ignore-memory", action="store_true", help="launch even when no GPU has gpu_min_free_mib free")
    p.add_argument("--repeat", action="store_true", help="allow a config already in the ledger (same seed; nondeterminism floor)")
    p.add_argument("--allow", nargs="*", default=[], help="constraint keys consciously overridden (give --reason)")
    p.add_argument("--reason", default="")
    p.add_argument("--smoke", type=float, metavar="MINUTES", help="code check: kill after N minutes, delete the log at collect")
    p.add_argument("--wait", action="store_true")
    p.set_defaults(func=cmd_launch)

    p = sub.add_parser("validate", help="check a config against constraints.yaml without launching")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--allow", nargs="*", default=[])
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("collect", help="finalize finished harness runs, import logs not in the ledger")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser("status", help="GPUs, running jobs, queue")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("queue", help="list queue.yaml")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_queue)

    p = sub.add_parser("report", help="results table vs references and goals")
    p.add_argument("--dataset", default="amazon_book")
    p.add_argument("--all", action="store_true", help="include baseline models")
    p.add_argument("--wide", action="store_true", help="also show @20")
    p.add_argument("--sort", default="ndcg@10")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("goals", help="standing of every (dataset, goal): target, best candidate, phase")
    p.add_argument("--datasets", nargs="*")
    p.set_defaults(func=cmd_goals)

    p = sub.add_parser("show", help="one record (without the curve and diag lines)")
    p.add_argument("id")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("diag", help="all [diag] lines of a run")
    p.add_argument("id")
    p.set_defaults(func=cmd_diag)

    p = sub.add_parser("note", help="attach an analysis (ANALYSIS_TEMPLATE.md) to a record")
    p.add_argument("id")
    p.add_argument("--file")
    p.add_argument("--text")
    p.set_defaults(func=cmd_note)

    p = sub.add_parser("mark", help="correct a record: --status invalid|done, --label, --reason")
    p.add_argument("id")
    p.add_argument("--status")
    p.add_argument("--label")
    p.add_argument("--reason")
    p.set_defaults(func=cmd_mark)

    p = sub.add_parser("kill", help="stop a harness run")
    p.add_argument("id")
    p.add_argument("--delete-log", action="store_true")
    p.set_defaults(func=cmd_kill)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
