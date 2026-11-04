---
name: harness-tick
description: One tick of the PERSONA-MOSAIC experiment harness (encoder/harness) - collect finished runs, analyse them against the goals, launch the next ready queue entries, append the journal. Use on a schedule, e.g. "/loop 30m /harness-tick".
---

Work from the repo's `encoder/` directory (e.g. /home/kyung/mosaic_v2/encoder on the old server, /home/work/minkyung/mosaic_v2/encoder on the Backend.AI container). Read harness/CLAUDE.md first; its rules override everything below.
`R=harness/run.py` (runs with the mosaic conda python via its shebang).
If that shebang path does not exist on this machine (after a server move), use `R="python harness/run.py"` with the
`mosaic` conda env activated, and read harness/HANDOFF.md once before the first tick.

1. `$R collect` — finalises harness runs that ended, imports logs started outside the harness.
2. For every run that just finished (collect lists them): `$R show ID`, `$R diag ID`, `$R report`; write the analysis
   following harness/ANALYSIS_TEMPLATE.md into a scratch file and attach it with `$R note ID --file FILE`.
   State explicitly whether the prediction held and which branch of the decision rule applies.
   A run marked `failed`: read its `.out` tail (`$R show ID` -> failure_tail), fix only what is tier A, otherwise report.
3. `$R goals`, `$R status` and `$R queue`. Launch every entry with status `ready` and tier `A`: `$R launch --queue qNNN`
   (no concurrency cap; run.py balances GPUs and staggers launches). A launch that prints `deferred` (exit 3) hit the
   GPU-memory guard: leave the entry `ready`, never pass `--ignore-memory`. Never launch `proposed` or tier `B` entries.
4. If no `ready` entry is left, add at most two new entries to harness/queue.yaml for the **current phase's open
   (dataset, goal) pairs** shown by `$R goals` (never for a dataset outside the current phase: `phase_lock` makes
   `launch` refuse it; Book waits until P1 is achieved) (hypothesis / prediction / decision_rule / contribution filled).
   Override-only entries are tier A: give them status `ready` and launch them in this same tick (auto-approved).
   Do not hold proposals "until results arrive" while a GPU has room: queue the next elimination experiment of the
   current phase instead; hold only when the memory guard defers or no hypothesis is left (then say so and propose tier B).
   Entries that need code changes are tier B: status `proposed`, not launched unless constraints.yaml says
   `auto_approve: {B: true}`. A tier-B entry must answer the three questions of CLAUDE.md 3-2 in its `needs` field
   (not a text-similarity-distribution teacher, not an augmented-graph contrast, not a baseline paper's loss); never
   propose anything in constraints.yaml `forbidden_loss_families` (ProMax SDR / S2DR, RLMRec profile KD).
   Shared methodology: a candidate run must match constraints.yaml `shared_structure` on every structural knob
   (report tags deviations `struct!=`); hyper-parameters (alignment weight, temperature, balance, layer_num, reg ...)
   may differ per dataset. A structural exploration that looks better must be run on all three datasets before it
   can be adopted (then update `shared_structure` and the yml defaults). No repeats: when `goals` shows
   "REACHED->add", add that (dataset, goal) to `achieved` in constraints.yaml with the config string.
5. Append a dated entry (<= 10 lines) to harness/JOURNAL.md: finished runs with verdicts, launches, proposals, open questions.
6. Reply to the user in <= 10 lines: what finished and the verdict, what was launched, what waits for approval.
   If nothing finished and nothing was launched, say so in one line.
