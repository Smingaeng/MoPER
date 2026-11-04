---
name: persona-mosaic-design
description: PERSONA-MOSAIC (GitHub Smingaeng/MOSAIC issue #2) spec and the design decisions taken in encoder/models/general_cf/persona_mosaic.py on 2026-10-26
metadata:
  type: project
---

PERSONA-MOSAIC = MOSAIC (encoder/models/general_cf/mosaic.py, CIKM'26 paper) where the experts are
LLM persona clusters. Spec source: https://github.com/Smingaeng/MOSAIC/issues/2 ("persona mosaic 초안").
Per the user (2026-10-26) only three things differ from MOSAIC: the LLM response (personas), the
expert count (K persona clusters), and the prior; gate / ExpertGCN structure stay identical.

Data facts (data/<dataset>/persona/persona_emb.pkl, symlinked from /home/kyung/PersonaMosaic/data):
- user personas only (record_ids 'u<uid>:p<j>', 1-3 per user, field_mean [records, 4096] unit vectors,
  mean of 4 views identity/tastes/aspects/behavior); 2 Book users and 3 Movie users have none.
- item personas (added 2026-10-26): persona/item_persona_emb.pkl = {'emb' [item_num, 4096] unit, row = iid,
  'valid' bool (6 Book / 7 Yelp / 5 Movie invalid), 'view_names'}; same space as user personas (mean cosine 0.99),
  user-mean-centred cosine to own audience 0.21 Book / 0.08 Yelp / 0.20 Movie vs ~0 random.
  itm_emb_np.pkl (4096-d profile) is the same embedder but weakly aligned.
- raw vectors are anisotropic (same-user vs random cosine 0.771 vs 0.763) -> centre before clustering.

Decisions in persona_mosaic.py (state on 2026-10-27; the file was edited outside my sessions on 10-27 02:39,
re-read it before changing anything):
- K chosen without validation data (user instruction 2026-10-26: "K값 선택 시 validation 사용 X"), search
  K=2..16, max_iter=300, rule `clustering_k_select`: silhouette (default) or SSE elbow. The user does NOT want
  K selected by SSE-vs-K curves ("K가 늘면 SSE가 주는 건 당연"); SSE-vs-K figures are only descriptive.
- `text_dim` (CLI `--d_text`, yml default 4096 = raw) = the clustering space: PCA fitted on the centred user
  persona pool, L2-normalised; k-means, closest-persona choice and the gate's cluster similarities live there.
- `prior_space: raw` (default) feeds MLP^k with the 4096-d centred/normalised persona + raw-space centroid
  (unit mean of the cluster members); `prior_space: pca` uses the d_text vectors. So "cluster in 32-d, train
  in 4096-d" (the advisor's plan pasted on 2026-10-27) = `--d_text 32` with the default prior_space.
- items: knob `item_text` (yml default profile, added 2026-10-29): `profile` = RLMRec itm_emb_np.pkl centred by
  its own mean (original code, verbatim); `persona` = persona/item_persona_emb.pkl processed exactly like user
  personas (user-mean centring, same PCA, to_prior), invalid items -> zero fallback row.
- training is NOT bit-reproducible on GPU despite reproducible: true: identical reruns match at epoch 0 and
  drift at the 4th decimal afterwards; Movie R@20 spread over 3 identical runs 0.0924-0.0943.
- 2026-10-29 comparison (single seed, R@20): Book persona .1637 / profile .1606 / MOSAIC .1661; Yelp
  .1291 / .1289 / .1293; Movie .0902 / .0941 / .0969. Item persona gives no clear gain.
- 2026-10-29 the user re-pointed MOSAIC inputs (intent/conf embeddings, community_assignments) to
  /home/kyung/PersonaMosaic/data/<ds>/mosaic/ (Qwen3-Embedding-8B); community labels are identical to the old
  CoMoE ones, so persona_mosaic results stay valid. New-data MOSAIC R@20: Book .1656, Movie .0985, Yelp .1301
  (Yelp embeddings generated 10-29 14:36, run 15:22: R@10 .0799 N@10 .0645 N@20 .0816, best epoch 102).
- 2026-10-29 persona-shuffle control (knobs persona_shuffle none|global|within_cluster, item_shuffle, shuffle_seed in
  persona_mosaic.py/yml; helper _shuffle_user_owners permutes who owns which persona SET, clustering unchanged,
  trained from scratch). Book, single seed, item_text=profile, R@20 / N@20: none .1619/.1012 (ep 363),
  item_global .1603/.1003, user_within_cluster .1625/.1019, user_global .1674/.1051 (ep 636). Shuffling
  personas does NOT hurt (global shuffle even best, ~ MOSAIC .1656): the encoder does not use persona content
  on Book; fix the encoder/gate before investing in persona generation (DPO / embedder fine-tuning).
- 2026-10-29 diagnosis (Book, checkpoints save/checkpoint/persona_mosaic/*diag_save*.pth, script in the
  10-29 session scratchpad): the shared gate COLLAPSES to one expert (prob 1.000 for every node; gate features
  have zero influence; balance_loss_weight 1e-3 does not prevent it) and the LayerWisePriorModulator learns a
  node-independent mask (lambda saturates at 1.0, tanh at +-1, std over nodes 1e-4; swapping personas at inference
  changes top-20 by <6% / 0%). So the prior path is inert by construction (multiplicative modulation of free
  per-node embeddings = reparametrisation). persona_align_weight sweep on Book: 5e-4 .1603, 5e-3 .1592,
  2e-2 .1611/.1606 (two runs), 5e-2 and 1e-1 collapse from epoch 0 (R@20 .075); 2e-2 + global persona shuffle
  collapses too (R@20 .040), so the alignment loss does see content but only as a constraint, never as a gain.
  Next step agreed as plan: additive prior injection into E^(0) (or a text score term) + a working gate.
- 2026-10-29 added prior_injection knob (modulate default = MOSAIC, additive = E^(0) = E_free + prior_add_scale*x^k_v
  without modulation, both), prior_add_scale 1.0, prior_add_free_embeds. Default verified bit-identical (5e-7) to the
  old code on the diag checkpoint. The user runs the additive experiments themselves (no results yet).
- 2026-10-29 run logs now carry '[diag]' lines (PersonaMosaic.diagnostics(), called by trainer.log_diagnostics):
  brief gate line every 10 evaluations (mean prob / top-1 share per expert, normalised entropy, TV-dist when
  zeroing cluster-sim or topology gate features) and, on the best model at the end, per-expert |E_free| vs |prior|
  norms, layer alpha, modulation lambda / mean|gate-1| / std over nodes, and the inference-only persona swap
  (top-20 overlap, embedding cos) for users and items. Read these before interpreting any persona_mosaic result:
  top-1 share 1.0 = collapsed gate; std over nodes 0 = prior ignored; swap overlap ~1 = output independent of personas.
- 2026-10-30 additive results (user's runs, Book, R@20): additive .1058 (best ep 48), additive no-free .1088,
  additive + global shuffle .0940; all early-stopped ~ep 50-100 at the level the baseline has at ep 30-60.
  Diag: |prior| row-norm 7-20 vs |E_free| 0.2-0.3 (0 with no-free) -> node embedding = shared MLP^k(persona);
  persona swap overlap 0.15-0.33 (content now drives output); real vs shuffled = +13% only. NOT score saturation
  (60-ep check: margins/BPR equal to modulate). Added prior_add_norm (unit-norm x * scale); 130-ep smoke:
  scale 1.0 R@20 .119 at ep 120, 0.3 .112, baseline .144 -> no plateau but slower; gate collapsed again.
  Conclusion: persona content alone is worth ~.11 on Book vs .16 for ID+graph; naive additive lets the
  projector take over memorisation. Next candidates: warm-start E_free from prior, stop-grad projector from BPR,
  or a separate score term beta * t_u.t_i; fix gate balance regardless.
- 2026-10-30 added text_score knob: y_ui = e_u.e_i + beta * sum_k w^k_u w^k_i cos(x^k_u, x^k_i) (gate-weighted
  expert persona projections, node embeddings untouched); text_score_beta_init / _learnable / text_score_norm.
  Log: '[diag] text score: beta=..., mean|beta*T.T| vs mean|e.e|, corr'. Default off = old code (diff 2e-6).
  The user runs it (decision rule agreed: beta ~0 / no gain -> drop content work, fix gate, evaluate structure
  path and sparsity groups; gain + shuffle drop -> invest in content: expert-wise text term, DPO/embedder).
  RESULT 2026-10-30 (Book, user's runs, ~26 min each, early stop ~ep 400): text_score real .1584/.0994
  (best ep 342), text_score + global persona shuffle .1530/.0942 (ep 318); baseline .1619/.1012. Mixed case:
  beta grew 1.0 -> 4.0 in BOTH runs (so beta growth is not evidence of usefulness: trainable MLP^0 lets the
  text term memorise training pairs), |beta T.T| 0.65 vs |e.e| 1.0 on random pairs, corr(text,graph) 0.38
  real vs 0.07 shuffled, swap top-20 overlap .57/.50 with embedding cos .996 (content reaches ranking only
  via the score term). Real beats shuffled by +3.5% (content carries signal) but both trail baseline (term
  adds overfitting capacity). Gate collapsed to expert 0 by ep ~30 again; expert 1 dead; modulator std over
  nodes 0. Agreed reading: content signal real but weak; next = fix gate first, then retest text term with
  frozen/low-capacity projection (stop-grad into MLP^k) so it cannot memorise.
- 2026-10-30 gate fix knobs (persona_mosaic.py/yml, trainer.set_epoch hook): gate_mode learned|cluster|residual
  (cluster: logits = gate_cluster_scale * z-scored cluster sims, no gate params; residual: MLP + sims),
  gate_warmup_epochs (uniform 1/K routing for the first N epochs; gate.force_uniform), balance value now in
  the [diag] gate line. 7-min Book smokes (deleted): cluster R@20 .1444 at ep 120 (= baseline), learned +
  warm-up 15 + balance 1e-1 .1480 at ep 120 with both experts alive (entropy .58-.72, top-1 share .47/.53,
  cluster-sim sensitivity rising .05 -> .16). The user runs the full experiments.
  Full runs 2026-10-30: gate=cluster test .1584/.0990 (best ep 366, val .1639; baseline val ~.165 / test
  .1619): both experts alive but no gain; expert 1 keeps the prior (|prior| 1.0 vs |E_free| .73), swap overlap
  .77/.58 (content now reaches the embeddings). warmup15 + balance 1e-1: val peaked .1596 (ep 204), gate
  node-specific (entropy .21-.63, cluster-sim sensitivity .28), but the logged TEST .1209 IS INVALID: the
  trainer rebuilds the model for the final test and __init__ set force_uniform=True (bug, fixed: fresh model
  starts False; only trainer.set_epoch turns it on). Result-file line [warmup15 balance1e-1] must be ignored
  and the run repeated. Both gate fixes trail the collapsed baseline on Book so far.
  2026-10-31 (my runs): gate=cluster + global persona shuffle .1616/.0999 (ep 525) > gate=cluster real
  .1584/.0990; gate=cluster scale 0.3 .1584/.0973 (ep 303). Shuffled run drops the prior (|prior| .09/.02)
  and recovers the baseline; real run keeps it and is worse. Same pattern as the collapsed-gate control
  (shuffle .1674 > real .1619). Conclusion: on Book the persona prior via modulation is at best noise even
  when routed/alive; softer routing does not help. Content investment (DPO/embedder) ruled out for Book.
  Open: whether MOSAIC's own LLM prior helps on Book (ablation), and the picture on yelp / movie.
- 2026-11-02 MOSAIC ablation on Book (knob llm_prior real|none|shuffle in mosaic.py/yml, my runs): real
  .1656/.1032 (ep 363), none (all LLM tables zeroed, align off) .1623/.1023 (ep 525), shuffle (rows permuted
  among users/items) .1569/.0970 (ep 363). The LLM prior is worth +0.003 R@20 (~2%) on Book in MOSAIC; a
  wrong description costs -0.005 vs none. Offline text-only retrieval AUC on Book test pairs (scratch
  content_auc.py): RLMRec profile pair .800, persona mean vs item profile .799 (current pairing), persona vs
  item persona .790, MOSAIC intent .771, conformity .543, popularity .638, shuffled persona .499; raw
  (uncentred) cosines drop to .68-.74. So persona content is as informative as profiles; the modulation +
  weak alignment path is what cannot use it. Book is a dataset where text adds ~2% at most to MOSAIC.
  User correction 2026-11-02: PERSONA-MOSAIC experts DO receive different text per expert (persona closest
  to cluster k (.) centroid_k; 89% of Book users have >=2 personas: 1176 have 1, 2496 have 2, 7326 have 3;
  items: one text (.) centroid_k). And Book PERSONA-MOSAIC is NOT below MOSAIC: .1672 (one-view persona
  file, 10-28), .1637 (item_text=persona), .1606/.1619 (profile items, current default) vs MOSAIC
  .1661/.1656, MOSAIC-none .1623. Everything on Book sits in a .157-.167 band with ~.003 run spread, so
  Book cannot discriminate the methods; don't claim content/structure effects from single Book runs.
- 2026-11-02 HARNESS: encoder/harness/ (CLAUDE.md = rules, run.py = launch/collect/report/queue/note/mark,
  constraints.yaml, references.yaml, queue.yaml, ledger.jsonl, JOURNAL.md, ANALYSIS_TEMPLATE.md) + project skill
  .claude/skills/harness-tick (for `/loop 30m /harness-tick`). Goals (user, 2026-11-02): metrics R@5/R@10/N@5/N@10,
  the MEAN relative gain over the four must be >= +4%: G1 vs max(MOSAIC, ProMax d=32) at d=64; G2 vs ProMax d=64 at d=64; G3 = PERSONA-MOSAIC
  d=32 vs ProMax d=32 (+4%); d=64 until G3. Gate must work AND have learnable parameters (gate_mode=cluster is analysis-only, never a
  candidate); expert concentration (collapse) is allowed (performance first); shuffle controls optional. Tier A (override-only) experiments are AUTO-APPROVED: the tick creates them as
  ready and launches them; tier B (code changes) still needs the user's yes (constraints auto_approve.B false). Seed fixed, no concurrency cap, baselines untouched.
  At 15:07 the user reset the working tree to commit d8186e5 (10-29): prior_injection, text_score, gate_mode,
  [diag] logging, trainer set_epoch, mosaic llm_prior are GONE from the tree; a reconstructed restore patch sits in
  encoder/harness/patches/2026-11-02-experimental-knobs.patch; the user had it APPLIED on 11-02 16:40 (uncommitted).
  persona_file knob (yml, default persona_emb.pkl; persona_emb_one.pkl = one-view, own cache prefix 'one_') added 11-02.
  2026-11-02 19:30 G1 (mean +4% over max(MOSAIC, ProMax d=32) on R@5/R@10/N@5/N@10) CONFIRMED twice with
  gate_mode=residual persona_align_weight=2e-2 persona_file=persona_emb_one.pkl (d=64): +4.5% / +5.2%
  (N@10 .0894/.0893). Gate collapses to one expert, prior ~.1 (content unused); gain = alignment regulariser +
  residual gate. 22:00: + prior_composition=persona (alignment target = MLP(persona), no centroid product) is the
  new confirmed best: G1 +8.1% / +7.7% (N@10 .0918 both), on par with ProMax d=64 (G2 mean +0.8/+0.5%). Movie: align
  2e-2 over-aligns (-8% vs MOSAIC; one-view Movie clusters K=10) -> per-dataset weight needed. Yelp 2e-2 -22%
  (K=16, graph dies). PRIORITY (user 23:10): P1 = G1 on Movie & Yelp first, then P2 = G2/G3 on all three
  (constraints.yaml phases/achieved; `run.py goals`). Yelp G1 target = ProMax d=32 +4% (N@10 .0701, above ProMax d=64). Dead ends on Book: align 3e-2 / temp .1 / item personas + align (collapse), splitting experts
  (cluster gate, balance .1, scale 3), text_score even with stop-grad, align 1e-2/5e-3 (worse than 5e-4).
- user personas: persona/persona_emb.pkl (4-view mean); the user dropped persona_emb_one.pkl on 2026-10-28.
- K=2 Lloyd-iteration check (2026-10-27, 5 seeds, both spaces): converges in 9-21 iterations; final SSE is
  8.4-11.1% (4096-d) / 11.6-15.2% (32-d) below the K=1 level vs 0.1-0.3% for a random split; 32-d and
  4096-d partitions agree 98.8-99.7% (ARI 0.95-0.99) and the 32-d partition attains the 4096-d optimum SSE;
  Movie has two local optima (n_init handles it). So 4096-d k-means does work at K=2; 32-d gives the same
  clusters with larger assignment margins. Figures: encoder/save/figure/persona_kmeans_k2_iterations.png.
- gate = MOSAIC gate; input = 4 topology features + K cluster similarities (gate_persona_similarity).
- alignment loss = gate-weighted (detached) InfoNCE to the selected experts' priors (persona_align_weight).
- k-means++/silhouette run in torch on GPU; cached in persona/persona_clusters_<spec>.pkl with a lock.
- per-dataset hyper-parameters were copied from mosaic.yml and are NOT tuned for persona_mosaic.

**Why:** these choices resolve gaps between the issue draft and the available data.
**How to apply:** when extending persona_mosaic, keep MOSAIC parity for gate/ExpertGCN and reuse
these knobs for ablations; see [[mosaic-v2-environment]] for how to run tests.
- 2026-11-03 02:20 user-approved tier-B knobs (kept in the tree, defaults = old behaviour): prior_composition=mean
  (user target = unit mean of all its personas) and persona_align_target=fixed (align to a constant PCA-64
  projection of the persona text, stop-grad). Running on Movie/Yelp for P1 (G1 there is still open: Movie best
  -4.3%, Yelp -5.1% vs targets). The tree also carries the restore patch (applied 11-02 16:40) + persona_file knob.
- 2026-11-03 15:25 user-approved item_text=audience (item target = unit mean of its training users' persona means;
  default profile unchanged). Yelp best so far: mean target + align 2e-3 (2 runs avg N@10 .0657, MOSAIC +2.1%, G1 -2.5%);
  Movie best: mean target + 5e-3 (-3.3% vs MOSAIC). phase_lock: no Book runs until P1 (Movie/Yelp G1) is achieved.
- 2026-11-03 23:50 user rules: ONE shared methodology across Book/Movie/Yelp (structural knobs fixed in
  constraints.yaml shared_structure: residual gate, one-view personas, MEAN persona target, profile items, learned
  align target, modulate, silhouette K, raw 4096-d, top_k 2); only hyper-parameters (align weight etc.) per dataset.
  No repeat runs; a single run >= +4% counts. Under the shared structure Book G1 = q051 +4.9% (mean target, 2e-2);
  Movie best mean 7e-3 -1.8%, Yelp best mean 2e-3 -2.5% (G1 open). Facet-target Book runs are explorations now.
- 2026-11-04 user: top_k free per dataset (>= 2); when adopting a structural change, re-tune hyper-parameters on
  datasets where it underperforms. Approved + implemented knobs: persona_align_side (both|user|item) and
  persona_align_decay_epochs (linear decay of the alignment weight to 0). Running on Movie/Yelp.

- 2026-11-04 11:22: 사용자 지시로 하네스 실험 종료(서버 이전). 재개 절차·중단 실행 목록은 encoder/harness/HANDOFF.md, 결과는 FINAL_REPORT.md. [[mosaic-v2-environment]]
