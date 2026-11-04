---
name: persona-mosaic-paper-notation
description: "PERSONA-MOSAIC paper (adapted from the MOSAIC CIKM'26 LaTeX) — notation chosen for the rewritten Section 4 on 2026-10-28, so later sections stay consistent"
metadata:
  node_type: memory
  type: project
  originSessionId: 8f4a51b0-7384-4dd9-a6ce-e924b3b54d9f
  modified: 2026-10-28T04:32:21.176Z
---

On 2026-10-28 the user asked to rewrite only the methodology section (Section 4) of the MOSAIC CIKM'26
LaTeX for persona mosaic, keeping every MOSAIC equation/notation that is unchanged (ExpertGCN h/g/z/alpha,
gate s_v/E_v/w^e_v/y_v, total loss L). Model name used in the text: PERSONA-MOSAIC. Abstract, intro,
preliminary study, experiments and conclusion were NOT touched (still say MOSAIC; Principle 1/2 refs kept).

Notation introduced (match it when writing other sections; trimmed on 2026-10-28 at the user's request,
"notation이 좀 많은 것 같은데 줄여줘"):
- persona embedding p in R^{d_p}; P_v = persona set of node v (items: one LLM persona each,
  |P_i| = 1, since items have no facets; code switched 2026-10-28, Section 4 text still says RLMRec profile); all mean-centred + l2-normalised.
- experts E = {1..K} = k-means clusters of the user personas; prototype c^e = normalised mean of the
  cluster (no equation); K = argmax mean silhouette over a candidate range (no validation data).
- closest persona p^e_v = argmax_{p in P_v} p^T c^e, similarity sigma^e_v = max; expert input
  tilde-x^e_v = p^e_v (.) c^e (the code's sqrt(d_p) scale is NOT written); x^e_v = MLP^e(tilde-x^e_v).
- gate input [t_v || sigma_v]; L_Align described in words only (gate-weighted InfoNCE to x^e_v of the
  selected experts), total loss L kept verbatim.
- Deliberately omitted: PCA / d_text (default d_text = 4096 = raw space, user: "PCA 안쓰지? 그런
  쓸데없는 거 빼"), K_min/K_max, P_u/n_u/P^e, the four-view averaging, users without personas, tau, B.
- Section/figure labels unchanged: fig:framework, subsec:diversify_data, subsec:diversify_recommenders.

**Why:** the user wants the persona version to reuse MOSAIC's formulation wherever the method is the same.
**How to apply:** when rewriting abstract/intro/experiments, reuse these symbols and keep the kept
equations verbatim; the method text follows encoder/models/general_cf/persona_mosaic.py (see
[[persona-mosaic-design]]); never write "segment" ([[clustering-not-segment-terminology]]).
