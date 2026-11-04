---
name: no-copying-baseline-contributions
description: Do not port a baseline's own contributions (ProMax SDR / S2DR, etc.) into PERSONA-MOSAIC; Book reference numbers and why Book is the development dataset
metadata:
  type: feedback
---

Never add a competing method's headline contribution to PERSONA-MOSAIC (2026-11-02: I proposed an SDR-style
distribution-alignment loss and launched ProMax SDR/S2DR ablations; the user stopped it: "SDR, S2DR는 ProMax의
기여점이잖아. 이걸 베끼면 안되지").

Reference points the user works from: ProMax on Book at its own d_emb=32 gives R@20 .1588 (the d=64 run
.1725 in save/result/promax_amazon_book.txt is not the user's reference); MOSAIC / PERSONA-MOSAIC are
~.166 on Book, i.e. above that. On Yelp both are below ProMax, which is why development happens on Book.

2026-11-03 the user restated it for the harness: ProMax-style losses (SDR = text-similarity distribution as a CE
teacher over the catalog; S2DR = InfoNCE between original and RAG-augmented graph embeddings) must never be added,
in any variant; RLMRec profile KD likewise. Recorded in encoder/harness/constraints.yaml `forbidden_loss_families`
and CLAUDE.md 3-2 (three screening questions every tier-B proposal must answer).

**Why:** the contribution must be the persona-cluster MoE itself; borrowing a baseline's loss makes the
comparison meaningless and the paper unpublishable.
**How to apply:** improvements must come from PERSONA-MOSAIC's own components (experts, gate, prior
injection, its existing alignment loss, evaluation protocol). Analysis-only runs of baselines are fine, but
ask before launching baseline ablations whose purpose is to decide what to port. See [[persona-mosaic-design]],
[[no-copying-sibling-projects]].
