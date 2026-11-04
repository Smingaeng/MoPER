# PERSONA-MOSAIC 하네스 최종 보고서 (실험 종료 2026-11-04 11:22)

진실 원천은 `harness/ledger.jsonl`이다. 이 문서는 종료 시점의 `run.py report --all` / `goals` 출력과 요약표를 모아 둔 것이며, 규칙·역사는 `harness/CLAUDE.md`, `harness/JOURNAL.md`, 다른 서버에서의 재개는 `harness/HANDOFF.md`를 본다.

목표: 네 지표(R@5, R@10, N@5, N@10)의 기준 대비 상대 이득 **평균 +4% 이상**. G1 = max(MOSAIC, ProMax d=32) 대비, G2 = ProMax d=64 대비, G3 = PERSONA-MOSAIC d=32 vs ProMax d=32. 후보는 `constraints.yaml`의 `shared_structure`와 일치하는 실행(공유 방법론)뿐이며, 벗어난 실행은 탐색(`struct!=`).

## 1. 요약표 — 데이터셋별 기준값과 PERSONA-MOSAIC 최고

| dataset | model / run | R@5 | R@10 | N@5 | N@10 | G1 mean | G2 mean | 비고 |
|---|---|---|---|---|---|---|---|---|
| amazon_book | MOSAIC (mosaic-amazon_book-1029-113029) | 0.0721 | 0.1112 | 0.0722 | 0.0855 | +0.0% | -6.7% | 기준값 |
| amazon_book | ProMax d=32 (promax-amazon_book-1026-203658) | 0.0677 | 0.1066 | 0.0699 | 0.0833 | -4.0% | -10.4% | 기준값 |
| amazon_book | ProMax d=64 (promax-amazon_book-1026-232159) | 0.0768 | 0.1200 | 0.0771 | 0.0921 | +7.2% | +0.0% | 기준값 |
| amazon_book | PERSONA-MOSAIC 최고 후보(공유 구조): persona_mosaic-amazon_book-1103-100050 | 0.0766 | 0.1165 | 0.0755 | 0.0891 | +4.9% **달성** | -2.1% | gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean persona_align_weight=0.02 |
| amazon_book | PERSONA-MOSAIC 최고 탐색(struct!=): persona_mosaic-amazon_book-1102-190205 | 0.0794 | 0.1188 | 0.0780 | 0.0918 | +8.1% **달성** | +0.8% |  |
| amazon_movie | MOSAIC (mosaic-amazon_movie-1029-113034) | 0.0415 | 0.0641 | 0.0361 | 0.0440 | +0.0% | +2.9% | 기준값 |
| amazon_movie | ProMax d=32 (promax-amazon_movie-1102-195510) | 0.0357 | 0.0561 | 0.0320 | 0.0387 | -12.4% | -9.9% | 기준값 |
| amazon_movie | ProMax d=64 (promax-amazon_movie-1102-203125) | 0.0394 | 0.0621 | 0.0355 | 0.0433 | -2.8% | +0.0% | 기준값 |
| amazon_movie | PERSONA-MOSAIC 최고 후보(공유 구조): persona_mosaic-amazon_movie-1104-010018 | 0.0427 | 0.0628 | 0.0363 | 0.0431 | -0.1% | +2.8% | gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean persona_align_weight=0.007 top_k=4 |
| amazon_movie | PERSONA-MOSAIC 최고 탐색(struct!=): persona_mosaic-amazon_movie-1104-095519 | 0.0396 | 0.0621 | 0.0347 | 0.0425 | -3.7% | -0.9% |  |
| yelp | MOSAIC (mosaic-yelp-1029-145331) | 0.0466 | 0.0799 | 0.0540 | 0.0645 | -4.4% | -7.2% | 기준값 |
| yelp | ProMax d=32 (promax-yelp-1102-195516) | 0.0491 | 0.0817 | 0.0575 | 0.0674 | +0.0% | -2.9% | 기준값 |
| yelp | ProMax d=64 (promax-yelp-1102-203128) | 0.0509 | 0.0834 | 0.0595 | 0.0692 | +3.0% | +0.0% | 기준값 |
| yelp | PERSONA-MOSAIC 최고 후보(공유 구조): persona_mosaic-yelp-1103-213052 | 0.0492 | 0.0815 | 0.0566 | 0.0665 | -0.7% | -3.6% | gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean persona_align_weight=0.002 top_k=4 |
| yelp | PERSONA-MOSAIC 최고 탐색(struct!=): persona_mosaic-yelp-1103-223019 | 0.0493 | 0.0807 | 0.0560 | 0.0660 | -1.4% | -4.2% |  |

`goals` 종료 시점:
```
current phase: P1 -> open pairs: amazon_movie:G1, yelp:G1
dataset       goal phase status                             target +4%   best candidate (mean gain)
amazon_book   G1   P1    ACHIEVED   R@5=0.0750 R@10=0.1157 N@5=0.0751 N@10=0.0889   persona_mosaic-amazon_book-1103-100050 (+4.9%) P2/structure: Book best with the mean-persona targ
amazon_book   G2   P2    open       R@5=0.0799 R@10=0.1248 N@5=0.0802 N@10=0.0957   persona_mosaic-amazon_book-1103-100050 (-2.1%) P2/structure: Book best with the mean-persona targ
amazon_book   G3   P2    open       R@5=0.0704 R@10=0.1109 N@5=0.0727 N@10=0.0867   -
amazon_movie  G1   P1    open       R@5=0.0431 R@10=0.0667 N@5=0.0375 N@10=0.0457   persona_mosaic-amazon_movie-1104-010018 (-0.1%) P1 movie: mean 7e-3 with top_k=4
amazon_movie  G2   P2    open       R@5=0.0409 R@10=0.0646 N@5=0.0370 N@10=0.0450   persona_mosaic-amazon_movie-1104-010018 (+2.8%) P1 movie: mean 7e-3 with top_k=4
amazon_movie  G3   P2    open       R@5=0.0372 R@10=0.0583 N@5=0.0332 N@10=0.0403   -
yelp          G1   P1    open       R@5=0.0511 R@10=0.0849 N@5=0.0598 N@10=0.0701   persona_mosaic-yelp-1103-213052 (-0.7%) P1 yelp: mean + 2e-3 with top_k=4
yelp          G2   P2    open       R@5=0.0530 R@10=0.0867 N@5=0.0618 N@10=0.0720   persona_mosaic-yelp-1103-213052 (-3.6%) P1 yelp: mean + 2e-3 with top_k=4
yelp          G3   P2    open       R@5=0.0511 R@10=0.0849 N@5=0.0598 N@10=0.0701   -
```

## 2. 데이터셋별 전체 보고 (`run.py report --dataset D --all`, 종료 시점)

### amazon_book
```
== amazon_book: references ==
                                  recall@5 recall@10  ndcg@5 ndcg@10
  MOSAIC                           0.0721  0.1112  0.0722  0.0855   (MOSAIC on the current data files, d=64 (tag new_mosaic_data))
  ProMax d=32                      0.0677  0.1066  0.0699  0.0833   (ProMax at its own embedding size (the user's reference))
  ProMax d=64                      0.0768  0.1200  0.0771  0.0921   (ProMax at d=64, same width as MOSAIC / PERSONA-MOSAIC)
  MOSAIC no-text                   0.0719  0.1096  0.0723  0.0851   (analysis only: llm_prior=none (all LLM tables zeroed))
  G1 target = max(MOSAIC, ProMax d=32) 0.0721  0.1112  0.0722  0.0855   candidates: persona_mosaic d=64
  G1 (mean gain >= +4%) e.g.       0.0750  0.1157  0.0751  0.0889
  G2 target = max(ProMax d=64)     0.0768  0.1200  0.0771  0.0921   candidates: persona_mosaic d=64
  G2 (mean gain >= +4%) e.g.       0.0799  0.1248  0.0802  0.0957
  G3 target = max(ProMax d=32)     0.0677  0.1066  0.0699  0.0833   candidates: persona_mosaic d=32
  G3 (mean gain >= +4%) e.g.       0.0704  0.1109  0.0727  0.0867
  noise floor (fixed seed, same config repeats; max of observed and default): recall@5=0.0030 recall@10=0.0030 ndcg@5=0.0020 ndcg@10=0.0020  [6 repeated configs]

== amazon_book: runs (sorted by ndcg@10; Gn = mean relative gain over that goal's target across the four metrics, * = goal reached; ctrl = control run, not a candidate) ==
id                                  d   ep recall@5 recall@10  ndcg@5 ndcg@10  dN@10% gate       G1      G2      G3  label / overrides
promax-amazon_book-1026-232159     64   96  0.0768  0.1200  0.0771  0.0921   +7.7% -        -       -       -     
persona_mosaic-amazon_book-1102-195703 64  306  0.0786  0.1185  0.0779  0.0918   +7.3% COLL     -       -       -     persona-only target REPEAT (confirm new best / G2 proximity)  [gate_mode=residual persona_align_weight=0.02 persona_file=persona_emb_one.pkl prior_composition=persona]  struct!=prior_composition=persona
persona_mosaic-amazon_book-1102-190205 64  318  0.0794  0.1188  0.0780  0.0918   +7.3% COLL     -       -       -     best + alignment target without the centroid product (prior = MLP(persona))  [gate_mode=residual persona_align_weight=0.02 persona_file=persona_emb_one.pkl prior_composition=persona]  struct!=prior_composition=persona
persona_mosaic-amazon_book-1102-210102 64  330  0.0762  0.1167  0.0760  0.0901   +5.4% COLL     -       -       -     persona-only target + align 2.5e-2 (push the alignment edge with the better target)  [gate_mode=residual persona_align_weight=0.025 persona_file=persona_emb_one.pkl prior_composition=persona]  struct!=prior_composition=persona
persona_mosaic-amazon_book-1102-170123 64  315  0.0756  0.1163  0.0752  0.0894   +4.6% COLL     -       -       -     residual gate + align 2e-2 + one-view personas  [gate_mode=residual persona_align_weight=0.02 persona_file=persona_emb_one.pkl]  struct!=prior_composition=product
persona_mosaic-amazon_book-1102-180202 64  489  0.0772  0.1158  0.0759  0.0893   +4.5% COLL     -       -       -     residual + align 2e-2 + one-view REPEAT (confirm best)  [gate_mode=residual persona_align_weight=0.02 persona_file=persona_emb_one.pkl]  struct!=prior_composition=product
persona_mosaic-amazon_book-1102-190202 64  318  0.0765  0.1162  0.0756  0.0893   +4.4% COLL     -       -       -     best + livelier modulation init (yelp values: lambda 5e-2, std 1e-2)  [gate_mode=residual persona_align_weight=0.02 persona_file=persona_emb_one.pkl layer_modulation_lambda_init=0.05 layer_modulation_init_std=0.01]  struct!=prior_composition=product
persona_mosaic-amazon_book-1103-100050 64  273  0.0766  0.1165  0.0755  0.0891   +4.1% COLL  * +4.9%   -2.1%    -     P2/structure: Book best with the mean-persona target (unify the target composition across datasets)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean persona_align_weight=0.02]
persona_mosaic-amazon_book-1102-155534 64  321  0.0767  0.1152  0.0749  0.0880   +3.0% COLL     -       -       -     residual gate + persona alignment 2e-2  [gate_mode=residual persona_align_weight=0.02]  struct!=persona_file=persona_emb.pkl,prior_composition=product
persona_mosaic-amazon_book-1029-163245 64  636  0.0756  0.1141  0.0748  0.0879   +2.8% -        -       -       -     control=user_global  [persona_shuffle=global]  ctrl
persona_mosaic-amazon_book-1102-173248 64  297  0.0753  0.1132  0.0740  0.0872   +1.9% COLL     -       -       -     residual gate + align 2e-2 REPEAT (confirm G1)  [gate_mode=residual persona_align_weight=0.02]  struct!=persona_file=persona_emb.pkl,prior_composition=product
persona_mosaic-amazon_book-1029-192856 64  660  0.0750  0.1120  0.0742  0.0871   +1.8% -        -       -       -     align=2.0e-2  [persona_align_weight=0.02]  struct!=gate_mode=learned,persona_file=persona_emb.pkl,prior_composition=product
persona_mosaic-amazon_book-1102-195706 64  144  0.0743  0.1143  0.0730  0.0871   +1.8% COLL     -       -       -     persona-only target + wide prior projector (hidden 4096)  [gate_mode=residual persona_align_weight=0.02 persona_file=persona_emb_one.pkl prior_composition=persona prior_hidden_size=4096]  struct!=prior_composition=persona
persona_mosaic-amazon_book-1102-155537 64  612  0.0738  0.1100  0.0741  0.0865   +1.1% COLL     -       -       -     align 2e-2 repeat with [diag] (3rd run)  [persona_align_weight=0.02]  struct!=gate_mode=learned,persona_file=persona_emb.pkl,prior_composition=product
persona_mosaic-amazon_book-1102-183121 64  516  0.0747  0.1137  0.0726  0.0862   +0.8% COLL     -       -       -     best + K by elbow rule (more experts, finer alignment targets)  [gate_mode=residual persona_align_weight=0.02 persona_file=persona_emb_one.pkl clustering_k_select=elbow]  struct!=prior_composition=product,clustering_k_select=elbow
persona_mosaic-amazon_book-1029-204103 64  648  0.0754  0.1114  0.0738  0.0862   +0.8% -        -       -       -     align=2e-2 diag_save  [persona_align_weight=0.02]  struct!=gate_mode=learned,persona_file=persona_emb.pkl,prior_composition=product
mosaic-amazon_book-1029-113029     64  363  0.0721  0.1112  0.0722  0.0855   +0.0% -        -       -       -     new_mosaic_data
mosaic-amazon_book-1102-012913     64  525  0.0719  0.1096  0.0723  0.0851   -0.5% -        -       -       -     [llm_prior=none]
persona_mosaic-amazon_book-1028-205254 64  519  0.0724  0.1096  0.0723  0.0848   -0.9% -        -       -       -     one-view persona file (persona_emb_one.pkl, deleted 10-28; not reproducible)  struct!=gate_mode=learned,persona_file=persona_emb.pkl,prior_composition=product
persona_mosaic-amazon_book-1029-001334 64  381  0.0717  0.1104  0.0711  0.0846   -1.0% -        -       -       -     item_text=persona (default at that time)  struct!=gate_mode=learned,persona_file=persona_emb.pkl,prior_composition=product
mosaic-amazon_book-1028-161554     64  453  0.0703  0.1108  0.0703  0.0844   -1.3% -        -       -       -     
persona_mosaic-amazon_book-1029-163023 64  465  0.0710  0.1088  0.0717  0.0843   -1.4% -        -       -       -     control=user_within_cluster  [persona_shuffle=within_cluster]  ctrl
persona_mosaic-amazon_book-1102-155528 64  366  0.0715  0.1078  0.0717  0.0841   -1.6% COLL     -       -       -     residual gate + item personas  [gate_mode=residual item_text=persona]  struct!=persona_file=persona_emb.pkl,prior_composition=product,item_text=persona
persona_mosaic-amazon_book-1102-152858 64  381  0.0696  0.1077  0.0704  0.0837   -2.2% -        -       -       -     baseline repeat (nondeterminism floor)  struct!=gate_mode=learned,persona_file=persona_emb.pkl,prior_composition=product
persona_mosaic-amazon_book-1102-155525 64  360  0.0702  0.1086  0.0703  0.0836   -2.2% COLL     -       -       -     one-view persona embedding (재현)  [persona_file=persona_emb_one.pkl]  struct!=gate_mode=learned,prior_composition=product
persona_mosaic-amazon_book-1102-183401 64  312  0.0707  0.1091  0.0708  0.0836   -2.2% COLL     -       -       -     text score stop-grad WITHOUT strong alignment (q005 companion)  [gate_mode=residual persona_file=persona_emb_one.pkl text_score=true text_score_detach=true]  struct!=prior_composition=product,text_score=True,text_score_detach=True
promax-amazon_book-1026-203658     32  120  0.0677  0.1066  0.0699  0.0833   -2.5% -        -       -       -     
persona_mosaic-amazon_book-1029-163249 64  372  0.0707  0.1081  0.0703  0.0833   -2.5% -        -       -       -     control=item_global  [item_shuffle=true]  ctrl
persona_mosaic-amazon_book-1029-162955 64  363  0.0701  0.1070  0.0703  0.0832   -2.7% -        -       -       -     control=none  struct!=gate_mode=learned,persona_file=persona_emb.pkl,prior_composition=product
persona_mosaic-amazon_book-1029-023448 64  360  0.0710  0.1064  0.0706  0.0828   -3.2% -        -       -       -     item_text=profile  struct!=gate_mode=learned,persona_file=persona_emb.pkl,prior_composition=product
persona_mosaic-amazon_book-1030-031033 64  342  0.0691  0.1073  0.0698  0.0827   -3.3% COLL     -       -       -     [text_score=true]  struct!=gate_mode=learned,persona_file=persona_emb.pkl,prior_composition=product,text_score=True
persona_mosaic-amazon_book-1029-192848 64  300  0.0695  0.1077  0.0689  0.0824   -3.6% -        -       -       -     diag_save  struct!=gate_mode=learned,persona_file=persona_emb.pkl,prior_composition=product
persona_mosaic-amazon_book-1031-011804 64  525  0.0684  0.1065  0.0690  0.0820   -4.1% ok       -       -       -     gate=cluster control=user_global  [gate_mode=cluster persona_shuffle=global]  ctrl
persona_mosaic-amazon_book-1030-220843 64  366  0.0692  0.1063  0.0692  0.0818   -4.4% ok       -       -       -     gate=cluster  [gate_mode=cluster]  n/c
persona_mosaic-amazon_book-1102-173251 64  165  0.0697  0.1049  0.0685  0.0809   -5.4% ok       -       -       -     residual gate + align 2e-2 + cluster logits x3 (route by cluster)  [gate_mode=residual persona_align_weight=0.02 gate_cluster_scale=3.0]  struct!=persona_file=persona_emb.pkl,prior_composition=product,gate_cluster_scale=3.0
persona_mosaic-amazon_book-1102-155855 64  189  0.0677  0.1051  0.0677  0.0807   -5.7% COLL     -       -       -     persona alignment 1e-2 (fills the 5e-3 -> 2e-2 gap)  [persona_align_weight=0.01]  struct!=gate_mode=learned,persona_file=persona_emb.pkl,prior_composition=product
persona_mosaic-amazon_book-1029-192851 64  147  0.0672  0.1054  0.0670  0.0805   -5.9% -        -       -       -     align=5.0e-3  [persona_align_weight=0.005]  struct!=gate_mode=learned,persona_file=persona_emb.pkl,prior_composition=product
mosaic-amazon_book-1102-012918     64  363  0.0700  0.1048  0.0676  0.0798   -6.7% -        -       -       -     [llm_prior=shuffle]  ctrl
persona_mosaic-amazon_book-1031-011809 64  303  0.0667  0.1054  0.0660  0.0798   -6.7% ok       -       -       -     gate=cluster scale=0.3  [gate_mode=cluster gate_cluster_scale=0.3]  n/c
persona_mosaic-amazon_book-1102-155531 64  486  0.0664  0.1044  0.0661  0.0793   -7.2% ok       -       -       -     residual gate (cluster logits + learned correction), balance 0.1  [gate_mode=residual balance_loss_weight=0.1]  struct!=persona_file=persona_emb.pkl,prior_composition=product
persona_mosaic-amazon_book-1030-031046 64  318  0.0652  0.1013  0.0648  0.0773   -9.7% COLL     -       -       -     [text_score=true persona_shuffle=global]  ctrl
persona_mosaic-amazon_book-1102-155540 32  348  0.0623  0.1007  0.0621  0.0757  -11.5% COLL     -       -       -     G3 reference - PERSONA-MOSAIC at d=32 (defaults)  [embedding_size=32]  struct!=gate_mode=learned,persona_file=persona_emb.pkl,prior_composition=product
persona_mosaic-amazon_book-1102-183358 64  186  0.0621  0.0990  0.0621  0.0749  -12.4% ok       -       -       -     best + text score with stop-grad projection (q005, user-approved code change)  [gate_mode=residual persona_align_weight=0.02 persona_file=persona_emb_one.pkl text_score=true text_score_detach=true]  struct!=prior_composition=product,text_score=True,text_score_detach=True
persona_mosaic-amazon_book-1103-103043 32   99  0.0607  0.1002  0.0608  0.0747  -12.7% COLL     -       -       -     P2 Book G3: best config at d=32 vs ProMax d=32  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=persona persona_align_weight=0.02 embedding_size=32]  struct!=prior_composition=persona
persona_mosaic-amazon_book-1103-110040 32  171  0.0591  0.0941  0.0592  0.0713  -16.6% COLL     -       -       -     P2 Book G3: best structure at d=32 with align 5e-3 (d=32-specific weight)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=persona persona_align_weight=0.005 embedding_size=32]  struct!=prior_composition=persona
persona_mosaic-amazon_book-1102-193107 64   96  0.0504  0.0793  0.0508  0.0604  -29.3% COLL     -       -       -     best + narrow prior projector (MLP^k hidden 2080 -> 256)  [gate_mode=residual persona_align_weight=0.02 persona_file=persona_emb_one.pkl prior_hidden_size=256]  struct!=prior_composition=product
persona_mosaic-amazon_book-1029-231027 64   48  0.0403  0.0667  0.0411  0.0502  -41.3% ok       -       -       -     [prior_injection=additive]  struct!=gate_mode=learned,persona_file=persona_emb.pkl,prior_composition=product,prior_injection=additive
persona_mosaic-amazon_book-1102-193110 64   63  0.0405  0.0678  0.0408  0.0502  -41.3% COLL     -       -       -     best + PCA-32 persona space for clustering, similarities and the alignment target  [gate_mode=residual persona_align_weight=0.02 persona_file=persona_emb_one.pkl prior_space=pca text_dim=32]  struct!=prior_composition=product,prior_space=pca,text_dim=32
persona_mosaic-amazon_book-1029-231056 64   51  0.0398  0.0675  0.0405  0.0500  -41.6% ok       -       -       -     [prior_injection=additive prior_add_free_embeds=false]  struct!=gate_mode=learned,persona_file=persona_emb.pkl,prior_composition=product,prior_injection=additive
persona_mosaic-amazon_book-1029-235852 64  102  0.0354  0.0577  0.0373  0.0447  -47.8% ok       -       -       -     [prior_injection=additive persona_shuffle=global]  ctrl
persona_mosaic-amazon_book-1029-192857 64    0  0.0306  0.0484  0.0309  0.0368  -57.0% -        -       -       -     align=5.0e-2  [persona_align_weight=0.05]  struct!=gate_mode=learned,persona_file=persona_emb.pkl,prior_composition=product
persona_mosaic-amazon_book-1102-155852 64    0  0.0307  0.0486  0.0309  0.0368  -57.0% ok       -       -       -     persona alignment 3e-2 (sweep point between 2e-2 and the 5e-2 collapse)  [persona_align_weight=0.03]  struct!=gate_mode=learned,persona_file=persona_emb.pkl,prior_composition=product
persona_mosaic-amazon_book-1029-192902 64    0  0.0304  0.0484  0.0307  0.0367  -57.1% -        -       -       -     align=1.0e-1  [persona_align_weight=0.1]  struct!=gate_mode=learned,persona_file=persona_emb.pkl,prior_composition=product
persona_mosaic-amazon_book-1102-170120 64    0  0.0279  0.0447  0.0282  0.0339  -60.3% ok       -       -       -     residual gate + align 2e-2 + item personas  [gate_mode=residual persona_align_weight=0.02 item_text=persona]  struct!=persona_file=persona_emb.pkl,prior_composition=product,item_text=persona
persona_mosaic-amazon_book-1102-183118 64    0  0.0271  0.0410  0.0265  0.0313  -63.4% ok       -       -       -     best + temperature 0.1 with align 1.5e-2 (q017 rule c)  [gate_mode=residual persona_align_weight=0.015 persona_file=persona_emb_one.pkl llm_align_temperature=0.1]  struct!=prior_composition=product
persona_mosaic-amazon_book-1102-180205 64    0  0.0270  0.0409  0.0264  0.0312  -63.5% ok       -       -       -     best + alignment InfoNCE temperature 0.1 (sharper targets)  [gate_mode=residual persona_align_weight=0.02 persona_file=persona_emb_one.pkl llm_align_temperature=0.1]  struct!=prior_composition=product
persona_mosaic-amazon_book-1102-152901 64    0  0.0177  0.0292  0.0163  0.0206  -75.9% -        -       -       -     CONTROL: align 2e-2 + within-cluster persona shuffle  [persona_align_weight=0.02 persona_shuffle=within_cluster]  ctrl
persona_mosaic-amazon_book-1029-204101 64    0  0.0152  0.0240  0.0133  0.0167  -80.5% -        -       -       -     align=2e-2 control=user_global  [persona_align_weight=0.02 persona_shuffle=global]  ctrl
  Gn = mean over R@5, R@10, N@5, N@10 of (run / target - 1); * = mean >= +4%. Differences below the noise floor are not meaningful. ctrl = control run, n/c = not a candidate (e.g. parameter-free gate), struct!= = deviates from constraints.shared_structure (exploration, never a goal candidate). gate: COLL = top-1 share >= .99 (allowed; performance first), - = no [diag] block in that log

not counted: persona_mosaic-amazon_book-1030-220835(invalid)
```

### amazon_movie
```
== amazon_movie: references ==
                                  recall@5 recall@10  ndcg@5 ndcg@10
  MOSAIC                           0.0415  0.0641  0.0361  0.0440   (tag new_mosaic_data)
  ProMax d=32                      0.0357  0.0561  0.0320  0.0387   (reference run 2026-11-02 (ProMax yml amazon_movie block))
  ProMax d=64                      0.0394  0.0621  0.0355  0.0433   (reference run 2026-11-02, d=64)
  G1 target = max(MOSAIC, ProMax d=32) 0.0415  0.0641  0.0361  0.0440   candidates: persona_mosaic d=64
  G1 (mean gain >= +4%) e.g.       0.0431  0.0667  0.0375  0.0457
  G2 target = max(ProMax d=64)     0.0394  0.0621  0.0355  0.0433   candidates: persona_mosaic d=64
  G2 (mean gain >= +4%) e.g.       0.0409  0.0646  0.0370  0.0450
  G3 target = max(ProMax d=32)     0.0357  0.0561  0.0320  0.0387   candidates: persona_mosaic d=32
  G3 (mean gain >= +4%) e.g.       0.0372  0.0583  0.0332  0.0403
  noise floor (fixed seed, same config repeats; max of observed and default): recall@5=0.0036 recall@10=0.0060 ndcg@5=0.0036 ndcg@10=0.0046  [5 repeated configs]

== amazon_movie: runs (sorted by ndcg@10; Gn = mean relative gain over that goal's target across the four metrics, * = goal reached; ctrl = control run, not a candidate) ==
id                                  d   ep recall@5 recall@10  ndcg@5 ndcg@10  dN@10% gate       G1      G2      G3  label / overrides
mosaic-amazon_movie-1028-161611    64  192  0.0427  0.0656  0.0365  0.0445   +1.2% -        -       -       -     
mosaic-amazon_movie-1029-113034    64  147  0.0415  0.0641  0.0361  0.0440   +0.0% -        -       -       -     new_mosaic_data
persona_mosaic-amazon_movie-1104-083235 64   96  0.0395  0.0654  0.0343  0.0434   -1.2% COLL    -2.2%   +0.6%    -     P1 movie: top_k=4 + 7e-3 with 4 propagation layers (movie block and MOSAIC use 3; 2 was -5.8%)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean top_k=4 persona_align_weight=0.007 layer_num=4]
persona_mosaic-amazon_movie-1103-163041 64   78  0.0403  0.0651  0.0347  0.0434   -1.4% COLL    -1.6%   +1.2%    -     P1 movie: mean target + 7e-3 (refine the sharp 5e-3 peak)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean persona_align_weight=0.007]
promax-amazon_movie-1102-203125    64   54  0.0394  0.0621  0.0355  0.0433   -1.6% -        -       -       -     ProMax d=64 reference (relaunch after OOM on GPU 0)  [embedding_size=64]
persona_mosaic-amazon_movie-1104-040050 64   87  0.0421  0.0643  0.0357  0.0432   -1.8% COLL    -0.2%   +2.7%    -     P1 movie: top_k=4 with align 8e-3 (between 7e-3 and 1e-2 at the new width)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean top_k=4 persona_align_weight=0.008]
persona_mosaic-amazon_movie-1104-033019 64   78  0.0409  0.0632  0.0355  0.0431   -2.0% COLL    -1.6%   +1.3%    -     P1 movie: top_k=4 with align 1e-2 (upper weight point at the new mixture width)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean top_k=4 persona_align_weight=0.01]
persona_mosaic-amazon_movie-1104-080347 64   66  0.0422  0.0624  0.0365  0.0431   -2.0% COLL    -0.4%   +2.5%    -     P1 movie: top_k=4 + 7e-3 with the yelp-block modulation init (lambda_init 5e-2, init_std 1e-2)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean top_k=4 persona_align_weight=0.007 layer_modulation_lambda_init=0.05 layer_modulation_init_std=0.01]
persona_mosaic-amazon_movie-1104-010018 64   96  0.0427  0.0628  0.0363  0.0431   -2.1% COLL    -0.1%   +2.8%    -     P1 movie: mean 7e-3 with top_k=4  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean persona_align_weight=0.007 top_k=4]
persona_mosaic-amazon_movie-1104-073120 64   96  0.0422  0.0625  0.0362  0.0430   -2.1% COLL    -0.6%   +2.3%    -     P1 movie: top_k=4 + 7e-3 with edge dropout keep_rate=0.8 (against early overfitting)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean top_k=4 persona_align_weight=0.007 keep_rate=0.8]
persona_mosaic-amazon_movie-1103-173049 64   72  0.0390  0.0628  0.0347  0.0430   -2.2% COLL    -3.5%   -0.7%    -     P1 movie: mean target + 8e-3 (q062 rule: one step up from the 7e-3 gain)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean persona_align_weight=0.008]
persona_mosaic-amazon_movie-1104-020109 64  102  0.0398  0.0636  0.0349  0.0429   -2.5% COLL    -2.6%   +0.2%    -     P1 movie: mean 7e-3 with top_k=8  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean persona_align_weight=0.007 top_k=8]
persona_mosaic-amazon_movie-1103-173052 64   87  0.0408  0.0637  0.0350  0.0428   -2.6% COLL    -1.9%   +1.0%    -     P1 movie: mean target 7e-3 REPEAT (confirm the -1.6%)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean persona_align_weight=0.007]
persona_mosaic-amazon_movie-1103-004818 64   84  0.0395  0.0633  0.0345  0.0428   -2.6% COLL    -3.3%   -0.4%    -     P1 movie/yelp: alignment target = MLP(mean of the user's personas) (new prior_composition=mean)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean persona_align_weight=0.005]
persona_mosaic-amazon_movie-1104-003047 64   54  0.0393  0.0621  0.0350  0.0428   -2.7% COLL    -3.5%   -0.7%    -     P1 movie: mean 7e-3 with top_k=3 (mixture width on Movie, K=10)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean persona_align_weight=0.007 top_k=3]
persona_mosaic-amazon_movie-1103-183052 64   87  0.0382  0.0632  0.0340  0.0426   -3.0% COLL    -4.5%   -1.8%    -     P1 movie: mean target + 6e-3 (fill the curve between 5e-3 and the 7e-3 peak)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean persona_align_weight=0.006]
persona_mosaic-amazon_movie-1104-095519 64   60  0.0396  0.0621  0.0347  0.0425   -3.3% COLL     -       -       -     P1 all datasets: prior_composition=mix — user target = unit(facet closest to cluster k + mean of all personas)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mix persona_align_weight=0.007 top_k=4]  struct!=prior_composition=mix
persona_mosaic-amazon_movie-1103-161958 64   60  0.0394  0.0624  0.0347  0.0425   -3.4% COLL     -       -       -     P1 movie: mean 5e-3 with the learned gate (gate-type check on Movie)  [gate_mode=learned persona_file=persona_emb_one.pkl prior_composition=mean persona_align_weight=0.005]  struct!=gate_mode=learned
persona_mosaic-amazon_movie-1104-043049 64   72  0.0415  0.0638  0.0350  0.0424   -3.5% COLL    -1.7%   +1.2%    -     P1 movie: mean 7e-3 with top_k=5 (fill the width curve between 4 and 6)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean top_k=5 persona_align_weight=0.007]
persona_mosaic-amazon_movie-1103-033055 64   78  0.0400  0.0637  0.0342  0.0424   -3.6% COLL    -3.3%   -0.4%    -     P1 movie: mean target 5e-3 REPEAT (is -3.3% vs -4.3% real?)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean persona_align_weight=0.005]
persona_mosaic-amazon_movie-1104-020106 64   81  0.0379  0.0637  0.0334  0.0422   -4.1% COLL    -5.2%   -2.5%    -     P1 movie: mean 7e-3 with top_k=6 (extend the mixture-width curve)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean persona_align_weight=0.007 top_k=6]
persona_mosaic-amazon_movie-1029-023439 64   54  0.0386  0.0596  0.0352  0.0421   -4.2% -        -       -       -     item_text=profile  struct!=gate_mode=learned,persona_file=persona_emb.pkl,prior_composition=product
persona_mosaic-amazon_movie-1102-210217 64   75  0.0396  0.0624  0.0341  0.0420   -4.6% COLL     -       -       -     TABLE: movie persona-only + align 5e-3 (movie default weight)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=persona persona_align_weight=0.005]  struct!=prior_composition=persona
persona_mosaic-amazon_movie-1103-043052 64   42  0.0380  0.0608  0.0343  0.0419   -4.7% COLL    -5.8%   -3.1%    -     P1 movie: mean target 5e-3 + 2 propagation layers (against early overfitting)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean persona_align_weight=0.005 layer_num=2]
persona_mosaic-amazon_movie-1027-203822 64   54  0.0375  0.0605  0.0342  0.0418   -4.9% -        -       -       -     struct!=gate_mode=learned,persona_file=persona_emb.pkl,prior_composition=product
persona_mosaic-amazon_movie-1104-070815 64   66  0.0408  0.0591  0.0355  0.0415   -5.5% COLL    -4.1%   -1.3%    -     P1 movie: top_k=4 + 7e-3 with layer_modulation_scale=2.0 (lambda sits at the 1.0 cap)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean top_k=4 persona_align_weight=0.007 layer_modulation_scale=2.0]
persona_mosaic-amazon_movie-1104-030106 64   51  0.0380  0.0607  0.0332  0.0411   -6.6% COLL    -7.1%   -4.4%    -     P1 movie: top_k=4 with align 5e-3 (re-tune the weight at the new mixture width)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean top_k=4 persona_align_weight=0.005]
persona_mosaic-amazon_movie-1103-023018 64   99  0.0379  0.0601  0.0335  0.0410   -6.7% COLL     -       -       -     P1 movie: FIXED target with align 2e-3 (q040b rule: lower weight after over-alignment)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=persona persona_align_weight=0.002 persona_align_target=fixed]  struct!=prior_composition=persona,persona_align_target=fixed
persona_mosaic-amazon_movie-1103-170049 64   84  0.0391  0.0608  0.0333  0.0410   -6.8% COLL    -6.3%   -3.6%    -     P1 movie: mean 5e-3 + gate warm-up 15 (all 10 experts trained before the gate chooses)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean persona_align_weight=0.005 gate_warmup_epochs=15]
persona_mosaic-amazon_movie-1103-230018 64   81  0.0389  0.0615  0.0332  0.0408   -7.2% COLL    -6.3%   -3.6%    -     P1 movie/yelp: alignment weight that DECAYS linearly to 0 over the first N epochs (strong early regulariser, free late training)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean persona_align_weight=0.02 persona_align_decay_epochs=60]
persona_mosaic-amazon_movie-1102-210105 64   72  0.0375  0.0597  0.0330  0.0408   -7.3% COLL     -       -       -     TABLE: persona-only best config on amazon_movie  [gate_mode=residual persona_align_weight=0.02 persona_file=persona_emb_one.pkl prior_composition=persona]  struct!=prior_composition=persona
persona_mosaic-amazon_movie-1102-220533 64  105  0.0378  0.0598  0.0327  0.0406   -7.6% COLL     -       -       -     P1 movie: 4-view personas + residual + persona-only + 5e-3 (no balance)  [gate_mode=residual prior_composition=persona persona_align_weight=0.005]  struct!=persona_file=persona_emb.pkl,prior_composition=persona
persona_mosaic-amazon_movie-1103-200108 64   54  0.0367  0.0597  0.0322  0.0403   -8.4% COLL    -9.4%   -6.7%    -     P1 movie: mean 7e-3 with a weaker layer-wise modulation (scale 0.5; movie block uses 1.0)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean persona_align_weight=0.007 layer_modulation_scale=0.5]
persona_mosaic-amazon_movie-1103-200111 64   93  0.0388  0.0591  0.0334  0.0402   -8.7% COLL    -7.6%   -4.9%    -     P1 movie: mean 7e-3 with the modulation OFF (scale 0: plain LightGCN experts + alignment)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean persona_align_weight=0.007 layer_modulation_scale=0.0]
persona_mosaic-amazon_movie-1103-013022 64   72  0.0370  0.0601  0.0321  0.0401   -8.7% COLL     -       -       -     P1 movie: align to a FIXED PCA-64 persona target  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=persona persona_align_weight=0.005 persona_align_target=fixed]  struct!=prior_composition=persona,persona_align_target=fixed
persona_mosaic-amazon_movie-1102-230016 64  111  0.0375  0.0592  0.0322  0.0398   -9.4% COLL     -       -       -     P1 movie: persona-only + align 2e-3 (below the movie default)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=persona persona_align_weight=0.002]  struct!=prior_composition=persona
persona_mosaic-amazon_movie-1102-195504 64   75  0.0387  0.0597  0.0325  0.0398   -9.4% COLL     -       -       -     G1 config (residual + align 2e-2 + one-view)  [gate_mode=residual persona_align_weight=0.02 persona_file=persona_emb_one.pkl]  struct!=prior_composition=product
persona_mosaic-amazon_movie-1029-023442 64   96  0.0362  0.0588  0.0316  0.0395  -10.1% -        -       -       -     [item_text=persona]  struct!=gate_mode=learned,persona_file=persona_emb.pkl,prior_composition=product,item_text=persona
persona_mosaic-amazon_movie-1103-135636 64   51  0.0346  0.0585  0.0310  0.0393  -10.5% ok       -       -       -     P1 movie: audience-persona item target (q057 companion)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean persona_align_weight=0.005 item_text=audience]  struct!=item_text=audience
promax-amazon_movie-1102-195510    64  102  0.0357  0.0561  0.0320  0.0387  -11.9% -        -       -       -     ProMax d=32 reference  [embedding_size=32]
persona_mosaic-amazon_movie-1102-210214 64   54  0.0359  0.0568  0.0312  0.0386  -12.2% COLL     -       -       -     TABLE: movie persona-only + align 1e-2 (dataset-specific weight)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=persona persona_align_weight=0.01]  struct!=prior_composition=persona
persona_mosaic-amazon_movie-1103-220139 64  132  0.0342  0.0573  0.0303  0.0381  -13.2% COLL     -       -       -     P1 movie: alignment on the USER side only (q070 companion), 7e-3  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean persona_align_weight=0.007 persona_align_side=user]  struct!=persona_align_side=user
persona_mosaic-amazon_movie-1102-214516 64   81  0.0338  0.0534  0.0303  0.0367  -16.6% ok       -       -       -     movie: 4-view personas + residual + persona-only + 5e-3 + balance 0.1 (keep 2 experts alive)  [gate_mode=residual prior_composition=persona persona_align_weight=0.005 balance_loss_weight=0.1]  struct!=persona_file=persona_emb.pkl,prior_composition=persona
persona_mosaic-amazon_movie-1103-123032 64  108  0.0311  0.0516  0.0263  0.0335  -23.9% COLL   -23.9%  -21.7%    -     P1 movie: mean target 5e-3 with a SOFTER InfoNCE (temperature 0.5)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean persona_align_weight=0.005 llm_align_temperature=0.5]
persona_mosaic-amazon_movie-1104-093037 64    3  0.0278  0.0435  0.0227  0.0285  -35.2% ok     -34.3%  -32.4%    -     P1 movie: top_k=4 + 7e-3 with reg_weight 3e-6 (embedding L2 x3 against early overfitting; 1e-5 collapsed)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean top_k=4 persona_align_weight=0.007 reg_weight=0.000003]
persona_mosaic-amazon_movie-1103-000029 64    3  0.0242  0.0386  0.0203  0.0255  -42.1% ok       -       -       -     P1 movie: persona-only + 5e-3 + reg_weight 1e-5 (against early overfitting)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=persona persona_align_weight=0.005 reg_weight=0.00001]  struct!=prior_composition=persona
  Gn = mean over R@5, R@10, N@5, N@10 of (run / target - 1); * = mean >= +4%. Differences below the noise floor are not meaningful. ctrl = control run, n/c = not a candidate (e.g. parameter-free gate), struct!= = deviates from constraints.shared_structure (exploration, never a goal candidate). gate: COLL = top-1 share >= .99 (allowed; performance first), - = no [diag] block in that log

not counted: mosaic-amazon_movie-1028-161605(incomplete), promax-amazon_movie-1102-195513(failed), persona_mosaic-amazon_movie-1102-220536(failed)
```

### yelp
```
== yelp: references ==
                                  recall@5 recall@10  ndcg@5 ndcg@10
  MOSAIC                           0.0466  0.0799  0.0540  0.0645   (tag new_mosaic_data)
  ProMax d=32                      0.0491  0.0817  0.0575  0.0674   (reference run 2026-11-02 (ProMax yml yelp block))
  ProMax d=64                      0.0509  0.0834  0.0595  0.0692   (reference run 2026-11-02, d=64)
  G1 target = max(MOSAIC, ProMax d=32) 0.0491  0.0817  0.0575  0.0674   candidates: persona_mosaic d=64
  G1 (mean gain >= +4%) e.g.       0.0511  0.0849  0.0598  0.0701
  G2 target = max(ProMax d=64)     0.0509  0.0834  0.0595  0.0692   candidates: persona_mosaic d=64
  G2 (mean gain >= +4%) e.g.       0.0530  0.0867  0.0618  0.0720
  G3 target = max(ProMax d=32)     0.0491  0.0817  0.0575  0.0674   candidates: persona_mosaic d=32
  G3 (mean gain >= +4%) e.g.       0.0511  0.0849  0.0598  0.0701
  noise floor (fixed seed, same config repeats; max of observed and default): recall@5=0.0030 recall@10=0.0030 ndcg@5=0.0020 ndcg@10=0.0020  [3 repeated configs]

== yelp: runs (sorted by ndcg@10; Gn = mean relative gain over that goal's target across the four metrics, * = goal reached; ctrl = control run, not a candidate) ==
id                                  d   ep recall@5 recall@10  ndcg@5 ndcg@10  dN@10% gate       G1      G2      G3  label / overrides
promax-yelp-1102-203128            64   84  0.0509  0.0834  0.0595  0.0692   +7.3% -        -       -       -     ProMax d=64 reference (relaunch after OOM on GPU 0)  [embedding_size=64]
promax-yelp-1102-195516            32  102  0.0491  0.0817  0.0575  0.0674   +4.6% -        -       -       -     ProMax d=32 reference  [embedding_size=32]
persona_mosaic-yelp-1103-213052    64  246  0.0492  0.0815  0.0566  0.0665   +3.1% COLL    -0.7%   -3.6%    -     P1 yelp: mean + 2e-3 with top_k=4  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean persona_align_weight=0.002 top_k=4]
persona_mosaic-yelp-1104-023129    64  249  0.0491  0.0810  0.0567  0.0665   +3.1% COLL    -0.9%   -3.8%    -     P1 yelp: mean + 2e-3 with top_k=6 (extend the mixture-width curve)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean persona_align_weight=0.002 top_k=6]
persona_mosaic-yelp-1103-033052    64  273  0.0482  0.0813  0.0561  0.0662   +2.7% COLL    -1.6%   -4.4%    -     P1 yelp: mean target + align 2e-3 (combine the two leading yelp variants)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean persona_align_weight=0.002]
persona_mosaic-yelp-1103-120125    64  201  0.0485  0.0808  0.0561  0.0661   +2.5% COLL    -1.6%   -4.5%    -     P1 yelp: mean target, align 5e-3 with a SOFTER InfoNCE (temperature 0.5)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean persona_align_weight=0.005 llm_align_temperature=0.5]
persona_mosaic-yelp-1103-223019    64  246  0.0493  0.0807  0.0560  0.0660   +2.3% COLL     -       -       -     P1 yelp/movie: alignment on the USER side only (drop the item-side term)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean persona_align_weight=0.005 persona_align_side=user]  struct!=persona_align_side=user
persona_mosaic-yelp-1103-053036    64  324  0.0485  0.0810  0.0558  0.0659   +2.2% COLL    -1.8%   -4.7%    -     P1 yelp: mean target + align 1e-3 (extend the weight curve downward)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean persona_align_weight=0.001]
persona_mosaic-yelp-1103-080051    64  141  0.0474  0.0811  0.0548  0.0659   +2.2% COLL    -2.8%   -5.6%    -     P1 yelp: mean target + 3e-3 (between 2e-3 and 5e-3)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean persona_align_weight=0.003]
persona_mosaic-yelp-1104-030018    64  258  0.0489  0.0809  0.0561  0.0658   +1.9% COLL    -1.6%   -4.4%    -     P1 yelp: mean + 2e-3 with top_k=8  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean persona_align_weight=0.002 top_k=8]
persona_mosaic-yelp-1104-003018    64  237  0.0478  0.0800  0.0555  0.0656   +1.7% COLL    -2.7%   -5.5%    -     P1 yelp: alignment 2e-2 decaying to 0 over 60 epochs (q071 companion)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean persona_align_weight=0.02 persona_align_decay_epochs=60]
persona_mosaic-yelp-1103-020018    64  147  0.0479  0.0797  0.0558  0.0656   +1.7% COLL    -2.6%   -5.4%    -     P1 yelp: alignment target = MLP(mean of the user's personas)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean persona_align_weight=0.005]
persona_mosaic-yelp-1104-060053    64  186  0.0479  0.0811  0.0553  0.0656   +1.6% COLL    -2.4%   -5.3%    -     P1 yelp: top_k=4 with align 3e-3 (fine point just above 2e-3)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean top_k=4 persona_align_weight=0.003]
persona_mosaic-yelp-1103-213049    64  318  0.0488  0.0808  0.0558  0.0655   +1.6% COLL    -1.9%   -4.7%    -     P1 yelp: mean + 2e-3 with top_k=3 (more experts in the mixture)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean persona_align_weight=0.002 top_k=3]
persona_mosaic-yelp-1103-050046    64  186  0.0479  0.0803  0.0553  0.0652   +1.1% COLL     -       -       -     P1 yelp: mean target 5e-3 + item personas (sharper item-side target)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean persona_align_weight=0.005 item_text=persona]  struct!=item_text=persona
persona_mosaic-yelp-1103-135633    64  135  0.0466  0.0793  0.0550  0.0652   +1.1% COLL     -       -       -     P1 yelp/movie: item-side target = mean persona of the item's training users (audience persona)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean persona_align_weight=0.002 item_text=audience]  struct!=item_text=audience
persona_mosaic-yelp-1103-080048    64  273  0.0469  0.0800  0.0549  0.0652   +1.1% COLL    -3.6%   -6.4%    -     P1 yelp: mean target + 2e-3 REPEAT (confirm yelp best, MOSAIC +2.9%)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean persona_align_weight=0.002]
persona_mosaic-yelp-1103-161955    64  132  0.0473  0.0800  0.0545  0.0652   +1.0% COLL     -       -       -     P1 yelp: audience item target with align 5e-3 (second weight point for audience)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean persona_align_weight=0.005 item_text=audience]  struct!=item_text=audience
persona_mosaic-yelp-1103-050043    64  273  0.0473  0.0794  0.0553  0.0652   +1.0% COLL    -3.4%   -6.2%    -     P1 yelp: mean target 5e-3 REPEAT (confirm the new yelp best, MOSAIC +1.9%)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean persona_align_weight=0.005]
persona_mosaic-yelp-1103-001332    64  303  0.0488  0.0793  0.0557  0.0649   +0.6% COLL     -       -       -     P1 yelp: persona-only + align 2e-3  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=persona persona_align_weight=0.002]  struct!=prior_composition=persona
persona_mosaic-yelp-1104-053022    64  147  0.0466  0.0796  0.0543  0.0647   +0.3% COLL    -4.3%   -7.1%    -     P1 yelp: top_k=4 with align 5e-3 (re-tune the weight at the new mixture width)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean top_k=4 persona_align_weight=0.005]
persona_mosaic-yelp-1103-113107    64  228  0.0483  0.0801  0.0548  0.0647   +0.3% COLL     -       -       -     P1 yelp: mean + 2e-3 with the learned (MOSAIC) gate instead of residual  [gate_mode=learned persona_file=persona_emb_one.pkl prior_composition=mean persona_align_weight=0.002]  struct!=gate_mode=learned
mosaic-yelp-1029-145331            64  102  0.0466  0.0799  0.0540  0.0645   +0.0% -        -       -       -     new_mosaic_data
persona_mosaic-yelp-1103-173055    64  285  0.0468  0.0775  0.0546  0.0644   -0.2% COLL    -4.8%   -7.6%    -     P1 yelp: mean + 2e-3 + gate warm-up 15  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean persona_align_weight=0.002 gate_warmup_epochs=15]
mosaic-yelp-1028-164222            64  276  0.0466  0.0780  0.0547  0.0644   -0.2% -        -       -       -     
persona_mosaic-yelp-1103-153043    64  255  0.0469  0.0793  0.0539  0.0641   -0.6% COLL    -4.6%   -7.4%    -     P1 yelp: mean target, align 1e-2 with temperature 0.5 (does the softer loss tolerate more weight?)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean persona_align_weight=0.01 llm_align_temperature=0.5]
persona_mosaic-yelp-1102-212220    64  222  0.0469  0.0778  0.0540  0.0639   -0.9% COLL     -       -       -     TABLE: yelp persona-only + align 5e-3 (yelp default weight)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=persona persona_align_weight=0.005]  struct!=prior_composition=persona
persona_mosaic-yelp-1027-203753    64  180  0.0469  0.0790  0.0533  0.0637   -1.2% -        -       -       -     struct!=gate_mode=learned,persona_file=persona_emb.pkl,prior_composition=product
persona_mosaic-yelp-1103-043049    64  159  0.0445  0.0784  0.0531  0.0637   -1.2% COLL    -6.6%   -9.3%    -     P1 yelp: mean target 5e-3 + 3 propagation layers (yelp block uses 2)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean persona_align_weight=0.005 layer_num=3]
persona_mosaic-yelp-1029-023455    64  192  0.0459  0.0798  0.0527  0.0636   -1.3% -        -       -       -     [item_text=persona]  struct!=gate_mode=learned,persona_file=persona_emb.pkl,prior_composition=product,item_text=persona
persona_mosaic-yelp-1103-001329    64  156  0.0465  0.0771  0.0536  0.0629   -2.4% COLL     -       -       -     P1 yelp: 4-view personas + residual + persona-only + 5e-3  [gate_mode=residual prior_composition=persona persona_align_weight=0.005]  struct!=persona_file=persona_emb.pkl,prior_composition=persona
persona_mosaic-yelp-1028-210649    64  105  0.0442  0.0772  0.0517  0.0624   -3.3% -        -       -       -     struct!=gate_mode=learned,persona_file=persona_emb.pkl,prior_composition=product
persona_mosaic-yelp-1102-214400    64  255  0.0459  0.0750  0.0523  0.0613   -5.0% COLL     -       -       -     TABLE: yelp persona-only + align 1e-2  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=persona persona_align_weight=0.01]  struct!=prior_composition=persona
persona_mosaic-yelp-1104-060050    64  231  0.0440  0.0749  0.0513  0.0609   -5.6% COLL    -9.8%  -12.4%    -     P1 yelp: top_k=4 with align 1e-2 (upper weight point at the new width)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean top_k=4 persona_align_weight=0.01]
persona_mosaic-yelp-1104-070820    64   60  0.0416  0.0712  0.0490  0.0585   -9.3% COLL   -14.0%  -16.5%    -     P1 yelp: top_k=4 + 2e-3 with layer_modulation_scale=2.0 (lambda sits at the 1.0 cap)  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean top_k=4 persona_align_weight=0.002 layer_modulation_scale=2.0]
persona_mosaic-yelp-1103-013025    64  102  0.0412  0.0720  0.0482  0.0580  -10.2% COLL     -       -       -     P1 yelp/movie: align to a FIXED (detached, frozen-projection) persona target instead of the learned MLP^k output  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=persona persona_align_weight=0.005 persona_align_target=fixed]  struct!=prior_composition=persona,persona_align_target=fixed
persona_mosaic-yelp-1103-163038    64  183  0.0399  0.0688  0.0475  0.0562  -12.8% ok     -17.1%  -19.5%    -     P1 yelp: mean + 2e-3 with hard routing (top_k=1) — is the .72/.02x15 mixture helping?  [gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean persona_align_weight=0.002 top_k=1]
persona_mosaic-yelp-1102-195507    64   57  0.0384  0.0637  0.0446  0.0526  -18.5% COLL     -       -       -     G1 config (residual + align 2e-2 + one-view)  [gate_mode=residual persona_align_weight=0.02 persona_file=persona_emb_one.pkl]  struct!=prior_composition=product
  Gn = mean over R@5, R@10, N@5, N@10 of (run / target - 1); * = mean >= +4%. Differences below the noise floor are not meaningful. ctrl = control run, n/c = not a candidate (e.g. parameter-free gate), struct!= = deviates from constraints.shared_structure (exploration, never a goal candidate). gate: COLL = top-1 share >= .99 (allowed; performance first), - = no [diag] block in that log

not counted: promax-yelp-1102-195519(failed), persona_mosaic-yelp-1102-210108(failed), persona_mosaic-yelp-1102-212223(failed), persona_mosaic-yelp-1103-170052(failed)
```

## 3. 종료 시 중단된 실행 (killed, 부분 결과: validation R@20 최고값)

- persona_mosaic-yelp-1104-083230: P1 yelp: top_k=4 with align 1e-3 (lower weight point at the new width) @persona_mosaic-yelp-1104-083230 — ep 235 / val R@20 0.1293 @ 201
- persona_mosaic-yelp-1104-090859: P1 yelp: top_k=4 + 2e-3 with a softer InfoNCE (temperature 0.5) at the new mixture width @persona_mosaic-yelp- — ep 160 / val R@20 0.1248 @ 159
- persona_mosaic-yelp-1104-100101: P1 all datasets: prior_composition=mix on yelp (q076 companion; same code change, same approval) @persona_mosa — ep 73 / val R@20 0.1234 @ 69
- persona_mosaic-yelp-1104-103152: P1 yelp: top_k=4 + 2e-3 with edge dropout keep_rate=0.8 (regularisation axis, untested on yelp) @persona_mosai — ep 46 / val R@20 0.1176 @ 42
- persona_mosaic-amazon_movie-1104-110056: P1 movie: mix target with align 1e-2 (per-dataset re-tune of the structural candidate before judging it) @pers — ep 67 / val R@20 0.0959 @ 66
