# PERSONA-MOSAIC 실험 하네스 — 운영 규칙

이 디렉터리(encoder/harness)의 작업과 `/harness-tick`에서는 이 문서가 최우선이다. 기계적으로 검사되는 부분은
`constraints.yaml`(run.py가 실행 전에 강제), 실행·기록은 `run.py`, 실험 대기열은 `queue.yaml`, 모든 결과는
`ledger.jsonl`, 결정의 역사는 `JOURNAL.md`, 분석 양식은 `ANALYSIS_TEMPLATE.md`. 기준값은 `references.yaml`.
실행 명령은 항상 `/home/kyung/mosaic_v2/encoder`에서 `harness/run.py ...` (shebang이 mosaic conda python을 쓴다).

## 0. 우선순위 — 현재 단계 (사용자, 2026-11-03 23:50 갱신)
- **2026-11-04 11:22 실험 종료(서버 이전).** 실행 0개, cron 중지. 새 서버에서 재개하려면 먼저 `harness/HANDOFF.md`를 따른다
  (링크 실체화, run.py shebang, `gpu_min_free_mib`, smoke 확인, 중단된 q092/q094/q096/q097/q098 재실행 여부).
- **방법론은 Book·Movie·Yelp에서 동일해야 한다.** 구조 노브(gate 종류, 페르소나 파일, 정렬 목표 구성, 아이템 텍스트, prior 주입,
  K 규칙, top_k 등)는 `constraints.yaml`의 `shared_structure` 값으로 고정이고, 데이터셋별로 달라도 되는 것은 하이퍼파라미터
  (정렬 가중치, 온도, balance, layer_num, reg 등)뿐이다. `shared_structure`에서 벗어난 실행은 '탐색'(report에 `struct!=`)이며
  목표 후보가 될 수 없다. 구조 변경을 채택하려면 세 데이터셋 모두에서 돌려 G1이 유지·개선될 때 `shared_structure`(와 yml 기본값)를
  바꾼다 — 그러면 과거 실행의 후보 여부가 자동으로 재평가된다.
- **P1 = 공유 구조로 Book·Movie·Yelp 모두 G1.** Book은 공유 구조(평균 목표)로 +4.9%(q051)라 달성, Movie(-1.8%)·Yelp(-2.5%)가 열려 있다.
  그 다음 **P2: 세 데이터셋의 G2와 G3**. 단계 정의는 `phases`, 달성 기록은 `achieved`.
- **재실행 평균은 하지 않는다** (사용자): 단일 실행이 평균 +4% 이상이면 달성으로 기록한다. 같은 설정의 반복은 더 이상 큐에 넣지 않는다.
- (사용자, 2026-11-04) `top_k`는 2 이상이면 되고 데이터셋별로 달라도 된다(구조 노브가 아님). 구조 변경이 한 데이터셋에서 좋게 나와
  다른 데이터셋에 적용했을 때 성능이 안 나오면, 그 데이터셋의 하이퍼파라미터(정렬 가중치 등)를 재조정해서 맞춘다 — 구조 채택을
  막는 이유가 아니다.
- 매 tick 처음에 `run.py goals`로 현재 단계와 열린 (데이터셋, 목표) 쌍을 확인하고, 제안·실행은 그 쌍을 겨냥한다.
  **P1이 끝나기 전에는 Book 실험을 올리지 않는다** (사용자, 2026-11-03 13:20; GPU가 비어도 마찬가지). `run.py launch`가
  현재 단계에 없는 데이터셋의 persona_mosaic 실행을 거부한다(`constraints.yaml` `phase_lock: true`; 참고용 베이스라인 실행은 예외).
- 달성 판정: 공유 구조의 한 실행이 평균 +4% 이상이면 `goals`가 "REACHED->add"를 보이고, tick이 `achieved`에 설정 문자열과 함께
  기록한다. 반복 실행으로 확인하지 않는다.
- 데이터셋별 하이퍼파라미터(정렬 가중치, balance 등)는 데이터셋마다 따로 맞춘다 (MOSAIC도 intent_align_weight를 데이터셋별로 둔다).
  구조는 반드시 공통(`shared_structure`).
- GPU 메모리 가드: `run.py launch`는 `gpu_min_free_mib`만큼 빈 GPU가 없으면 "deferred"로 끝나고(exit 3) 큐 항목은 ready로 남는다.
  `--ignore-memory`는 쓰지 않는다 (외부 사용자 프로세스가 GPU당 ~42 GB를 쓰는 동안 OOM이 3회 발생했다).

## 1. 목표와 지표
- 대상: `persona_mosaic` (encoder/models/general_cf/persona_mosaic.py). 데이터셋 1순위 `amazon_book`.
  `yelp`, `amazon_movie`는 보고만 한다 (두 모델 모두 ProMax에 못 미쳐 Book에서 개발 중).
- 지표: test **R@5, R@10, N@5, N@10 네 개 모두**. @20은 참고값. 모델 선택과 early stop은 공유 trainer가
  validation R@20으로 하며 바꾸지 않는다 (베이스라인과 같은 조건).
- 폭: 모든 목표는 네 지표의 기준값 대비 상대 이득의 **평균이 +4% 이상** (`constraints.yaml`의 `goal_margin`). 한 지표가
  4% 미만이어도 평균이 넘으면 달성. 잡음 폭(.002~.003)보다 훨씬 크므로 단일 실행으로 판정할 수 있다.
- **G1 (1차)**: 가장 높은 baseline 대비 +4%. 기준값은 지표별로 MOSAIC과 ProMax(d=32) 중 큰 값. PERSONA-MOSAIC은 d=64.
- **G2 (2차)**: ProMax(d=64) 대비 +4%. PERSONA-MOSAIC은 d=64.
- **G3 (3차)**: PERSONA-MOSAIC **d=32**가 ProMax(d=32) 대비 +4%. G3 전까지 PERSONA-MOSAIC은 d=64로 실험한다
  (`embedding_size=32` override는 G3용 실행에만 쓴다; constraints가 64/32 외의 값을 거부한다).
- 판정은 `run.py report`가 한다. `G1`/`G2`/`G3` 열 = 네 지표의 그 목표 기준값 대비 상대 이득의 **평균**,
  `*` = 평균 +4% 달성. 보고서에 `*` 없는 숫자를 "달성"이라고 쓰지 않는다.
- 통제 실행(`persona_shuffle`, `item_shuffle`, `llm_prior`)은 후보가 아니고 **필수도 아니다**. 내용 효과를 따로 주장하고
  싶을 때만 쓴다. 성능이 우선이다.
- seed는 2025 고정. 여러 seed를 돌리지 않는다. 같은 설정 재실행(`--repeat`)도 하지 않는다 (2026-11-03 23:50).

## 2. 기여와 근거 — 무엇을 증명해야 하는가
| 기여 (사용자 정의) | 필요한 근거 | 상태 (2026-11-02) |
|---|---|---|
| expert를 hand-craft하지 않음 (k-means 페르소나 클러스터, K는 validation 없이 선택) | (a) MOSAIC의 수작업 4-expert 대비 평균 +4% (= G1) (b) 선택: 실제 페르소나 > 셔플 페르소나 | (a) 미달: 최고 평균 이득 +2.3%. (b) Book에서 셔플 ≥ 실제 (3회), 필수 아님 |
| MoE 구조 | gate가 동작(가중 라우팅)하는 상태에서의 성능. expert 선택이 한쪽에 몰려도 허용 (덜 몰리면 좋지만 추후 목표) | 기본 learned gate는 ep 30 전 expert 0으로 몰림. cluster/warm-up으로 분산시켰으나 성능 이득 없음 |
| 균등이 아닌 가중 gate (학습 파라미터 필수) | gate 엔트로피·민감도 진단 + persona swap 민감도. 균등 gate·파라미터 없는 gate는 금지(비교 대상도 아님) | learned gate는 expert 0으로 몰림; learned+warm-up은 노드별 분기. gate=cluster 실행 3개는 분석용(n/c) |
| intent·conformity가 페르소나에 녹아 있음 | 텍스트 정보량 비교(offline AUC) + 모델이 그 정보를 쓰는 증거(셔플 하락) | AUC persona .799 ≥ intent .771 ≫ conformity .543. 모델이 못 쓰는 것이 문제 |
모든 큐 항목은 `contribution`에 어느 주장을 위한 실험인지 적는다. 주장을 지지하지 못하는 결과도 그대로 적는다.

## 3. 불변 규칙
1. **베이스라인 수정 금지**: mosaic.py, promax.py와 그 yml, trainer/, data_utils/, loss·metric·base_model, train_encoder.py,
   data/. 베이스라인 실행은 참고값용으로만 (`--allow baseline`), "무엇을 이식할지" 정하는 베이스라인 ablation은 하지 않는다.
   (harness/patches/…knobs.patch 안의 mosaic.py `llm_prior` 노브는 분석 전용이고 기본값 real = 동작 불변. 적용 여부는 사용자 결정.)
2. **베이스라인 기여점 이식 금지 — 특히 ProMax 식의 손실은 추가하지 않는다** (사용자, 2026-11-03). 목록은
   `constraints.yaml`의 `forbidden_loss_families`.
   - **SDR** (ProMax의 Semantic Distribution Regularization): 유저·아이템 텍스트 유사도를 softmax한 분포를 교사로 삼아
     CF 점수의 유저→아이템 분포에 cross-entropy를 거는 손실. 온도·엔트로피 가중·부분 카탈로그·다른 텍스트 등 어떤 변형도 금지.
   - **S2DR** (ProMax의 Structure-Semantic Dual Regularization): 원 그래프와 텍스트/RAG로 증강한 그래프(new_trn_rag_mat)에서
     얻은 두 임베딩 사이의 InfoNCE(유저·아이템·교차). 증강 그래프를 만들어 대조하는 어떤 변형도 금지.
   - 같은 이유로 RLMRec의 profile KD(생성형·대조형 지식 증류)도 새 손실로 넣지 않는다.
   - **허용되는 것**: PERSONA-MOSAIC이 이미 가진 구성요소를 바꾸는 일 — persona_align(InfoNCE, MOSAIC 계승)의 가중치·온도·
     목표(prior 구성, 고정/학습 목표, facet 선택), gate 종류, prior 주입 방식, expert 정의·클러스터링, 자체 점수 항.
   - **판별 질문** (tier B 제안서에 답을 적는다): (1) 새 손실이 텍스트 유사도 "분포"를 교사로 쓰는가? (2) 증강 그래프와
     대조하는가? (3) 베이스라인 논문의 식을 그대로 옮기는가? 하나라도 예이면 금지.
3. **gate에는 학습 가능한 파라미터가 있어야 한다** (사용자, 2026-11-02). 파라미터 없는 라우팅(`gate_mode=cluster`: 클러스터
   유사도만으로 고정)은 분석용일 뿐 후보가 아니며 constraints가 실행을 거부한다. 허용되는 gate는 `learned`(MOSAIC MLP)와
   `residual`(MLP + 클러스터 유사도 logits). **균등 gate 금지**: `gate_cluster_scale=0`, 고정 1/K 라우팅을 최종 모델로 쓰는
   어떤 설정도. (warm-up은 일시 스케줄이라 허용)
4. **seed 고정 2025**, 다중 seed 금지. **K는 `clustering_k_select` 규칙으로만** (`expert_num`, `clustering_seed` override 금지).
5. 용어: cluster / clustering (segment 금지). `/home/kyung/PersonaMosaic`의 코드를 참조·복사하지 않는다.
6. 동시 실행 제한 없음. run.py가 GPU를 자동 배정(하네스 작업 수 → 메모리 순)하고 실행 간 3초 간격을 지킨다.
   수동(사용자) 실행은 건드리지 않는다 (`kill`은 하네스 실행만).
7. 로그는 `save/log/<model>/`에 남기고 ledger로 파싱한다. 코드 검증은 `--smoke 분`으로만 돌리며 그 로그는 collect가 지운다.
   결과 파일 `save/result/*.txt`는 trainer가 쓰는 부산물일 뿐, 진실 원천은 ledger다.

## 4. 권한 등급
- **A (자율, 자동 승인)**: constraints 범위 안의 override만 바꾸는 실험. 에이전트가 가설·예측·결정 규칙을 적어 큐에
  `ready`로 올리고 같은 tick에 실행한다 (사용자 2026-11-02: "자동 승인"). harness/ 아래 파일 편집, scratch 분석 스크립트 포함.
- **B (승인 후)**: persona_mosaic.py / persona_mosaic.yml 변경이 필요한 실험. 큐에 `tier: B`, `status: proposed`로 올리고
  diff와 `--smoke` 결과를 보인 뒤 사용자가 yes 하면 `launch --queue qNNN --approved`. constraints.yaml의 `auto_approve.B`를
  true로 바꾸면 에이전트가 구현 → smoke → JOURNAL에 diff 기록 → 실행까지 스스로 한다 (현재 false).
- **C (금지)**: 3-1의 파일, 데이터·클러스터 캐시 삭제, 평가 프로토콜 변경, 다른 사람의 실행 중단.

## 5. 실험 프로토콜
1. **실행 전**: hypothesis(무엇이 왜), prediction(수치 범위), decision_rule(결과 A → X, B → Y), contribution.
   없으면 run.py가 거부한다 (`--no-hypothesis`는 참고·재실행 전용).
2. **실행**: `run.py launch --queue qNNN` 또는 `--override ...`. 같은 설정은 거부 (`--repeat` 예외).
3. **종료 후**: `run.py collect` → `show` / `diag` / `report` → ANALYSIS_TEMPLATE.md 양식으로 작성 → `run.py note ID --file`.
   예측 적중 여부와 decision_rule의 어느 가지인지 반드시 적는다.
4. **해석 규칙**: 차이가 잡음 폭(기본 R .003 / N .002, 관측값이 크면 그 값) 미만이면 "차이 없음". 편차 안의 숫자로
   결론을 내리지 않는다. 한 번의 실행으로 "되고/안 되고"를 단정하지 않는다. 좋은 지표만 골라 쓰지 않는다.
5. gate는 동작해야 한다(가중 라우팅, 균등 금지). 그러나 expert 선택이 한쪽에 몰리는 것(top-1 share ≥ .99, 보고서 `gate` 열
   `COLL`)은 **허용**한다. 성능이 우선이고 분산은 추후 목표다. `gate` 열은 정보용이며 판정에 쓰지 않는다.
6. 셔플 통제 짝은 필수가 아니다. 내용 효과를 명시적으로 주장할 때만 같은 설정의 `persona_shuffle=within_cluster` 짝을 돌린다.
7. 큐가 비면 ≤2개를 새로 올린다. 대상은 `run.py goals`가 보여 주는 **현재 단계의 열린 쌍**이다. override만 바꾸는 것(tier A)은
   `ready`로 올리고 바로 실행한다(메모리 가드에 걸리면 다음 tick에 재시도). **GPU에 자리가 있는데 큐가 비어 있으면 "결과 대기"를
   이유로 보류하지 않는다** (사용자, 2026-11-03 18:10): 기대값이 낮은 소거 실험이라도 현재 단계의 실험을 올린다. 보류는
   메모리 가드에 걸릴 때, 또는 올릴 가설이 정말 하나도 없을 때(그 경우 JOURNAL에 "소진"이라고 적고 tier B 제안을 올린다)뿐이다. 코드 변경이 필요하면 tier B `proposed`로 올리고
   실행하지 않는다 (`auto_approve.B`가 true가 아닌 한). 제안 수는 tick당 2개까지로 묶어, 결과 없이 실험만 쌓이는 것을 막는다.
8. 이상 징후를 먼저 의심한다: 재빌드 후 validation 급락(= 진단·초기화 버그), 같은 초 로그 충돌, best epoch 0, 학습 붕괴.
   의심스러운 결과는 `run.py mark ID --status invalid --reason ...`로 표에서 뺀다.

## 6. 명령 요약
```
harness/run.py status                      # GPU, 실행 중, 큐
harness/run.py collect                     # 끝난 실행 확정 + 하네스 밖에서 시작된 로그 import
harness/run.py report [--wide] [--all] [--dataset yelp]
harness/run.py launch --queue q003         # ready·tier A 항목 실행
harness/run.py launch --model persona_mosaic --dataset amazon_book --override gate_mode=cluster --label ... \
                      --hypothesis ... --prediction ... --decision ...
harness/run.py launch ... --smoke 3        # 3분 코드 검증, 로그는 collect가 삭제
harness/run.py validate --model persona_mosaic --dataset amazon_book --override k=v   # 제약 검사만
harness/run.py show ID | diag ID | note ID --file f.md | mark ID --status invalid --reason "..." | kill ID
```
`/loop 30m /harness-tick` 으로 tick을 반복한다 (.claude/skills/harness-tick/SKILL.md).

## 7. 파일
- `run.py` 실행·파싱·보고 (유일한 실행 진입점) · `constraints.yaml` 기계 검사 규칙 · `references.yaml` 기준값
- `queue.yaml` 가설이 붙은 실험 대기열 · `ledger.jsonl` 모든 실행 기록(진실 원천) · `JOURNAL.md` 날짜별 결정
- `ANALYSIS_TEMPLATE.md` 분석 양식 · `runs/<id>.out` 각 실행의 stdout/stderr · `state.json` 마지막 실행 시각

## 8. 코드 상태 — 반드시 확인
- 2026-11-02 15:07 작업 트리 전체가 커밋 d8186e5(10-29 22:36)로 되돌려졌다. 그 뒤에 넣었던 실험 노브
  (prior_injection / prior_add_*, text_score*, gate_mode / gate_cluster_scale / gate_warmup_epochs, `[diag]` 진단 로깅,
  trainer의 set_epoch 훅, mosaic.py의 분석용 llm_prior)는 현재 코드에 없다. 커밋 상태에 있는 노브는 persona_shuffle /
  item_shuffle / shuffle_seed, persona_align_weight, item_text, text_dim 등 yml에 보이는 것뿐이다.
- 복원 패치: `harness/patches/2026-11-02-experimental-knobs.patch` (백업에서 재구성, `git apply --check` 통과, 단위 smoke 통과).
  적용은 사용자 결정이다 (`cd /home/kyung/mosaic_v2 && git apply encoder/harness/patches/2026-11-02-experimental-knobs.patch`).
  에이전트는 이 패치를 스스로 적용하지 않는다 (tier B). 적용 전에는 그 노브를 쓰는 큐 항목(q003, q004, q006, q008)은 proposed로 둔다.
- `run.py validate`는 현재 yml의 키만 허용하므로, 패치 적용 여부가 자동으로 반영된다.
- 2026-11-04 09:50: `prior_composition=mix`(유저 행 = unit(facet_k + 평균)) 추가 — harness/patches/2026-11-04-q076-mix-composition.patch,
  사용자 승인(09:48), smoke 10 epoch 통과. `shared_structure`에 채택되기 전까지는 탐색(report `struct!=`)이다.

## 9. 현재 상태 (2026-11-02 23:10 시작, 2026-11-04 11:22 종료 시점까지 갱신; 상세는 JOURNAL.md, 요약표는 FINAL_REPORT.md)
- Movie: 최고는 mean 목표 + align 7e-3 (2회 평균 .0406/.0644/.0349/.0431, MOSAIC -1.8%, ProMax d=64 +0.8%; 11-03 20:30).
  가중치 곡선 5e-3 -3.3 < 7e-3 -1.8 > 8e-3 -3.5. 그 밖의 모든 변형(view, balance, reg, 2 layers, 온도, 고정 목표, audience, learned gate,
  warm-up)은 더 나쁨.
  정렬 2e-2는 Movie·Yelp에서 그래프를 죽인다(과한 정렬) → 데이터셋별 가중치 필수. 고정(detach) 정렬 목표는 Movie -7~-9%,
  Yelp -14%로 폐기(11-03 06:00). Movie G1에 남은 tier A 축: 백본 깊이(q045)뿐.
- Yelp: G1 타깃은 ProMax d=32 +4% (N@10 .0701, ProMax d=64 .0692보다 높음). PERSONA 최고 = mean 목표 + 5e-3
  (.0479/.0797/.0558/.0656) → mean + 2e-3 2회 평균 .0476/.0807/.0555/.0657 (MOSAIC +2.1%, G1 -2.5%; 11-03 15:00 확정).
  막힌 축: 정렬 1e-2 이상, 온도 .1/.5, 고정 목표, 4-view, 3 layers, learned gate, item persona/audience, top_k=1.
- 11-04 05:00: top_k(데이터셋별 하이퍼파라미터, >=2)가 새 지렛대 — Movie top_k=4 → MOSAIC -0.1%, Yelp top_k=4 → G1 -0.7%.
  side=user(Movie -13%)·가중치 감쇠(Movie -5.9%, Yelp -2.7%)는 폐기. 폭 곡선 확정(07:08) — Movie 2 -1.8/3 -3.5/4 -0.1/5 -1.7/6 -5.2/8 -2.6,
  Yelp 2 -1.6/3 -1.9/4 -0.7/6 -0.9/8 -1.6 (Yelp는 2 이상 평탄). 둘 다 top_k=4. Movie 가중치 축(top_k=4): 5e-3 -7.1/7e-3 -0.1/8e-3 -0.2/1e-2 -1.6 소진.
  변조 상한 축: Movie scale 2.0 → -4.1%(최종 lambda 1.0~1.24, 상한 1.0이 사실상 최적) → Movie 1.0 유지·축 종료(08:03); yelp scale 2.0은 진행 중.
  08:32: keep_rate .8 -0.6%, 변조 초기값 -0.4% → 둘 다 잡음 안(축 종료). 09:30: layer_num 4 → -2.2%(@5 하락, 과평활) → 3층 확정, 깊이 축 종료.
  09:47: reg 3e-6 → -34.3%(붕괴, best epoch 3; 1e-5와 같음) → reg 축 종료. **Movie tier A 소진**(JOURNAL 09:50 목록). 남은 것은 구조 변경뿐:
  q076 mix: Movie -3.7%(mean -0.1% 대비 -3.6점, prior 의존 감소; 10:31) → Movie 재조정 1점(mix + 1e-2, q098)과 Yelp mix(q096, 10:01 실행 중) 결과로 최종 판정.
  Movie·Yelp 모두 mean 이상이 아니면 mix 폐기(공유 구조 mean 유지). q098 movie mix + 1e-2 11:00 실행 중. Yelp 소거: keep_rate .8(q097) 실행 중. Movie·Yelp가 mean 이상이면 Book 1회(phase_lock 예외, 사용자 확인) 뒤 채택 판단.
  Yelp top_k=4 가중치: 2e-3 -0.7 / 3e-3 -2.4(잡음 안) / 5e-3 -4.3 / 1e-2 -9.8(11:00; |prior| 1/9, swap .992 — 센 정렬은 투영을 줄이고 민감도를 없앤다) → 2e-3 위로 단조 하락, 1e-3 진행 중. Yelp scale 2.0 → G1 -14.0%(09:08):
  lambda가 1.5~1.8까지 올라가 층 배율 1+lambda·tanh가 음수가 됨 → 붕괴. lambda ≤ 1(상한 1.0)은 배율 비음수 조건이라 성능 상한이 아님. 변조 상한 축 양쪽 종료.
  GPU 가드(11-04 08:03): yelp 프로세스 21~35 GB, movie 15 GB 관측 → `gpu_min_free_mib` yelp 36000 / movie 16000.
- Book에서 모든 변형이 R@20 .157~.167 안. 텍스트 기여 상한은 MOSAIC 기준 .003 (2%).
- 2026-11-02 18:00: `gate_mode=residual persona_align_weight=2e-2`가 G1을 평균 +4.2%로 처음 달성(.0767/.1152/.0749/.0880).
  19:30 **G1 확정**: `gate_mode=residual persona_align_weight=2e-2 persona_file=persona_emb_one.pkl` (d=64) — 두 실행
  +4.5% / +5.2% (.0756/.1163/.0752/.0894, .0772/.1158/.0759/.0893), 네 지표 모두 +4% 이상. 4-view 변형은 +4.2% / +2.6%로 경계.
  20:30 **새 최고**: 위 설정 + `prior_composition=persona`(정렬 목표에서 centroid 곱 제거) → .0794/.1188/.0780/.0918,
  G1 +8.1%, **G2 +0.8%**. 22:00 재실행 q024가 +7.7% / +0.5%로 재현 → **persona-only 목표가 현재 G1 확정 설정**
  (`gate_mode=residual persona_align_weight=2e-2 persona_file=persona_emb_one.pkl prior_composition=persona`, d=64).
  ProMax d=64와 동급(G2 평균 0~+1%); G2 달성선(+4%)까지 약 3점. Book에서 mean 목표는 .0891(-2.7점)로 persona-only 유지(11-03 12:00).
  Movie·Yelp는 정렬 2e-2가 과함 → 데이터셋별 가중치(Movie 5e-3, Yelp 2e-3)와 mean 목표가 최선.
  막힌 축: expert 분산(cluster, balance .1, scale 3; Movie balance .1도 -17%), 정렬 가중치(2.5e-2 +5.3%, 3e-2 붕괴; 2e-2 정점),
  온도 .1·item persona(붕괴), text_score(stop-grad 포함), modulation 초기값(무변화), projector 용량(256 -29%, 4096 -5%), elbow K, PCA-32. learned gate + align 2e-2는 +2.3%/+2.0%. 정렬 가중치는 2e-2가 상한(3e-2, 5e-2 붕괴),
  1e-2/5e-3은 기본보다 나쁨(비단조). 두 expert를 실제로 나눠 쓰면(gate=cluster, residual+balance .1) 손해.
  `gate_mode=cluster` 실행들은 파라미터 없는 gate라 후보에서 제외(n/c).
- G2(ProMax d=64 대비 평균 +4%)는 현재 최고 대비 약 10점 위 (align 2e-2의 평균 이득 -4.6%).
- 열린 질문: 텍스트가 AUC .80의 신호인데 모델이 못 쓰는 이유 — 상호작용 많은 유저는 그래프가 이미 알고 있음(상한 2%) vs
  전달 경로가 닫혀 있음. 희소 유저 그룹별 지표(미구현)가 이를 가를 수 있다.
