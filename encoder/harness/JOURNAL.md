# PERSONA-MOSAIC 실험 일지 (하네스)

매 tick 끝에 날짜·내용 10줄 이내로 추가한다. 세부 수치는 ledger.jsonl (`run.py show ID`), 규칙은 CLAUDE.md.

## 2026-11-02 하네스 개설 — 지금까지의 사실 (Book, seed 2025, 단일 실행)
- 기준값: MOSAIC R@5/R@10/N@5/N@10 = .0721/.1112/.0722/.0855 (R@20 .1656). ProMax d=32 .0677/.1066/.0699/.0833 (.1588). ProMax d=64 .0768/.1200/.0771/.0921 (.1725).
- PERSONA-MOSAIC 기본 설정(profile 아이템, 4-view 페르소나): .1606/.1619 (R@20); N@10 .0828~.0832. 설정별 최고는 one-view 페르소나 .1672 (10-28, 재현 불가: 파일 삭제됨), item_text=persona .1637.
- 잡음 폭: 같은 설정 재실행 R@20 .0013~.0016, N@10 ~.0005 (GPU 비결정성). constraints.yaml 기본값 R .003 / N .002로 보수적으로 둠.
- 통제: persona_shuffle=global이 실제 페르소나와 같거나 높음 (기준선 .1674 vs .1619; cluster gate .1616 vs .1584). 내용 효과 주장은 현재 근거 없음.
- gate: 기본 learned gate는 ep 30 전에 expert 0으로 붕괴(expert 1의 E_free, prior 모두 0). gate_mode=cluster / gate_warmup+balance 0.1로 살렸으나 성능 이득 없음 (cluster .1584; warm-up은 val .1596).
- prior 경로: modulate는 노드 무관 마스크로 퇴화(std over nodes 0). additive는 MLP^k가 E_free를 밀어내 .106~.109로 붕괴. text_score는 beta 1→4로 커지며 외우기(셔플도 동일) → .1584.
- 텍스트 정보량(offline AUC, test 양성 vs 임의): persona-mean vs item profile .799 = RLMRec profile 쌍 .800 > MOSAIC intent .771 >> conformity .543; 인기도 .638; 셔플 .499. 센터링 필수(raw .68~.74).
- MOSAIC ablation(분석용 노브 llm_prior, 기본값 real=변경 없음): none .1623, shuffle .1569 vs real .1656 → Book에서 텍스트 기여 상한 ≈ .003 (2%).
- 무효 결과: 결과 파일의 [warmup15 balance1e-1] test .1209는 재빌드 시 균등 라우팅으로 평가된 버그(수정됨). 무시.
- 결정: 베이스라인 기여(ProMax SDR/S2DR) 이식 금지. 균등 gate 금지. 지표는 R@5/R@10/N@5/N@10. G1 = MOSAIC·ProMax32 이상, G2 = ProMax64 이상.
- 열린 질문: 텍스트가 .80의 신호인데 왜 모델이 못 쓰는가 — (1) 상호작용 많은 유저는 그래프가 이미 알고 있음(Book 텍스트 상한 2%), (2) 전달 경로(모듈레이터·약한 정렬)가 닫혀 있음. 희소 유저 그룹별 지표가 (1)을 가를 수 있음 (하네스 report에 아직 없음).

## 2026-11-02 15:30 하네스 가동 준비
- 로그 40개를 ledger로 역이관. 무효 처리: [warmup15 balance1e-1] (균등 라우팅 평가 버그). 라벨 보정: one-view 실행, item_text=persona 실행.
- 새 지표 기준으로 본 사실: persona_align_weight=2e-2가 두 번 모두 G1(MOSAIC·ProMax32 이상)을 네 지표에서 만족 (N@10 .0871/.0862 vs .0855). 다만 잡음 폭 안(yes~)이고 gate 진단 없음. 셔플 짝(global)은 ep 0 붕괴라 해석 불가 → q007(within_cluster 짝), q008(재실행+진단).
- 코드: 15:07에 작업 트리가 d8186e5로 되돌려져 실험 노브·진단이 사라짐. 복원 패치를 harness/patches/에 두었고 적용은 사용자 결정. 패치가 필요한 큐 항목은 proposed(tier B)로 내림.
- 하네스 launch→collect 경로는 2.5분 smoke로 검증 (로그 자동 삭제).

## 2026-11-02 16:00 목표 재정의 (사용자)
- 모든 목표는 네 지표(R@5/R@10/N@5/N@10)의 상대 이득 **평균 +4% 이상** (16:10 정정: 각각이 아니라 평균). G1 = 가장 높은 baseline(지표별 MOSAIC·ProMax d=32 중 큰 값) +4%; G2 = ProMax d=64 +4%; G3 = PERSONA-MOSAIC d=32가 ProMax d=32 +4%. G3 전까지 PERSONA-MOSAIC은 d=64.
- gate는 동작해야 하지만 expert 선택 쏠림은 허용(성능 우선, 분산은 추후). 셔플 통제 짝은 필수 아님.
- 이 기준으로 현재 위치: 최고 후보 align 2e-2의 평균 이득 +2.3% / +2.0% (G1까지 약 2점), G2는 평균 -4.6% (약 9점). 어떤 설정도 G1 근처가 아니다. 큐에 q009(d=32 참고, proposed) 추가.

## 2026-11-02 16:30 gate 조건 (사용자)
- gate에는 학습 가능한 파라미터가 있어야 한다. gate_mode=cluster(파라미터 없는 고정 라우팅)는 분석용으로만 남기고 후보에서 제외(report에 n/c). constraints가 cluster 실행을 거부. 큐의 cluster 항목(q003, q005, q006)은 residual(MLP + 클러스터 logits)로 변경.

## 2026-11-02 16:50 패치 적용, 큐 가동 (사용자 결정)
- 복원 패치 적용(git apply): residual gate, prior_injection, text_score, [diag], set_epoch, mosaic llm_prior(분석용) 복귀. 커밋은 보류.
- persona_file 노브 추가(persona_mosaic.py/yml): data/<ds>/persona/ 아래 다른 피클 선택, 비기본 파일은 자체 클러스터 캐시(prefix). 사용자가 persona_emb_one.pkl 링크 추가 → q002 ready.
- q002(one-view), q003(residual+item persona), q004(residual+balance .1), q006(residual+align 2e-2), q008(align 2e-2 재실행+diag), q009(d=32 참고) 실행. q005(text_score_detach, 코드 필요)만 proposed.
- /loop 30m /harness-tick 시작.

## 2026-11-02 16:05 tick
- 끝난 실행 없음. 7개 실행 중 (q001 ep 319 val .1667; q002/q003/q004/q006/q008/q009 ep 4~7).
- ready 항목 없음 → 제안 2건(proposed, tier A): q010 align 3e-2 (2e-2와 5e-2 붕괴 사이), q011 align 1e-2 (5e-3 → 2e-2 간격). 둘 다 승인 전 미실행.
- 대기: q005(text_score_detach, 코드 변경 필요).
- 16:15 사용자: tier A 자동 승인. constraints auto_approve {A: true, B: false}; q010(align 3e-2), q011(align 1e-2) 실행. 코드 변경(tier B)은 계속 승인 필요.

## 2026-11-02 16:30 tick
- 끝난 실행 없음. 9개 실행 중 (GPU 0: 4개 60 GB, GPU 1: 5개 66 GB, 둘 다 util 100%). 새 항목은 추가하지 않음 (GPU 포화; 결과가 나온 뒤 제안).
- 점검: align 2e-2 계열이 ep 16~22에 best@0인 것은 이상 아님. 10-29의 2e-2 실행도 ep 0 .0735 → ep 24 .048까지 내려갔다가 회복해 .161에 도달했고, 이번 3번째 실행이 그 곡선을 소수점 셋째 자리까지 재현 중. 5e-2는 같은 구간에서 더 깊이 내려가 회복 못 함. residual+align 2e-2는 ep 21 .0451로 2e-2(.0481)와 5e-2(.0413) 사이 — 회복 여부가 관건.
- 대기: q005(tier B).

## 2026-11-02 17:00 tick
- 끝남: q001 기준선 재실행 .0696/.1077/.0704/.0837 (G1 -2.8%). 기본 설정 4회의 폭 R@10 .0013, N@10 .0013 → noise floor 기본값(R .003/N .002) 유지. 예측 적중.
- 끝남: q010 align 3e-2 → 붕괴(best ep 0, R@10 .0486 = 5e-2와 동일). 2e-2가 정렬 가중치의 상한. diag: E_free .13 vs prior .70, swap overlap .98 — 정렬이 그래프 임베딩을 덮어 씀. gate는 살아 있음.
- 진행 중 주목: residual gate + align 2e-2가 ep 138에 val R@20 .1645 (같은 epoch에서 기존 align 2e-2 .1469, 기준선 .1461) — 지금까지 가장 큰 초반 선행. align 1e-2도 ep 120 .1577로 선행. 정렬이 학습을 앞당기고 residual gate가 그 위에서 더 간다.
- 새 실행 없음(7개 진행, GPU 포화). residual+align 결과(1~1.5h 내)를 보고 결합 축을 정한다. 대기: q005(tier B).

## 2026-11-02 17:30 tick
- 끝남: q011 align 1e-2 .0677/.1051/.0677/.0807 (G1 -6.1%). 예측 실패. 가중치-성능 비단조: 5e-4 .0832, 5e-3 .0805, 1e-2 .0807, 2e-2 .0871, 3e-2 붕괴. diag: 1e-2는 prior .05 vs E_free 1.24(페르소나 무시, swap overlap 1.0), 3e-2는 prior .70 vs E_free .13(그래프 소멸). 공존 구간이 2e-2 근처로 좁다. val은 ep 231에도 상승 중이었으나 R@20 선택 기준으로 조기 종료(trainer 불변).
- 진행 중: residual+align 2e-2 val .1718@270 (MOSAIC val 최고 ~.169 수준 초과), residual+item persona .1651@273, one-view .1647@273, residual+balance .1610@318, align 2e-2 재실행 .1562@309, d=32 .1543@303.
- 실행(자동 승인, tier A): q012 residual+align 2e-2+item persona, q013 residual+align 2e-2+one-view. 8개 실행 중.
- 대기: q005(tier B).

## 2026-11-02 18:00 tick — G1 첫 달성
- **q006 residual gate + align 2e-2: .0767/.1152/.0749/.0880 → G1 평균 +4.2% (*)**. 여유 0.2점 → q014 재실행으로 확인. G2 -2.8%. diag: gate는 expert 1로 100% 쏠림(learned는 expert 0), prior .09 vs E_free 1.09(페르소나 내용은 거의 안 쓰임, swap .96) → 이득은 정렬 정규화 + expert 1 prior를 목표로 삼은 효과.
- q002 one-view .0836 N@10 (재현 실패, 기본과 동일) → 단독 폐기. q003 residual+item persona .0841 (차이 없음). q004 residual+balance .1 .0793 (두 expert 생존 시 손해, gate=cluster와 같은 결론). q009 d=32 기본 .0757 (G3 -8.5%). q012 residual+align+item persona 붕괴(아이템 페르소나가 정렬 압력을 키움).
- 실행(자동): q014 residual+align 2e-2 재실행, q015 residual+align 2e-2 + gate_cluster_scale 3.0 (클러스터별 라우팅이 정렬 효과를 키우는지 = MoE 근거 시험). 진행 중: q008 align 2e-2 재실행(ep 556), q013 residual+align+one-view(ep 160, val .1671 선행).
- 열린 질문: G2(+4% over ProMax d=64, N@10 .0957)까지 8점. 페르소나 내용이 쓰이지 않는 상태의 상한은 정렬 정규화 효과뿐 → 내용을 쓰게 하는 경로(q005 text_score_detach, tier B)가 다음 후보.

## 2026-11-02 18:30 tick — 최고 갱신
- **q013 residual + align 2e-2 + one-view: .0756/.1163/.0752/.0894 (R@20 .1694) → G1 평균 +4.5%, 네 지표 모두 +4% 이상.** 현재 최고. G2 -2.5% (N@10 .0957까지 .0063). gate는 expert 0 쏠림(q006은 expert 1) → residual 이득은 expert 선택이 아니라 붕괴 전 초반 라우팅 효과로 추정. prior .09(내용 미사용).
- q008 align 2e-2 3회차 +1.3% (learned+align 평균 +1.9%, 3회) → learned gate로는 G1 미달 확정. prior .06, swap 1.00: 순수 정규화 효과.
- q015 residual + scale 3: -4.9%. 라우팅이 분산(.17/.83)되자 q004 수준으로 하락. Book에서 expert 분산은 세 번째 확인된 손해 → gate 축 종료.
- 실행(자동): q016 q013 재실행(확인), q017 q013 + InfoNCE 온도 .1(목표 선명화; 붕괴 위험 있음). 진행 중: q014 q006 재실행(ep 283, val .1720 — q006과 동일 pace).
- 열린 질문: G2까지 .0063 N@10. 내용이 쓰이지 않는 상태(prior ~.09)에서 남은 지렛대는 정렬 목표의 질(view, 온도)뿐. 내용을 쓰게 하는 경로는 코드 변경(q005 text_score_detach 등, tier B) 필요.

## 2026-11-02 19:00 tick
- q014 residual+align 2e-2 재실행: +2.6% (q006 +4.2%) → 4-view 변형은 G1 경계(두 실행 평균 +3.4%). 개별 지표 폭은 잡음 안(R@10 .002, N@10 .0008)이지만 평균 이득이 1.6점 흔들림.
- q017 온도 .1: 붕괴(3e-2형). 정렬 유효 강도의 유효 구간이 좁다는 세 번째 확인.
- 실행(자동): q018 best + 온도 .1 + 가중치 1.5e-2 (q017 규칙 c), q019 best + clustering_k_select=elbow (4-view에서 K=5; expert 수가 정렬 목표를 세밀하게 하는지). 진행 중: q016 one-view 재실행(ep 382, val .1735로 q013보다 약간 위).
- 열린 질문: G1 확정은 q016에 달림. G2(.0957)까지 .0063은 knob로는 어려워 보임 → tier B 후보(q005 text_score_detach) 검토 요청.

## 2026-11-02 19:05 q005 실행 (사용자 승인: 코드 변경 후 복구)
- persona_mosaic.py/yml에 text_score_detach 노브 추가(텍스트 항 안의 x^k에 stop-gradient; 단위 검사: 텍스트 항→MLP^k gradient 2.95e5 → 0, beta·gate gradient 유지). 두 실행이 모델을 빌드한 뒤 두 파일을 변경 전 상태로 복구했고(diff 0), 재현용 diff는 harness/patches/2026-11-02-q005-text_score_detach.patch.
- 실행: q005 best(residual+align 2e-2+one-view) + text_score(detach), q005b residual+one-view + text_score(detach) (정렬 기본값; 텍스트 항 단독 효과 분리).
- 진행 중: q016 one-view 재실행 ep 448 val .1743(q013 .1720 초과), q018 온도 .1+1.5e-2 ep 31(.0641@0 — 하강 구간), q019 elbow K ep 13.

## 2026-11-02 19:30 tick — G1 확정
- **q016 one-view 재실행 .0772/.1158/.0759/.0893 → 평균 +5.2%**. q013(+4.5%)과 함께 두 실행 모두 네 지표 +4% 이상 → **G1 확정 설정: gate_mode=residual, persona_align_weight=2e-2, persona_file=persona_emb_one.pkl (d=64)**. G2까지 N@10 .0064 / R@10 .009.
- q018 온도 .1 + 1.5e-2: 붕괴 → 온도 축 종료. q005 best + text_score(stop-grad): .0749 (-12.8%). beta는 stop-grad에서도 1→4.25(외우기가 아니라 코사인 크기 때문), 텍스트 항이 gate를 .80/.20으로 갈라 expert 1을 prior 지배로 몰고 초반 학습을 막음 → text_score 계열 폐기, 코드 복구 유지.
- 실행(자동): q020 best + modulation 초기값 yelp 값(prior가 임베딩에 들어가게), q021 best + prior_composition=persona(centroid 곱 제거). 진행 중: q019 elbow K(ep 142, val .1501 — best pace .1645@138보다 아래), q005b(ep 247, .1634).
- 열린 질문: G2는 내용이 임베딩에 쓰여야 가능해 보이는데 지금까지 내용을 쓰게 한 모든 시도(additive, text_score, 분산 라우팅)는 손해. q020이 마지막 knob 수준 시도.

## 2026-11-02 20:00 tick
- q005b 텍스트 항(stop-grad), 정렬 기본값: .0836 = 기본 설정. 내용이 랭킹에 닿아도(swap .75, corr .52) 정확도 불변 → 텍스트 점수는 그래프와 중복 정보. text_score 계열 폐기 확정(코드 복구 유지).
- 진행 중: q020 modulation 초기값(val .1726@267), q021 centroid 곱 제거(.1734@291) — 둘 다 최고 설정 pace(.1720@270)와 동급. q019 elbow K .1603@363(아래).
- 실행(자동): q022 best + prior_hidden_size 256, q023 best + prior_space=pca text_dim=32 — 둘 다 '정렬 목표 용량' 가설. knob 수준 축은 이 둘로 거의 소진.

## 2026-11-02 20:30 — 새 최고: 정렬 목표에서 centroid 곱 제거
- **q021 best + prior_composition=persona: .0794/.1188/.0780/.0918 (R@20 .1698) → G1 +8.1%, G2 +0.8% (ProMax d=64와 동급: R@5/N@5는 위, R@10/N@10은 1% 안쪽 아래)**. 내용은 여전히 미사용(prior .06, swap .999) — 정렬 목표의 모양(페르소나 자체 vs 페르소나⊙centroid)만으로 3점.
- q020 modulation 초기값: +4.9% (변화 없음) → modulation 경로 종료. q022 hidden 256: -29% → 목표 생성기는 고용량이어야 함.
- 실행(자동): q024 q021 재실행(확인), q025 q021 + prior_hidden_size 4096. 교차 데이터셋 표(사용자 요청): G1 설정을 amazon_movie/yelp에, ProMax d=32/64를 amazon_movie/yelp에 실행 중(6개).
- 주의: data/{amazon_movie,yelp}/persona/persona_emb_one.pkl 링크는 사용자가 15:51에 이미 만들어 둔 것(에이전트가 data/를 건드리지 않음).

## 2026-11-02 21:00 tick
- 끝난 실행 없음. 10개 실행 중. GPU 0 80.1/81.6 GB(외부 사용자 프로세스 4개가 21 GB씩 점유 + yelp persona 20 GB) → OOM 위험. 새 실행 없음(ready 없음, GPU 포화로 제안 보류).
- ProMax 4개(movie/yelp d=32/64)는 CPU PCA 전처리 중(5분 경과, GPU 470 MiB). 로그는 전처리 후 생성.
- 진행 중 주목: q024(persona-only 재실행) ep 22, q025(hidden 4096) ep 13 — 둘 다 2e-2 초반 하강 구간. yelp G1 설정 ep 4 val .0228@0, movie ep 19 .0533@3.

## 2026-11-02 21:30 tick
- q019 elbow K=5: +1.8% (최고보다 3점 아래). 5개 expert 중 1개로 쏠리고 그 클러스터 centroid 합성이 목표를 더 흐림 → persona-only(centroid 제거)가 +8.1%인 것과 일관. K 축 종료.
- q023 PCA-32: -41.9% → 저차원/저용량 목표는 실패(q022와 함께 '목표 용량은 커야' 확정).
- ProMax d=64 movie/yelp: GPU 0 OOM(외부 프로세스 4×21 GB)으로 실패 → GPU 1에 재실행. d=32 둘은 ep 163 진행 중.
- 진행 중: q024 persona-only 재실행 ep 310 val .1745(q021 .1734@291 이상), q025 hidden 4096 ep 160 .1721@144(선행). movie G1 설정 ep 115 .0930@75, yelp ep 28 .0347.
- 새 제안 보류(GPU 0 포화). q024/q025 종료 후 persona-only 기반 제안 예정.

## 2026-11-02 22:00 tick — persona-only 목표 확정 (Book)
- **q024 persona-only 재실행 .0786/.1185/.0779/.0918 → G1 +7.7%, G2 +0.5%**. q021(+8.1/+0.8)과 함께 두 번 모두 * → 새 G1 확정 설정: residual + align 2e-2 + one-view + prior_composition=persona. ProMax d=64와 동급(평균 0~+1%), G2 달성선(+4%)까지 3점.
- q025 hidden 4096: +2.2% (persona-only보다 -.0047) → 용량 축 종료(256 -29%, 2080 최선, 4096 -5%).
- 표: Movie G1 설정 .0387/.0597/.0325/.0398 — MOSAIC 대비 -8.2% (2e-2가 Movie에서 과한 정렬: ep 54까지 val .025 정체, E_free .025 vs prior .60; one-view Movie는 K=10). ProMax d=32 movie .0357/.0561/.0320/.0387(MOSAIC 아래), yelp .0491/.0817/.0575/.0674(MOSAIC +4.5%).
- 실행(자동): q026 Book persona-only + align 2.5e-2, q027/q028 persona-only 설정을 movie/yelp에, q029/q030 movie persona-only + align 1e-2/5e-3(데이터셋별 가중치). 진행 중: yelp G1 설정(ep 79), ProMax d=64 movie/yelp(재실행).

## 2026-11-02 22:25 교차 데이터셋 표 진행
- ProMax d=64 참고값: movie .0394/.0621/.0355/.0433 (MOSAIC -1.6%), yelp .0509/.0834/.0595/.0692 (MOSAIC +7.3%). Movie에서는 MOSAIC이 최강, Yelp에서는 ProMax d=64가 최강.
- Movie PERSONA: G1 설정 -8.2%, persona-only 2e-2 -8.1%, persona-only 1e-2 -12.6%. 2e-2의 긴 초반 억제(ep 54까지)가 빨리 수렴하는 Movie(MOSAIC best ep 147)에 치명적. q030(5e-3) 진행 중(val .0955@54 ≈ MOSAIC .0986@54).
- Yelp PERSONA: G1 설정(2e-2) val .1106@100 vs MOSAIC .1339 → 과한 정렬. persona-only 2e-2(q028)는 GPU 0 OOM으로 실패, 재실행 안 함. q031(5e-3), q032(1e-2) 실행.
- 결론 방향: Book에서 확정한 정렬 가중치 2e-2는 Movie·Yelp로 이식되지 않는다. 데이터셋별 가중치(MOSAIC의 intent_align_weight처럼)로 표를 채운다.

## 2026-11-02 22:30 tick
- q032 yelp persona-only + 1e-2: OOM 실패(물리 GPU 1; 외부 프로세스가 GPU당 2×21 GB, yelp 실행은 21~24 GB). yelp G1 설정(195507, best@57, ep 106 → patience 임박) 종료 후 GPU 0에 자동 재실행하도록 대기 걸어 둠.
- 진행 중: Book q026(2.5e-2) ep 178 val .1657@165(2e-2 pace .1645@138과 비슷, 붕괴 아님), Movie q030(5e-3) ep 97 .0978@75(MOSAIC best val .1018), yelp q031(5e-3) ep 13.
- 새 실행 없음(GPU 메모리 한계). 제안 보류.

## 2026-11-02 22:50
- yelp G1 설정(2e-2): .0384/.0637/.0446/.0526 (-22%) — E_free .003, 그래프 소멸(과한 정렬, K=16). q032(1e-2) GPU 0에 재실행 시작, q031(5e-3) ep 37 val .1139.
- Movie 5e-3 persona-only: .0396/.0624/.0341/.0420 → MOSAIC -4.3% (Movie 변형 중 최고, 10-29 기본 설정 .0421과 동급). best ep 75 vs MOSAIC 147 → 조기 과적합 가설 → q033 (4-view + balance 0.1, 두 expert 유지) 실행.
- Book q026(2.5e-2) ep 265 val .1723 — 붕괴 없이 2e-2 pace.

## 2026-11-02 23:00 tick
- 끝난 실행 없음. 4개 진행: Book 2.5e-2 ep 385 val .1740@330(2e-2 최고 .1745와 동급), yelp 5e-3 ep 55 .1201@54(MOSAIC .1263@54), yelp 1e-2 ep 16 .0359(초반 하강), movie balance .1 ep 67 .0930@60(MOSAIC .0986@54).
- GPU 1 79.5 GB(외부 42 GB + yelp 24 + movie 12) — 새 실행 없음.

## 2026-11-02 23:15 우선순위 변경 (사용자)
- P1 = Movie·Yelp의 G1 최우선, P2 = 세 데이터셋의 G2·G3. constraints.yaml에 phases/achieved(Book G1 기록), GPU 메모리 가드(gpu_min_free_mib: movie 14 GB, yelp 27 GB) 추가. run.py에 `goals`(단계·목표별 현황) 추가, launch가 메모리 부족 시 deferred(exit 3).
- 큐: P1용 q034(movie 4-view), q035(movie 2e-3), q036(movie reg 1e-5), q037(yelp 4-view), q038(yelp 2e-3). 메모리 가드에 따라 순차 실행.
- 현황(goals): movie G1 타깃 N@10 .0457/R@10 .0667, 최고 -4.3%. yelp G1 타깃 N@10 .0701/R@10 .0849(ProMax d=32 +4%, ProMax d=64보다 높음), 최고 -5.1%.

## 2026-11-02 23:30 tick
- Book q026 2.5e-2: +5.3% (< persona-only 2e-2 +8.1/+7.7) → Book 정렬 가중치 축 최종 종료(2e-2 정점).
- Movie q033 4-view + balance .1: -17% (gate가 10 expert에 분산, E_free 축소) → Movie에서도 expert 분산은 손해(balance 축 종료). q035(2e-3) OOM → ready로 복귀.
- 진행 중: Movie q034 4-view 5e-3 ep 112 val .1003@105 (MOSAIC best val .1018 근접), yelp 5e-3 ep 97 .1262@96 (MOSAIC .1339), yelp 1e-2 ep 55 .1184.
- 대기(메모리 가드): q035, q036, q037, q038. GPU 0 1.9 GB / GPU 1 14 GB 여유.

## 2026-11-03 00:00 tick
- Movie q034 4-view 5e-3: -8.1% (one-view 5e-3 -4.3%보다 아래; K=10 동일). Movie view 축 종료, one-view 유지.
- 실행: q035 movie persona-only 2e-3 (OOM 후 재실행). 대기(메모리): q036 movie reg 1e-5, q037 yelp 4-view, q038 yelp 2e-3.
- 진행 중: yelp 5e-3 ep 151 val .1287@147 (MOSAIC .1339, ProMax d64 .1355), yelp 1e-2 ep 100 .1212@60(정체).
- 열린 질문(Movie G1): 모든 knob 변형이 -4% 이하. 남은 tier A 축은 reg(q036)와 2e-3(q035). 그 뒤에는 구조 변경(tier B) 없이는 Movie G1 달성 근거가 없음.

## 2026-11-03 00:30 tick
- 끝난 실행 없음, 새 실행 없음(메모리 가드: GPU 0 1.6 GB, GPU 1 14 GB 여유). 대기: q036, q037, q038.
- 진행 중: yelp 5e-3 ep 208 val .1299@186(MOSAIC .1339), yelp 1e-2 ep 136 .1223@117, movie 2e-3 ep 139 .0917@111(5e-3의 .0955@54보다 아래 → 2e-3은 더 약함).

## 2026-11-03 01:00 tick
- Movie q035 2e-3: -9.4% (5e-3 -4.3%가 정점) → Movie 정렬 축 종료.
- 실행: q036 movie reg 1e-5. yelp 메모리 기준 27→25 GB로 조정(실측 21~24 GB) 후 q037/q038 재시도.
- 진행 중: yelp 5e-3 ep 265 val .1300@222 (MOSAIC .1339), yelp 1e-2 ep 187 .1235.
- Movie G1 남은 tier A: q036 하나. 실패 시 Movie는 구조 변경(tier B) 제안으로 전환: 후보 = (1) Movie에서 gate 붕괴 전 expert 2개를 유지하되 분산 손해를 피하는 top_k=1 하드 라우팅(코드 불필요, 사실 tier A) (2) 정렬 목표를 유저 페르소나 평균이 아닌 facet별로 선택(코드).

## 2026-11-03 01:20
- Yelp q031 persona-only + 5e-3: .0469/.0778/.0540/.0639 → MOSAIC 동급(-0.9%), G1 타깃(ProMax d=32 +4%) 대비 -5.1%. 10-27 기본 설정과 동일: Book 구조는 yelp에서 중립.
- Movie q036 reg 1e-5: 학습 실패(-42%, E_free 수축). Movie G1 tier A 축 전부 소진(정렬 2e-3/5e-3/1e-2/2e-2, view, balance, reg).
- 실행: q037 yelp 4-view 5e-3, q038 yelp 2e-3 (GPU 1). 진행 중: yelp 1e-2(ep 211, val .1235 — 5e-3 .1300보다 아래).
- 제안(tier B, 승인 대기): q039 prior_composition=mean(유저 페르소나 평균 목표), q040 persona_align_target=fixed(고정·detach 텍스트 투영 목표). 둘 다 코드 변경 필요.
- 표(현재 최고): Movie persona-only 5e-3 .0396/.0624/.0341/.0420 (MOSAIC -4.3%), Yelp persona-only 5e-3 .0469/.0778/.0540/.0639 (MOSAIC -0.9%, ProMax32 -5.1%).

## 2026-11-03 01:30 tick
- 끝난 실행 없음, 새 실행 없음. 진행 중: yelp 1e-2 ep 244(val .1235 정체), yelp 4-view 5e-3 ep 16, yelp 2e-3 ep 16.
- ready 없음. P1의 남은 tier A 축이 없어 새 제안은 tier B(q039 mean 목표, q040 고정 목표) 승인 대기.

## 2026-11-03 01:45 규칙 추가 (사용자)
- ProMax 식의 손실(SDR: 텍스트 유사도 분포를 교사로 한 CE, S2DR: 증강 그래프 대조 InfoNCE)과 RLMRec profile KD는 어떤 변형으로도 추가하지 않는다. constraints.yaml forbidden_loss_families + sdr_weight/s2dr_weight 금지 키, CLAUDE.md 3-2 판별 질문 3개, tier B 제안서의 needs에 답 기재 의무. q039/q040에 판별 답 추가.

## 2026-11-03 02:20 q039/q040 승인·구현·실행
- 코드(persona_mosaic.py/yml, 기본값은 기존 동작): prior_composition=mean(유저 = 모든 페르소나의 단위 평균, 아이템 불변), persona_align_target=fixed(정렬 목표 = 유저 페르소나 풀의 PCA-64 고정 투영, stop-grad; InfoNCE·gate 가중 불변). smoke: mean cos 1.0; fixed PCA 81.7% var, 목표 no-grad, MLP^k로의 직접 gradient 0. 패치 저장. 코드는 유지(사용자가 복구를 요구하지 않음).
- 실행(tier B 승인): q039 movie mean 5e-3, q040 yelp fixed 5e-3, q039b yelp mean 5e-3, q040b movie fixed 5e-3 (메모리 가드에 따라 일부 deferred).

## 2026-11-03 02:30 tick
- 끝난 실행 없음, 새 실행 없음(메모리 가드). 대기(승인됨): q040 yelp fixed, q039b yelp mean, q040b movie fixed.
- 진행 중: movie mean 5e-3 ep 55 val .0968@54 (one-view 5e-3 .0955@54, MOSAIC .0986@54 — 약간 위), yelp 4-view ep 52 .1187, yelp 2e-3 ep 49 .1159, yelp 1e-2 ep 292 .1241 정체.

## 2026-11-03 03:00 tick
- Movie q039 mean 목표 5e-3: .0395/.0633/.0345/.0428 → MOSAIC -3.3% (Movie 최고 후보; persona-only 5e-3 -4.3%와 floor 안). 평균 목표는 Movie에서 무차별~약간 위.
- Yelp q032 1e-2: -8.2% (5e-3 -5.1% 아래; E_free 0 — yelp는 1e-2도 과함). yelp 정렬 상향 축 종료.
- 실행(승인됨, 외부 프로세스가 GPU 0에서 빠져 메모리 확보): q040b movie fixed 5e-3, q040 yelp fixed 5e-3. 대기: q039b yelp mean.
- 진행 중: yelp 4-view ep 85 val .1232, yelp 2e-3 ep 85 .1237.

## 2026-11-03 03:30 tick
- Movie q040b 고정 PCA-64 목표 5e-3: -9.2% (ep 54까지 억제 후 회복; 고정 목표는 유효 강도가 커 5e-3이 과함). 규칙대로 q041 movie fixed 2e-3 실행.
- 실행: q039b yelp mean 5e-3 (GPU 0), q041 movie fixed 2e-3. 진행 중: yelp 4-view ep 121 val .1250, yelp 2e-3 ep 118 .1267, yelp fixed 5e-3 ep 40 .1067(억제 조짐).

## 2026-11-03 04:00 tick
- 끝난 실행 없음. 실행: q041 movie fixed 2e-3. 진행 중(yelp 4개): 4-view ep 154 val .1273, 2e-3 ep 151 .1296(5e-3 .1300@222 pace와 유사), fixed 5e-3 ep 73 .1194, mean 5e-3 ep 31 .1163.

## 2026-11-03 04:30 tick
- 끝난 실행 없음, 새 실행 없음(GPU 포화, ready 없음; 5개 결과 대기 중이라 제안 보류).
- 진행 중: yelp 2e-3 ep 184 val .1302@180 (5e-3 .1300@222보다 이른 도달), yelp mean 5e-3 ep 58 .1248@57 (5e-3 .1201@54 위), yelp 4-view ep 190 .1283, yelp fixed 5e-3 ep 100 .1205, movie fixed 2e-3 ep 100 .0963@99 (5e-3 persona-only .0955@54, MOSAIC .1018).

## 2026-11-03 05:00 tick
- Yelp q037 4-view(K=15) 5e-3: .0629 N@10 (G1 -6.1%, MOSAIC -2.4%) — one-view(.0639) 아래 → yelp view 축 종료.
- Movie q041 fixed 2e-3: -7.2% (억제는 짧아졌으나 끝점 낮음, prior 0) → Movie 고정 목표 폐기. 고정 목표는 MLP^k 공동 훈련 효과를 잃는다.
- 실행(승인 코드): q042 yelp mean + 2e-3, q043 movie mean 5e-3 재실행. 진행 중: yelp 2e-3 ep 226 .1302, yelp mean 5e-3 ep 85 .1273(선행), yelp fixed 5e-3 ep 127 .1207(뒤처짐).

## 2026-11-03 05:30 tick
- 끝난 실행 없음, 새 실행 없음(5개 진행, ready 없음, 결과 대기로 제안 보류).
- 진행 중: yelp mean 5e-3 ep 115 val .1301@105 (5e-3 persona-only가 ep 222에 도달한 .1300을 105에 도달 — 가장 빠른 yelp 곡선), yelp 2e-3 ep 265 .1308@249, yelp fixed 5e-3 .1207 정체(폐기 방향), yelp mean+2e-3 ep 34 .1146, movie mean 재실행 ep 106 .0980@78.

## 2026-11-03 06:00 tick
- Yelp q040 고정 목표 5e-3: -14.5% (임베딩 소멸) → Movie(-7.2%, -9.2%)와 함께 **고정 목표 폐기**(노브는 기본값 learned로 유지).
- Movie q043 mean 재실행 .0424 N@10 (-3.3%, 1회차 -3.3%) → **Movie 표 설정 = mean 목표 5e-3** (persona-only보다 1점 위, 재현). Movie G1 미달 유지.
- 실행(tier A, 백본 깊이): q044 yelp mean 5e-3 + 3 layers, q045 movie mean 5e-3 + 2 layers. 진행 중: yelp mean 5e-3 ep 178 val .1314@147(yelp 최고 val), yelp 2e-3 ep 304 .1309, yelp mean+2e-3 ep 73 .1243.

## 2026-11-03 06:30 tick — Yelp 최고 갱신
- **Yelp q039b mean 목표 5e-3: .0479/.0797/.0558/.0656 → MOSAIC +1.9%** (G1 -2.6%, G2 -5.4%). yelp에서 처음 MOSAIC 위. gate는 .73/.018×15로 완전 쏠림 아님.
- Movie q045 2 layers: -5.8% (더 빠른 과적합) → Movie 백본 축 종료. Movie G1 축 전부 소진(-3.3% 상한, mean 5e-3).
- 실행(승인 코드): q046 yelp mean 5e-3 재실행, q047 yelp mean + item personas. 진행 중: yelp 2e-3 ep 340 .1309, yelp mean+2e-3 ep 112 .1286, yelp mean+3 layers ep 28.
- 남은 과제: Yelp G1까지 2.6점, Movie G1까지 3.3점. 둘 다 knob 범위 밖일 가능성이 크며, 다음 tier B 후보는 '유저 평균 + facet 혼합 목표' 또는 Movie의 조기 과적합 자체를 겨냥한 변경.

## 2026-11-03 07:00 tick
- Yelp q038 persona-only 2e-3: .0488/.0793/.0557/.0649 → MOSAIC +1.9%, G1 -2.6% (mean 5e-3과 동급). yelp 정렬은 낮을수록 좋음(2e-3 > 5e-3 > 1e-2 > 2e-2).
- 실행(승인 코드): q048 yelp mean + 1e-3. 진행 중: yelp mean+2e-3 ep 166 val .1311, mean+3 layers ep 49 .1199, mean 재실행 ep 25, mean+item persona ep 25.
- Yelp G1(-2.6%)까지의 남은 축: 가중치 하향(1e-3), 결합(mean+2e-3), 3 layers, item persona — 모두 진행 중.

## 2026-11-03 07:30 tick
- 끝난 실행 없음, 새 실행 없음(5개 yelp 진행, GPU 포화). 진행: mean+2e-3 ep 205 val .1311@147(정체), mean+3 layers ep 73 .1236, mean 재실행 ep 52 .1234, mean+item persona ep 49 .1207, mean+1e-3 ep 37 .1074.

## 2026-11-03 08:00 tick
- 끝난 실행 없음, 새 실행 없음(5개 yelp 진행). mean+2e-3 ep 244 val .1319@222(yelp 최고 val 갱신), mean+3 layers ep 94 .1257, mean 재실행 ep 76 .1280(1회차 .1273@75와 동일 pace), mean+item persona ep 76 .1243, mean+1e-3 ep 73 .1219.

## 2026-11-03 08:30 tick
- 끝난 실행 없음, 새 실행 없음(5개 yelp 진행). mean+2e-3 ep 283 val .1321@273, mean+3 layers ep 115 .1266, mean 재실행 ep 103 .1288, mean+item persona ep 103 .1269, mean+1e-3 ep 112 .1264.

## 2026-11-03 09:00 tick
- 끝난 실행 없음, 새 실행 없음(5개 yelp 진행). mean+2e-3 ep 319 val .1321@273(정체 → 곧 종료), mean+1e-3 ep 148 .1292, mean 재실행 ep 130 .1291, mean+3 layers ep 136 .1287, mean+item persona ep 127 .1284.

## 2026-11-03 09:30 tick — Yelp 최고 갱신
- **Yelp q042 mean + 2e-3: .0482/.0813/.0561/.0662 → MOSAIC +2.9%, G1 -1.6%** (두 요인이 더해짐). yelp 표 후보.
- 실행(승인 코드): q049 mean+2e-3 재실행, q050 mean+3e-3. 진행 중: mean 5e-3 재실행 ep 157 .1310, mean+1e-3 ep 214 .1301, mean+3 layers .1288, mean+item persona .1286.

## 2026-11-03 10:00 tick
- 끝난 실행 없음, 새 실행 없음(6개 yelp 진행, GPU 포화). mean 5e-3 재실행 ep 181 .1310@147, mean+1e-3 ep 241 .1305, mean+3 layers .1298, mean+item persona .1287, mean+2e-3 재실행 ep 22, mean+3e-3 ep 22.

## 2026-11-03 10:30 tick
- 끝난 실행 없음, 새 실행 없음(6개 yelp 진행, GPU 포화). mean 5e-3 재실행 ep 208 .1311, mean+1e-3 ep 268 .1306, mean+3 layers .1298, mean+item persona .1291, mean+2e-3 재실행 ep 49 .1193, mean+3e-3 ep 49 .1202.

## 2026-11-03 11:00 tick
- Yelp q044 mean 5e-3 + 3 layers: .0637 (MOSAIC -1.3%, G1 -6.6%) → 2 layers 유지, yelp 백본 축 종료.
- 새 실행 없음(결과 5개 대기: mean 5e-3 재실행 ep 235 val .1317, mean+1e-3 ep 292 .1314, mean+item persona .1291, mean+2e-3 재실행 ep 73 .1242, mean+3e-3 ep 73 .1246).

## 2026-11-03 11:30 tick
- Yelp q047 mean 5e-3 + item personas: .0652 (G1 -2.8%, MOSAIC +1.1%) — profile과 동급 → yelp item_text=profile 유지.
- 실행: q051 Book best + mean 목표(구조 통일 확인; GPU 여유 활용). 진행 중: yelp mean 5e-3 재실행 ep 301 val .1318, mean+1e-3 ep 319 .1314, mean+2e-3 재실행 ep 100 .1276, mean+3e-3 ep 97 .1278.

## 2026-11-03 12:00 tick
- Yelp q046 mean 5e-3 재실행 .0652 (1회차 .0656) → 재현 폭 .0004; mean 5e-3 = MOSAIC +1.5%(2회 평균). yelp 표 후보는 mean+2e-3(.0662, 재실행 중).
- Book q051 mean 목표: .0891 N@10 (G1 +4.9%) — persona-only .0918보다 2.7점 아래(floor 초과) → Book은 facet 하나(persona-only) 유지. 목표 구성은 데이터셋별(Book facet / Movie·Yelp mean): Book K=2는 facet을 갈라 선택이 유의미, K=10/16에서는 평균이 안정적.
- 실행(GPU 0 비어 P2 활용): q052 Book best @ d=32 (G3 거리 측정). 진행 중: yelp mean+1e-3 ep 343 val .1318, mean+2e-3 재실행 ep 124 .1296, mean+3e-3 ep 124 .1295.

## 2026-11-03 12:30 tick
- Book q052 best @ d=32: .0747 N@10 (G3 -10.0%, d=32 기본 .0757보다 아래) — d=32에서 2e-2는 과한 정렬(ep 48까지 억제). 실행: q053 d=32 + 5e-3.
- 진행 중: yelp mean+1e-3 ep 367 val .1318, mean+2e-3 재실행 ep 148 .1301, mean+3e-3 ep 148 .1304.

## 2026-11-03 13:00 tick
- Yelp q048 mean+1e-3: .0659 (G1 -1.8%) — 2e-3과 동급 → yelp 가중치 축 종료(1e-3~5e-3 평탄, 2e-3 고정). G1까지 1.6점은 가중치로 불가.
- Book q053 d=32 + 5e-3: .0713 (G3 -13.6%) — 2e-2(.0747)보다도 아래, d=32 기본(.0757)이 최선. Book 구조 이득은 d=64 전용. G3 보류.
- 실행: q054 yelp mean+2e-3 + learned gate(gate 종류 ablation, 성능 가능성). 진행 중: yelp mean+2e-3 재실행 ep 175 val .1301, mean+3e-3 ep 175 .1304.
- 열린 질문: Yelp G1(-1.6%)과 Movie G1(-3.3%)은 현재 knob·승인 코드로 소진에 가까움. 다음 tier B 후보가 필요.

## 2026-11-03 13:20 규칙 변경 (사용자)
- P1(Movie·Yelp G1)이 끝날 때까지 Book 실험 금지 — GPU가 비어도. constraints phase_lock: true, run.py launch가 현재 단계 밖 데이터셋의 persona_mosaic 실행을 거부(검증: Book launch 거부 확인). CLAUDE.md §0의 "GPU 남을 때" 예외 삭제.

## 2026-11-03 13:30 tick
- Yelp q050 mean+3e-3: .0659 (G1 -1.8%) → yelp mean 가중치 곡선 평탄(1e-3~5e-3), 2e-3 고정. 가중치 축 종료.
- 실행(tier A): q055 yelp mean 5e-3 + 온도 .5(부드러운 InfoNCE; 미시험 방향). 제안(tier B, 승인 대기): q057 item_text=audience — 아이템 목표 = 훈련 유저들의 페르소나 평균.
- 진행 중: yelp mean+2e-3 재실행 ep 226 val .1315, yelp learned gate ep 76 .1251.

## 2026-11-03 14:00 tick
- 끝난 실행 없음. 실행(tier A): q056 movie mean 5e-3 + 온도 .5 (yelp q055와 짝). 승인 대기: q057(item_text=audience).
- 진행 중: yelp mean+2e-3 재실행 ep 265 val .1317(1회차 .1321@273과 동일 pace), yelp learned gate ep 157 .1304(residual .1301@147과 동급), yelp 온도 .5 ep 34 .1025.

## 2026-11-03 14:30 tick
- Movie q056 온도 .5: -24% → 온도 축 완전 종료(.2 고정). yelp 온도 .5(q055)도 뒤처짐(ep 69 .1184).
- 새 실행 없음. Movie G1: knob·승인 코드 축 전부 소진(-3.3% 상한). 승인 대기: q057 item_text=audience.
- 진행 중: yelp mean+2e-3 재실행 ep 304 val .1326@273(1회차 초과), yelp learned gate ep 205 .1316@168(residual과 동급 이상 — 결과 주목).

## 2026-11-03 15:00 tick
- Yelp q049 mean+2e-3 재실행: .0652 N@10 (1회차 .0662) → 2회 평균 .0657, MOSAIC +2.1%, G1 -2.5%. yelp 표 설정 확정(mean + 2e-3). 1회차의 +2.9%는 유리한 편차 포함.
- 새 실행 없음(tier A 축 소진; q057 승인 대기). 진행 중: yelp learned gate ep 286 val .1323@228(residual 2회 .1321/.1326과 동급), yelp 온도 .5 ep 118 .1252(뒤처짐).

## 2026-11-03 15:25 q057 승인·구현·실행
- item_text=audience 구현(persona_mosaic.py/yml; 기본값 profile 불변): 아이템 행 = 훈련 상호작용(trn_mat)의 유저들이 가진 페르소나 평균의 단위 평균(클러스터 공간·prior 공간 각각), 훈련 유저 없는 아이템은 0. smoke(Book): 아이템 0(훈련 유저 9명) 행이 기대값과 cos 1.0, 손실·진단 정상. 패치 저장.
- 실행(승인): q057 yelp mean+2e-3+audience, q057b movie mean+5e-3+audience.

## 2026-11-03 16:00 tick
- Yelp q054 learned gate: .0647 (G1 -3.2%) — residual(2회 평균 .0657)보다 1점 아래 → residual 유지, gate 축 종료.
- 새 실행 없음. 진행 중: yelp 온도 .5 ep 184 val .1293(뒤처짐), yelp audience ep 1, movie audience ep 49 .0960@45(mean 5e-3 .0968@54과 동급 pace).

## 2026-11-03 16:30 tick
- Movie q057b audience 아이템 목표: .0393 N@10 (MOSAIC -10.7%) — mean 5e-3(-3.3%)보다 7점 아래 → Movie에서 audience 폐기. yelp q057(ep 33 val .1192, mean+2e-3 .1146@34보다 약간 위) 결과 대기.
- 새 실행 없음. 진행 중: yelp 온도 .5 ep 217 .1304(뒤처짐).

## 2026-11-03 17:00 tick
- 끝난 실행 없음, 새 실행 없음(외부 프로세스가 GPU 0을 다시 67 GB 점유). 진행 중: yelp audience ep 61 val .1244@60 (mean+2e-3 .1243@72보다 이른 도달), yelp 온도 .5 ep 244 .1304 정체.

## 2026-11-03 17:30 tick
- Yelp q055 mean 5e-3 + 온도 .5: .0661 N@10 (G1 -1.7%) — mean+2e-3과 동급(Movie와 달리 손해 아님). 실행: q058 yelp mean 1e-2 + 온도 .5 (부드러운 손실이 더 센 가중치를 수용하는지).
- 진행 중: yelp audience ep 97 val .1295@96 (mean+2e-3 .1296@120보다 빠름 — 주목).

## 2026-11-03 18:00 tick
- 끝난 실행 없음, 새 실행 없음. 진행 중: yelp audience ep 121 val .1299@114, yelp 1e-2+온도 .5 ep 25 .1013(초반 하강 구간).

## 2026-11-03 18:10 규칙 보강 (사용자)
- GPU에 자리가 있는데 큐가 비어 있으면 "결과 대기"로 보류하지 않는다(소거 실험이라도 올린다). CLAUDE.md 5-7, SKILL 4 수정.
- 실행: q059 yelp audience + 5e-3, q060 movie mean 5e-3 + learned gate.

## 2026-11-03 18:30 tick
- 끝난 실행 없음. 실행(소거): q061 yelp mean+2e-3 top_k=1(혼합 효과 소거), q062 movie mean 7e-3(정점 정밀화). 진행 중: yelp audience 2e-3 ep 148 val .1320@135(mean+2e-3 .1311@147보다 위), yelp 1e-2+온도 .5 ep 49 .1159, yelp audience 5e-3 ep 7, movie learned gate ep 37 .0930.

## 2026-11-03 19:00 tick
- Movie q060 learned gate: .0425 (-3.3%) = residual → Movie gate 축 종료.
- 실행(소거): q063 movie mean 5e-3 + warm-up 15, q064 yelp mean+2e-3 + warm-up 15. 진행 중: yelp audience 2e-3 ep 181 val .1320@135, yelp audience 5e-3 ep 31 .1233, yelp top_k=1 ep 22, yelp 1e-2+온도 .5 ep 82 .1241, movie 7e-3 ep 88 .0979@78.

## 2026-11-03 19:30 tick
- Yelp q057 audience 2e-3: .0652 (G1 -3.9%) — profile과 동급(val 최고였으나 test 불변) → q059(5e-3) 결과까지 보류, 동급이면 audience 폐기.
- **Movie q062 mean 7e-3: .0403/.0651/.0347/.0434 → MOSAIC -1.6%** (Movie 최고, R@10은 MOSAIC 초과). 실행: q065 8e-3, q066 7e-3 재실행.
- yelp warm-up(q064) OOM → 재실행. 진행 중: yelp 1e-2+온도 .5 ep 118 .1264, yelp audience 5e-3 ep 61 .1269, yelp top_k=1 ep 52 .1053(뒤처짐), movie warm-up ep 139 .0938@84(뒤처짐).

## 2026-11-03 20:00 tick
- Movie q063 warm-up 15: .0410 (-5.5%) → Movie warm-up 축 종료. 새 실행 없음(GPU 포화: 77/62 GB, 6개 진행).
- 진행 중: movie 7e-3 재실행 ep 97 val .0985@87, movie 8e-3 ep 106 .0980@72, yelp audience 5e-3 ep 88 .1295@84, yelp 1e-2+온도 .5 ep 145 .1280, yelp top_k=1 ep 76 .1071(뒤처짐), yelp warm-up ep 25.

## 2026-11-03 20:30 tick — Movie 표 설정 확정
- Movie q065 8e-3: -3.5%; q066 7e-3 재실행: -2.0% → **7e-3 2회 평균 .0406/.0644/.0349/.0431 = MOSAIC -1.8%, G2 +0.8%**. Movie 표 설정 = residual + one-view + mean 목표 + 7e-3. 정점 7e-3 확정(가중치 축 종료). 실행: q067 6e-3(곡선 채움).
- 진행 중: yelp audience 5e-3 ep 106 val .1298, yelp 1e-2+온도 .5 ep 181 .1301, yelp top_k=1 .1101(뒤처짐), yelp warm-up ep 58 .1145.

## 2026-11-03 21:00 tick
- 끝난 실행 없음, 새 실행 없음(GPU 0 76 GB·GPU 1 51 GB, yelp 25 GB 불가). 진행 중: yelp 1e-2+온도 .5 ep 214 .1301 정체, yelp audience 5e-3 ep 127 .1298 정체, yelp top_k=1 .1118(뒤처짐), yelp warm-up ep 91 .1212, movie 6e-3 ep 70 .0969@51.

## 2026-11-03 21:30 tick
- 끝난 실행 없음, 새 실행 없음(GPU 포화). 진행 중: yelp audience 5e-3 ep 145 .1316@132, yelp 1e-2+온도 .5 ep 247 .1307, yelp warm-up ep 124 .1243, yelp top_k=1 .1118 정체(혼합이 유효하다는 신호), movie 6e-3 ep 145 .0983@87.

## 2026-11-03 22:00 tick
- Movie q067 6e-3: -4.3% → 7e-3 좁은 정점 확정, Movie 가중치 축 종료.
- 실행(소거, 새 축 'modulation 강도'): q068 movie 7e-3 + layer_modulation_scale .5, q069 scale 0(modulation off). 근거: Movie·Yelp E_free 노름 .003(Book 1.0) — 퇴화한 마스크가 차원을 과하게 끄는지.
- 진행 중: yelp audience 5e-3 ep 166 .1316, yelp 1e-2+온도 .5 ep 280 .1312, yelp warm-up ep 157 .1243, yelp top_k=1 .1132.

## 2026-11-03 22:30 tick
- 끝난 실행 없음, 새 실행 없음(GPU 포화, 6개 진행). 주목: movie modulation scale .5가 ep 54에 val .1001(7e-3 scale 1.0의 .0985@87보다 이르고 높음 — MOSAIC .1018 근접). scale 0은 .0968@93. yelp audience 5e-3 .1316 정체, yelp 1e-2+온도 .5 .1312 정체, yelp warm-up .1249, yelp top_k=1 .1147(뒤처짐).

## 2026-11-03 23:00 tick — P1 tier A 소진
- Yelp q058 1e-2+온도 .5: -4.6% → 온도 축 최종 종료. Yelp q059 audience 5e-3: .0652(동급) → audience 폐기(아이템 텍스트 profile 유지).
- Movie q068 modulation .5: -8.5%, q069 modulation off: -8.6% → modulation 경로 필요, scale 1.0 유지. (정정: 노름 .003은 미사용 expert 값; 사용 expert는 노름 ~1, 마스크 활성.)
- **P1 현황**: Movie 최고 mean 7e-3 (2회 평균 -1.8% vs MOSAIC), Yelp 최고 mean 2e-3 (2회 평균 -2.5% vs ProMax d=32 +4% 타깃). knob·승인 코드 축 소진(가중치, 온도, view, 깊이, gate 종류, warm-up, balance, reg, 고정 목표, audience, modulation).
- 제안(tier B, 승인 대기): q070 persona_align_side=user(아이템 항 제거), q071 정렬 가중치 선형 감쇠(초반 강, 후반 0). 진행 중: yelp top_k=1(.1162, 뒤처짐), yelp warm-up(.1271).

## 2026-11-03 23:30 tick — yelp에서 MoE 혼합의 첫 증거
- Yelp q061 top_k=1: .0562 N@10 (G1 -16.6%) — 정정: balance 손실이 하드 라우팅을 16 expert에 균등 분산(각 1/16 데이터) → 'expert 분할은 손해'의 네 번째 확인이지 혼합 유효성의 증거는 아님. 실행: q072 top_k=3, q073 top_k=4 (새 축 '혼합 폭').
- 진행 중: yelp warm-up ep 265 .1274. 승인 대기: q070, q071.

## 2026-11-03 23:50 규칙 변경 (사용자) — 공유 방법론, 단일 실행
- 방법론은 세 데이터셋에서 동일(구조 노브 = constraints shared_structure: residual gate, one-view, mean 목표, profile 아이템, learned 정렬 목표, modulate, silhouette K, raw 4096, top_k 2 …); 하이퍼파라미터(정렬 가중치 등)만 데이터셋별. 벗어난 실행은 report에 struct!= 표시, 후보 불가. 구조 채택은 세 데이터셋 검증 후 shared_structure 갱신.
- 재실행 평균 폐지: 단일 실행 +4% = 달성("REACHED->add" → achieved 기록).
- 영향: Book 후보는 mean 목표 실행(q051 +4.9%, ACHIEVED)으로 바뀜(facet 목표 +8.1%는 struct!=). P1 = 세 데이터셋 G1; 열린 쌍 Movie(-1.8%), Yelp(-2.5%). 진행 중 yelp top_k=3/4는 구조 탐색(채택하려면 Book·Movie도 필요).

## 2026-11-04 00:20 승인 반영 (사용자): q070/q071 구현·실행, top_k 규칙
- top_k는 >= 2이면 데이터셋별 하이퍼파라미터(shared_structure에서 제거, value_rules min 2). 구조 채택 시 다른 데이터셋에서 성능이 안 나오면 그 데이터셋 하이퍼파라미터를 재조정.
- 코드(persona_mosaic.py/yml, 기본값 불변): persona_align_side(both|user|item), persona_align_decay_epochs(N>0: 가중치 선형 감쇠, set_epoch). smoke: both = user + item 분해 정확, decay 60: 2e-2 → 1e-2(ep30) → 0(ep60). 패치 저장.
- 실행(승인): q070 yelp side=user 5e-3, q070b movie side=user 7e-3, q071 movie 2e-2 decay 60, q071b yelp 2e-2 decay 60.

## 2026-11-04 00:30 tick
- 끝난 실행 없음. 승인 실행 q070(yelp side=user), q071(movie decay), q071b(yelp decay)는 메모리 가드로 대기(GPU 0 1.8 GB, GPU 1 0 GB 여유 — 외부 프로세스). 진행 중: yelp top_k=3/4 ep 34, yelp warm-up ep 319 .1287, movie side=user ep 1.

## 2026-11-04 01:00 tick
- Yelp q064 warm-up 15: .0644 (G1 -4.5%) → warm-up 축 종료(Movie와 동일).
- 실행(승인): q070 yelp side=user 5e-3. 대기(메모리): q071 movie decay, q071b yelp decay. 진행 중: yelp top_k=3 ep 63 .1224, top_k=4 .1238(top_k=2 .1243@72 pace와 동급), movie side=user ep 178 .0944@132(mean 7e-3 .0985보다 아래).

## 2026-11-04 01:30 tick
- Movie q070b side=user 7e-3: .0381 N@10 (MOSAIC -13%) → 아이템 쪽 정렬 항은 필수(빼면 11점 손해). yelp q070(진행 중 ep 42 .1193)도 같은 방향이면 side 축 폐기.
- 실행(승인): q071 movie 2e-2 decay 60. 대기: q071b yelp decay. 진행 중: yelp top_k=3 ep 96 .1263, top_k=4 .1280(top_k=2 .1276@93 동급).

## 2026-11-04 02:00 tick
- 끝난 실행 없음, 새 실행 없음(q071b yelp decay는 메모리 대기). 진행 중: yelp top_k=3 ep 126 .1280, top_k=4 .1287 (top_k=2 .1288@126과 동급), yelp side=user ep 72 .1252(mean+2e-3 .1243@72과 동급), movie decay ep 133 .0964@81(7e-3 .0985보다 아래).

## 2026-11-04 02:30 tick
- Movie q071 2e-2 decay 60: .0408 (-5.9%) — 상수 7e-3(-1.8%)보다 아래 → 감쇠 스케줄은 Movie에서 폐기 방향(yelp q071b 대기, 메모리 23.4 GB < 25 GB로 deferred).
- 진행 중: yelp top_k=3 ep 166 .1297, top_k=4 .1307@147(top_k=2 .1302@180과 동급), yelp side=user ep 127 .1295@96(2e-3 .1296@120과 동급).

## 2026-11-04 03:00 tick
- 끝난 실행 없음. 실행: q071b yelp 2e-2 decay 60(승인), q074/q075 movie top_k=3/4(소거; GPU 1 여유 활용). 진행 중: yelp top_k=3 ep 186 .1302, top_k=4 .1309, yelp side=user ep 141 .1309 — 셋 다 top_k=2 기준(.1326 최고)과 동급.

## 2026-11-04 03:30 tick
- 끝난 실행 없음. 실행: q075 movie top_k=4. 진행 중: yelp top_k=4 ep 222 .1324(top_k=2 최고 .1326 동급), top_k=3 .1309, side=user .1309@141 정체, yelp decay ep 24 .0363(초반 억제), movie top_k=3 ep 106 .0986@54(7e-3 .0985 동급).

## 2026-11-04 04:00 tick
- Movie q074 top_k=3: .0428 (-2.6%) — top_k=2와 floor 안 → Movie top_k=2 유지. Movie tier A 축 소진(GPU 1 24 GB 여유에도 올릴 가설 없음 → tier B 제안).
- 제안(tier B, 승인 대기): q076 prior_composition=mix(facet + mean 단위 합) — Book(facet 우세)과 Movie·Yelp(mean 우세)를 하나의 공유 구조로 묶을 후보.
- 진행 중: yelp top_k=4 ep 246 .1326(top_k=2 최고와 동일), top_k=3 .1312, side=user .1316, yelp decay ep 45 .1201(억제 해제), movie top_k=4 ep 96 .0984.

## 2026-11-04 04:30 tick — Movie 최고 갱신
- **Movie q075 mean 7e-3 + top_k=4: .0427/.0628/.0363/.0431 → MOSAIC -0.1%** (top_k=2 -1.8%; R@5·N@5는 MOSAIC 초과). Movie top_k=4 채택(데이터셋별 하이퍼파라미터). G1까지 4점.
- 실행: q077 movie top_k=6, q078 top_k=8 (혼합 폭 곡선 확장). 진행 중: yelp top_k=4 .1326(최고 동급), top_k=3 .1313, side=user .1316, yelp decay ep 87 .1263.

## 2026-11-04 05:00 tick — Yelp 최고 갱신
- **Yelp q073 mean+2e-3 top_k=4: .0492/.0815/.0566/.0665 → G1 -0.7%, MOSAIC +3.9%** (top_k=2 -1.6%). yelp top_k=4 채택. Movie와 같은 방향(top-4 혼합 +1.7점).
- Yelp q070 side=user 5e-3: .0660 (-1.4%, both와 동급) → Movie -13%와 함께 side 축 폐기(both 유지).
- 실행: q079 yelp top_k=6, q080 top_k=8. 진행 중: yelp top_k=3 ep 318 .1321, yelp decay ep 120 .1280, movie top_k=6 ep 81 .0976, top_k=8 .0992.
- P1 현황: Book 달성(+4.9%), Movie -0.1%, Yelp -0.7% — 둘 다 1점 안.

## 2026-11-04 05:30 tick
- Yelp q072 top_k=3: .0655 (-1.6%) → 4 채택 유지. Movie q077 top_k=6: -3.9%, q078 top_k=8: -2.4% → Movie top_k=4 확정(곡선 2 -1.8, 3 -2.6, 4 -0.1, 6 -3.9, 8 -2.4).
- 실행: q080 yelp top_k=8, q081 movie top_k=4 + 5e-3, q082 movie top_k=4 + 1e-2 (새 혼합 폭에서 가중치 재조정). 진행 중: yelp top_k=6 ep 21, yelp decay ep 147 .1294.

## 2026-11-04 06:00 tick
- 끝난 실행 없음. 실행: q082 movie top_k=4 + 1e-2. 진행 중: movie top_k=4 + 5e-3 ep 100 .0978@51, yelp top_k=6 ep 42 .1196, top_k=8 ep 60 .1216(top_k=4 .1238@63 pace보다 약간 아래), yelp decay ep 156 .1295.

## 2026-11-04 06:30 tick
- Movie q081 top_k=4 + 5e-3: .0411 (-5.5%) → 7e-3 유지. 실행: q083 movie top_k=4 + 8e-3. 진행 중: movie top_k=4 + 1e-2 ep 136 .0966@78(아래 조짐), yelp top_k=6 ep 84 .1266, top_k=8 ep 87 .1272(top_k=4 .1295@84 아래), yelp decay ep 222 .1310.

## 2026-11-04 07:00 tick
- Movie q082 top_k=4 + 1e-2: .0431 (-1.6%) → 7e-3 유지(top-4는 센 정렬을 top-2보다 잘 견딤: 1e-2가 -12.6% → -1.6%). 실행: q084 movie top_k=5.
- 진행 중: movie top_k=4 + 8e-3 ep 100 .0971@87, yelp top_k=6 ep 112 .1296, top_k=8 ep 160 .1299, yelp decay ep 256 .1314.

## 2026-11-04 07:30 tick
- Movie q083 top_k=4 + 8e-3: -0.3% (7e-3 -0.1%와 동급); q084 top_k=5: -1.4% → Movie 표 설정 = mean + top_k=4 + 7e-3 (G1 -0.1%). Movie 가중치·폭 축 소진; 남은 후보 q076(mix 목표, 승인 대기).
- 큐: q085 yelp top_k=4 + 5e-3 (메모리 가드 가능). 진행 중: yelp top_k=6 ep 102 .1296, top_k=8 ep 168 .1305, yelp decay ep 228 .1314.

## 2026-11-04 08:00 tick
- Yelp q071b 2e-2 decay 60: .0656 (G1 -2.7%) — 상수 2e-3과 동급, Movie -5.9% → 감쇠 스케줄 축 폐기.
- 실행: q085 yelp top_k=4 + 5e-3. 진행 중: yelp top_k=6 ep 177 .1313, top_k=8 ep 258 .1310 (top_k=4 최고 .1326 아래).

## 2026-11-04 08:30 tick
- Yelp q080 top_k=8: .0658 (G1 -1.4%) → 4 유지. 실행: q086 yelp top_k=4 + 1e-2, q087 top_k=4 + 3e-3 (새 폭에서 가중치 곡선). 진행 중: yelp top_k=6 ep 249 .1320, yelp top_k=4 + 5e-3 ep 33 .1169.

## 2026-11-04 09:00 tick
- 끝난 실행 없음, 새 실행 없음(GPU 포화: 57/69 GB, yelp 25 GB 자리 없음). 진행 중: yelp top_k=6 ep 249 .1320 정체, yelp top_k=4+5e-3 ep 63 .1236(top_k=4 2e-3 .1238@63과 동급), top_k=4+1e-2 ep 30 .1104, top_k=4+3e-3 .1120.

## 2026-11-04 07:08 tick — Yelp 폭 축 소진, 변조 상한 축 개시
- Yelp q079 top_k=6: .0491/.0810/.0567/.0665 → G1 -0.9% (top_k=4 -0.7%와 잡음 안, 차이 없음) → **yelp top_k=4 확정**. 폭 곡선 1 -17.1, 2 -1.6, 3 -1.9, 4 -0.7, 6 -0.9, 8 -1.6: 2 이상은 평탄. gate는 ep 30에 expert 9로 쏠림(공통), swap overlap .974.
- 진단 발견: Movie·Yelp 최고 실행 모두 사용 expert의 modulation lambda가 모든 층에서 1.000 = `layer_modulation_scale` 상한에 붙어 있고 tanh도 포화(mean|gate-1| .87~.995). Movie scale 곡선 0 -7.6 / 0.5 -9.4 / 1.0 -0.1 → 상한을 올리는 것이 남은 하이퍼파라미터 축.
- 실행: q088 movie top_k=4+7e-3+scale 2.0, q089 yelp top_k=4+2e-3+scale 2.0 (GPU 1). 진행 중: yelp top_k=4 + 5e-3 ep 136 .1284@129, + 1e-2 ep 73 .1218, + 3e-3 ep 70 .1244.
- 승인 대기: q076 mix 목표(tier B). Movie 가중치·폭 축은 소진 상태(7e-3 -0.1 / 8e-3 -0.2).

## 2026-11-04 07:31 tick
- 끝난 실행 없음. 실행: q090 movie top_k=4+7e-3 + keep_rate 0.8 (edge dropout, 어느 데이터셋에서도 처음; 조기 과적합 대응; GPU 0). yelp는 GPU 자리 없음(0: 24.6 GB, 1: 10 GB free < 25 GB).
- 진행 중: yelp top_k=4 + 5e-3 ep 166 .1304@147(2e-3의 .1326 아래 pace), + 1e-2 ep 103 .1225, + 3e-3 ep 103 .1284@102, movie scale 2.0 ep 79 .0958@66(1.0의 .0984@96보다 아래 pace), yelp scale 2.0 ep 19.
- 승인 대기: q076 mix 목표(tier B).

## 2026-11-04 08:03 tick — Movie 변조 상한 축 종료
- Movie q088 scale 2.0: .0408/.0591/.0355/.0415 → G1 -4.1% (scale 1.0 -0.1%, R@10·N@10 잡음 밖 하락). 최종 lambda [1.24,1.09,1.00,1.03] — 상한 2.0을 거의 안 씀, 1.0 상한이 사실상 자유 최적값. 손해는 초기 궤적(쏠림 ep 30→60, |prior| ↑, item swap overlap .74)에서. → Movie scale 1.0 유지, 축 종료. Movie 곡선 0 -7.6 / 0.5 -9.4 / 1.0 -0.1 / 2.0 -4.1.
- 실행: q091 movie top_k=4+7e-3 + 변조 초기값(yelp 블록 값 lambda_init 5e-2, init_std 1e-2; Movie에서 처음) GPU 1.
- 하네스: 외부 사용자 프로세스가 사라져 GPU 메모리는 전부 하네스 실행. yelp 프로세스가 21~35 GB까지 커져 가드를 yelp 36 GB / movie 16 GB로 올림(OOM 예방; yelp 추가 실행은 자리 날 때까지 deferred).
- 진행 중: yelp top_k=4 + 5e-3 ep 199 .1304@147(2e-3 .1326 아래), + 1e-2 ep 130 .1226, + 3e-3 ep 130 .1286@129, yelp scale 2.0 ep 52 .1175(2e-3의 .1204@51 약간 아래), movie keep_rate .8 ep 103 .0985@96(기본 .0984@96와 동급).
- 승인 대기: q076 mix 목표(tier B).

## 2026-11-04 08:32 tick — Movie tier A 축 거의 소진
- Yelp q085 top_k=4 + 5e-3: .0466/.0796/.0543/.0647 → G1 -4.3% (2e-3 -0.7% 대비 -3.6점, 잡음 밖) → 2e-3 유지. top_k=2와 달리 top_k=4에서 센 정렬은 손해(Movie와 반대).
- Movie q090 keep_rate .8: G1 -0.6%, best epoch 96 그대로 → 잡음 안, edge dropout 축 종료. Movie q091 변조 초기값(5e-2/1e-2): G1 -0.4% → 잡음 안, 축 종료 (|prior| 두 배, item swap overlap .81→.70으로 prior를 더 쓰지만 지표 불변).
- 반복 관찰: prior 경로를 더 세게 쓰게 하는 변경(scale 2.0, 초기값 ↑)은 swap 민감도를 올리지만 성능은 같거나 하락 → 페르소나 정보가 그래프와 중복이라는 가설 쪽.
- 실행: q092 yelp top_k=4 + 1e-3 (가중치 곡선 아래 점; GPU 1), q093 movie top_k=4 + 7e-3 + layer_num 4 (Movie 마지막 tier A 축; GPU 0).
- 진행 중: yelp top_k=4 + 1e-2 ep 163 .1226(나쁨), + 3e-3 ep 160 .1310@147(2e-3 .1326 아래), yelp scale 2.0 ep 94 .1185@60(정체, 나쁨).
- Movie: q093이 잡음 안이면 tier A 소진 → tier B(q076 mix 목표) 승인만 남음. 승인 대기: q076.

## 2026-11-04 09:08 tick — 변조 상한 축 양쪽 종료
- Yelp q089 scale 2.0: .0416/.0712/.0490/.0585 → G1 -14.0% (2e-3·scale 1.0 -0.7% 대비 -13.3점). 사용 expert 9의 lambda가 1.5~1.8까지 올라 층 배율 1+lambda·tanh가 음수(최소 -0.84)가 됨 → 임베딩 붕괴, val .1185@60 정체. 예측(-3~+2%) 빗나감, "더 나쁨 → 축 종료" 가지. 교훈: lambda ≤ 1 상한은 배율 비음수 조건이지 성능 상한이 아니다(Movie 2.0도 -4.1%). 변조 상한 축 Movie·Yelp 모두 종료.
- 실행: q094 yelp top_k=4 + 2e-3 + 온도 .5 (새 폭에서 온도 축; GPU 1). q095 movie top_k=4 + 7e-3 + reg 3e-6 (과적합 대응; 1e-5는 옛 구조에서 붕괴) → 메모리 가드 deferred(빈 GPU 8.9/7.3 GB < 16 GB), ready 유지.
- 진행 중: yelp 1e-2 ep 190 .1243@174(나쁨), yelp 3e-3 ep 190 .1317@168(2e-3 .1326 아래), yelp 1e-3 ep 49 .1160, movie layer_num 4 ep 85 .0973@84(3층 .0984@96 아래 pace).
- Movie tier A 축: layer_num 4·reg 3e-6 뒤에는 남는 가설 없음(가중치·폭·scale·초기값·keep_rate·온도 .5·warm-up·balance·side·decay·고정 목표 모두 소진). 둘이 잡음 안이면 "소진" 기록 → tier B(q076 mix 목표)만 남음. 승인 대기: q076.

## 2026-11-04 09:30 tick — Movie 깊이 축 종료
- Movie q093 layer_num 4: .0395/.0654/.0343/.0434 → G1 -2.2% (3층 -0.1%; @5 하락 경계 밖, @10 잡음 안 → 과평활). 예측 범위 안이지만 이득 없음 → 3층 확정. 깊이 곡선 2 -5.8 / 3 -0.1 / 4 -2.2.
- 실행: q095 movie top_k=4 + 7e-3 + reg 3e-6 (GPU 0; 전 tick deferred 분). 그 뒤 빈 메모리 GPU0 ~8 GB / GPU1 ~6 GB → yelp(36 GB) 추가 불가, 새 항목은 올리지 않음(메모리 가드 보류).
- 진행 중: yelp 1e-2 ep 223 .1243@174(나쁨), yelp 3e-3 ep 220 .1317@168(2e-3 .1326 아래), yelp 1e-3 ep 100 .1258@99, yelp 온도 .5 ep 25 .0954.
- Movie tier A: reg 3e-6이 마지막 축. 잡음 안이면 "소진" → tier B(q076 mix 목표)만 남음. 승인 대기: q076.

## 2026-11-04 09:50 — q076 승인·구현·실행, Movie tier A 소진
- 사용자 09:48 "승인해" → `prior_composition=mix` 구현(유저 행 = unit(facet_k + 평균), 아이템 불변, swap 진단은 index·mean 테이블 함께 permute; PRIOR_COMPOSITIONS·docstring·yml 주석·constraints enum). 패치 harness/patches/2026-11-04-q076-mix-composition.patch(74줄). smoke 3분: prior=mix 요약, 10 epoch, val R@20 .057→.063@3, NaN·오류 없음.
- 실행: q076 movie mix(top_k=4 추가: 현 Movie 하이퍼파라미터) GPU 0. q096 yelp mix(2e-3, top_k=4; 같은 승인의 세-데이터셋 후보) → 메모리 가드 deferred(36 GB 자리 없음), ready 유지. Book은 phase_lock — Movie·Yelp가 mean 이상이면 사용자에게 예외를 묻는다.
- Movie q095 reg 3e-6: .0278/.0435/.0227/.0285 → G1 -34.3%, best epoch 3, |E_free| .07~.11로 임베딩 붕괴(1e-5와 같은 양상). reg 축 종료. gate가 균등으로 남은 유일한 실행(엔트로피 .84) → 쏠림은 임베딩 성장 의존.
- **Movie tier A 소진**: 가중치(5e-3~1e-2)·폭(2~8)·scale(0~2)·초기값·keep_rate·깊이(2~4)·reg·온도 .5·warm-up·balance·side·decay·고정 목표·audience·learned gate 모두 잡음 안 또는 하락. 남은 것은 구조 변경(mix 실행 중) 뿐.
- 진행 중: yelp 1e-2(ep ~240, .1243@174), 3e-3(.1317), 1e-3(ep ~110), 온도 .5(ep ~40), movie mix(시작).

## 2026-11-04 10:01 tick — Yelp mix 실행, 3e-3 차이 없음
- Yelp q087 top_k=4 + 3e-3: .0479/.0811/.0553/.0656 → G1 -2.4% (2e-3 -0.7%와 네 지표 모두 잡음 안 → 차이 없음, 2e-3 유지). 곡선 2e-3 -0.7 / 3e-3 -2.4 / 5e-3 -4.3 / 1e-2(val .1246, 나쁨) / 1e-3 진행 중 → 정점 ≤ 2e-3.
- 실행: q096 yelp mix(2e-3, top_k=4; 승인된 q076의 Yelp 짝) GPU 0. 그 뒤 빈 메모리 GPU0 ~10 GB / GPU1 ~6 GB → 추가 실행 없음.
- Movie mix(q076) 초반 곡선 점검: val R@20 ep 3 .0634 → 24 .0531, mean 최고 실행도 같은 구간에서 .0622 → .0519(ep 33 .0469까지 내려갔다가 ep 96 .0984로 회복) → 정상 Movie 궤적, 버그 신호 없음. 결과는 ~1시간 뒤.
- Yelp 남은 tier A(reg·keep_rate·초기값 약화)는 기대값이 낮다(Movie에서 reg는 붕괴, 나머지는 무변화). mix 결과까지 새 제안 보류(메모리도 없음).
- 진행 중: yelp 1e-2 ep 253 .1246@231, yelp 1e-3 ep 136 .1281@129, yelp 온도 .5 ep 61 .1129@60, movie mix ep 22, yelp mix 시작.

## 2026-11-04 10:31 tick — Movie mix 결과: mean보다 -3.6점
- Movie q076 mix + 7e-3: .0396/.0621/.0347/.0425 → G1 평균 ≈ -3.7% (수동 계산; struct!=라 report Gn 비움), mean 7e-3(-0.1%) 대비 R@5 -.0031(경계), 나머지 잡음 안. swap 민감도 .982(mean .966)로 prior 의존이 오히려 감소, 3층째 변조 꺼짐, best epoch 60. 예측(-3~0%) 하단 바로 밖. Movie 단독으로는 "폐기" 가지이나 사용자 규칙(데이터셋별 재조정)대로 1e-2 한 점(q098) + Yelp mix 결과 뒤 최종 판정.
- 실행: q097 yelp top_k=4 + 2e-3 + keep_rate .8 (Yelp 미시험 정규화 축, 소거; GPU 0). q098 movie mix + 1e-2 → 큐(메모리 가드면 다음 tick).
- 진행 중: yelp 1e-2 ep 280 .1246@231(종료 임박), yelp 1e-3 ep 172 .1291@153, yelp 온도 .5 ep 97 .1201@93, yelp mix ep 25 .1060@24, yelp keep_rate 시작.
- 판정 계획: Yelp mix(~3h) ≥ mean(-0.7%, 잡음 안) 이고 Movie 재조정이 mean 이상이면 Book 예외 요청; 아니면 mix 폐기, 공유 구조 mean 유지.

## 2026-11-04 11:00 tick — Yelp 가중치 상한 확정
- Yelp q086 top_k=4 + 1e-2: .0440/.0749/.0513/.0609 → G1 -9.8% (2e-3 -0.7% 대비 네 지표 -.005~-.007). |prior| .054(2e-3의 .48의 1/9), swap overlap .992 — 정렬을 세게 걸수록 투영이 작아지고 페르소나 민감도도 떨어진다. 곡선 2e-3 -0.7 / 3e-3 -2.4 / 5e-3 -4.3 / 1e-2 -9.8(단조 하락); 1e-3(val .1293@201) 결과로 가중치 축 종료.
- 실행: q098 movie mix + 1e-2 (구조 후보 재조정 점; GPU 0). 그 뒤 빈 메모리 GPU0 ~7 GB / GPU1 ~6 GB → 추가 없음.
- 진행 중: yelp 1e-3 ep 211 .1293@201, yelp 온도 .5 ep 136 .1236@135, yelp mix ep 55 .1202@54 (mean 2e-3은 ep 51 .1204 → 같은 pace), yelp keep_rate .8 ep 28 .1103@27, movie mix 1e-2 시작.
- 관찰(세 실행 일관): 정렬 가중치 ↑ → |prior| ↓, swap 민감도 ↓, 성능 ↓. 페르소나 목표가 그래프를 돕는 범위는 매우 좁다(Yelp 1e-3~2e-3).

## 2026-11-04 11:22 — 실험 종료 (서버 이전)
- 사용자 지시로 진행 중 5개 실행 중단(`kill` → ledger `killed`): yelp 1e-3(ep 235, val .1293@201), yelp 온도 .5(ep 160, .1248), yelp mix(ep 73, .1234), yelp keep_rate .8(ep 46, .1176), movie mix 1e-2(ep 67, .0959). `/loop 30m /harness-tick` cron(0190b636) 중지. GPU 비움.
- 정리 산출물: `FINAL_REPORT.md`(요약표·데이터셋별 report·중단 실행), `results_all.csv`(155건), `HANDOFF.md`(이전 절차: 링크 실체화, shebang, GPU 가드, smoke, 루프 재개, 재실행 우선순위), `environment.yml`/`requirements-frozen.txt`, `agent_memory/`(Claude 메모리 사본).
- 종료 시점 결론: Book G1 달성(공유 구조 +4.9%), Movie -0.1%, Yelp -0.7%. Movie tier A 소진. mix 구조 후보는 Movie -3.7%로 불리, Yelp·Movie 재조정 미완(중단). 다음 서버에서 q092(yelp 1e-3) 재실행이 1순위, 그 다음 mix 판정(q096/q098).
