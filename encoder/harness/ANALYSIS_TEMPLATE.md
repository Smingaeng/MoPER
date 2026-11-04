# 실행 분석 템플릿 (run.py note ID --file 로 ledger에 저장)

항목을 빠뜨리지 말고, 숫자는 `run.py show ID` / `run.py diag ID` / `run.py report` 에서 가져온다.
잡음 폭(noise floor) 미만의 차이는 "차이 없음"으로 적는다. 좋은 숫자만 고르지 않는다.

## <id> <label>
- 설정: model / dataset / overrides
- 결과(test): R@5 R@10 N@5 N@10 (R@20 N@20) ; best epoch ; 학습 분
- 기준 대비: G1 / G2 (/ G3) 기준값 대비 네 지표 상대 이득과 그 평균(= report의 Gn 열) ; 평균 +4% 달성 여부 ; 잡음 폭
- 예측 적중: yes / partial / no — 예측 "…" vs 실제 "…"
- gate (정보용, 판정과 무관): 쏠림 여부(top-1 share), 엔트로피, cluster-sim / topology 민감도, balance 값
- prior 사용: expert별 |prior| vs |E_free| (user/item), modulation std over nodes, layer alpha
- 내용 민감도 (있을 때만): persona swap top-20 overlap / 임베딩 cos ; 통제 짝이 있으면 실제 vs 셔플 Δ
- 결정 규칙 적용: decision_rule → 실제 결과가 해당하는 가지 → 다음 행동 1개
- 기여 근거: 이 실행이 어느 주장(experts-not-handcrafted / moe / weighted-gate / intent-conformity-in-personas)을 지지·반박하는가
- 이상 징후: 로그 오류, 재빌드 후 validation 급락(= 진단 버그), 같은 초 로그 충돌 등
