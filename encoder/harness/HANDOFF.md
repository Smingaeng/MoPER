# 다른 서버에서 하네스 재개하기 (HANDOFF, 2026-11-04 11:22 작성)

종료 시점 상태: 하네스 실행 0개(진행 중이던 5개는 `kill`로 중단, ledger에 `killed`로 기록), `/loop 30m /harness-tick` cron 중지,
GPU 비움. 결과 요약은 `harness/FINAL_REPORT.md`, 전체 실행 표는 `harness/results_all.csv`, 진실 원천은 `harness/ledger.jsonl`.

## 0. 빠른 경로: bai-vscode = Backend.AI 세션 (gpu.cnu.ac.kr, user `work`) — 2026-11-04 14:41 갱신
`gpu.cnu.ac.kr:443`은 **Backend.AI 웹 UI**다(`/config.toml`: `openPortToPublic = false`, `allowNonAuthTCP = true`). 세션의 SSH 포트(10506)는
웹 UI에서 SSH 앱을 연 클라이언트(노트북)에게만 프록시되고 공개로 바꿀 수 없으며, 세션이 종료되면 포트 자체가 사라진다(다음엔 번호가 바뀔 수
있다). 그래서 이 서버(168.188.128.124)에서 `ssh bai-vscode`는 세션이 떠 있어도 timeout이고 **이 서버에서 밀어 넣는 `migrate.sh`는 쓸 수 없다.**
대신 **컨테이너 안에서 이 서버로부터 당겨온다**(이 서버는 22번 포트에 sshd가 떠 있다; kyung 비밀번호 필요).

Backend.AI 영속성 주의: 세션의 `/home/work`는 세션이 끝나면 사라지고 마운트한 vfolder만 남는다.
1. 데이터 vfolder(예: `mosaic`, 20 GB 이상)를 만들어 세션에 마운트하고 리포와 miniconda를 그 안에 둔다 → `/home/work/mosaic/mosaic_v2`.
2. 점(.)으로 시작하는 vfolder는 모든 세션에 `~/.<이름>`으로 자동 마운트된다. `.claude`(로그인·메모리)와 `.local`(claude CLI)을 vfolder로 만들면
   세션을 새로 띄워도 로그인과 메모리가 남는다. 없으면 세션마다 다시 로그인한다(`CLAUDE_CONFIG_DIR`로 vfolder 안을 가리켜도 된다; CLI 전용).
3. 세션 자원: GPU 1개 이상, RAM 넉넉히(Yelp 학습 수 GB), idle timeout이 있으면 루프가 끊기므로 길게 잡는다.

컨테이너 터미널(웹 터미널 또는 VS Code 앱)에서:
```bash
timeout 5 bash -c '</dev/tcp/168.188.128.124/22' && echo reachable || echo blocked    # 0) 이 서버에 닿는지
D=/home/work/mosaic                                                                    # 마운트한 vfolder
# 1) 당겨오기: 링크 실체화(-L), data/steam 제외, 약 6.6 GB. rsync가 없으면 아래 tar 한 줄.
rsync -aL --info=progress2 --exclude 'data/steam' kyung@168.188.128.124:/home/kyung/mosaic_v2/ $D/mosaic_v2/
#   ssh kyung@168.188.128.124 'cd /home/kyung && tar -chf - --exclude=mosaic_v2/data/steam mosaic_v2' | tar -xf - -C $D
# 2) 설치(멱등; miniconda도 vfolder 안에)
MINICONDA_DIR=$D/miniconda3 bash $D/mosaic_v2/encoder/harness/remote_setup.sh $D/mosaic_v2 --smoke
```
`remote_setup.sh`: miniconda(없을 때만, `MINICONDA_DIR`) → env `mosaic` python 3.10.22 → `pip -r requirements-min.txt`(torch 2.8.0 cu128) →
`harness/.python`·`CLAUDE.md`·`SKILL.md`의 `/home/kyung/mosaic_v2` 치환 → Claude Code 메모리를 `~/.claude/projects/<slug>/memory/`에 복사
(slug = 리포 절대경로의 영숫자 외 문자를 `-`로: `/home/work/mosaic/mosaic_v2` → `-home-work-mosaic-mosaic-v2`) + 모델 설정 →
`run.py status/goals` → Claude Code CLI(`~/.local/bin/claude`, `SKIP_CLI=1`로 생략) → `--smoke`면 §4 smoke 실행·로그 확인.
GPU가 80 GB가 아니면 `constraints.yaml`의 `gpu_min_free_mib`를 보라고 알려준다. 다시 실행해도 안전하다.

**14:40 결과: 컨테이너 → 이 서버 22번은 `Connection refused`**(이 서버 ufw 활성; 캠퍼스 ACL일 수도 있다).
**14:41 결정: 코드는 GitHub로, 데이터·로그는 노트북 역터널로 스트리밍.** 이 서버 디스크가 가득 차서(2.1 GB 남음) 아카이브를 만들 수 없고,
이 서버에는 GitHub 푸시 자격(토큰/키)이 없다.

**14:48 터널 연결 성공.** 컨테이너 = `main1`, user `work`, vfolder `/home/work/minkyung`(NAS 20 TB, 15 TB 여유), GPU A100 80 GB × 1,
시스템 python 3.10.12, rsync·git 있음, conda 없음(remote_setup.sh가 vfolder 안에 miniconda를 설치). 터널이 열린 김에 **코드 트리(.git 포함,
브랜치 `migrate-20261104` 커밋 그대로)와 데이터·로그를 통째로 rsync**했으므로 아래 (A)의 GitHub 단계는 선택이고 (C)의 git init/fetch는 필요 없다
(이미 같은 커밋이 있다; 나중에 어느 쪽에서든 `git push -u origin migrate-20261104`). GPU 1장이라 동시 실행 수는 `gpu_min_free_mib`가 정한다.

(A) 코드 — 비공개 리포 `Smingaeng/mosaic_v2`. 이 서버에서 브랜치 **`migrate-20261104`**에 작업 트리 전체(실험 노브 + `encoder/harness/` + `.claude/`)를
커밋해 두었다. 푸시는 VS Code 통합 터미널(이 서버)에서 한다(VS Code가 GitHub 인증을 넘겨준다):
```bash
cd /home/kyung/mosaic_v2 && git push -u origin migrate-20261104
```
git이 무시하는 것은 GitHub에 없다: `data/`, `encoder/save/`(log·result·checkpoint), `encoder/harness/runs/`, `state.json` → (B)로 옮긴다.

(B) 데이터 + ledger가 가리키는 로그(약 6.3 GB) — 노트북에서 역터널을 열어 두면 이 서버에서 스트리밍한다:
```bash
# 노트북 (끝날 때까지 열어 둔다)
ssh -N -R 2222:gpu.cnu.ac.kr:10506 kyung@168.188.128.124
# 이 서버. 전부 git-ignored 경로라 (C)보다 먼저 가도 checkout과 충돌하지 않는다.
cd /home/kyung/mosaic_v2 && rsync -aLR --info=progress2 -e "ssh -p 2222 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
  ./data/amazon_book ./data/amazon_movie ./data/yelp ./encoder/save/log ./encoder/save/result ./encoder/harness/runs \
  work@localhost:/home/work/minkyung/mosaic_v2/
```

(C) 컨테이너(VS Code 터미널)에서 — 같은 디렉터리에 코드를 받고 설치(비공개 리포라 fetch에 GitHub 인증이 필요; VS Code 터미널이면 인증 창이 뜬다):
```bash
D=/home/work/minkyung; mkdir -p $D/mosaic_v2 && cd $D/mosaic_v2
git init -q && git remote add origin https://github.com/Smingaeng/mosaic_v2.git && git fetch -q origin \
  && git checkout -q -b migrate-20261104 origin/migrate-20261104
MINICONDA_DIR=$D/miniconda3 bash encoder/harness/remote_setup.sh $D/mosaic_v2 --smoke     # (B)가 끝난 뒤 (smoke는 data/가 있어야 돈다)
```
`migrate.sh`(이 서버에서 통째로 밀어 넣기)는 역터널이 있으면 여전히 쓸 수 있다: `SSH_OPTS="-p 2222 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
bash encoder/harness/migrate.sh work@localhost /home/work/minkyung/mosaic_v2 --smoke`.

에이전트 재개 — 사람이 해야 하는 것은 로그인과 루프 시작 두 가지다:
1. 로그인. (a) VS Code → Remote-SSH `bai-vscode`(웹 UI에서 SSH 앱을 연 상태) → 폴더 **`/home/work/mosaic/mosaic_v2`(리포 루트; 메모리 slug가
   이 경로에서 나온다)** → Claude Code 확장 로그인. 또는 (b) 컨테이너 터미널에서 `tmux new -s harness` → `cd $D/mosaic_v2/encoder && claude`
   → 로그인. 30분 루프를 몇 시간 돌리려면 (b)가 안전하다(노트북 절전·VS Code 재접속과 무관).
2. `/loop 30m /harness-tick`. 첫 tick은 §5대로(q092 먼저 재실행). smoke(§4 또는 `--smoke`)가 통과한 뒤에 시작한다.
3. GPU가 바뀌면 같은 환경이라도 수치가 소수점 넷째 자리에서 흔들린다(§2-1): Book 후보 `1103-100050` 설정을 한 번 재실행해
   ledger 값과 비교한 뒤 실험을 재개한다.

컨테이너에서 remote_setup.sh를 돌리며 고친 것(둘 다 커밋됨): (1) 최신 Miniconda는 defaults 채널에 Anaconda 약관 동의를 요구해
`conda create`가 `CondaToSNonInteractiveError`로 멈춘다 → env 생성을 `--override-channels -c conda-forge`로 바꿨다(python 3.10.22 그대로).
(2) 컨테이너의 nvidia-smi가 `utilization.gpu`를 `[Not Found]`로 내놓아 `run.py gpu_table()`이 ValueError → 숫자가 아닌 필드는 0으로 읽도록
고쳤다(메모리 필드는 정상이라 GPU 메모리 가드는 그대로 동작).

**15:07 설치 완료 (컨테이너).** `remote_setup.sh --smoke` 통과: env `/home/work/minkyung/miniconda3/envs/mosaic`(python 3.10.22, torch 2.8.0+cu128,
A100 80 GB 1장, 드라이버 580.173.02), `run.py status/goals` 표가 이 서버와 동일, Claude Code CLI 2.1.318(`~/.local/bin/claude`), 3분 smoke 52 epoch.
smoke의 초기 검증 곡선은 Book 후보 `1103-100050`의 로그와 소수점 넷째 자리에서만 다르다(같은 seed·데이터; A100 vs H100):
| epoch | 이 서버 R@5 / R@10 / R@20 | 컨테이너 R@5 / R@10 / R@20 |
|---|---|---|
| 0 | .0326 / .0539 / .0848 | .0326 / .0537 / .0848 |
| 12 | .0198 / .0322 / .0472 | .0195 / .0315 / .0473 |
vfolder 사용량: miniconda 8.2 GB + 리포 6.6 GB. 컨테이너 git 트리는 이 서버의 `migrate-20261104` 최신 커밋과 동일하게 맞춰 두었다.

**영속성(중요).** `/home/work/.claude`(로그인·메모리)와 `/home/work/.local/bin/claude`는 세션이 끝나면 사라진다. 두 가지 중 하나:
- CLI(tmux) 경로: `export CLAUDE_CONFIG_DIR=/home/work/minkyung/.claude` 후 `claude` — 메모리·설정 사본을 이 경로에도 넣어 두었다(로그인은 처음 한 번).
- VS Code 확장 경로: 웹 UI에서 이름이 `.claude`, `.local`인 vfolder를 만들어 두면 다음 세션부터 `~/.claude`, `~/.local`로 자동 마운트된다.
  새 세션에서는 `bash encoder/harness/remote_setup.sh /home/work/minkyung/mosaic_v2`를 한 번 더 돌리면 빠진 것(메모리 사본, CLI)만 채운다.

## 1. 옮길 것 (git clone만으로는 부족하다)
작업 트리에는 **커밋되지 않은 변경**이 있다: 실험 노브(persona_mosaic.py / persona_mosaic.yml / mosaic.py의 분석용 llm_prior /
trainer.py의 set_epoch·진단 훅)와 미추적 디렉터리 `encoder/harness/`, `.claude/`. 반드시 작업 트리를 통째로 복사한다
(또는 옮기기 전에 커밋한다 — 하네스 파일 커밋은 사용자 결정).

```bash
# 이 서버에서 (data/의 심볼릭 링크를 실체 파일로 풀어서 복사: -L). steam은 미사용이라 제외해도 된다.
rsync -aL --info=progress2 --exclude 'data/steam' /home/kyung/mosaic_v2/ NEW_HOST:/NEW/PATH/mosaic_v2/
```
- `data/<dataset>/` 안의 파일은 거의 전부 `/home/kyung/PersonaMosaic/data/...`(상호작용 행렬, RLMRec 프로필 임베딩, 페르소나
  임베딩 `persona_emb.pkl` / `persona_emb_one.pkl` / `item_persona_emb.pkl`)와 `/home/kyung/CoMoE/data/...`(MOSAIC의 intent /
  conformity / community 임베딩)로 가는 링크다. `-L` 없이 복사하면 새 서버에서 전부 깨진다. 실체 크기 약 8 GB. 전체 링크 목록은 §7.
- 페르소나 클러스터 캐시 `data/<dataset>/persona/persona_clusters_*.pkl`은 일반 파일이며 없으면 첫 실행 때 재생성된다(k-means, 수 분).
- 로그: `encoder/save/log/{mosaic,promax,persona_mosaic}/*.log`(ledger의 `log_path`가 가리킴), `encoder/harness/runs/*.out`(stderr).
  둘 다 상대 경로로 기록되어 있어 트리를 통째로 옮기면 `run.py show/diag`가 그대로 동작한다.
- Claude Code 메모리(선택): 이 서버의 `~/.claude/projects/-home-kyung-mosaic-v2/memory/`를 `encoder/harness/agent_memory/`에
  복사해 두었다. 새 서버에서는 `~/.claude/projects/<새 절대경로의 영숫자 외 문자('/', '_')를 '-'로 바꾼 이름>/memory/`에 넣으면 같은 메모리로 시작한다(`remote_setup.sh`가 한다).
  규칙 자체는 `harness/CLAUDE.md`에 모두 있으므로 없어도 동작한다.

## 2. 환경
| 항목 | 값 (이 서버) |
|---|---|
| python | 3.10.22 (conda env `mosaic`) |
| torch / CUDA | 2.8.0+cu128 / 12.8 |
| numpy | 2.0.2 |
| GPU | H100 PCIe 80 GB × 2 |

### 2-1. 새 서버 베이스 이미지와 설치 (2026-11-04 추가)
코드가 import 하는 외부 패키지는 torch, numpy, scipy, PyYAML, tqdm 뿐이다(torch_scatter/torch_sparse/cupy는 설치만 돼 있고 쓰지 않는다).
그러므로 이미지에 들어 있는 PyTorch 버전은 중요하지 않다. **이미지 위에 python 3.10 env를 따로 만들고 torch 2.8.0(cu128)을 설치**하면
ledger 수치를 낸 환경과 같아진다. 이미지의 torch 2.10/2.11 + python 3.12를 그대로 쓰지 말 것(ledger/references 수치와 비교 불가, 고정 패키지 설치 불가).
- 이미지: 시스템 python 3.10 + CUDA 12.x 이미지(NGC 24.04 계열: PyTorch 2.3 / Python 3 / CUDA 12.4)가 가장 가깝다.
  CUDA 13.x 이미지(25.12, 26.03)도 호스트 드라이버가 더 새것일 뿐이라 같은 방법으로 동작한다.
- 부팅 후 확인: `nvidia-smi` 드라이버 ≥ 570 권장(이 서버 570.181; cu128 휠은 525 이상이면 minor-version compatibility로 동작).
  torch 2.8.0 cu128 휠의 arch list는 sm_70~sm_120(A100, H100, Blackwell 모두 포함).
- 설치(권장, 최소):
  ```bash
  conda create -n mosaic --override-channels -c conda-forge python=3.10.22 -y   # defaults 채널은 Anaconda 약관 동의를 요구(CondaToSNonInteractiveError); conda-forge는 불필요. venv도 가능
  conda activate mosaic
  pip install -r encoder/harness/requirements-min.txt
  python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"   # 2.8.0+cu128 12.8 True
  ```
- 전체 복제가 필요하면 `pip install -r encoder/harness/requirements-frozen.txt`(torch_scatter/torch_sparse용 `-f` 줄 포함) 또는
  `conda env create -n mosaic -f encoder/harness/environment.yml`.
- 클러스터 캐시 `data/<dataset>/persona/persona_clusters_*.pkl`은 rsync로 같이 옮긴다. 새 GPU/torch에서 재생성하면 k-means 결과가
  달라질 수 있으므로 지우지 말 것. 환경이 같아도 GPU가 다르면 수치는 소수점 넷째 자리 수준에서 흔들린다: Book 후보
  `1103-100050` 설정을 한 번 그대로 재실행해 ledger 값과 비교한 뒤 실험을 재개하는 것이 안전하다.

## 3. 새 서버에서 고칠 것 (순서대로)
1. **`encoder/harness/.python`**(git 무시, 한 줄)에 env의 python 절대경로를 적는다(`remote_setup.sh`가 쓴다). `run.py`는 `#!/usr/bin/env python3`로 시작해
   `.python`이 있으면 그 인터프리터로 자기 자신을 다시 실행하므로 어느 셸에서든 `harness/run.py ...`가 된다. 실행 명령에는 `sys.executable`을 쓴다.
2. **`encoder/harness/constraints.yaml`의 `gpu_min_free_mib`**: GPU 메모리에 맞춘다. 관측 피크(프로세스당): amazon_book ≈ 9 GB,
   amazon_movie ≈ 15 GB, yelp 21~35 GB(학습 후반에 커진다). 80 GB 기준 현재값 `{default: 9000, amazon_movie: 16000, yelp: 36000}`.
3. (필요 시) `.claude/skills/harness-tick/SKILL.md`의 `R=harness/run.py` 설명 — `.python`을 못 쓰면 env를 활성화하고 `R="python harness/run.py"`.
4. `encoder/harness/state.json`은 마지막 실행 시각일 뿐이다(그대로 두거나 삭제).

## 4. 동작 확인 (실험 재개 전)
```bash
cd /NEW/PATH/mosaic_v2/encoder
python harness/run.py status                          # GPU와 실행 0개가 보여야 한다
python harness/run.py report --dataset amazon_book    # 표가 이 서버와 같게 나와야 한다 (ledger가 그대로이므로)
python harness/run.py goals
# 3분 smoke (phase_lock·중복 검사 면제, 로그는 collect가 지운다). 클러스터 캐시가 없으면 k-means부터 돈다.
python harness/run.py launch --model persona_mosaic --dataset amazon_book \
  --override gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean persona_align_weight=0.02 \
  --label "smoke new server" --no-hypothesis --smoke 3
python harness/run.py collect
```
smoke 로그(`save/log/persona_mosaic/amazon_book_*.log`)에 `prior=mean on 4096-d raw text embeddings; ... gate mode=residual` 요약과
`Epoch 3 Validation` 줄이 나오면 데이터·환경이 맞는 것이다.

## 5. 루프 재개
Claude Code를 리포 루트(`/NEW/PATH/mosaic_v2`)에서 열고 `/loop 30m /harness-tick` (스킬: `.claude/skills/harness-tick/SKILL.md`,
규칙: `encoder/harness/CLAUDE.md` — tick은 이 문서를 먼저 읽는다). 첫 tick에서 정할 것:
- 종료 때 중단된 5개 실행을 다시 돌릴지. 재실행하려면 `harness/queue.yaml`에서 해당 항목(q092, q094, q096, q097, q098)의
  `status: launched` → `ready`로 바꾸고 `launch --queue qNNN`(q096은 tier B라 `--approved`). 같은 설정의 `killed` 기록 때문에 중복
  검사에 걸리면 `--repeat`를 붙인다(중단 재실행은 "반복 금지" 규칙의 예외).
  | 큐 | 실행 | 중단 시점 | 의미 |
  |---|---|---|---|
  | q092 | yelp top_k=4 + 1e-3 | ep 235, val R@20 .1293@201 (2e-3는 .1326@246) | Yelp 가중치 곡선의 마지막 점 — **가장 먼저 재실행** |
  | q094 | yelp top_k=4 + 2e-3 + 온도 .5 | ep 160, .1248@159 | 새 폭에서 온도 축 소거 |
  | q096 | yelp mix(2e-3, top_k=4) | ep 73, .1234@69 (mean 2e-3는 ep 69쯤 .122) | 구조 후보 mix의 Yelp 판정 |
  | q097 | yelp keep_rate .8 | ep 46, .1176@42 | 소거 |
  | q098 | movie mix + 1e-2 | ep 67, .0959@66 (mean 7e-3는 .0984@96) | mix의 Movie 재조정 점 |
- 그 다음은 `CLAUDE.md` §0·§9와 `JOURNAL.md` 마지막 항목이 가리키는 대로: P1(Movie·Yelp G1)이 열려 있고 Movie의 tier A 축은 소진,
  남은 것은 구조 변경(mix 판정 → 폐기 시 새 tier B 제안)뿐이다.

## 6. 연구 상태 한 장 (상세는 FINAL_REPORT.md / JOURNAL.md)
- 공유 구조(residual gate, persona_emb_one, mean 목표, profile 아이템, modulate 주입, silhouette K, raw 4096-d)에서
  Book G1 달성(+4.9%, q051; facet 목표는 +8.1%이나 Movie·Yelp에서 나빠 공유 구조가 아님). Movie G1 -0.1%, Yelp G1 -0.7% (top_k=4).
- 데이터셋별 하이퍼파라미터: Book align 2e-2 / Movie 7e-3, top_k 4, 3층 / Yelp 2e-3, top_k 4, 2층. 세 데이터셋 모두 가중치 창이 좁고
  센 정렬은 투영을 줄이며 성능·페르소나 민감도를 함께 떨어뜨린다.
- 소진된 축: 가중치, 폭(top_k), 변조 상한(λ ≤ 1은 배율 비음수 조건), 변조 초기값, keep_rate, 깊이, reg(3e-6부터 붕괴), 온도, warm-up,
  balance, side, decay, 고정 목표, audience, learned gate, 4-view, elbow K, PCA, text_score, additive 주입, projector 용량.
- 금지: 베이스라인 수정, ProMax SDR/S2DR·RLMRec KD 이식, 파라미터 없는 gate, 균등 gate, seed 변경, expert_num 직접 지정, 반복 평균.

## 7. 파일 안내
| 파일 | 역할 |
|---|---|
| `harness/CLAUDE.md` | 운영 규칙(최우선). §0 우선순위, §3 불변 규칙, §8 코드 상태, §9 현재 상태 |
| `harness/run.py` | 유일한 실행 진입점: launch / collect / status / report / goals / show / diag / note / mark / kill |
| `harness/constraints.yaml` | 기계 검사 규칙: shared_structure, phases, achieved, value_rules, gpu_min_free_mib, 금지 손실 |
| `harness/references.yaml` | 기준값 실행 id (MOSAIC, ProMax d=32/64, 데이터셋별) |
| `harness/queue.yaml` | q001~q098 가설·예측·결정 규칙·상태 |
| `harness/ledger.jsonl` | 모든 실행 기록(진실 원천, 155건; 분석 노트 포함) |
| `harness/results_all.csv` | ledger를 표로 내보낸 것 |
| `harness/FINAL_REPORT.md` | 종료 시점 요약표 + 데이터셋별 전체 report + 중단 실행 |
| `harness/JOURNAL.md` | 날짜별 결정 기록 |
| `harness/patches/*.patch` | 코드 변경 이력(복원 패치, mean/fixed 목표, audience, side/decay, mix). 현재 트리에 모두 적용됨 |
| `harness/agent_memory/` | Claude Code 메모리 사본(§1) |
| `harness/environment.yml`, `requirements-frozen.txt` | 환경 |
| `harness/requirements-min.txt` | 코드가 실제로 쓰는 패키지만 고정(torch 2.8.0, numpy, scipy, PyYAML, tqdm) — 새 서버 설치용 |
| `harness/remote_setup.sh`, `migrate.sh`, `agent_settings.json` | 새 서버 설치 스크립트(§0; Backend.AI에서는 컨테이너 안에서 당겨온 뒤 remote_setup.sh), 밀어 넣기용 migrate.sh(역터널·CLI 경유 때만), Claude Code 모델 설정 사본 |
| `.claude/skills/harness-tick/SKILL.md` | tick 절차 |

## 8. data/ 심볼릭 링크 목록 (이 서버 기준; rsync -L로 실체화하거나 새 서버에서 다시 만든다)
- `data/amazon_book/community_assignments.pkl -> /home/kyung/CoMoE/data/amazon_book/community_assignments.pkl`
- `data/amazon_book/community_conf_emb.pkl -> /home/kyung/CoMoE/data/amazon_book/community_conf_emb.pkl`
- `data/amazon_book/community_intent_emb.pkl -> /home/kyung/CoMoE/data/amazon_book/community_intent_emb.pkl`
- `data/amazon_book/item_conf_emb.pkl -> /home/kyung/CoMoE/data/amazon_book/item_conf_emb.pkl`
- `data/amazon_book/item_intent_emb_3.pkl -> /home/kyung/CoMoE/data/amazon_book/item_intent_emb_3.pkl`
- `data/amazon_book/itm_emb_np.pkl -> /home/kyung/PersonaMosaic/data/amazon_book/itm_emb_np.pkl`
- `data/amazon_book/new_trn_rag_mat.pkl -> /home/kyung/PersonaMosaic/data/amazon_book/new_trn_rag_mat.pkl`
- `data/amazon_book/persona/item_persona_emb.pkl -> /home/kyung/PersonaMosaic/data/amazon_book/persona/item_persona_emb.pkl`
- `data/amazon_book/persona/persona_emb.pkl -> /home/kyung/PersonaMosaic/data/amazon_book/persona/persona_emb.pkl`
- `data/amazon_book/persona/persona_emb_one.pkl -> /home/kyung/PersonaMosaic/data/amazon_book/persona/persona_emb_one.pkl`
- `data/amazon_book/trn_mat.pkl -> /home/kyung/PersonaMosaic/data/amazon_book/trn_mat.pkl`
- `data/amazon_book/tst_mat.pkl -> /home/kyung/PersonaMosaic/data/amazon_book/tst_mat.pkl`
- `data/amazon_book/user_conf_emb.pkl -> /home/kyung/CoMoE/data/amazon_book/user_conf_emb.pkl`
- `data/amazon_book/user_intent_emb_3.pkl -> /home/kyung/CoMoE/data/amazon_book/user_intent_emb_3.pkl`
- `data/amazon_book/usr_emb_np.pkl -> /home/kyung/PersonaMosaic/data/amazon_book/usr_emb_np.pkl`
- `data/amazon_book/val_mat.pkl -> /home/kyung/PersonaMosaic/data/amazon_book/val_mat.pkl`
- `data/amazon_movie/community_assignments.pkl -> /home/kyung/CoMoE/data/amazon_movie/community_assignments.pkl`
- `data/amazon_movie/community_conf_emb.pkl -> /home/kyung/CoMoE/data/amazon_movie/community_conf_emb.pkl`
- `data/amazon_movie/community_intent_emb.pkl -> /home/kyung/CoMoE/data/amazon_movie/community_intent_emb.pkl`
- `data/amazon_movie/item_conf_emb.pkl -> /home/kyung/CoMoE/data/amazon_movie/item_conf_emb.pkl`
- `data/amazon_movie/item_intent_emb_3.pkl -> /home/kyung/CoMoE/data/amazon_movie/item_intent_emb_3.pkl`
- `data/amazon_movie/itm_emb_np.pkl -> /home/kyung/PersonaMosaic/data/amazon_movie/itm_emb_np.pkl`
- `data/amazon_movie/new_trn_rag_mat.pkl -> /home/kyung/PersonaMosaic/data/amazon_movie/new_trn_rag_mat.pkl`
- `data/amazon_movie/persona/item_persona_emb.pkl -> /home/kyung/PersonaMosaic/data/amazon_movie/persona/item_persona_emb.pkl`
- `data/amazon_movie/persona/persona_emb.pkl -> /home/kyung/PersonaMosaic/data/amazon_movie/persona/persona_emb.pkl`
- `data/amazon_movie/persona/persona_emb_one.pkl -> /home/kyung/PersonaMosaic/data/amazon_movie/persona/persona_emb_one.pkl`
- `data/amazon_movie/trn_mat.pkl -> /home/kyung/PersonaMosaic/data/amazon_movie/trn_mat.pkl`
- `data/amazon_movie/tst_mat.pkl -> /home/kyung/PersonaMosaic/data/amazon_movie/tst_mat.pkl`
- `data/amazon_movie/user_conf_emb.pkl -> /home/kyung/CoMoE/data/amazon_movie/user_conf_emb.pkl`
- `data/amazon_movie/user_intent_emb_3.pkl -> /home/kyung/CoMoE/data/amazon_movie/user_intent_emb_3.pkl`
- `data/amazon_movie/usr_emb_np.pkl -> /home/kyung/PersonaMosaic/data/amazon_movie/usr_emb_np.pkl`
- `data/amazon_movie/val_mat.pkl -> /home/kyung/PersonaMosaic/data/amazon_movie/val_mat.pkl`
- `data/yelp/community_assignments.pkl -> /home/kyung/CoMoE/data/yelp/community_assignments.pkl`
- `data/yelp/community_conf_emb.pkl -> /home/kyung/CoMoE/data/yelp/community_conf_emb.pkl`
- `data/yelp/community_intent_emb.pkl -> /home/kyung/CoMoE/data/yelp/community_intent_emb.pkl`
- `data/yelp/item_conf_emb.pkl -> /home/kyung/CoMoE/data/yelp/item_conf_emb.pkl`
- `data/yelp/item_intent_emb_3.pkl -> /home/kyung/CoMoE/data/yelp/item_intent_emb_3.pkl`
- `data/yelp/itm_emb_np.pkl -> /home/kyung/PersonaMosaic/data/yelp/itm_emb_np.pkl`
- `data/yelp/new_trn_rag_mat.pkl -> /home/kyung/PersonaMosaic/data/yelp/new_trn_rag_mat.pkl`
- `data/yelp/persona/item_persona_emb.pkl -> /home/kyung/PersonaMosaic/data/yelp/persona/item_persona_emb.pkl`
- `data/yelp/persona/persona_emb.pkl -> /home/kyung/PersonaMosaic/data/yelp/persona/persona_emb.pkl`
- `data/yelp/persona/persona_emb_one.pkl -> /home/kyung/PersonaMosaic/data/yelp/persona/persona_emb_one.pkl`
- `data/yelp/trn_mat.pkl -> /home/kyung/PersonaMosaic/data/yelp/trn_mat.pkl`
- `data/yelp/tst_mat.pkl -> /home/kyung/PersonaMosaic/data/yelp/tst_mat.pkl`
- `data/yelp/user_conf_emb.pkl -> /home/kyung/CoMoE/data/yelp/user_conf_emb.pkl`
- `data/yelp/user_intent_emb_3.pkl -> /home/kyung/CoMoE/data/yelp/user_intent_emb_3.pkl`
- `data/yelp/usr_emb_np.pkl -> /home/kyung/PersonaMosaic/data/yelp/usr_emb_np.pkl`
- `data/yelp/val_mat.pkl -> /home/kyung/PersonaMosaic/data/yelp/val_mat.pkl`
