#!/usr/bin/env bash
# PERSONA-MOSAIC harness: set up the copied working tree on a NEW server. Idempotent (safe to re-run).
#   bash <repo>/encoder/harness/remote_setup.sh [REPO_DIR] [--smoke]
#   REPO_DIR  repo root (default: two levels above this script)
#   --smoke   also run the 3-minute amazon_book smoke of HANDOFF.md §4 and check its log
# Steps: conda (miniconda if absent) -> env "mosaic" python 3.10.22 -> pip -r requirements-min.txt (torch 2.8.0 cu128)
#        -> run.py shebang + CLAUDE.md / SKILL.md paths -> Claude Code memory + model settings -> harness status/goals
#        -> Claude Code CLI (~/.local/bin/claude; SKIP_CLI=1 to skip) -> optional smoke.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "${1:-$HERE/../..}" && pwd)
SMOKE=${2:-}
ENV_NAME=${ENV_NAME:-mosaic}
PY_VER=${PY_VER:-3.10.22}
MINICONDA_DIR=${MINICONDA_DIR:-$HOME/miniconda3}      # Backend.AI: put it inside a mounted vfolder, /home/work is ephemeral
CC=${CLAUDE_CONFIG_DIR:-$HOME/.claude}; [ -z "$CC" ] && CC=$HOME/.claude                # Claude Code config dir (memory, login); honour CLAUDE_CONFIG_DIR
log() { printf '\n== %s\n' "$*"; }

log "repo $REPO"
[ -f "$REPO/encoder/harness/run.py" ] || { echo "run.py not found under $REPO (rsync first)" >&2; exit 1; }

log "1. python env ($ENV_NAME, python $PY_VER)"
if command -v conda >/dev/null 2>&1; then
  . "$(conda info --base)/etc/profile.d/conda.sh"
elif [ -x "$MINICONDA_DIR/bin/conda" ]; then
  . "$MINICONDA_DIR/etc/profile.d/conda.sh"
else
  echo "no conda: installing miniconda to $MINICONDA_DIR"
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "$HOME/miniconda.sh"
  bash "$HOME/miniconda.sh" -b -p "$MINICONDA_DIR"
  rm -f "$HOME/miniconda.sh"
  . "$MINICONDA_DIR/etc/profile.d/conda.sh"
  conda init bash >/dev/null          # so new shells (VS Code terminals) see conda
fi
if conda env list | awk '{print $1}' | grep -x "$ENV_NAME" >/dev/null; then
  echo "env $ENV_NAME exists"
else
  conda create -y -n "$ENV_NAME" "python=$PY_VER"
fi
set +u; conda activate "$ENV_NAME"; set -u
PY=$(python -c 'import sys; print(sys.executable)')
if ! python -c 'import torch, numpy, scipy, yaml, tqdm' 2>/dev/null; then
  pip install -r "$REPO/encoder/harness/requirements-min.txt"
fi
python - <<'PYEOF'
import sys, torch, numpy, scipy
print(f"python {sys.version.split()[0]}  torch {torch.__version__}  cuda {torch.version.cuda}  numpy {numpy.__version__}  scipy {scipy.__version__}")
print("cuda available:", torch.cuda.is_available(), "| gpus:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i); print(f"  gpu{i}: {p.name} {p.total_memory/2**30:.0f} GiB")
PYEOF

log "2. machine-specific paths"
sed -i "1s|^#!.*|#!$PY|" "$REPO/encoder/harness/run.py"
chmod +x "$REPO/encoder/harness/run.py"
for f in "$REPO/.claude/skills/harness-tick/SKILL.md" "$REPO/encoder/harness/CLAUDE.md"; do
  if [ -f "$f" ]; then sed -i "s|/home/kyung/mosaic_v2|$REPO|g" "$f"; fi
done
echo "run.py shebang: $(head -1 "$REPO/encoder/harness/run.py")"

log "3. Claude Code memory + settings"
SLUG=$(printf '%s' "$REPO" | sed 's#[^A-Za-z0-9]#-#g')     # /home/work/mosaic_v2 -> -home-work-mosaic-v2
MEM="$CC/projects/$SLUG/memory"
mkdir -p "$MEM"
for f in "$REPO"/encoder/harness/agent_memory/*.md; do
  b=$(basename "$f")
  if [ ! -e "$MEM/$b" ]; then
    sed -e "s|/home/kyung/miniconda3/envs/mosaic/bin/python|$PY|g" -e "s|/home/kyung/mosaic_v2|$REPO|g" "$f" > "$MEM/$b"
    if [ "$b" = "mosaic-v2-environment.md" ]; then
      printf '\n- Moved to a new server on 2026-11-04 (remote_setup.sh): python %s; check GPU count/memory with nvidia-smi before assuming the old "GPU 1 for tests" note.\n' "$PY" >> "$MEM/$b"
    fi
  fi
done
echo "memory: $MEM ($(ls "$MEM" | wc -l) files; Claude Code must be opened at $REPO for this slug)"
if [ ! -f "$CC/settings.json" ] && [ -f "$REPO/encoder/harness/agent_settings.json" ]; then
  mkdir -p "$CC"; cp "$REPO/encoder/harness/agent_settings.json" "$CC/settings.json"
  echo "settings: model preference copied to $CC/settings.json"
fi
[ -n "${CLAUDE_CONFIG_DIR:-}" ] && echo "NOTE: export CLAUDE_CONFIG_DIR=$CC before running claude (tmux/CLI); the VS Code extension needs ~/.claude itself (Backend.AI: automount vfolder named .claude)"

log "4. GPU + memory guard"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv,noheader
  MIN=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1 | tr -d ' ')
  if [ "${MIN:-0}" -lt 60000 ]; then
    echo "NOTE: smallest GPU < 60 GB: review gpu_min_free_mib in encoder/harness/constraints.yaml (peaks: book 9 GB, movie 15 GB, yelp 21-35 GB)"
  fi
else
  echo "nvidia-smi not found"
fi

log "5. harness sanity (status / goals)"
cd "$REPO/encoder"
python harness/run.py status | head -8
python harness/run.py goals | head -12

log "6. Claude Code CLI"
if [ "${SKIP_CLI:-0}" = "1" ]; then
  echo "skipped (SKIP_CLI=1)"
elif command -v claude >/dev/null 2>&1; then
  echo "claude: $(command -v claude)"
elif [ -x "$HOME/.local/bin/claude" ]; then
  echo "claude: $HOME/.local/bin/claude (add ~/.local/bin to PATH)"
elif curl -fsSL https://claude.ai/install.sh | bash; then
  echo "claude CLI installed (~/.local/bin/claude)"
else
  echo "claude CLI install failed: optional, the VS Code Claude Code extension (Remote-SSH) brings its own binary"
fi

if [ "$SMOKE" = "--smoke" ]; then
  log "7. smoke: amazon_book, 3 min (k-means cache is built first if missing)"
  python harness/run.py launch --model persona_mosaic --dataset amazon_book \
    --override gate_mode=residual persona_file=persona_emb_one.pkl prior_composition=mean persona_align_weight=0.02 \
    --label "smoke new server" --no-hypothesis --smoke 3
  for _ in $(seq 1 40); do                       # wait up to 20 min
    sleep 30
    if python harness/run.py status | grep -q "running (0)"; then break; fi
  done
  LOGF=$(ls -t save/log/persona_mosaic/amazon_book_*.log 2>/dev/null | head -1 || true)
  if [ -n "$LOGF" ]; then
    echo "log: $LOGF"; grep -E "prior=|gate mode|Validation" "$LOGF" | head -6 || true
  else
    echo "smoke log not found"
  fi
  python harness/run.py collect | tail -3
fi

log "DONE"
cat <<NEXT
Next (manual):
  0. Backend.AI: everything under a mounted vfolder survives, /home/work does not (repo: $REPO, conda: $MINICONDA_DIR, claude config: $CC).
  1. Claude Code login on this server: either VS Code Remote-SSH + Claude Code extension (open folder $REPO),
     or   tmux new -s harness ; cd $REPO/encoder ; claude   (CLI survives laptop sleep; recommended for the 30-min loop)
  2. In Claude Code:  /loop 30m /harness-tick     (first tick follows HANDOFF.md §5: re-run q092 first)
  3. Before resuming experiments, re-run the Book candidate 1103-100050 config once and compare with the ledger (HANDOFF §2-1).
NEXT
