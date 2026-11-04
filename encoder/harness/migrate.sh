#!/usr/bin/env bash
# Copy the mosaic_v2 working tree to a new server and set it up there. Runs on the OLD server.
#   bash encoder/harness/migrate.sh [HOST] [REMOTE_DIR] [--smoke]
#   HOST        ssh alias or user@host (default bai-vscode; the alias must be in ~/.ssh/config, or pass SSH_OPTS="-F file")
#   REMOTE_DIR  destination on HOST (default: $HOME/mosaic_v2 there)
#   --smoke     run the 3-minute amazon_book smoke after setup (pass "" for REMOTE_DIR to keep the default)
# Env: SRC (default /home/kyung/mosaic_v2), SSH_OPTS (extra ssh options), MINICONDA_DIR (default: next to REMOTE_DIR),
#      CLAUDE_CONFIG_DIR, SKIP_CLI (passed to remote_setup.sh).
# Via a laptop reverse tunnel (Backend.AI, see HANDOFF §0):
#   SSH_OPTS="-p 2222 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" bash migrate.sh work@localhost /home/work/<vfolder>/mosaic_v2 --smoke
# Password auth is fine: one ControlMaster connection is shared by every step, so the password is asked once.
set -euo pipefail
HOST=${1:-bai-vscode}; REMOTE_DIR=${2:-}; SMOKE=${3:-}
SRC=${SRC:-/home/kyung/mosaic_v2}
SSH="ssh ${SSH_OPTS:-} -o ControlMaster=auto -o ControlPath=/tmp/mosaic-migrate-%C -o ControlPersist=15m"
rs() { $SSH "$HOST" "$@"; }

echo "== 1/4 preflight: $HOST"
rs 'echo "connected: $(hostname) as $(id -un), HOME=$HOME"' || { echo "cannot reach $HOST (container up? port open?)" >&2; exit 2; }
REMOTE_HOME=$(rs 'echo $HOME')
REMOTE_DIR=${REMOTE_DIR:-$REMOTE_HOME/mosaic_v2}
rs "nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo 'no nvidia-smi'; df -h '$REMOTE_HOME' | tail -1"
NEED=$(cd "$SRC" && du -sL -BM --exclude=data/steam . | cut -f1)
echo "copy $NEED -> $HOST:$REMOTE_DIR  (symlinks dereferenced, data/steam excluded)"

echo "== 2/4 rsync"
rsync -aL --info=progress2 --exclude 'data/steam' --exclude '__pycache__' -e "$SSH" "$SRC/" "$HOST:$REMOTE_DIR/"

echo "== 3/4 remote setup (remote_setup.sh)"
# conda next to the repo by default (Backend.AI: both inside the mounted vfolder); CLAUDE_CONFIG_DIR / SKIP_CLI pass through
MINICONDA_DIR=${MINICONDA_DIR:-$(dirname "$REMOTE_DIR")/miniconda3}
rs "MINICONDA_DIR='$MINICONDA_DIR' CLAUDE_CONFIG_DIR='${CLAUDE_CONFIG_DIR:-}' SKIP_CLI='${SKIP_CLI:-0}' bash '$REMOTE_DIR/encoder/harness/remote_setup.sh' '$REMOTE_DIR' $SMOKE"

echo "== 4/4 done: $HOST:$REMOTE_DIR"
$SSH -O exit "$HOST" 2>/dev/null || true
