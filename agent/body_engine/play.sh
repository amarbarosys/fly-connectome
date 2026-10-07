#!/usr/bin/env bash
# Interactive play (docs/plans/PLAY.md). gpu-wait runs the job detached from this terminal, so this script keeps the
# terminal raw and relays it: keystrokes -> $IO/keys, the job's HUD $IO/hud -> screen. usage: play.sh [world [SEED]]
set -euo pipefail
cd "$(dirname "$0")/../.."
IO=.work/agent/body_engine/play/io
mkdir -p "$IO"; rm -f "$IO/keys" "$IO/hud"; mkfifo "$IO/keys" "$IO/hud"
old=$(stty -g)
cleanup() { stty "$old"; kill "${pids[@]}" 2>/dev/null || true; }
trap cleanup EXIT
cat "$IO/hud" & pids=($!)
stty raw -echo
cat > "$IO/keys" & pids+=($!)
env_args=(--env PYTHONPATH="$PWD" --env PLAY_IO="$PWD/$IO")
for v in WAYLAND_DISPLAY DISPLAY XDG_RUNTIME_DIR $(compgen -v PLAY_ || true); do [ -n "${!v:-}" ] && env_args+=(--env "$v=${!v}"); done
rc=0
gpu-wait run --priority 90 --vram 4 --timeout 3600 "${env_args[@]}" -- .venv/bin/python -m agent.body_engine.play "$@" </dev/null || rc=$?
[ $rc -eq 0 ] || echo -e "\r\nFAIL play: exit $rc (gpu-wait logs for the job)\r"
exit $rc
