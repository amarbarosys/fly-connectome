#!/usr/bin/env bash
# Drive interactive play without a person: hold W (kitty press/release) for HOLD seconds, then q; the HUD lines land in
# .work/agent/body_engine/play/io/hud.log. Checks the path a player meets: keys in, engine + render + mpv, HUD out.
# usage: tools/play_drive.sh [world [SEED]]
set -euo pipefail
cd "$(dirname "$0")/.."
IO=$PWD/.work/agent/body_engine/play/io; HOLD=${HOLD:-8}
mkdir -p "$IO"; rm -f "$IO/keys" "$IO/hud"; mkfifo "$IO/keys" "$IO/hud"
cat "$IO/hud" > "$IO/hud.log" &
exec 3<>"$IO/keys"
env_args=(--env PYTHONPATH="$PWD" --env PLAY_IO="$IO")
for v in WAYLAND_DISPLAY DISPLAY XDG_RUNTIME_DIR; do [ -n "${!v:-}" ] && env_args+=(--env "$v=${!v}"); done
gpu-wait run --priority 90 --vram 4 --timeout 600 "${env_args[@]}" -- .venv/bin/python -m agent.body_engine.play "$@" </dev/null &
job=$!
until grep -q "real time" "$IO/hud.log" 2>/dev/null; do kill -0 $job 2>/dev/null || { echo "FAIL play_drive: play exited before its HUD"; exit 1; }; sleep 0.5; done
printf '\x1b[119u' >&3; sleep "$HOLD"; printf '\x1b[119;1:3u' >&3; sleep 2; printf 'q' >&3
wait $job || { echo "FAIL play_drive: play exit $?"; exit 1; }
tr '\r' '\n' < "$IO/hud.log" | grep "real time" | tail -3
