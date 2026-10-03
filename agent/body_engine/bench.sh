#!/usr/bin/env bash
# S1 gate timing: the full contact-free step (smooth.mojo STAGE 5) at 1k / 16k / 65k animals, random states.
# usage: agent/body_engine/bench.sh build MJCF     (CPU: export, states, one binary per batch size)
#        gpu-wait run --priority 90 --timeout 1200 -- agent/body_engine/bench.sh run MJCF
# Correctness is check.sh step's job; run it first (a speed number from an unchecked build is not reported).
set -euo pipefail
cd "$(dirname "$0")/../.."
mode=$1; mjcf=$2; X=.work/agent/body_engine/bench/$(basename "$mjcf" .xml); fail=0
for n in 1024 16384 65536; do
  d=$X/$n; mkdir -p "$d"
  if [ "$mode" = build ]; then
    .venv/bin/python -c "
import sys, numpy as np
from agent.body_engine.model import export, VSCALE
from mj_oracle import states, velocities
m = export('$mjcf', '$d')
states(m, $n).astype(np.float32).T.copy().tofile('$d/qpos.bin')
velocities(m, $n, scale=VSCALE).astype(np.float32).T.copy().tofile('$d/qvel.bin')" >/dev/null || { echo "FAIL export $n"; fail=$((fail+1)); continue; }
    ~/.local/bin/mojo build $(cat "$d/sizes.txt") -DFLIES=$n -DSTAGE=5 agent/body_engine/smooth.mojo -o "$d/step" 2>&1 | grep -v warning | grep . && { echo "FAIL build $n"; fail=$((fail+1)); } || true
  else
    "$d/step" "$d/" || { echo "FAIL run $n"; fail=$((fail+1)); }
  fi
done
[ $fail -eq 0 ] || { echo "FAIL $fail/3 batch sizes"; exit 1; }
