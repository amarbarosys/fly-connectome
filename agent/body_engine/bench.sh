#!/usr/bin/env bash
# Engine timing at 1k / 16k / 65k animals. STAGE 5 (S1 gate): full contact-free step, random states.
# STAGE 6 (S2 probe): foot contacts + Newton + step, distinct states from actuated rollouts (model.standing).
# usage: agent/body_engine/bench.sh build MJCF [STAGE]   (CPU: export, states, one binary per batch size)
#        gpu-wait run --priority 90 --timeout 1200 -- agent/body_engine/bench.sh run MJCF [STAGE]
# Correctness is check.sh step's job; run it first (a speed number from an unchecked build is not reported).
set -euo pipefail
cd "$(dirname "$0")/../.."
mode=$1; mjcf=$2; stage=${3:-5}; X=.work/agent/body_engine/bench/$(basename "$mjcf" .xml)-$stage; fail=0
for n in 1024 16384 65536; do
  d=$X/$n; mkdir -p "$d"
  if [ "$mode" = build ]; then
    .venv/bin/python -c "
import sys, numpy as np
from agent.body_engine.model import export, standing, FEET, VSCALE
from mj_oracle import states, velocities
m = export('$mjcf', '$d', FEET if $stage == 6 else None)
import os
if not os.path.exists('$d/qvel.bin'):  # states are seeded: reuse them, the export is always fresh
    q, v = standing(m, $n) if $stage == 6 else (states(m, $n), velocities(m, $n, scale=VSCALE))
    q.astype(np.float32).T.copy().tofile('$d/qpos.bin')
    v.astype(np.float32).T.copy().tofile('$d/qvel.bin')" >/dev/null || { echo "FAIL export $n"; fail=$((fail+1)); continue; }
    ~/.local/bin/mojo build $(cat "$d/sizes.txt") -DFLIES=$n -DSTAGE=$stage agent/body_engine/smooth.mojo -o "$d/step" 2>&1 | grep -v warning | grep . && { echo "FAIL build $n"; fail=$((fail+1)); } || true
  else
    "$d/step" "$d/" || { echo "FAIL run $n"; fail=$((fail+1)); }
  fi
done
[ $fail -eq 0 ] || { echo "FAIL $fail/3 batch sizes"; exit 1; }
