#!/usr/bin/env bash
# usage: gpu-wait run --priority 90 --timeout 1800 -- agent/body_engine/stand.sh
# live check + upright on the ramp (terrain() default, slope 0.1/0.05) and on 0.1 mm bumps (8 mm, 4 octaves)
set -euo pipefail
cd ~/Projects/science/fly-connectome
B=.work/agent/gpu-body-port/body-v2/body.xml; fail=0
.venv/bin/python -m agent.body_engine.live check || fail=$((fail+1))
for p in "(0.1, 0.05, 0, 8.0, 0, 0, 2.0, 0.5)" "(0.0, 0.0, 0.1, 8.0, 4, 1, 2.0, 0.5)"; do
  T=$(PYTHONPATH=$PWD .venv/bin/python -c "from kinestride.model import terrain; print(terrain('$B', $p)[0])" | tail -1)
  echo "=== upright $p"
  .venv/bin/python -m agent.body_engine.live upright "$T" || fail=$((fail+1))
done
[ $fail = 0 ] || { echo "FAIL $fail/3 gates"; exit 1; }
