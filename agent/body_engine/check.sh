#!/usr/bin/env bash
# Every ported stage against MuJoCo on two bodies (fly: free + hinge; 8-leg spider test model:
# free + ball + hinge + slide), plus a mutation that must fail (hinge angle sign flipped).
# usage: gpu-wait run --timeout 1200 -- agent/body_engine/check.sh [STAGE]
set -euo pipefail
cd "$(dirname "$0")/../.."
stage=${1:-kinematics}; M=.venv/bin/python; X=.work/agent/body_engine; fail=0
$M -m agent.body_engine.model check .work/agent/gpu-body-port/body-v2/body.xml $X/fly "$stage" 64 || { echo "FAIL fly $stage"; fail=$((fail+1)); }
$M -m agent.body_engine.model check agent/body_engine/models/spider_test.xml $X/spider "$stage" 64 || { echo "FAIL spider $stage"; fail=$((fail+1)); }
if [ "$stage" = kinematics ]; then
  src=agent/body_engine/kinematics.mojo; cp "$src" $X/kin_orig.mojo; trap 'cp $X/kin_orig.mojo '"$src" EXIT
  sed -i 's/var h = 0.5 \* (qpos/var h = -0.5 * (qpos/' "$src"
  if $M -m agent.body_engine.model check .work/agent/gpu-body-port/body-v2/body.xml $X/mutant kinematics 64 >/dev/null 2>&1; then
    echo "FAIL mutant: flipped hinge sign passed the gate"; fail=$((fail+1)); else echo "PASS mutant caught"; fi
fi
[ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
echo "PASS all $stage checks"
