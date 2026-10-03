#!/usr/bin/env bash
# Every ported stage against MuJoCo on two bodies (fly: free + hinge; 8-leg spider test model:
# free + ball + hinge + slide), plus a mutation that must fail (hinge angle sign flipped).
# usage: gpu-wait run --timeout 1200 -- agent/body_engine/check.sh [STAGE]
set -euo pipefail
cd "$(dirname "$0")/../.."
stage=${1:-kinematics}; M=.venv/bin/python; X=.work/agent/body_engine; fail=0
$M -m agent.body_engine.model check .work/agent/gpu-body-port/body-v2/body.xml $X/fly "$stage" 64 || { echo "FAIL fly $stage"; fail=$((fail+1)); }
$M -m agent.body_engine.model check agent/body_engine/models/spider_test.xml $X/spider "$stage" 64 || { echo "FAIL spider $stage"; fail=$((fail+1)); }
if [ "$stage" = kinematics ]; then  # the gate must fail with the hinge angle's sign flipped
  ~/iTools/bin/mutant-check agent/body_engine/smooth.mojo 's/var h = 0.5 \* (qpos/var h = -0.5 * (qpos/' -- \
    $M -m agent.body_engine.model check .work/agent/gpu-body-port/body-v2/body.xml $X/mutant kinematics 64 || fail=$((fail+1))
fi
if [ "$stage" = crb ]; then  # the gate must fail with the hinge cdof lever arm reversed (anchor - com)
  ~/iTools/bin/mutant-check agent/body_engine/smooth.mojo 's/var off = com - anchor/var off = anchor - com/' -- \
    $M -m agent.body_engine.model check .work/agent/gpu-body-port/body-v2/body.xml $X/mutant crb 64 || fail=$((fail+1))
fi
if [ "$stage" = factorM ]; then  # the gate must fail with the elimination update's sign flipped
  ~/iTools/bin/mutant-check agent/body_engine/smooth.mojo 's/var scl = -get(res, f, L + adr) \* invD/var scl = get(res, f, L + adr) * invD/' -- \
    $M -m agent.body_engine.model check .work/agent/gpu-body-port/body-v2/body.xml $X/mutant factorM 64 || fail=$((fail+1))
fi
if [ "$stage" = rne ]; then  # the gate must fail with the gyroscopic term's sign flipped (cvel x* I cvel)
  ~/iTools/bin/mutant-check agent/body_engine/smooth.mojo 's/acc) + cross_force(cvel/acc) - cross_force(cvel/' -- \
    $M -m agent.body_engine.model check .work/agent/gpu-body-port/body-v2/body.xml $X/mutant rne 64 || fail=$((fail+1))
fi
if [ "$stage" = step ]; then  # the gate must fail with the body-frame angular velocity integrated as world-frame
  ~/iTools/bin/mutant-check agent/body_engine/smooth.mojo 's/mul(q, Q4(cos(half), axis\[0\] \* s, axis\[1\] \* s, axis\[2\] \* s))/mul(Q4(cos(half), axis[0] * s, axis[1] * s, axis[2] * s), q)/' -- \
    $M -m agent.body_engine.model check .work/agent/gpu-body-port/body-v2/body.xml $X/mutant step 64 || fail=$((fail+1))
fi
[ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
echo "PASS all $stage checks"
