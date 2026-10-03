#!/usr/bin/env bash
# Every ported stage against MuJoCo on two bodies (fly: free + hinge; 8-leg spider test model:
# free + ball + hinge + slide), plus a mutation that must fail (hinge angle sign flipped).
# usage: gpu-wait run --timeout 1200 -- [env BODY=path.xml] agent/body_engine/check.sh [STAGE]
set -euo pipefail
cd "$(dirname "$0")/../.."
stage=${1:-kinematics}; M=.venv/bin/python; X=.work/agent/body_engine; fail=0
if [ "$stage" = cube ]; then  # closed 40 mm cube of planes (floor, 4 walls, ceiling; 330 pairs), states on every face
  C=$($M -c "from agent.body_engine.model import cube; print(cube('.work/agent/gpu-body-port/body-v2/body.xml'))")
  $M -m agent.body_engine.model check $C $X/cube actuated 64 || { echo "FAIL cube"; fail=$((fail+1)); }
  # the gate must fail when the bounding-sphere skip drops pairs that touch (rbound added instead of subtracted)
  ~/iTools/bin/mutant-check agent/body_engine/smooth.mojo 's/(pos2 - pos1)).reduce_add() - mf\[pf + 12\] > margin/(pos2 - pos1)).reduce_add() + mf[pf + 12] > margin/' -- \
    $M -m agent.body_engine.model check $C $X/mutant actuated 64 || fail=$((fail+1))
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all cube checks"; exit 0
fi
if [ "$stage" = loop ]; then  # K steps per launch, state kept on the card. K=1 vs mj_step (one step: the accuracy
  # gate); one K=10 launch must equal ten K=1 launches byte for byte (the loop). K=10 vs ten chained mj_step is not a
  # gate: the engine's ~5e-6 per-step float32 error grows ~12x in 10 steps on contact-rich states (diag/k_floor.py)
  F=.work/agent/gpu-body-port/body-v2/body.xml
  $M -m agent.body_engine.model check $F $X/fly actuated 64 1 || { echo "FAIL fly loop K=1"; fail=$((fail+1)); }
  ~/.local/bin/mojo build $(cat $X/fly/sizes.txt) -DFLIES=64 -DSTAGE=7 -DK=10 agent/body_engine/smooth.mojo -o $X/fly/k10
  cp $X/fly/state10.bin $X/fly/k1x10.bin && $X/fly/k10 $X/fly/
  cmp $X/fly/state.bin $X/fly/k1x10.bin && echo "PASS K=10 launch == 10 K=1 launches, byte-equal" || { echo "FAIL loop K=10 vs 10 x K=1"; fail=$((fail+1)); }
  # the gate must fail with the velocity write-back dropped (each step restarts from the launch's qvel)
  ~/iTools/bin/mutant-check agent/body_engine/smooth.mojo 's/qvel\[v \* F + f\] += mf\[TSTEP\]/qvel[v * F + f] += 0 * mf[TSTEP]/' -- \
    $M -m agent.body_engine.model check .work/agent/gpu-body-port/body-v2/body.xml $X/mutant actuated 64 1 || fail=$((fail+1))
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all loop checks"; exit 0
fi
B=${BODY:-.work/agent/gpu-body-port/body-v2/body.xml}  # BODY=path checks another MJCF (e.g. the spider from models/spider.py)
$M -m agent.body_engine.model check $B $X/$(basename $B .xml) "$stage" 64 || { echo "FAIL $(basename $B .xml) $stage"; fail=$((fail+1)); }
if [ "$stage" = contact ] || [ "$stage" = actuated ]; then  # plane-mesh foot pairs; the spider test body has none
  echo "spider: no plane-mesh contact pairs or actuators, $stage stage is fly-only"
else
  $M -m agent.body_engine.model check agent/body_engine/models/spider_test.xml $X/spider "$stage" 64 || { echo "FAIL spider $stage"; fail=$((fail+1)); }
fi
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
if [ "$stage" = contact ]; then  # the gate must fail with the contact spring pushing the wrong way
  ~/iTools/bin/mutant-check agent/body_engine/smooth.mojo 's/- K \* imp \* (dist - imargin)/+ K * imp * (dist - imargin)/' -- \
    $M -m agent.body_engine.model check .work/agent/gpu-body-port/body-v2/body.xml $X/mutant contact 64 || fail=$((fail+1))
fi
if [ "$stage" = actuated ]; then  # the gate must fail with the servo bias (-kp q) dropped
  ~/iTools/bin/mutant-check agent/body_engine/smooth.mojo 's/mf\[af\] \* c + mf\[af + 1\] + mf\[af + 2\] \* length/mf[af] * c + mf[af + 1]/' -- \
    $M -m agent.body_engine.model check .work/agent/gpu-body-port/body-v2/body.xml $X/mutant actuated 64 || fail=$((fail+1))
fi
[ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
echo "PASS all $stage checks"
