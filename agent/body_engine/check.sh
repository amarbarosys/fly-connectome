#!/usr/bin/env bash
# Every ported stage against MuJoCo on two bodies (fly: free + hinge; 8-leg spider test model:
# free + ball + hinge + slide), plus a mutation that must fail (hinge angle sign flipped).
# usage: gpu-wait run --timeout 1200 -- [env BODY=path.xml] agent/body_engine/check.sh [STAGE]
set -euo pipefail
cd "$(dirname "$0")/../.."
stage=${1:-kinematics}; M=.venv/bin/python; X=.work/agent/body_engine; fail=0
if [ "$stage" = cube ]; then  # closed 40 mm cube of planes (floor, 4 walls, ceiling; 330 pairs), states on every face
  C=$($M -c "from agent.body_engine.model import cube; print(cube('${BODY:-.work/agent/gpu-body-port/body-v2/body.xml}'))")
  $M -m agent.body_engine.model check $C $X/cube-$(basename ${BODY:-body.xml} .xml) actuated 64 || { echo "FAIL cube"; fail=$((fail+1)); }
  # the gate must fail when the bounding-sphere skip drops pairs that touch (rbound added instead of subtracted)
  ~/iTools/bin/mutant-check agent/body_engine/smooth.mojo 's/(pos2 - pos1)).reduce_add() - mf\[pf + 12\] > margin/(pos2 - pos1)).reduce_add() + mf[pf + 12] > margin/' -- \
    $M -m agent.body_engine.model check $C $X/mutant actuated 64 || fail=$((fail+1))
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all cube checks"; exit 0
fi
if [ "$stage" = terrain ]; then  # heightfield ground (tangent plane) on a tilted ramp vs MuJoCo on the same tilted plane
  for B in ${BODY:-.work/agent/gpu-body-port/body-v2/body.xml .work/agent/body_engine/spider-body/spider.xml}; do
    read T P < <($M -c "from agent.body_engine.model import terrain; print(*terrain('$B'))")
    ORACLE=$P $M -m agent.body_engine.model check $T $X/terrain-$(basename $B .xml) actuated 64 || { echo "FAIL terrain $(basename $B .xml)"; fail=$((fail+1)); }
  done
  # the gate must fail with the slope's normal tilted the wrong way (fly: the spider's known field would fail anyway)
  read T P < <($M -c "from agent.body_engine.model import terrain; print(*terrain('.work/agent/gpu-body-port/body-v2/body.xml'))")
  ~/iTools/bin/mutant-check agent/body_engine/smooth.mojo 's/normal = V3(-hg\[1\], -hg\[2\], 1, 0)/normal = V3(hg[1], hg[2], 1, 0)/' -- \
    env ORACLE=$P $M -m agent.body_engine.model check $T $X/mutant actuated 64 || fail=$((fail+1))
  # MEASURED, not a gate: tangent plane vs MuJoCo's own mjc_ConvexHField on bumpy terrain (0.3 mm, 8 mm wavelength,
  # 4 octaves; heightfield sampled every 0.31 mm). The tangent plane is not that algorithm; the stand gate is live check
  N=$($M -c "from agent.body_engine.model import terrain; print(terrain('.work/agent/gpu-body-port/body-v2/body.xml', (0, 0, 0.3, 8.0, 4, 1, 2.0, 0.5))[0])")
  $M -m agent.body_engine.model check $N $X/terrain-noise actuated 64 2>&1 | grep -E "max rel err|ncon" | sed 's/^\(PASS\|FAIL\)/MEASURED/' || true
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all terrain checks"; exit 0
fi
if [ "$stage" = groom ]; then  # the 7 sphere-sphere grooming pairs (front-leg rubs on head, antennae, each other)
  $M agent/body_engine/groom.py $X/groom.npz || { echo "FAIL groom states"; exit 1; }
  STATES=$X/groom.npz $M -m agent.body_engine.model check .work/agent/gpu-body-port/body-v2/body.xml $X/groom actuated 64 || { echo "FAIL groom"; fail=$((fail+1)); }
  # the gate must fail with the sphere-sphere normal reversed
  ~/iTools/bin/mutant-check agent/body_engine/smooth.mojo 's/normal = dif \/ len/normal = -dif \/ len/' -- \
    env STATES=$X/groom.npz $M -m agent.body_engine.model check .work/agent/gpu-body-port/body-v2/body.xml $X/mutant actuated 64 || fail=$((fail+1))
  # and with noslip skipped (the fly ships noslip 5; MuJoCo runs it)
  ~/iTools/bin/mutant-check agent/body_engine/smooth.mojo 's/^        noslip(mi, mf, res, f)$/        pass/' -- \
    env STATES=$X/groom.npz $M -m agent.body_engine.model check .work/agent/gpu-body-port/body-v2/body.xml $X/mutant actuated 64 || fail=$((fail+1))
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all groom checks"; exit 0
fi
if [ "$stage" = world ]; then  # fly + spider in one model (model.combine): engine vs MuJoCo on it, and MuJoCo merged vs
  # each animal alone (constraint islands: the merged solve must equal the separate ones)
  # spider 8 mm and 200 mm from the fly: each animal computes relative to its own origin (smooth.mojo origin())
  for off in 200 8; do
    W=$($M -c "from agent.body_engine.model import combine; print(combine('.work/agent/gpu-body-port/body-v2/body.xml', '.work/agent/body_engine/spider-body/spider.xml', ($off, 0, 0), 'world$off'))")
    $M -m agent.body_engine.model check $W $X/world$off actuated 64 || { echo "FAIL world spider at $off mm"; fail=$((fail+1)); }
    if [ $off = 200 ]; then  # the gate must fail when the spider uses the fly's origin (one origin per model)
      ~/iTools/bin/mutant-check agent/body_engine/smooth.mojo 's/    var r = Int(mi\[ROOT + b\])/    var r = Int(mi[ROOT + 1])/' -- \
        $M -m agent.body_engine.model check $W $X/mutant actuated 64 || fail=$((fail+1))
    fi
  done
  $M -c "
import numpy as np, mujoco
from agent.body_engine.model import load, standing
W, F, S = '$W', '.work/agent/gpu-body-port/body-v2/body.xml', '.work/agent/body_engine/spider-body/spider.xml'
w, f, s = load(W), load(F), load(S)
for m in (w, f, s): m.opt.noslip_iterations = 0  # the merge gives the spider the fly's 5; MuJoCo vs MuJoCo here, islands only
q, v = standing(w, 8)
dw, df, ds = mujoco.MjData(w), mujoco.MjData(f), mujoco.MjData(s)
err = 0.0
for n in range(8):
    dw.qpos[:], dw.qvel[:] = q[n], v[n]; df.qpos[:], df.qvel[:] = q[n, :f.nq], v[n, :f.nv]
    ds.qpos[:], ds.qvel[:] = q[n, f.nq:], v[n, f.nv:]; ds.qpos[:3] -= (8, 0, 0)
    for d, m in ((dw, w), (df, f), (ds, s)):
        d.ctrl[:] = 0; mujoco.mj_forward(m, d)
    for a, b in ((dw.qfrc_constraint[:f.nv], df.qfrc_constraint), (dw.qfrc_constraint[f.nv:], ds.qfrc_constraint)):
        err = max(err, np.abs(a - b).max() / np.abs(b).max()) if np.abs(b).max() > 0 else err  # no contact: nothing to compare
# MEASURED, not a gate (forces, not qacc: qacc of a near-static stance is the small rest of large cancelling forces).
# fly 1e-6, spider 7.3e-5 in float64, unchanged at tolerance 0 and with the spider's meaninertia: cause open.
print(f'MEASURED MuJoCo merged vs separate qfrc_constraint per animal: max rel err {err:.1e} over 8 states')" 
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all world checks"; exit 0
fi
if [ "$stage" = loop ]; then  # K steps per launch, state kept on the card. K=1 vs mj_step (one step: the accuracy
  # gate); one K=10 launch must equal ten K=1 launches byte for byte (the loop). K=10 vs ten chained mj_step is not a
  # gate: the engine's ~5e-6 per-step float32 error grows ~12x in 10 steps on contact-rich states (diag/k_floor.py)
  F=.work/agent/gpu-body-port/body-v2/body.xml
  $M -m agent.body_engine.model check $F $X/fly actuated 64 1 || { echo "FAIL fly loop K=1"; fail=$((fail+1)); }
  ~/.local/bin/mojo build $(cat $X/fly/sizes.txt) -DFLIES=64 -DSTAGE=7 -DK=10 -DWAVE=${WAVE:-0} agent/body_engine/smooth.mojo -o $X/fly/k10
  cp $X/fly/state10.bin $X/fly/k1x10.bin && $X/fly/k10 $X/fly/
  cmp $X/fly/state.bin $X/fly/k1x10.bin && echo "PASS K=10 launch == 10 K=1 launches, byte-equal" || { echo "FAIL loop K=10 vs 10 x K=1"; fail=$((fail+1)); }
  # the gate must fail with the velocity write-back dropped (each step restarts from the launch's qvel)
  ~/iTools/bin/mutant-check agent/body_engine/smooth.mojo 's/qvel\[v \* F + f\] += mf\[TSTEP\]/qvel[v * F + f] += 0 * mf[TSTEP]/' -- \
    $M -m agent.body_engine.model check .work/agent/gpu-body-port/body-v2/body.xml $X/mutant actuated 64 1 || fail=$((fail+1))
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all loop checks"; exit 0
fi
if [ "$stage" = spiderlink ]; then  # brain<->body link on the spider (docs/plans/SPIDER_BRAIN.md unit 1): kernel gait vs oracle, engine vs MuJoCo
  # under the brain's ctrl stream, closed-loop walk/turn/stand. Mutants: the ctrl write skipped; the gait kernel's yaw sign flipped
  L="$M -m agent.body_engine.spider_link check"
  $L || { echo "FAIL spiderlink"; fail=$((fail+1)); }
  ~/iTools/bin/mutant-check agent/body_engine/live.mojo 's/ctx.enqueue_copy(u_b, ctrl_h)/pass/' -- $L || fail=$((fail+1))
  ~/iTools/bin/mutant-check agent/body_engine/spider_brain.mojo 's/out = side \* YAW/out = -side * YAW/' -- $L || fail=$((fail+1))
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all spiderlink checks"; exit 0
fi
if [ "$stage" = fluid ]; then  # Flybody flight model: mj_fluid (ellipsoid wings + inertia boxes) inside qfrc_passive, wing velocities up to flapping rate
  R=~/Projects/science/fly-connectome; FL=${FLIGHT:-$R/.work/flight-body}
  [ -f $FL/flight.xml ] || $R/.work/neural-flies/flight-venv/bin/python agent/body_engine/models/flight_export.py $FL
  VSCALE=${VSCALE:-100} $M -m agent.body_engine.model check $FL/flight.xml $X/flight rne 64 || { echo "FAIL flight fluid"; fail=$((fail+1)); }
  # each mutant must fail: Kutta lift sign, Magnus sign, added-mass sign, inertia-box drag sign
  for mut in 's/    fo += cross(circ, v)/    fo -= cross(circ, v)/' 's/    fo += cross(w, v) \* (magnus/    fo -= cross(w, v) * (magnus/' \
             's/    var fo = cross(lin_mom, w)/    var fo = -cross(lin_mom, w)/' 's/        fr\[3\] -= 0.5 \* rho/        fr[3] += 0.5 * rho/'; do
    ~/iTools/bin/mutant-check agent/body_engine/smooth.mojo "$mut" -- \
      env VSCALE=${VSCALE:-100} $M -m agent.body_engine.model check $FL/flight.xml $X/mutant rne 64 || fail=$((fail+1))
  done
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all fluid checks"; exit 0
fi
if [ "$stage" = loopk ]; then  # engine vs ITSELF, no MuJoCo: one K=10 launch == ten K=1 launches byte for byte, on BODY (default fly)
  # with ENGINE_OPT options (integrator, solver). One-step MuJoCo parity is the separate per-option stages (integrator, loop)
  B=${BODY:-.work/agent/gpu-body-port/body-v2/body.xml}; D=$X/loopk-$(basename $B .xml)
  ST=actuated; [ "$(basename $B)" = spider_test.xml ] && ST=step  # the spider test body has no actuators
  ENGINE_ONLY=1 $M -m agent.body_engine.model check $B $D $ST 64 1 | tail -1
  ~/.local/bin/mojo build $(cat $D/sizes.txt) -DFLIES=64 -DSTAGE=$($M -c "from agent.body_engine.model import STAGES; print(STAGES['$ST'][0])") -DK=10 -DWAVE=${WAVE:-0} agent/body_engine/smooth.mojo -o $D/k10
  cp $D/state10.bin $D/k1x10.bin && $D/k10 $D/ >/dev/null
  cmp $D/state.bin $D/k1x10.bin && echo "PASS loopk $(basename $B .xml) ${ENGINE_OPT:-defaults}: K=10 launch == 10 K=1 launches, byte-equal (engine vs engine, not MuJoCo)" || { echo "FAIL loopk $(basename $B .xml)"; exit 1; }
  exit 0
fi
if [ "$stage" = integrator ]; then  # implicitfast (mj_implicitSkip) vs MuJoCo with opt.integrator=implicitfast, fly + spider + 2-animal world
  F=.work/agent/gpu-body-port/body-v2/body.xml; SP=agent/body_engine/models/spider_test.xml
  SB=.work/agent/body_engine/spider-body/spider.xml; I=$X/integrator; KV=0.01  # KV: the shipped servos have kv 0 = Euler's qH; 0.01 makes h gear^2 kv ~ M's diagonal
  echo "COVERAGE integrator=implicitfast (3): fly step + actuated (airborne states, kv=$KV, shipped kv=0 in step), spider_test step, world fly+spider at 8 and 200 mm"
  echo "COVERAGE not covered: standalone free body (midpoint, refused at export), implicit (full), RK4, tendon-damping and fluid derivatives (refused), spring-only tendons (accepted, no velocity derivative, no parity case), contact states (MEASURED only)"
  $M -m agent.body_engine.reject_check || { echo "FAIL reject"; fail=$((fail+1)); }
  ~/iTools/bin/mutant-check agent/body_engine/model.py 's/m.opt.density or m.opt.viscosity or m.tendon_damping.any()/False/' -- $M -m agent.body_engine.reject_check || fail=$((fail+1))
  export ENGINE_OPT=integrator=3,kv=$KV
  A=$($M -m agent.body_engine.model airborne $F $I/fly | tail -1)
  STATES=$A $M -m agent.body_engine.model check $F $I/fly actuated 64 || { echo "FAIL fly actuated implicitfast"; fail=$((fail+1)); }
  ENGINE_OPT=integrator=3 $M -m agent.body_engine.model check $F $I/fly-step step 64 || { echo "FAIL fly step implicitfast"; fail=$((fail+1)); }
  ENGINE_OPT=integrator=3 $M -m agent.body_engine.model check $SP $I/spider step 64 || { echo "FAIL spider step implicitfast"; fail=$((fail+1)); }
  # the gate must fail with the actuator velocity term of qH (qDeriv) added instead of subtracted
  ~/iTools/bin/mutant-check agent/body_engine/smooth.mojo 's/-= h \* mf\[af + 8\] \* mf\[af + 8\] \* mf\[af + 3\]/+= h * mf[af + 8] * mf[af + 8] * mf[af + 3]/' -- \
    env STATES=$A $M -m agent.body_engine.model check $F $I/mutant actuated 64 || fail=$((fail+1))
  # islands: two animals in one model. Each animal's dofs get their own qH terms; the engine must match MuJoCo on the merged model
  # at 200 mm (two islands) and 8 mm (the spider's origin 8 mm away), and fail when the spider uses the fly's origin
  for off in 200 8; do
    W=$($M -c "from agent.body_engine.model import combine; print(combine('$F', '$SB', ($off, 0, 0), 'iworld$off'))")
    AW=$($M -m agent.body_engine.model airborne $W $I/world$off | tail -1)
    STATES=$AW $M -m agent.body_engine.model check $W $I/world$off actuated 64 || { echo "FAIL implicitfast world at $off mm"; fail=$((fail+1)); }
    if [ $off = 200 ]; then
      ~/iTools/bin/mutant-check agent/body_engine/smooth.mojo 's/    var r = Int(mi\[ROOT + b\])/    var r = Int(mi[ROOT + 1])/' -- \
        env STATES=$AW $M -m agent.body_engine.model check $W $I/mutant actuated 64 || fail=$((fail+1))
    fi
  done
  # MEASURED, not a gate: contact states (active-set float32 noise, fly kv 0.01: qacc_euler 2.4e-5, qfrc_constraint 1.2e-4 vs a floor-lifted bar of 1.3e-5, 6.4e-5)
  $M -m agent.body_engine.model check $F $I/fly-contact actuated 64 2>&1 | grep -E "max rel err" | grep -E "qacc|qfrc_constraint" | sed 's/^\(PASS\|FAIL\)/MEASURED/' || true
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all integrator checks (implicitfast)"; exit 0
fi
if [ "$stage" = tendon ]; then  # Flybody flight model, airborne (no contacts) with a spring and damper on its two fixed tendons: full step, actuators incl. trntype 3
  R=~/Projects/science/fly-connectome; FL=${FLIGHT:-$R/.work/flight-body}
  [ -f $FL/flight.xml ] || $R/.work/neural-flies/flight-venv/bin/python agent/body_engine/models/flight_export.py $FL
  A=$($M -c "from agent.body_engine.model import airborne_mjcf; print(airborne_mjcf('$FL/flight.xml'))")
  $M -m agent.body_engine.model check $A $X/airborne actuated 64 || { echo "FAIL flight tendon"; fail=$((fail+1)); }
  # each mutant must fail: tendon length sign, passive coef sign, actuator moment sign, spring dead zone dropped
  for mut in 's/        length += c \* qpos/        length -= c * qpos/' 's/\] += mf\[TNF + w\] \* frc/] -= mf[TNF + w] * frc/' \
             's/+= gear \* mf\[TNF + w\] \* force/-= gear * mf[TNF + w] * force/' 's/var x = lv\[0\] - hi if lv\[0\] > hi else (lv\[0\] - lo if lv\[0\] < lo else Float32(0))/var x = lv[0] - hi/'; do
    ~/iTools/bin/mutant-check agent/body_engine/smooth.mojo "$mut" -- $M -m agent.body_engine.model check $A $X/mutant actuated 64 || fail=$((fail+1))
  done
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all tendon checks"; exit 0
fi
B=${BODY:-.work/agent/gpu-body-port/body-v2/body.xml}  # BODY=path checks another MJCF (e.g. the spider from models/spider.py)
$M -m agent.body_engine.model check $B $X/body-$(basename $B .xml) "$stage" 64 || { echo "FAIL $(basename $B .xml) $stage"; fail=$((fail+1)); }
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
