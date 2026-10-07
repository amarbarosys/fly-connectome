#!/usr/bin/env bash
# Every ported stage against MuJoCo on two bodies (fly: free + hinge; 8-leg spider test model:
# free + ball + hinge + slide), plus a mutation that must fail (hinge angle sign flipped).
# usage: gpu-wait run --timeout 1200 -- [env BODY=path.xml] agent/body_engine/check.sh [STAGE]
set -euo pipefail
cd "$(dirname "$0")/../.."
if [ -z "${CHECK_LOCK_HELD:-}" ]; then  # a stage that calls another (solver -> loopk) passes the lock down
  mkdir -p .work; exec 9>>.work/check.lock  # one stage per worktree: mutant-check edits sources in place (SHARED rule 13)
  flock -n 9 || { echo "FAIL check.sh: another stage is running in $(pwd) ($(tail -n 1 .work/check.lock)); run stages one at a time"; exit 1; }
  echo "pid $$ stage ${1:-kinematics} since $(date +%T)" > .work/check.lock
  export CHECK_LOCK_HELD=1
fi
stage=${1:-kinematics}; M=.venv/bin/python; X=.work/agent/body_engine; fail=0
[ "${QUICK:-0}" = 1 ] && { echo "QUICK=1: no mutants (all states kept, same bars): an iteration run, NOT a gate (verify_commit runs the full stage)"; export QUICK; trap 'echo "QUICK=1 run finished: rerun without QUICK before READY"' EXIT; }
if [ "$stage" = convex ]; then  # GJK + EPA narrowphase (convex.mojo, CPU build) vs MuJoCo mj_collision, per pair kind: float64 exact, float32 bars
  KINDS=${KINDS:-ellipsoid box}
  for kind in $KINDS; do
    $M -m agent.body_engine.convex_check $kind 6000 f64 1e-9 1 || { echo "FAIL convex $kind f64"; fail=$((fail+1)); }
    $M -m agent.body_engine.convex_check $kind 6000 f32 1e-4 1 || { echo "FAIL convex $kind f32"; fail=$((fail+1)); }
  done
  # the gate must fail with the EPA contact normal flipped
  ~/iTools/bin/mutant-check agent/body_engine/convex.mojo 's/return normalize3(st.x1 - st.x2)/return normalize3(st.x2 - st.x1)/' -- \
    $M -m agent.body_engine.convex_check ellipsoid 600 f64 1e-9 3 || fail=$((fail+1))
  # box-mesh only: the gate must fail with the perturbation search's negative tilt dropped (contact count)
  case " $KINDS " in *" box "*) ~/iTools/bin/mutant-check agent/body_engine/convex.mojo 's/var sgn = S(-1) if angle_id == 0 else S(1)/var sgn = S(1)/' -- \
    $M -m agent.body_engine.convex_check box 600 f64 1e-9 3 || fail=$((fail+1));; esac
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all convex checks"; exit 0
fi
if [ "$stage" = hfield ]; then  # exact heightfield contact (mjc_ConvexHField: GJK + EPA per cell prism, convex.mojo, CPU build) on a bumpy seeded terrain vs MuJoCo's own hfield geom
  $M -m agent.body_engine.hfield_check 6000 f64 1 || { echo "FAIL hfield f64"; fail=$((fail+1)); }
  $M -m agent.body_engine.hfield_check 6000 f32 1 || { echo "FAIL hfield f32"; fail=$((fail+1)); }
  # the gate must fail with the contact normal flipped, and with the prism height test inverted (cells under the mesh skipped)
  ~/iTools/bin/mutant-check agent/body_engine/convex.mojo 's/return normalize3(st.x1 - st.x2)/return normalize3(st.x2 - st.x1)/' -- \
    $M -m agent.body_engine.hfield_check 600 f64 3 || fail=$((fail+1))
  ~/iTools/bin/mutant-check agent/body_engine/convex.mojo 's/if o1.prism\[3\]\[2\] < zmin and/if o1.prism[3][2] > zmin and/' -- \
    $M -m agent.body_engine.hfield_check 600 f64 3 || fail=$((fail+1))
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all hfield checks"; exit 0
fi
if [ "$stage" = arena ]; then  # the original arena (box walls, stripes, beacons; ellipsoid stones): every pair incl. box-mesh and ellipsoid-mesh (GJK + EPA), states moved into walls and stones
  B=.work/agent/gpu-body-port/body-v2/body.xml
  # bar REL=3e-3 (default 1e-5), measured: qacc 7.1e-4, qacc_euler 2.0e-3 (3 x the float32 input floor is 3.5e-4 / 1.5e-3). `row_diag.py` on the same
  # states: convex contacts' aref within 6e-5 of the state's largest row, plane contacts 1.3e-4, ncon equal in all 64 states: the rest is the
  # engine's float32 solver on states with a 0.03 mm deep wall or stone contact, not the convex pairs. The gate must fail with the convex normal flipped.
  for st in contact actuated; do
    CONVEX=1 REL=3e-3 $M -m agent.body_engine.model check $B $X/arena-$st $st 64 || { echo "FAIL arena $st"; fail=$((fail+1)); }
  done
  # the gate must fail with the convex contact normal flipped
  ~/iTools/bin/mutant-check agent/body_engine/smooth.mojo 's/var normal = cs.normal\[count\].cast\[DType.float32\]()/var normal = -cs.normal[count].cast[DType.float32]()/' -- \
    env CONVEX=1 REL=3e-3 $M -m agent.body_engine.model check $B $X/mutant contact 64 || fail=$((fail+1))
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all arena checks"; exit 0
fi
if [ "$stage" = stone ]; then  # a stone in the walking fly's path through the live transport (fly_link): engine vs MuJoCo path effect, convex-off mutant
  $M -m agent.body_engine.stone_check || { echo "FAIL stone: log above"; exit 1; }
  exit 0
fi
if [ "$stage" = wall ]; then  # a wall square across the walking fly's path (box-vs-mesh pairs), same gates and mutant as stone
  $M -m agent.body_engine.stone_check wall || { echo "FAIL wall: log above"; exit 1; }
  exit 0
fi
if [ "$stage" = convexstep ]; then  # box and ellipsoid stones in a CHAINED engine run vs MuJoCo mj_step: fly (arena walls and stones) and spider (stones under its feet, model.stones)
  # per body: (1) 20 steps in one launch, qpos and qvel vs 20 chained mj_step, states whose own MuJoCo trajectory moves under a 1e-6 perturbation dropped (model.check,
  # at most half, listed); (2) the full one-step fields (ncon, qacc, qfrc_constraint, ...) on the states the engine reached after 3 steps (spider: 1), so forces are gated
  # on engine-evolved contacts too. Bars are measured stated bars: trajectories REL 1e-3, forces REL 1e-4 (measured fly qfrc_constraint 1.1e-5, qacc_euler 2.9e-5 on engine-evolved states; arena's 3e-3 is its deep 0.03 mm contacts).
  FLY=.work/agent/gpu-body-port/body-v2/body.xml
  SP=$($M -c "from agent.body_engine.model import stones; print(stones('.work/agent/body_engine/spider-body/spider.xml', 'spider_stones.xml'))")
  N=${CVS_COUNT:-64}  # states per body (CVS_COUNT=8: a short probe before the full run)
  for B in $FLY $SP; do
    n=$(basename $B .xml); S=; M_MIN=; EK=3; [ $B = $SP ] && { S=STONES=1; EK=1; M_MIN=CONVEX_MIN=$((N / 16)); }  # the spider's feet hover at the 1 mm margin: its stone contacts toggle within 3 steps, so its evolved states are 1 step on and keep few of them (stone contacts there are gated by the K=20 trajectory; the force check needs 1/16 of the states to hold one)
    env CONVEX=1 REL=1e-3 $S $M -m agent.body_engine.model check $B $X/cvs-$n-20 actuated $N 20 || { echo "FAIL convexstep $n K=20"; fail=$((fail+1)); }
    env CONVEX=1 REL=1e-3 $S SAVE_STATES=$X/cvs-$n.npz $M -m agent.body_engine.model check $B $X/cvs-$n-3 actuated $N $EK || { echo "FAIL convexstep $n K=$EK"; fail=$((fail+1)); }
    env CONVEX=1 REL=${CVS_FREL:-1e-4} $S $M_MIN STATES=$X/cvs-$n.npz $M -m agent.body_engine.model check $B $X/cvs-$n-f actuated $N || { echo "FAIL convexstep $n forces on engine states"; fail=$((fail+1)); }
  done
  # the gate must fail with the EPA contact normal flipped, on the fly and on the spider (32 states each: a flipped normal moves every kept state)
  ~/iTools/bin/mutant-check agent/body_engine/convex.mojo 's/return normalize3(st.x1 - st.x2)/return normalize3(st.x2 - st.x1)/' -- \
    env CONVEX=1 REL=1e-3 $M -m agent.body_engine.model check $FLY $X/mutant actuated ${CVS_MUTANT_COUNT:-32} 20 || fail=$((fail+1))
  ~/iTools/bin/mutant-check agent/body_engine/convex.mojo 's/return normalize3(st.x1 - st.x2)/return normalize3(st.x2 - st.x1)/' -- \
    env CONVEX=1 REL=1e-3 STONES=1 $M -m agent.body_engine.model check $SP $X/mutant actuated ${CVS_MUTANT_COUNT:-32} 20 || fail=$((fail+1))
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all convexstep checks"; exit 0
fi
if [ "$stage" = cube ]; then  # closed 40 mm cube of planes (floor, 4 walls, ceiling; 330 pairs), states on every face
  C=$($M -c "from agent.body_engine.model import cube; print(cube('${BODY:-.work/agent/gpu-body-port/body-v2/body.xml}'))")
  $M -m agent.body_engine.model check $C $X/cube-$(basename ${BODY:-body.xml} .xml) actuated 64 || { echo "FAIL cube"; fail=$((fail+1)); }
  # the gate must fail when the bounding-sphere skip drops pairs that touch (rbound added instead of subtracted)
  ~/iTools/bin/mutant-check agent/body_engine/smooth.mojo 's/(pos2 - pos1)).reduce_add() - mf\[pf + 12\] > margin/(pos2 - pos1)).reduce_add() + mf[pf + 12] > margin/' -- \
    $M -m agent.body_engine.model check $C $X/mutant actuated 64 || fail=$((fail+1))
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all cube checks"; exit 0
fi
if [ "$stage" = terrain_exact ]; then  # exact heightfield contact in the step (GJK + EPA per cell prism on the terrain function) on bumpy seeded terrain vs MuJoCo's OWN hfield geom (not the plane twin), fly and spider
  # bar REL=4e-4 (default 1e-5): the engine takes node heights from the float32 terrain function, MuJoCo from its float32 table
  # (hfield_check ... func: 1e-7 mm apart, 1e-4 of a contact depth); measured fly qacc 2.1e-4, others <= 3.4e-5. The gate must fail with the contact normal flipped
  for B in ${BODY:-.work/agent/gpu-body-port/body-v2/body.xml .work/agent/body_engine/spider-body/spider.xml}; do
    N=$($M -c "from agent.body_engine.model import terrain; print(terrain('$B', (0, 0, 0.3, 8.0, 4, 1, 2.0, 0.5))[0])")
    for st in contact actuated; do
      HFEXACT=1 REL=4e-4 $M -m agent.body_engine.model check $N $X/terrain-exact-$(basename $B .xml)-$st $st 64 || { echo "FAIL terrain_exact $(basename $B .xml) $st"; fail=$((fail+1)); }
    done
  done
  N=$($M -c "from agent.body_engine.model import terrain; print(terrain('.work/agent/gpu-body-port/body-v2/body.xml', (0, 0, 0.3, 8.0, 4, 1, 2.0, 0.5))[0])")
  ~/iTools/bin/mutant-check agent/body_engine/convex.mojo 's/return normalize3(st.x1 - st.x2)/return normalize3(st.x2 - st.x1)/' -- \
    env HFEXACT=1 REL=4e-4 $M -m agent.body_engine.model check $N $X/mutant contact 64 || fail=$((fail+1))
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all terrain_exact checks"; exit 0
fi
if [ "$stage" = terrain ]; then  # heightfield ground (tangent plane) on a tilted ramp vs MuJoCo on the same tilted plane
  for B in ${BODY:-.work/agent/gpu-body-port/body-v2/body.xml .work/agent/body_engine/spider-body/spider.xml}; do
    read T P < <($M -c "from agent.body_engine.model import terrain; print(*terrain('$B'))")
    # spider bar 3e-5, measured (main 6e6d0e9, deterministic over reruns): qfrc_constraint 1.19e-5 = 6.2 x its float32 floor 1.92e-6
    # (the open spider qfrc gap, also on the flat floor), qLD 9.3e-6, qacc 5.4e-6, the rest below 6e-6. Fly stays at the 1e-5 default.
    case $B in *spider*) rel=3e-5;; *) rel=${REL:-1e-5};; esac
    ORACLE=$P REL=$rel $M -m agent.body_engine.model check $T $X/terrain-$(basename $B .xml) actuated 64 || { echo "FAIL terrain $(basename $B .xml)"; fail=$((fail+1)); }
  done
  # the gate must fail with the slope's normal tilted the wrong way, on each body at its own bar
  for B in .work/agent/gpu-body-port/body-v2/body.xml .work/agent/body_engine/spider-body/spider.xml; do
    read T P < <($M -c "from agent.body_engine.model import terrain; print(*terrain('$B'))")
    case $B in *spider*) rel=3e-5;; *) rel=1e-5;; esac
    ~/iTools/bin/mutant-check agent/body_engine/smooth.mojo 's/normal = V3(-hg\[1\], -hg\[2\], 1, 0)/normal = V3(hg[1], hg[2], 1, 0)/' -- \
      env ORACLE=$P REL=$rel $M -m agent.body_engine.model check $T $X/mutant-$(basename $B .xml) actuated 64 || fail=$((fail+1))
  done
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
if [ "$stage" = equality ] || [ "$stage" = equality_contact ]; then  # connect, weld, joint equalities (models/eq_test.xml: random
  # states violate them; Jdot v needs velocities) and, for equality_contact, with an active sphere-sphere contact and noslip 5
  # in the same Newton solve (mixed H rows); both vs mj_step, float32-floor bars as the contact stage
  E=agent/body_engine/models/eq_test.xml; [ "$stage" = equality_contact ] && E=agent/body_engine/models/eq_contact.xml
  $M -m agent.body_engine.model check $E $X/$stage equality 64 || { echo "FAIL $stage"; fail=$((fail+1)); }
  S=agent/body_engine/smooth.mojo
  # the gate must fail with: the anchor sign flipped, the weld rotation conjugate dropped, the joint polynomial sign flipped,
  # the Jdot v correction sign flipped, the reference acceleration dropped
  for mut in 's/var p0 = rot(B0\[1\], m3(mf, ef))/var p0 = rot(B0[1], -m3(mf, ef))/' \
             's/var nq1 = Q4(qb\[0\], -qb\[1\], -qb\[2\], -qb\[3\])/var nq1 = Q4(qb[0], qb[1], qb[2], qb[3])/' \
             's/cpos\[0\] -= mf\[ef + 25\] \* dif/cpos[0] += mf[ef + 25] * dif/' \
             's/jdv\[k\] = jd0\[k\] - jd1\[k\]/jdv[k] = jd0[k] + jd1[k]/' \
             's/ - get(res, f, OEQA + rbase + k)/ - 0 * get(res, f, OEQA + rbase + k)/'; do
    ~/iTools/bin/mutant-check $S "$mut" -- $M -m agent.body_engine.model check $E $X/mutant equality 64 || fail=$((fail+1))
  done
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all $stage checks"; exit 0
fi
if [ "$stage" = mocap ]; then  # two mocap bodies (random mocap_pos/quat per state, quats unnormalised) welded/connected to the animal
  E=agent/body_engine/models/eq_mocap.xml
  $M -m agent.body_engine.model check $E $X/mocap mocap 64 || { echo "FAIL mocap"; fail=$((fail+1)); }
  # the gate must fail with every mocap body reading the first one's pose, and with the quaternion components swapped
  for mut in 's/var mrow = NU + 7 \* mid/var mrow = NU + 7 * 0/' \
             's/xquat = Q4(ctrl\[(mrow + 3) \* F + f\], ctrl\[(mrow + 4) \* F + f\]/xquat = Q4(ctrl[(mrow + 4) * F + f], ctrl[(mrow + 3) * F + f]/'; do
    ~/iTools/bin/mutant-check agent/body_engine/smooth.mojo "$mut" -- $M -m agent.body_engine.model check $E $X/mutant mocap 64 || fail=$((fail+1))
  done
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all mocap checks"; exit 0
fi
if [ "$stage" = gravcomp ]; then  # body gravcomp (mj_gravcomp) on the fly and the spider test body (body_gravcomp 1, .5, 0, .8 in turn)
  # and on models/gravcomp_act.xml, where joints with actgravcomp take the actuator route (engine_forward.c:770; needs nu > 0)
  S=agent/body_engine/smooth.mojo; G=agent/body_engine/models/gravcomp_act.xml; mkdir -p $X/gc
  FG=$($M -c "from agent.body_engine.model import with_gravcomp; print(with_gravcomp('${BODY:-.work/agent/gpu-body-port/body-v2/body.xml}', '$X/gc/fly.xml'))")
  SG=$($M -c "from agent.body_engine.model import with_gravcomp; print(with_gravcomp('agent/body_engine/models/spider_test.xml', '$X/gc/spider.xml'))")
  for B in $FG $SG $G; do for st in rne step; do
    $M -m agent.body_engine.model check $B $X/gc/$(basename $B .xml)-$st $st 64 || { echo "FAIL gravcomp $(basename $B .xml) $st"; fail=$((fail+1)); }
  done; done
  $M -m agent.body_engine.model check $G $X/gc/act actuated 64 || { echo "FAIL gravcomp actuator route"; fail=$((fail+1)); }
  # the gate must fail with gravcomp skipped, with its sign flipped, with the actuator route dropped and with actgravcomp ignored
  ~/iTools/bin/mutant-check $S 's/^        gravcomp(mi, mf, res, f)$/        pass/' -- $M -m agent.body_engine.model check $FG $X/mutant rne 64 || fail=$((fail+1))
  ~/iTools/bin/mutant-check $S 's/-(mf\[MASS + i\] \* gc)/(mf[MASS + i] * gc)/' -- $M -m agent.body_engine.model check $FG $X/mutant rne 64 || fail=$((fail+1))
  ~/iTools/bin/mutant-check $S 's/res\[(OQACT + d) \* F + f\] += get(res, f, OGC + d)/res[(OQACT + d) * F + f] += 0 * get(res, f, OGC + d)/' -- $M -m agent.body_engine.model check $G $X/mutant actuated 64 || fail=$((fail+1))
  ~/iTools/bin/mutant-check $S '0,/if mi\[CGC + j\] != 0:/s//if mi[CGC + j] != 9:/' -- $M -m agent.body_engine.model check $G $X/mutant rne 64 || fail=$((fail+1))
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all gravcomp checks"; exit 0
fi
if [ "$stage" = sptendon ]; then  # spatial tendons (mj_tendon, mju_wrap): site-site, pulleys, sphere and cylinder wraps with sidesites (inside the
  # sphere too: wrap_inside), a fixed tendon beside them, a tendon servo; fixture models/tendon_spatial.xml (random states, then the actuated path
  # on rollout states) and the fly and spider with three spatial tendons added (model.with_spatial_tendons), vs mj_step on the tendon outputs
  # (ten_length, ten_velocity, dense ten_J) and everything the tendon spring and damper reach
  S=agent/body_engine/smooth.mojo; T=agent/body_engine/models/tendon_spatial.xml; mkdir -p $X/st
  FT=$($M -c "from agent.body_engine.model import with_spatial_tendons; print(with_spatial_tendons('${BODY:-.work/agent/gpu-body-port/body-v2/body.xml}', '$X/st/fly.xml'))")
  ST=$($M -c "from agent.body_engine.model import with_spatial_tendons; print(with_spatial_tendons('agent/body_engine/models/spider_test.xml', '$X/st/spider.xml'))")
  for B in $T $FT $ST; do
    $M -m agent.body_engine.model check $B $X/st/$(basename $B .xml) sptendon 64 || { echo "FAIL sptendon $(basename $B .xml)"; fail=$((fail+1)); }
  done
  $M -m agent.body_engine.model check $T $X/st/act sptendon_act 64 || { echo "FAIL sptendon_act"; fail=$((fail+1)); }
  # the gate must fail with: the wrap side flipped, the tangent solutions swapped, the moment arm sign flipped, the pulley divisor dropped,
  # the cylinder's height correction dropped, the inside wrap's angle sign flipped
  for mut in 's/good = ux \* side\[0\] + uy \* side\[1\]/good = -(ux * side[0] + uy * side[1])/' \
             's/var sgn: Float32 = 1 if i == 0 else -1/var sgn: Float32 = -1 if i == 0 else 1/' \
             's/tjac(mi, qpos, res, f, row, bb, pb, dif, 1 \/ divisor, frame)/tjac(mi, qpos, res, f, row, bb, pb, dif, -1 \/ divisor, frame)/' \
             's/divisor = mf\[TNF + w0\]/divisor = 1/' \
             's/r1\[2\] = p0\[2\] + (p1\[2\] - p0\[2\]) \* (L0 + wlen) \/ (L0 + wlen + L1)/r1[2] = p0[2]/' \
             's/ang = asin64(z) - asin64(A \* z)/ang = asin64(z) + asin64(A * z)/'; do
    ~/iTools/bin/mutant-check $S "$mut" -- $M -m agent.body_engine.model check $T $X/mutant sptendon 64 || fail=$((fail+1))
  done
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all sptendon checks"; exit 0
fi
if [ "$stage" = loop ]; then  # K steps per launch, state kept on the card. K=1 vs mj_step (one step: the accuracy
  # gate); one K=10 launch must equal ten K=1 launches byte for byte (the loop). K=10 vs ten chained mj_step is not a
  # gate: the engine's ~5e-6 per-step float32 error grows ~12x in 10 steps on contact-rich states (diag/k_floor.py)
  F=.work/agent/gpu-body-port/body-v2/body.xml
  # one step is not a chained trajectory: the chaotic-state filter (for K>1 at wide bars) flags 52/64 states at this 1e-5 bar
  CHAOS_SEEDS=0 $M -m agent.body_engine.model check $F $X/fly actuated 64 1 || { echo "FAIL fly loop K=1"; fail=$((fail+1)); }
  ~/iTools/bin/mojo-cached build $(cat $X/fly/sizes.txt) -DFLIES=64 -DSTAGE=7 -DK=10 -DWAVE=${WAVE:-0} agent/body_engine/smooth.mojo -o $X/fly/k10
  cp $X/fly/state10.bin $X/fly/k1x10.bin && $X/fly/k10 $X/fly/
  cmp $X/fly/state.bin $X/fly/k1x10.bin && echo "PASS K=10 launch == 10 K=1 launches, byte-equal" || { echo "FAIL loop K=10 vs 10 x K=1"; fail=$((fail+1)); }
  # the gate must fail with the velocity write-back dropped (each step restarts from the launch's qvel)
  ~/iTools/bin/mutant-check agent/body_engine/smooth.mojo 's/qvel\[v \* F + f\] += mf\[TSTEP\]/qvel[v * F + f] += 0 * mf[TSTEP]/' -- \
    env CHAOS_SEEDS=0 $M -m agent.body_engine.model check .work/agent/gpu-body-port/body-v2/body.xml $X/mutant actuated 64 1 || fail=$((fail+1))
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
if [ "$stage" = spectate ]; then  # spectator hooks in live.mojo (docs/plans/RENDERER.md, -DFOLLOW=1 -DHUD=1): camera follows the walking spider, HUD bars show the
  # state buffer, hooks off changes no byte, frame cost split + leak receipt through spectator_integration_check.py. Mutants: camera target offset,
  # HUD fill off by one pixel, the follow hook compiled in by default (the hooks-off frame must differ from the pre-hook live.mojo)
  S="$M -m agent.body_engine.spectate_check"
  $M agent/body_engine/spectator_check.py || { echo "FAIL spectator_check"; fail=$((fail+1)); }
  $S check || { echo "FAIL spectate"; fail=$((fail+1)); }
  ~/iTools/bin/mutant-check agent/body_engine/spectate.mojo 's|tx /= Float32(NROOT)|tx = tx / Float32(NROOT) + 30.0|' -- $S clip || fail=$((fail+1))
  ~/iTools/bin/mutant-check agent/body_engine/spectate.mojo 's|Float32(x - BAR_X) + 0.5 < fill.width|Float32(x - BAR_X) + 1.5 < fill.width|' -- $S hud || fail=$((fail+1))
  ~/iTools/bin/mutant-check agent/body_engine/live.mojo 's|get_defined_int\["FOLLOW", 0\]|get_defined_int["FOLLOW", 1]|' -- $S off || fail=$((fail+1))
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all spectate checks"; exit 0
fi
if [ "$stage" = game ]; then  # game layer per step (-DGAME=1, docs/plans/GAME_LAYER.md) on fly and spider: engine energy/stamina rows == game_events.replay on
  # MuJoCo's per-step work, death at the oracle's step, servos and adhesion exactly 0 after it, HUD bars show the engine rows.
  # Mutants: the work term skipped; the dead gate removed
  G="$M -m agent.body_engine.game_check check"
  $G || { echo "FAIL game"; fail=$((fail+1)); }
  ~/iTools/bin/mutant-check agent/body_engine/smooth.mojo 's|work += abs(get(res, f, OQACT + v) \* qvel\[v \* F + f\]) \* mf\[TSTEP\]|work += 0|' -- $G fly || fail=$((fail+1))
  ~/iTools/bin/mutant-check agent/body_engine/smooth.mojo 's|if get(res, f, OGAME + 3) != 0:|if get(res, f, OGAME + 3) > 5:|' -- $G fly || fail=$((fail+1))
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all game checks"; exit 0
fi
if [ "$stage" = flylink ]; then  # brain-driven fly in live, host transport (BRAIN=1; docs/plans/BRAIN_BODY_LOOP.md): resident Shiu -> 2 DN drives -> FlyGym gait mapping -> 50 ctrl.
  # Gates: walks > 1 body length with tilt < 20 deg, opposite turn drives give opposite yaw, silenced DN and held ctrl differ from the live brain. Needs the Shiu worker:
  # submit with --vram 4. Mutants: ctrl write skipped, DN counts zeroed before the readout, left and right drive merged
  export PYTHONPATH=$PWD
  L="$M -m agent.body_engine.fly_link check"
  $L || { echo "FAIL flylink"; fail=$((fail+1)); }
  ~/iTools/bin/mutant-check agent/body_engine/live.mojo 's/ctx.enqueue_copy(u_b, ctrl_h)/pass/' -- $L || fail=$((fail+1))
  ~/iTools/bin/mutant-check agent/body_engine/fly_link.py 's/cum.append(side_counts(brain.advance(k), edges))/cum.append(side_counts(brain.advance(k) * 0, edges))/' -- $L || fail=$((fail+1))
  ~/iTools/bin/mutant-check agent/body_engine/fly_link.py 's/sig = np.clip(drive\[n\], 0, 1)/sig = np.clip(np.full(2, drive[n].mean()), 0, 1)/' -- $L || fail=$((fail+1))
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all flylink checks"; exit 0
fi
if [ "$stage" = world_hud ]; then  # plume world in the live frame (-DWORLD=1, docs/plans/WORLD.md): odor sites sample the plume on the card, the HUD prints the
  # concentrations and draws the sources from the same descriptor, fly (fly/l_odor, fly/r_odor) and spider (leg I tarsus tips), decoy present and absent.
  # Mutants: food/decoy swapped in the sampler, the presence flag ignored, mini-map y flipped, text colour back to white (illegible on the pale scene)
  export PYTHONPATH=$PWD
  W="$M -m agent.body_engine.world_check check"
  $W || { echo "FAIL world_hud"; fail=$((fail+1)); }
  ~/iTools/bin/mutant-check agent/body_engine/spectate.mojo 's|    plume\[2 \* s\] = c\[0\]|    plume[2 * s] = c[1]|' -- $W spider || fail=$((fail+1))
  ~/iTools/bin/mutant-check agent/body_engine/spectate.mojo 's|Int32(src\[8\]), src\[9\])|Int32(1), src[9])|' -- $W spider || fail=$((fail+1))
  ~/iTools/bin/mutant-check agent/body_engine/hud.mojo 's|center_y - (source_y - target_y) \* scale|center_y + (source_y - target_y) * scale|' -- $W spider || fail=$((fail+1))
  ~/iTools/bin/mutant-check agent/body_engine/spectate.mojo 's|comptime TEXT_RGB = UInt32(0x202020)|comptime TEXT_RGB = UInt32(0xFFFFFF)|' -- $W spider || fail=$((fail+1))  # white text on the pale floor: invisible
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all world_hud checks"; exit 0
fi
if [ "$stage" = worldgame ]; then  # `live.py watch K world-game` as a user meets it (watch_world.py check): follow camera, bars from the engine's own game rows, plume text, mini-map in one frame.
  # Mutants: the HUD reads a zero buffer instead of the engine rows, the mini-map launch dropped
  export PYTHONPATH=$PWD
  W="$M -m agent.body_engine.watch_world check"
  $W || { echo "FAIL worldgame"; fail=$((fail+1)); }
  ~/iTools/bin/mutant-check agent/body_engine/live.mojo 's|gs_src = o_b.create_sub_buffer\[DType.float32\](eng.OGAME \* eng.F, 4 \* eng.F)|gs_src = gs_b|' -- $W || fail=$((fail+1))
  ~/iTools/bin/mutant-check agent/body_engine/live.mojo 's|ctx.enqueue_function(kmm, img, tpose, troot, tsrc, grid_dim=(spec.MAP \* spec.MAP + 63) // 64, block_dim=64)|pass|' -- $W || fail=$((fail+1))
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all worldgame checks"; exit 0
fi
if [ "$stage" = flygait ]; then  # the FlyGym gait on the card (fly_gait.mojo, live.mojo -DBRAIN=3): the kernel vs HybridTurningController on a 400-step drive sequence, max abs err <= 1e-5.
  # Mutants: the coupling's phase difference reversed (sin(a - pi) = -sin(a + pi): the bias sign itself cannot be a mutant), the adhesion window, the drive sign
  export PYTHONPATH=$PWD
  G="$M -m agent.body_engine.fly_gait check"
  $G || { echo "FAIL flygait"; fail=$((fail+1)); }
  ~/iTools/bin/mutant-check agent/body_engine/fly_gait.mojo 's|sin64(theta\[j\] - theta\[i\] - 3.141592653589793)|sin64(theta[i] - theta[j] - 3.141592653589793)|' -- $G || fail=$((fail+1))
  ~/iTools/bin/mutant-check agent/body_engine/fly_gait.mojo 's|ph < swing_end)|ph < swing_end - 0.5)|' -- $G || fail=$((fail+1))
  ~/iTools/bin/mutant-check agent/body_engine/fly_gait.mojo 's|freq\[leg\] = FREQ if d >= 0 else -FREQ|freq[leg] = FREQ|' -- $G || fail=$((fail+1))
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all flygait checks"; exit 0
fi
if [ "$stage" = shiures ]; then  # the resident Shiu brain on a caller-owned DeviceContext (agent/body_engine/shiu_resident.mojo) vs ShiuLive.advance: counts and voltages bit-equal. Mutant: a tick counter that skips ticks
  export PYTHONPATH=$PWD
  R="$M -m agent.verify_shiu_resident"
  $R || { echo "FAIL shiures"; fail=$((fail+1)); }
  ~/iTools/bin/mutant-check agent/body_engine/shiu_resident.mojo 's|            self.t += 1|            self.t += 2|' -- $R || fail=$((fail+1))
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all shiures checks"; exit 0
fi
if [ "$stage" = flylink3 ]; then  # brain-driven fly, ONE DeviceContext (live.mojo -DBRAIN=3): fly_link's gates plus no host copy per step (rocprofv3 --memory-copy-trace). Submit with --vram 4.
  # Mutants: the brain step dropped, the silence flag ignored, left and right drive merged
  export PYTHONPATH=$PWD
  L="$M -m agent.body_engine.fly_link3 check"
  $L || { echo "FAIL flylink3"; fail=$((fail+1)); }
  ~/iTools/bin/mutant-check agent/body_engine/live.mojo 's|            fly3.step(ctx, TileTensor(u_b, eng.LU), fr, max(eng.K, 1))|            pass|' -- $L || fail=$((fail+1))
  ~/iTools/bin/mutant-check agent/body_engine/fly_brain.mojo 's|    if cfg\[2\] != 0 or not active:|    if not active:|' -- $L || fail=$((fail+1))
  ~/iTools/bin/mutant-check agent/body_engine/fly_brain.mojo 's|drive\[1\] = min(max(rr /|drive[1] = min(max(rl /|' -- $L || fail=$((fail+1))
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all flylink3 checks"; exit 0
fi
if [ "$stage" = hudclip ]; then  # one 300-frame clip with every HUD overlay on (follow camera, game-state bars from the engine rows, plume text, mini-map), checked frame by frame
  # (hud_clip.py), hooks off byte-equal to main's live.mojo. Mutants: the mini-map source x and y swapped (marker at the wrong place), a state bar one pixel short
  export PYTHONPATH=$PWD
  H="$M -m agent.body_engine.hud_clip"
  $H || { echo "FAIL hudclip"; fail=$((fail+1)); }
  ~/iTools/bin/mutant-check agent/body_engine/spectate.mojo 's|project_source(src\[5 \* k\], src\[5 \* k + 1\], tx, ty,|project_source(src[5 * k + 1], src[5 * k], tx, ty,|' -- $H || fail=$((fail+1))
  ~/iTools/bin/mutant-check agent/body_engine/spectate.mojo 's|Float32(x - BAR_X) + 0.5 < fill.width|Float32(x - BAR_X) + 1.5 < fill.width|' -- $H || fail=$((fail+1))
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all hudclip checks"; exit 0
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
  ~/iTools/bin/mojo-cached build $(cat $D/sizes.txt) -DFLIES=64 -DSTAGE=$($M -c "from agent.body_engine.model import STAGES; print(STAGES['$ST'][0])") -DK=10 -DWAVE=${WAVE:-0} agent/body_engine/smooth.mojo -o $D/k10
  cp $D/state10.bin $D/k1x10.bin && $D/k10 $D/ >/dev/null
  cmp $D/state.bin $D/k1x10.bin && echo "PASS loopk $(basename $B .xml) ${ENGINE_OPT:-defaults}: K=10 launch == 10 K=1 launches, byte-equal (engine vs engine, not MuJoCo)" || { echo "FAIL loopk $(basename $B .xml)"; exit 1; }
  exit 0
fi
if [ "$stage" = integrator ]; then  # implicitfast (mj_implicitSkip) vs MuJoCo with opt.integrator=implicitfast, fly + spider + 2-animal world
  F=.work/agent/gpu-body-port/body-v2/body.xml; SP=agent/body_engine/models/spider_test.xml
  SB=.work/agent/body_engine/spider-body/spider.xml; I=$X/integrator; KV=0.01  # KV: the shipped servos have kv 0 = Euler's qH; 0.01 makes h gear^2 kv ~ M's diagonal
  echo "COVERAGE integrator=implicitfast (3): fly step + actuated (airborne states, kv=$KV, shipped kv=0 in step), spider_test step, world fly+spider at 8 and 200 mm"
  echo "COVERAGE not covered: standalone free body (midpoint, refused at export), implicit (full), RK4, tendon-damping and fluid derivatives (refused), spring-only tendons (accepted, no velocity derivative, no parity case), contact states (MEASURED only)"
  # fluid derivatives (mjd_ellipsoidFluid, mjd_inertiaBoxFluid blocks of qDeriv) on the Flybody flight model, step fields under both integrators;
  # mutants: the ellipsoid viscous-drag diagonal sign, the inertia-box linear drag sign (both leave the forward forces alone)
  R=~/Projects/science/fly-connectome; FL=${FLIGHT:-$R/.work/flight-body}
  for integ in 2 3; do
    VSCALE=${VSCALE:-100} ENGINE_OPT=integrator=$integ,cone=0 $M -m agent.body_engine.model check $FL/flight.xml $I/fluid$integ step 64 || { echo "FAIL fluid derivative integrator=$integ"; fail=$((fail+1)); }
  done
  ~/iTools/bin/mutant-check agent/body_engine/smooth.mojo 's/    e0\[0\] -= lincoef; e1\[1\] -= lincoef; e2\[2\] -= lincoef/    e0[0] += lincoef; e1[1] += lincoef; e2[2] += lincoef/' -- \
    env VSCALE=${VSCALE:-100} ENGINE_OPT=integrator=2,cone=0 $M -m agent.body_engine.model check $FL/flight.xml $I/mutant step 64 || fail=$((fail+1))
  ~/iTools/bin/mutant-check agent/body_engine/smooth.mojo 's/            B\[(3 + k) \* 7\] = -3 \* pi \* diam \* mu/            B[(3 + k) * 7] = 3 * pi * diam * mu/' -- \
    env VSCALE=${VSCALE:-100} ENGINE_OPT=integrator=3,cone=0 $M -m agent.body_engine.model check $FL/flight.xml $I/mutant step 64 || fail=$((fail+1))
  # tendon damping under both implicit integrators (mjd_passive_vel's tendon term, the J' damping J block of qDeriv on D's / M's pattern): models/tendon_spatial.xml
  # (five spatial + one fixed tendon, every one damped) at random states; mutant: the block's sign
  T=agent/body_engine/models/tendon_spatial.xml
  for integ in 2 3; do
    ENGINE_OPT=integrator=$integ $M -m agent.body_engine.model check $T $I/tendon$integ sptendon 64 || { echo "FAIL tendon damping implicit integrator=$integ"; fail=$((fail+1)); }
  done
  ~/iTools/bin/mutant-check agent/body_engine/smooth.mojo 's/res\[(OQLU + e) \* F + f\] += h \* damp \* jk/res[(OQLU + e) * F + f] -= h * damp * jk/' -- \
    env ENGINE_OPT=integrator=2 $M -m agent.body_engine.model check $T $I/mutant sptendon 64 || fail=$((fail+1))
  ~/iTools/bin/mutant-check agent/body_engine/smooth.mojo 's/res\[(OQH + e) \* F + f\] += h \* damp \* jk/res[(OQH + e) * F + f] -= h * damp * jk/' -- \
    env ENGINE_OPT=integrator=3 $M -m agent.body_engine.model check $T $I/mutant sptendon 64 || fail=$((fail+1))
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
if [ "$stage" = implicit ]; then  # full implicit (mj_implicit: qDeriv with mjd_rne_vel, LU) and the implicitfast midpoint of standalone free bodies, vs MuJoCo
  F=.work/agent/gpu-body-port/body-v2/body.xml; SP=agent/body_engine/models/spider_test.xml; MP=agent/body_engine/models/midpoint.xml
  SB=.work/agent/body_engine/spider-body/spider.xml; I=$X/implicit; KV=0.01
  echo "COVERAGE integrator=implicit (2): fly step (shipped kv 0) + actuated (airborne states, kv=$KV), spider_test step (free, ball, hinge, slide), world fly+spider at 8 and 200 mm"
  echo "COVERAGE integrator=implicitfast (3) midpoint: four standalone free bodies (models/midpoint.xml: simple, aligned, coupled, coupled and rotated), contact-free; contact eligibility in the midpoint_contact part"
  echo "COVERAGE tendon damping derivative: models/tendon_spatial.xml step under integrator 2 and 3, sign mutants"
  echo "COVERAGE fluid derivatives: Flybody flight.xml step under integrator 2 and 3, drag-sign mutants"
  echo "COVERAGE not covered: contact states under full implicit (MEASURED only, active-set float32 noise as the implicitfast gate), WAVE=1"
  $M -m agent.body_engine.reject_check || { echo "FAIL reject"; fail=$((fail+1)); }
  ~/iTools/bin/mutant-check agent/body_engine/model.py 's/m.opt.density or m.opt.viscosity or m.tendon_damping.any()/False/' -- $M -m agent.body_engine.reject_check || fail=$((fail+1))
  export ENGINE_OPT=integrator=2
  $M -m agent.body_engine.model check $F $I/fly-step step 64 || { echo "FAIL fly step implicit"; fail=$((fail+1)); }
  $M -m agent.body_engine.model check $SP $I/spider step 64 || { echo "FAIL spider step implicit"; fail=$((fail+1)); }
  export ENGINE_OPT=integrator=2,kv=$KV
  A=$($M -m agent.body_engine.model airborne $F $I/fly | tail -1)
  STATES=$A $M -m agent.body_engine.model check $F $I/fly actuated 64 || { echo "FAIL fly actuated implicit"; fail=$((fail+1)); }
  # the gate must fail without the Coriolis derivative (mjd_rne_vel's term of qLU), with the damping term of qLU flipped, and without Dcacc inherited from the parent
  export ENGINE_OPT=integrator=2
  ~/iTools/bin/mutant-check agent/body_engine/smooth.mojo 's/+= h \* (fr \* cd).reduce_add()/+= 0 * h * (fr * cd).reduce_add()/' -- \
    $M -m agent.body_engine.model check $F $I/mutant step 64 || fail=$((fail+1))
  ~/iTools/bin/mutant-check agent/body_engine/smooth.mojo 's/+= h \* (fr \* cd).reduce_add()/+= 0 * h * (fr * cd).reduce_add()/' -- \
    $M -m agent.body_engine.model check $SP $I/mutant step 64 || fail=$((fail+1))
  ~/iTools/bin/mutant-check agent/body_engine/smooth.mojo 's/Int(mi\[DDG + v\])) \* F + f\] += h \* mf\[DAMP + v\]/Int(mi[DDG + v])) * F + f] -= h * mf[DAMP + v]/' -- \
    $M -m agent.body_engine.model check $SP $I/mutant step 64 || fail=$((fail+1))
  ~/iTools/bin/mutant-check agent/body_engine/smooth.mojo 's/res\[(ODCA + 6 \* badr + r) \* F + f\] = get(res, f, ODCA + 6 \* padr + r)/res[(ODCA + 6 * badr + r) * F + f] = 0/' -- \
    $M -m agent.body_engine.model check $SP $I/mutant step 64 || fail=$((fail+1))
  # islands: two animals in one model (own LU rows each), at 200 mm (two islands) and 8 mm
  export ENGINE_OPT=integrator=2,kv=$KV
  for off in 200 8; do
    W=$($M -c "from agent.body_engine.model import combine; print(combine('$F', '$SB', ($off, 0, 0), 'imworld$off'))")
    AW=$($M -m agent.body_engine.model airborne $W $I/world$off | tail -1)
    STATES=$AW $M -m agent.body_engine.model check $W $I/world$off actuated 64 || { echo "FAIL implicit world at $off mm"; fail=$((fail+1)); }
  done
  # midpoint (implicitfast, mj_midpoint)
  export ENGINE_OPT=integrator=3
  $M -m agent.body_engine.model check $MP $I/mid step 64 || { echo "FAIL midpoint step"; fail=$((fail+1)); }
  ~/iTools/bin/mutant-check agent/body_engine/smooth.mojo 's/if not mid_eligible(mi, res, f, b):/if True:/' -- \
    $M -m agent.body_engine.model check $MP $I/mutant step 64 || fail=$((fail+1))
  ~/iTools/bin/mutant-check agent/body_engine/smooth.mojo 's/var dw_body = rot(iquat, 2 \* delta)/var dw_body = rot(iquat, delta)/' -- \
    $M -m agent.body_engine.model check $MP $I/mutant step 64 || fail=$((fail+1))
  ~/iTools/bin/mutant-check agent/body_engine/smooth.mojo 's/var du = rot(iquat, 2 \* (x - cross(delta, r_com)))/var du = rot(iquat, 2 * x)/' -- \
    $M -m agent.body_engine.model check $MP $I/mutant step 64 || fail=$((fail+1))
  ~/iTools/bin/mutant-check agent/body_engine/smooth.mojo 's/force \/ mass + rot(rot_x2i, m3(mf, GRAV))/force \/ mass/' -- \
    $M -m agent.body_engine.model check $MP $I/mutant step 64 || fail=$((fail+1))
  # midpoint_contact: p and q touch in about half the states (live contact: ordinary implicitfast step), r never does (always midpoint)
  MC=agent/body_engine/models/midpoint_contact.xml
  $M -m agent.body_engine.midpoint_states $I/mc-states.npz 64
  STATES=$I/mc-states.npz $M -m agent.body_engine.model check $MC $I/mc contact 64 || { echo "FAIL midpoint contact"; fail=$((fail+1)); }
  ~/iTools/bin/mutant-check agent/body_engine/smooth.mojo 's/if not mid_eligible(mi, res, f, b):/if True:/' -- \
    env STATES=$I/mc-states.npz $M -m agent.body_engine.model check $MC $I/mutant contact 64 || fail=$((fail+1))
  ~/iTools/bin/mutant-check agent/body_engine/smooth.mojo 's/or Int(mi\[GBODY + Int(mi\[CPH + 11 \* p + 1\])\]) == b:/or False:/' -- \
    env STATES=$I/mc-states.npz $M -m agent.body_engine.model check $MC $I/mutant contact 64 || fail=$((fail+1))
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all implicit checks"; exit 0
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
if [ "$stage" = rk4 ]; then  # RK4 (mj_RungeKutta) vs MuJoCo with opt.integrator=RK4: fly, spider, fly + spider world
  F=.work/agent/gpu-body-port/body-v2/body.xml; SP=agent/body_engine/models/spider_test.xml
  SB=.work/agent/body_engine/spider-body/spider.xml; I=$X/rk4
  echo "COVERAGE integrator=RK4 (1): fly step (contact-free, dt 1e-6: h B / M 0.059, a wrong weight still moves qacc_euler by far more than 1e-5; dt 1e-5 leaves cdof_dot at 1.4e-5 float32 noise), spider_test step; fly actuated on standing states (contacts + servos + adhesion, dt 1e-7); world fly+spider at 8 and 200 mm (dt 1e-7)"
  echo "COVERAGE not covered: contact and servo parity at dt >= 1e-6 (stiff contacts: MuJoCo's own ncon flips between its 4 forward passes and float32 follows chaotically: dt 1e-6 qfrc_constraint 4.9e-5 vs 3.9e-5 bar, dt 1e-5 ncon off), RK4 on the shipped dt 1e-4 (explicit dof damping is unstable in MuJoCo itself: h B / M 5.9), actuator velocity terms, qH (stale in MuJoCo after RK4, not compared)"
  ENGINE_OPT=integrator=1,dt=1e-6 $M -m agent.body_engine.model check $F $I/fly-step step 64 || { echo "FAIL fly step rk4"; fail=$((fail+1)); }
  ENGINE_OPT=integrator=1,dt=1e-6 $M -m agent.body_engine.model check $SP $I/spider step 64 || { echo "FAIL spider step rk4"; fail=$((fail+1)); }
  # the gate must fail with the RK4 weights wrong (F1 weight 2 -> 1)
  ~/iTools/bin/mutant-check agent/body_engine/smooth.mojo 's/(get(res, f, ORKF + v) + 2 \* get(res, f, ORKF + NV + v)/(get(res, f, ORKF + v) + 1 * get(res, f, ORKF + NV + v)/' -- \
    env ENGINE_OPT=integrator=1,dt=1e-6 $M -m agent.body_engine.model check $F $I/mutant step 64 || fail=$((fail+1))
  export ENGINE_OPT=integrator=1,dt=1e-7
  A=$(env -u ENGINE_OPT $M -m agent.body_engine.model airborne $F $I/fly 0 | tail -1)  # standing states from an Euler rollout (RK4 cannot roll out the shipped model)
  STATES=$A $M -m agent.body_engine.model check $F $I/fly actuated 64 || { echo "FAIL fly actuated rk4"; fail=$((fail+1)); }
  # the gate must fail with the stage-1 velocity step dropped (X_1 qvel = v0 + h A F0 -> v0)
  ~/iTools/bin/mutant-check agent/body_engine/smooth.mojo 's/qvel\[v \* F + f\] = get(res, f, ORKV + v) + h \* c \* get(res, f, ORKF + s \* NV + v)/qvel[v * F + f] = get(res, f, ORKV + v)/' -- \
    env ENGINE_OPT=integrator=1,dt=1e-6 $M -m agent.body_engine.model check $F $I/mutant step 64 || fail=$((fail+1))
  # islands: two animals in one model, at 200 mm (two islands) and 8 mm; the gate must fail when the spider uses the fly's origin
  for off in 200 8; do
    W=$($M -c "from agent.body_engine.model import combine; print(combine('$F', '$SB', ($off, 0, 0), 'rworld$off'))")
    AW=$(env -u ENGINE_OPT $M -m agent.body_engine.model airborne $W $I/world$off 0 | tail -1)
    STATES=$AW $M -m agent.body_engine.model check $W $I/world$off actuated 64 || { echo "FAIL rk4 world at $off mm"; fail=$((fail+1)); }
    if [ $off = 200 ]; then
      ~/iTools/bin/mutant-check agent/body_engine/smooth.mojo 's/    var r = Int(mi\[ROOT + b\])/    var r = Int(mi[ROOT + 1])/' -- \
        env STATES=$AW $M -m agent.body_engine.model check $W $I/mutant actuated 64 || fail=$((fail+1))
    fi
  done
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all rk4 checks"; exit 0
fi
if [ "$stage" = cone ]; then  # elliptic friction cone on the walking fly: Newton (noslip 0) and Newton + noslip 5 (QCQP over the friction ellipse), MuJoCo run to convergence
  for ns in 0 5; do
    E=$($M -c "from agent.body_engine.model import elliptic; print(elliptic('.work/agent/gpu-body-port/body-v2/body.xml', $ns))")
    $M -m agent.body_engine.model check $E $X/elliptic$ns actuated 64 || { echo "FAIL elliptic cone noslip $ns"; fail=$((fail+1)); }
  done
  E0=$($M -c "from agent.body_engine.model import elliptic; print(elliptic('.work/agent/gpu-body-port/body-v2/body.xml', 0))")
  E5=$($M -c "from agent.body_engine.model import elliptic; print(elliptic('.work/agent/gpu-body-port/body-v2/body.xml', 5))")
  # Newton mutants (noslip 0): cone force tangent sign, Dm without (1 + mu^2), line-search slope sign, top-zone bound doubled
  for mut in 's/force = V3(f0, -f0 \/ T \* U\[1\] \* fr\[1\], -f0 \/ T \* U\[2\] \* fr\[2\], 0)/force = V3(f0, f0 \/ T * U[1] * fr[1], f0 \/ T * U[2] * fr[2], 0)/' \
             's/(mu \* mu \* (1 + mu \* mu))/(mu * mu)/' 's/e0 += Dm \* (N - mu \* T) \* (N1 - mu \* T1)/e0 -= Dm * (N - mu * T) * (N1 - mu * T1)/' \
             's/if N >= mu \* T or (T <= 0 and N >= 0):/if N >= 2 * mu * T or (T <= 0 and N >= 0):/'; do
    ~/iTools/bin/mutant-check agent/body_engine/smooth.mojo "$mut" -- $M -m agent.body_engine.model check $E0 $X/mutant actuated 64 || fail=$((fail+1))
  done
  # noslip mutants (noslip 5): QCQP residual cross term sign (tangent axes share mu, so an axis swap is an equivalent mutant), dual-finish sign, friction update dropped
  for mut in 's/res0 - (a00 \* o1 + a01 \* o2)/res0 - (a00 * o1 - a01 * o2)/' 's/var df = get(res, f, OFRN + row) - get(res, f, OFRC + row)/var df = get(res, f, OFRN + row) + get(res, f, OFRC + row)/' \
             's/res\[(OFRN + r + 1) \* F + f\] = n\[0\]/res[(OFRN + r + 1) * F + f] = o1/'; do
    ~/iTools/bin/mutant-check agent/body_engine/smooth.mojo "$mut" -- $M -m agent.body_engine.model check $E5 $X/mutant actuated 64 || fail=$((fail+1))
  done
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all cone checks"; exit 0
fi
if [ "$stage" = cone_convex ]; then  # elliptic friction cone on convex pairs: the fly in the arena (box walls, ellipsoid stones, CONVEX=1), Newton noslip 0 and 5,
  # MuJoCo run to convergence (model.elliptic), one-step fields on 64 states moved into walls and stones. The contact tail (emit_contact) is shared by every pair kind,
  # so the cone port is the same code; this gate is what lets export accept the pair. Mutant: the cone force tangent sign (the cone gate's first mutant) on convex states.
  for ns in 0 5; do
    E=$($M -c "from agent.body_engine.model import elliptic; print(elliptic('.work/agent/gpu-body-port/body-v2/body.xml', $ns))")
    env CONVEX=1 $M -m agent.body_engine.model check $E $X/cone-convex$ns actuated 64 || { echo "FAIL cone_convex noslip $ns"; fail=$((fail+1)); }
  done
  E0=$($M -c "from agent.body_engine.model import elliptic; print(elliptic('.work/agent/gpu-body-port/body-v2/body.xml', 0))")
  ~/iTools/bin/mutant-check agent/body_engine/smooth.mojo 's/force = V3(f0, -f0 \/ T \* U\[1\] \* fr\[1\], -f0 \/ T \* U\[2\] \* fr\[2\], 0)/force = V3(f0, f0 \/ T * U[1] * fr[1], f0 \/ T * U[2] * fr[2], 0)/' -- \
    env CONVEX=1 $M -m agent.body_engine.model check $E0 $X/mutant actuated 32 || fail=$((fail+1))
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all cone_convex checks"; exit 0
fi
if [ "$stage" = cone_hfield ]; then  # elliptic friction cone on exact-heightfield pairs (HFEXACT=1): the terrain_exact terrain for the fly and the spider, Newton noslip 0 and 5,
  # MuJoCo run to convergence (model.elliptic on the terrain xml), one-step fields on 64 states moved onto the terrain. Same contact tail as cone_convex. Bar REL=5e-4 (terrain_exact uses 4e-4, float32 terrain function vs MuJoCo table): fly noslip 0 qacc_euler 4.55e-4 on three Newton active-set switch states (59, 13, 30), median 5.9e-6, deterministic over two runs (DIAG=1); the other three runs within 1.2e-4.
  # Mutant: the cone force tangent sign (cone_convex's mutant) on the fly's exact-heightfield states.
  for B in .work/agent/gpu-body-port/body-v2/body.xml .work/agent/body_engine/spider-body/spider.xml; do
    N=$($M -c "from agent.body_engine.model import terrain; print(terrain('$B', (0, 0, 0.3, 8.0, 4, 1, 2.0, 0.5))[0])")
    for ns in 0 5; do
      E=$($M -c "from agent.body_engine.model import elliptic; print(elliptic('$N', $ns))")
      env HFEXACT=1 REL=5e-4 $M -m agent.body_engine.model check $E $X/cone-hfield-$(basename $B .xml)-$ns actuated 64 || { echo "FAIL cone_hfield $(basename $B .xml) noslip $ns"; fail=$((fail+1)); }
    done
  done
  N=$($M -c "from agent.body_engine.model import terrain; print(terrain('.work/agent/gpu-body-port/body-v2/body.xml', (0, 0, 0.3, 8.0, 4, 1, 2.0, 0.5))[0])")
  E0=$($M -c "from agent.body_engine.model import elliptic; print(elliptic('$N', 0))")
  ~/iTools/bin/mutant-check agent/body_engine/smooth.mojo 's/force = V3(f0, -f0 \/ T \* U\[1\] \* fr\[1\], -f0 \/ T \* U\[2\] \* fr\[2\], 0)/force = V3(f0, f0 \/ T * U[1] * fr[1], f0 \/ T * U[2] * fr[2], 0)/' -- \
    env HFEXACT=1 REL=5e-4 $M -m agent.body_engine.model check $E0 $X/mutant actuated 32 || fail=$((fail+1))
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all cone_hfield checks"; exit 0
fi
if [ "$stage" = spidereye ]; then  # native vision (docs/plans/SPIDER_BRAIN.md unit 2): rendered prey vs an independent analytic pinhole,
  # own-body occlusion vs MuJoCo ray casting, per-animal separation, environment-moved prey. Mutants: eye axis yaw flipped in the kernel,
  # prey path sign flipped, eye table yaw flipped (caught against the gate's own hand-written INTENDED geometry)
  L="$M -m agent.body_engine.spider_eyes check"
  $L || { echo "FAIL spidereye"; fail=$((fail+1)); }
  ~/iTools/bin/mutant-check agent/body_engine/eyes.mojo 's/var tgt = eye + rot(bq, ax)/var tgt = eye + rot(bq, V3(ax[0], -ax[1], ax[2], 0))/' -- $L || fail=$((fail+1))
  ~/iTools/bin/mutant-check agent/body_engine/eyes.mojo 's/= env\[k \* F + f\] + v0 \* dt/= -env[k * F + f] + v0 * dt/' -- $L || fail=$((fail+1))
  ~/iTools/bin/mutant-check agent/body_engine/spider_eyes.py 's/t\[i, 3:6\] = (np.cos(np.radians(yaw)), np.sin/t[i, 3:6] = (np.cos(np.radians(yaw)), -np.sin/' -- $L || fail=$((fail+1))
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all spidereye checks"; exit 0
fi
if [ "$stage" = jump ]; then  # spider jump (docs/plans/SPIDER_BRAIN.md unit 3): engine vs MuJoCo on the same ctrl stream, published take-off ranges,
  # ballistic flight and distance. Mutants: burst scaled to 0, weak servos (JUMP_S 1), ctrl write skipped
  L="$M -m agent.body_engine.spider_jump check"
  $L || { echo "FAIL jump"; fail=$((fail+1)); }
  ~/iTools/bin/mutant-check agent/body_engine/spider_jump.py 's/^BURST_SCALE = 1.0/BURST_SCALE = 0.0/' -- $L || fail=$((fail+1))
  ~/iTools/bin/mutant-check agent/body_engine/models/spider.py 's/^JUMP_S = 95.6/JUMP_S = 1.0/' -- $L || fail=$((fail+1))
  ~/iTools/bin/mutant-check agent/body_engine/live.mojo 's/ctx.enqueue_copy(u_b, ctrl_h)/pass/' -- $L || fail=$((fail+1))
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all jump checks"; exit 0
fi
if [ "$stage" = solver ]; then  # CG and PGS (mj_solCG, mj_solPGS) vs MuJoCo with opt.solver set, next to Newton (the default)
  F=.work/agent/gpu-body-port/body-v2/body.xml; SB=.work/agent/body_engine/spider-body/spider.xml; I=$X/solver
  echo "COVERAGE solver=CG (1) and PGS (0), integrator Euler, pyramidal cone, noslip 5 as shipped: fly actuated on standing states (feet, grooming pairs, servos, adhesion), fly + spider world at 200 and 8 mm, two-fly world at 200 mm (two kinematic trees = two MuJoCo islands: PGS per island with its own shuffle)"
  echo "COVERAGE not covered: equality and friction-loss rows (PGS/CG export refuses them), elliptic cones, solver x non-Euler integrator, contact-free step (the solver is not in it), CG at float32 stop tests (CG runs its iteration cap: MuJoCo's tests fire early in float32, see smooth.mojo), K>0 chains vs MuJoCo (loopk below is engine vs engine)"
  A=$(env -u ENGINE_OPT $M -m agent.body_engine.model airborne $F $I/fly 0 | tail -1)  # standing states from the shipped Euler/Newton rollout
  for SV in 1 0; do
    export ENGINE_OPT=solver=$SV
    STATES=$A $M -m agent.body_engine.model check $F $I/fly$SV actuated 64 || { echo "FAIL fly actuated solver=$SV"; fail=$((fail+1)); }
    for off in 200 8; do
      W=$($M -c "from agent.body_engine.model import combine; print(combine('$F', '$SB', ($off, 0, 0), 'sworld$off'))")
      AW=$(env -u ENGINE_OPT $M -m agent.body_engine.model airborne $W $I/world$off 0 | tail -1)
      STATES=$AW $M -m agent.body_engine.model check $W $I/world$SV$off actuated 64 || { echo "FAIL solver=$SV world at $off mm"; fail=$((fail+1)); }
    done
  done
  # islands: two flies 200 mm apart = two kinematic trees = two MuJoCo islands (own shuffle, own stop). The merged two-fly world differs from its engine
  # twin by 7e-5 in MuJoCo itself (Newton too: WORLD.md, open), so this test alone runs at REL=2e-4; PGS over both flies' rows as one block (the island loop run over all rows for every root) must fail here
  FW=$($M -c "from agent.body_engine.model import combine; print(combine('$F', '$F', (200, 0, 0), 'ffworld'))")
  AF=$(env -u ENGINE_OPT $M -m agent.body_engine.model airborne $FW $I/ffworld 0 | tail -1)
  for SV in 1 0; do
    REL=2e-4 ENGINE_OPT=solver=$SV STATES=$AF $M -m agent.body_engine.model check $FW $I/ff$SV actuated 64 || { echo "FAIL solver=$SV two-fly world"; fail=$((fail+1)); }
  done
  ~/iTools/bin/mutant-check agent/body_engine/smooth.mojo 's/            if Int(mi\[ROOT + Int(mi\[CPH + 11 \* rp + 2\])\]) == troot:/            if True:/' -- \
    env REL=2e-4 ENGINE_OPT=solver=0 STATES=$AF $M -m agent.body_engine.model check $FW $I/mutant actuated 64 || fail=$((fail+1))
  # mutants: PGS projection (force clamped at 0 -> min), CG Polak-Ribiere beta sign
  ~/iTools/bin/mutant-check agent/body_engine/smooth.mojo 's/var new = max(old - rs \* get(res, f, OGA + r), Float32(0))/var new = min(old - rs * get(res, f, OGA + r), Float32(0))/' -- \
    env ENGINE_OPT=solver=0 STATES=$A $M -m agent.body_engine.model check $F $I/mutant actuated 64 || fail=$((fail+1))
  ~/iTools/bin/mutant-check agent/body_engine/smooth.mojo 's/var beta = max(num \/ max(den, Float32(1e-15)), Float32(0))/var beta = max(-num \/ max(den, Float32(1e-15)), Float32(0))/' -- \
    env ENGINE_OPT=solver=1 STATES=$A $M -m agent.body_engine.model check $F $I/mutant actuated 64 || fail=$((fail+1))
  for SV in 1 0; do ENGINE_OPT=solver=$SV $0 loopk || fail=$((fail+1)); done  # K=10 vs 10 x K=1, engine vs engine, per solver
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all solver checks (CG, PGS)"; exit 0
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
