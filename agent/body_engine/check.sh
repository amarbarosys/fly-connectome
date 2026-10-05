#!/usr/bin/env bash
# Every ported stage against MuJoCo on two bodies (fly: free + hinge; 8-leg spider test model:
# free + ball + hinge + slide), plus a mutation that must fail (hinge angle sign flipped).
# usage: gpu-wait run --timeout 1200 -- [env BODY=path.xml] agent/body_engine/check.sh [STAGE]
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p .work; exec 9>.work/check.lock  # one stage per worktree: mutant-check edits sources in place (SHARED rule 13)
flock -n 9 || { echo "FAIL check.sh: another stage is running in $(pwd); run stages one at a time"; exit 1; }
stage=${1:-kinematics}; M=.venv/bin/python; X=.work/agent/body_engine; fail=0
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
