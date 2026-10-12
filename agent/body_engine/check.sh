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
KS=$($M -c "import kinestride, os; print(os.path.dirname(kinestride.__file__))")  # the physics core: ~/Projects/mojo/kinestride
case " flygait flylink flylink3 game hudclip jump shiures spectate spidereye spiderlink stone wall world world_hud worldgame " in *" $stage "*) ;; *) echo "FAIL check.sh: $stage is a physics stage: run it in Kinestride ($KS/check.sh $stage)"; exit 1;; esac
if [ "$stage" = stone ]; then  # a stone in the walking fly's path through the live transport (fly_link): engine vs MuJoCo path effect, convex-off mutant
  $M -m agent.body_engine.stone_check || { echo "FAIL stone: log above"; exit 1; }
  exit 0
fi
if [ "$stage" = wall ]; then  # a wall square across the walking fly's path (box-vs-mesh pairs), same gates and mutant as stone
  $M -m agent.body_engine.stone_check wall || { echo "FAIL wall: log above"; exit 1; }
  exit 0
fi
if [ "$stage" = world ]; then  # fly + spider in one model (model.combine): engine vs MuJoCo on it, and MuJoCo merged vs
  # each animal alone (constraint islands: the merged solve must equal the separate ones)
  # spider 8 mm and 200 mm from the fly: each animal computes relative to its own origin (smooth.mojo origin())
  for off in 200 8; do
    W=$($M -c "from kinestride.model import combine; print(combine('.work/agent/gpu-body-port/body-v2/body.xml', '.work/agent/body_engine/spider-body/spider.xml', ($off, 0, 0), 'world$off'))")
    $M -m kinestride.model check $W $X/world$off actuated 64 || { echo "FAIL world spider at $off mm"; fail=$((fail+1)); }
    if [ $off = 200 ]; then  # the gate must fail when the spider uses the fly's origin (one origin per model)
      ~/iTools/bin/mutant-check $KS/smooth.mojo 's/    var r = Int(mi\[ROOT + b\])/    var r = Int(mi[ROOT + 1])/' -- \
        $M -m kinestride.model check $W $X/mutant actuated 64 || fail=$((fail+1))
    fi
  done
  $M -c "
import numpy as np, mujoco
from kinestride.model import load, standing
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
if [ "$stage" = spiderlink ]; then  # brain<->body link on the spider (docs/plans/SPIDER_BRAIN.md unit 1): kernel gait vs oracle, engine vs MuJoCo
  # under the brain's ctrl stream, closed-loop walk/turn/stand. Mutants: the ctrl write skipped; the gait kernel's yaw sign flipped
  L="$M -m agent.body_engine.spider_link check"
  $L || { echo "FAIL spiderlink"; fail=$((fail+1)); }
  # the ctrl write skipped on both paths: the device copy (GPU step) and the host list (CPU step; spider_link runs on the
  # host up to 256 animals since d38a9ad, where the device copy alone is never read and its mutant survived)
  ~/iTools/bin/mutant-check agent/body_engine/live.mojo 's/ctx.enqueue_copy(u_b, ctrl_h)/pass/;s/uv.unsafe_ptr().unsafe_bitcast\[Float32\]()\[k\] = ctrl_h\[k\]/pass/' -- $L || fail=$((fail+1))
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
  ~/iTools/bin/mutant-check $KS/smooth.mojo 's|work += abs(get(res, f, OQACT + v) \* qvel\[v \* F + f\]) \* mf\[TSTEP\]|work += 0|' -- $G fly || fail=$((fail+1))
  ~/iTools/bin/mutant-check $KS/smooth.mojo 's|if get(res, f, OGAME + 3) != 0:|if get(res, f, OGAME + 3) > 5:|' -- $G fly || fail=$((fail+1))
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
if [ "$stage" = flylink3 ]; then  # brain-driven fly, ONE DeviceContext (live.mojo -DBRAIN=3): fly_link's gates on host physics, then no host copy per step on the all-GPU build (rocprofv3 --memory-copy-trace). Prebuild on the CPU: python -m agent.body_engine.fly_link3 build. Submit with --vram 4.
  # Mutants (each stops at the behaviour gate it breaks): the brain step dropped, the silence flag ignored, left and right drive merged
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
if [ "$stage" = spidereye ]; then  # native vision (docs/plans/SPIDER_BRAIN.md unit 2): rendered prey vs an independent analytic pinhole,
  # own-body occlusion vs MuJoCo ray casting, per-animal separation, environment-moved prey. Mutants: eye axis yaw flipped in the kernel,
  # prey path sign flipped, eye table yaw flipped (caught against the gate's own hand-written INTENDED geometry)
  L="$M -m agent.body_engine.spider_eyes check"
  $L || { echo "FAIL spidereye"; fail=$((fail+1)); }
  ~/iTools/bin/mutant-check $KS/eyes.mojo 's/var tgt = eye + rot(bq, ax)/var tgt = eye + rot(bq, V3(ax[0], -ax[1], ax[2], 0))/' -- $L || fail=$((fail+1))
  ~/iTools/bin/mutant-check $KS/eyes.mojo 's/= env\[k \* F + f\] + v0 \* dt/= -env[k * F + f] + v0 * dt/' -- $L || fail=$((fail+1))
  ~/iTools/bin/mutant-check agent/body_engine/spider_eyes.py 's/t\[i, 3:6\] = (np.cos(np.radians(yaw)), np.sin/t[i, 3:6] = (np.cos(np.radians(yaw)), -np.sin/' -- $L || fail=$((fail+1))
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all spidereye checks"; exit 0
fi
if [ "$stage" = jump ]; then  # spider jump (docs/plans/SPIDER_BRAIN.md unit 3): engine vs MuJoCo on the same ctrl stream, published take-off ranges,
  # ballistic flight and distance. Mutants: burst scaled to 0, weak servos (JUMP_S 1), ctrl write skipped
  L="$M -m agent.body_engine.spider_jump check"
  $L || { echo "FAIL jump"; fail=$((fail+1)); }
  ~/iTools/bin/mutant-check agent/body_engine/spider_jump.py 's/^BURST_SCALE = 1.0/BURST_SCALE = 0.0/' -- $L || fail=$((fail+1))
  ~/iTools/bin/mutant-check $KS/models/spider.py 's/^JUMP_S = 95.6/JUMP_S = 1.0/' -- $L || fail=$((fail+1))
  ~/iTools/bin/mutant-check agent/body_engine/live.mojo 's/ctx.enqueue_copy(u_b, ctrl_h)/pass/' -- $L || fail=$((fail+1))
  [ $fail -eq 0 ] || { echo "FAIL $fail checks"; exit 1; }
  echo "PASS all jump checks"; exit 0
fi
echo "FAIL check.sh: stage $stage fell through"; exit 1
