"""Brain-driven fly in the live engine, host transport (docs/plans/BRAIN_BODY_LOOP.md step 4, unit 5a): live.mojo -DBRAIN=1 on body-v2.

  python -m agent.body_engine.fly_link build
  gpu-wait run --shared --vram 4 --priority 60 --timeout 3000 -- .venv/bin/python -m agent.body_engine.fly_link check

Per control step (K physics steps, 0.1 ms each): the resident Shiu brain (agent/shiu.py ShiuLive.advance, state kept on the card across
steps) advances K ticks under a looming-pathway drive (LC4 + LPLC2, 150 Hz Poisson, per side); the descending neurons' spikes of the last
50 ms, summed per side, are normalised by their rate under the symmetric drive (one constant per side, measured once from the
symmetric run's own DN rates, so equal input gives equal full drive) into two drives in [0, 1]; flyloop's gait mapping (FlyGym
HybridTurningController, 42 leg servos by joint name, 6 stance adhesion values, the two head servos held) turns them into the 50 ctrl
values; they go over stdin to live.mojo, which launches K steps and returns a sensor packet (ctrl, qpos, qvel, contacts) every step.
Brain and body do not feed back in this unit (no sensors yet): the same resident brain serves every condition as separate columns.

Conditions (each its own engine process, stepped concurrently): sym (both eyes 150 Hz), left (left eye 150 Hz, right 40 Hz), right (mirror),
silenced (sym brain, DN counts zeroed before the readout), held (sym brain; the ctrl frozen at t = 0.5 s, the brain keeps running).
Gates (1 s of simulated time):
  walk     sym: the thorax moves > 1 body length along its start heading, max tilt < 20 deg
  turn     left and right: yaw change of opposite sign, each beyond 10 deg
  silenced the silenced-DN fly moves < 25 % of sym; held: its last 0.5 s moves < 25 % of sym's last 0.5 s
Reported: start height (live.prepare's standing root z vs flyloop's FlyGym 1.8 and body-v2 2.1), cost per frame split physics / neural / gait /
render+copy (engine side, PROFILE build, frames to /dev/null).
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

from agent.body_engine import live, spectate_check as sc
from agent.body_engine.render_check import MJCF

OUT = Path('.work/agent/body_engine/fly-link')
K, STEPS, DT = 10, 1000, 1e-4
HZ, WEAK, WINDOW = 150.0, 40.0, 50  # looming drive per eye, the weak eye, DN rate window in control steps
LOOM = ('LC4', 'LPLC2')
BODY_LENGTH = 4.077  # model units (mm), flyloop's measurement on body-v2 (tools/fly_gait_oracle.py --rollout)


def prepare(out=None, mjcf=MJCF, convex=True, flat=None, world=None):
    """live.prepare's standing fly, then settled into FlyGym's stance: the standing pose (servos at the standing angles, thorax pitched 66 deg) falls
    over when the controller's first targets (up to 2.4 rad away) arrive in one step, in MuJoCo as well (flyloop's `--rollout` preflight FAILs the
    same way: tilt 170 deg). So the servos are ramped from the standing angles to the controller's drive-0 pose over 0.3 s and held 0.3 s (CPU,
    MuJoCo, engine's noslip 0); that state and ctrl are the engine's start."""
    d = out or OUT / 'fly'
    cam, m = live.prepare(d, mjcf=mjcf, convex=convex, flat=flat, world=world)  # walls and stones collide too (check.sh convexstep)
    np.array([1, 2.0], np.float32).tofile(d / 'roots.bin')
    if not (d / 'settled').exists():
        q = np.fromfile(d / 'qpos.bin', np.float32).astype(float)
        v = np.fromfile(d / 'qvel.bin', np.float32).astype(float)
        np.r_[q, v].astype(np.float32).tofile(d / 'standing.bin')
        g = Gait(m, d, K)
        stand, dflt = g.held.astype(float), g.ctrl((0.0, 0.0)).astype(float)
        sm = m
        if flat:  # MuJoCo's rollout on the sampled heightfield jams the feet (live.prepare): settle on the flat floor, then lift
            sm = live.load(flat)
            q, v = (x[0] for x in live.standing(live.load(flat), 2))  # live.prepare's flat stance, before its lift
            sm.opt.noslip_iterations = 0
        st = mujoco.MjData(sm)
        st.qpos[:], st.qvel[:] = q, v
        ramp = int(0.3 / (K * DT))
        for i in range(ramp + int(0.3 / (K * DT))):
            st.ctrl[:] = stand + min(i / ramp, 1.0) * (dflt - stand)
            for _ in range(K):
                mujoco.mj_step(sm, st)
        if flat:
            st.qvel[:] = 0  # dropped from rest onto the bumps
            live.lift(m, st.qpos)
        st.qpos.astype(np.float32).tofile(d / 'settled-qpos.bin')
        st.qvel.astype(np.float32).tofile(d / 'settled-qvel.bin')
        dflt.astype(np.float32).tofile(d / 'settled-ctrl.bin')
        (d / 'settled').write_text('standing -> FlyGym drive-0 pose, ramp 0.3 s + hold 0.3 s\n')
    for n in ('qpos', 'qvel', 'ctrl'):  # live.prepare rewrites the files each call: the settled state is the engine's start
        (d / f'{n}.bin').write_bytes((d / f'settled-{n}.bin').read_bytes())
    return d, m, cam


class Gait:
    """flyloop's mapping (tools/fly_gait_oracle.py rollout): HybridTurningController -> 42 leg position servos + 6 stance adhesion values."""
    def __init__(self, m, d, k, slew_s=0.2, trim=0.02):
        """slew_s: the drive moves at most 1 per slew_s (a step from 0 to 1 at gait onset turned the fly -38 deg in 0.5 s; ramped
        +0.7). trim: drive x (1 - trim, 1 + trim) on (left, right): equal drives pulled right about 7 deg/s in MuJoCo and in the
        engine alike (body + controller, tools/gait_drift.py); 0.02 holds 2 s within 3 deg."""
        self.rate, self.gain, self.sig = k * DT / slew_s if slew_s else np.inf, np.array([1 - trim, 1 + trim]), np.zeros(2)
        from flygym_demo.complex_terrain import HybridTurningController, PreprogrammedSteps
        from flygym_demo.complex_terrain.common import get_default_locomotion_dof_order
        from flygym_demo.complex_terrain.hybrid_controller import HybridControllerObservation
        from tools.fly_gait_oracle import LEGS, joint_name
        dofs = (('coxa', 'yaw'), ('coxa', 'pitch'), ('coxa', 'roll'), ('trochanterfemur', 'pitch'), ('trochanterfemur', 'roll'), ('tibia', 'pitch'), ('tarsus1', 'pitch'))
        self.order = get_default_locomotion_dof_order()
        names = [joint_name(leg, seg, axis) for leg in LEGS for seg, axis in dofs]
        act = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        self.act, self.adh = [act(n + '-position') for n in names], [act(f'fly/{leg}_tarsus5-adhesion') for leg in LEGS]
        if min(self.act + self.adh) < 0:
            raise SystemExit('FAIL body-v2 lacks a gait or adhesion actuator')
        self.held = np.fromfile(d / 'ctrl.bin', np.float32).astype(np.float64)  # the standing ctrl: head servos stay, adhesion on
        self.steps = PreprogrammedSteps()
        self.obs = HybridControllerObservation(thorax_z=0.0, tarsus5_z=np.zeros(6), stumbling_contact_forces=np.zeros((6, 3, 3)), fly_heading=np.array([1.0, 0.0, 0.0]))
        self.controller = HybridTurningController(timestep=k * DT, preprogrammed_steps=self.steps, output_dof_order=self.order, enable_adhesion=True)
        self.controller.reset(seed=4, init_phases=np.array([0, np.pi, 0, np.pi, 0, np.pi]), init_magnitudes=np.zeros(6))

    def ctrl(self, signal):
        self.sig += np.clip(np.asarray(signal, float) - self.sig, -self.rate, self.rate)
        a = self.controller.step(self.sig * self.gain, self.obs)
        c = self.held.copy()
        c[self.act], c[self.adh] = a.joint_angles, a.adhesion_onoff.astype(float)
        return c.astype(np.float32)


class Engine:
    """live.mojo -DBRAIN=1 for F flies in one process: ctrl (F, NU) in over stdin, one sensor packet per step out ((NU+nq+nv+ns) x F,
    row-major by field); no frame drawn (brain=1 loops read no pixels). F > 1 runs in its own dir with the state tiled to F columns.
    flags: extra build flags (-DGAME=1 adds the 4 game rows, kept in self.game (F, 4)); frames go to `out` every `render`-th step."""
    def __init__(self, d, k, tag, frames, F=1, flags=(), out='/dev/null', render=0, q0=None):
        sizes = (d / 'sizes.txt').read_text()
        self.nu, self.nq, self.nv, self.ns = (int(re.search(rf'-D{n}=(\d+)', sizes).group(1)) for n in ('NU', 'NQ', 'NV', 'NSLOT'))
        if F > 1:
            src, d = d, d.parent / f'{d.name}-f{F}'
            d.mkdir(exist_ok=True)
            for f in src.glob('*'):
                if f.is_file() and f.suffix in ('.bin', '.txt'):
                    a = f.read_bytes()
                    if f.name in ('qpos.bin', 'qvel.bin', 'ctrl.bin'):
                        a = np.tile(np.frombuffer(a, np.float32).reshape(-1, 1), (1, F)).astype(np.float32).tobytes()
                    (d / f.name).write_bytes(a)
            if q0 is not None:  # (F, nq) start poses, one per animal (F > 1 only: its own dir)
                np.asarray(q0, np.float32).T.tofile(d / 'qpos.bin')
        b = sc.build(d, 'fly', list(flags), k, brain=1, flies=F)
        self.p = subprocess.Popen(['bash', '-c', 'exec "$0" "$@" 4>"$OUT"', str(b), str(d) + '/', str(frames), *sc.CAM, str(render), '1'],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env={**os.environ, 'OUT': str(out)})
        self.ng = 4 if '-DGAME=1' in flags else 0
        self.F, self.size = F, 4 * (self.nu + self.nq + self.nv + self.ns + self.ng) * F
        self.tag = tag

    def send(self, ctrl):
        """ctrl: (NU,) for one fly or (F, NU)."""
        self.p.stdin.write(np.ascontiguousarray(np.asarray(ctrl, np.float32).reshape(self.F, self.nu).T).tobytes())
        self.p.stdin.flush()

    def recv(self):
        """qpos: (nq,) for F == 1, else (F, nq)."""
        buf = self.p.stdout.read(self.size)
        if len(buf) != self.size:
            raise SystemExit(f'FAIL engine {self.tag} ended early: {self.p.stderr.read().decode()[-300:]}')
        a = np.frombuffer(buf, np.float32)
        q = a[self.nu * self.F:(self.nu + self.nq) * self.F].reshape(self.nq, self.F).T.astype(float)
        self.game = a[len(a) - self.ng * self.F:].reshape(self.ng, self.F).T
        return q[0] if self.F == 1 else q

    def close(self):
        self.p.stdin.close()
        err = self.p.stderr.read().decode()
        self.p.wait()
        m = re.findall(r'COST(?:@\d+)? physics (\S+) neural_control (\S+) render (\S+) copy (\S+) display (\S+)', err)  # last split: a stopped engine prints only COST@n
        return {k: float(v) for k, v in zip(('physics', 'neural_control', 'render', 'copy', 'display'), m[-1])} if m else {}


def brain_setup(start=True):
    """Resident Shiu on the real wiring, columns: 0 sym, 1 left, 2 right. Returns (brain, side -> readout slice, n)."""
    from agent import shiu_g1
    from agent.experiment import cells
    from agent.shiu import ShiuLive
    c = cells()
    data = shiu_g1.flywire()
    graph = next(shiu_g1.arms(data, 'flywire'))[1]
    n = len(data[3])
    eye = {s: np.flatnonzero(np.isin(c.type, LOOM) & (c.side == s)) for s in ('left', 'right')}
    dn = {s: np.flatnonzero((c.super == 'descending') & (c.side == s)) for s in ('left', 'right')}
    brain = ShiuLive(*graph, n=n, flies=3, seed=71)
    for col, (hl, hr) in enumerate(((HZ, HZ), (HZ, WEAK), (WEAK, HZ))):
        brain.stimulate(eye['left'], hl, fly=col, key=col)
        brain.stimulate(eye['right'], hr, fly=col, key=col)
    readout = np.concatenate([col * n + dn[s] for col in range(3) for s in ('left', 'right')]).astype(np.int32)
    sizes = [len(dn['left']), len(dn['right'])] * 3
    if start:
        brain.start([1], readout)
    return brain, np.cumsum([0] + sizes), n


def side_counts(cum, edges):
    """cumulative per-neuron counts -> (3 columns, 2 sides) cumulative DN spikes."""
    return np.array([[cum[edges[2 * col + s]:edges[2 * col + s + 1]].sum() for s in range(2)] for col in range(3)], np.int64)


def run(steps=STEPS, k=K, names=('sym', 'left', 'right', 'silenced', 'held'), flags=()):
    d, m, cam = prepare()
    q0 = np.fromfile(d / 'qpos.bin', np.float32).astype(float)
    root = live.roots(m)[0]
    gait = {n: Gait(m, d, k) for n in names}
    engines = {n: Engine(d, k, n, steps, flags=flags) for n in names}
    brain, edges, _ = brain_setup()
    cum = [np.zeros((3, 2), np.int64)]
    rate_ref = None
    tr = {n: [q0] for n in names}
    drives = {n: [] for n in names}
    cost = {'neural': 0.0, 'gait': 0.0, 'engine': 0.0}
    frozen = None
    for i in range(steps):
        t = time.perf_counter()
        cum.append(side_counts(brain.advance(k), edges))
        cost['neural'] += time.perf_counter() - t
        w = min(i + 1, WINDOW)
        rate = (cum[-1] - cum[-1 - w]).astype(float) / (w * k * DT)  # DN spikes per second, per column and side
        if rate_ref is None and i + 1 == WINDOW:
            rate_ref = np.maximum(rate[0], 1.0)  # the symmetric run's own DN rates after the first window: equal input gives equal full drive
        ref = rate_ref if rate_ref is not None else np.ones(2)
        zero = rate[0] * 0  # the first window is the brain's warm-up: the body stands (drive 0) until it has filled
        warm = rate_ref is None
        drive = {'sym': zero if warm else rate[0] / ref, 'left': zero if warm else rate[1] / ref, 'right': zero if warm else rate[2] / ref,
                 'silenced': zero, 'held': zero if warm else rate[0] / ref}
        t = time.perf_counter()
        ctrl = {}
        for n in names:
            if n == 'held' and i >= STEPS // 2:
                frozen = ctrl_prev['held'] if frozen is None else frozen
                ctrl[n] = frozen
            else:
                sig = np.clip(drive[n], 0, 1)
                drives[n].append(sig)
                ctrl[n] = gait[n].ctrl(sig)
        ctrl_prev = ctrl
        cost['gait'] += time.perf_counter() - t
        t = time.perf_counter()
        for n in names:
            engines[n].send(ctrl[n])
        for n in names:
            tr[n].append(engines[n].recv())
        cost['engine'] += time.perf_counter() - t
    brain.close()
    engine_cost = {n: engines[n].close() for n in names}
    return d, m, root, names, {n: np.array(v) for n, v in tr.items()}, {n: np.array(v) for n, v in drives.items()}, cost, engine_cost, rate_ref


def metrics(m, root, tr):
    """The thorax body's own pose per frame (MuJoCo kinematics of the engine's qpos): forward along its start heading, yaw, tilt of its up axis from vertical."""
    th = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'fly/c_thorax')
    s = mujoco.MjData(m)
    pos, yaw, tilt = [], [], []
    for q in tr:
        s.qpos[:] = q
        mujoco.mj_kinematics(m, s)
        r = s.xmat[th].reshape(3, 3)
        pos.append(s.xpos[th].copy())
        yaw.append(np.arctan2(r[1, 0], r[0, 0]))
        tilt.append(np.degrees(np.arccos(np.clip(r[2, 2], -1, 1))))
    pos, yaw = np.array(pos), np.unwrap(yaw)
    h0 = np.array([np.cos(yaw[0]), np.sin(yaw[0])])
    return {'forward': (pos[:, :2] - pos[0, :2]) @ h0, 'yaw_deg': np.degrees(yaw - yaw[0]), 'tilt_max': float(max(tilt)), 'tilt0': float(tilt[0]), 'z0': float(pos[0, 2]), 'z_end': float(pos[-1, 2])}


def check():
    d, m, root, names, tr, drives, cost, engine_cost, rate_ref = run()
    res = {n: metrics(m, root, tr[n]) for n in names}
    half = STEPS // 2
    f = {n: float(res[n]['forward'][-1]) for n in names}
    last = {n: float(res[n]['forward'][-1] - res[n]['forward'][half]) for n in names}
    yaw = {n: float(res[n]['yaw_deg'][-1]) for n in names}
    ok = True
    walk = f['sym'] > BODY_LENGTH and res['sym']['tilt_max'] < 20
    print(('PASS' if walk else 'FAIL'), f"walk: sym forward {f['sym']:.2f} mm = {f['sym'] / BODY_LENGTH:.2f} body lengths in {STEPS * K * DT:g} s (> 1), max tilt {res['sym']['tilt_max']:.1f} deg (< 20)")
    turn = yaw['left'] * yaw['right'] < 0 and min(abs(yaw['left']), abs(yaw['right'])) > 10
    print(('PASS' if turn else 'FAIL'), f"turn: yaw change left-eye-strong {yaw['left']:+.1f} deg, right-eye-strong {yaw['right']:+.1f} deg (opposite signs, each beyond 10)")
    sil = abs(f['silenced']) < 0.25 * f['sym']
    print(('PASS' if sil else 'FAIL'), f"silenced DN: forward {f['silenced']:.2f} mm vs sym {f['sym']:.2f} (< 25 %)")
    held = abs(last['held']) < 0.25 * last['sym']
    print(('PASS' if held else 'FAIL'), f"held ctrl from 0.5 s: last-half forward {last['held']:.2f} mm vs live {last['sym']:.2f} (< 25 %)")
    ok = walk and turn and sil and held
    z0 = res['sym']['z0']
    print(f"start: standing root z 0.528 (live.prepare), settled into the FlyGym stance thorax z {z0:.3f}, tilt {res['sym']['tilt0']:.1f} deg (flyloop: body-v2 2.1, FlyGym 1.8: thorax body origin, not comparable to the root qpos z); after 1 s z {res['sym']['z_end']:.3f}; DN reference rate per side (spikes/s) {np.round(rate_ref).tolist()}")
    per = lambda s: 1000 * s / STEPS
    # cost: one fly alone (the 5 concurrent engines slow each other), 100 control steps, frames drawn every step
    _, _, _, _, _, _, c1, e1, _ = run(100, K, ('sym',), flags=('-DPROFILE=1',))  # only this run pays the per-phase syncs
    e = e1['sym']
    cost1 = {'physics': e['physics'], 'neural_shiu_advance': 1000 * c1['neural'] / 100, 'gait_controller': 1000 * c1['gait'] / 100, 'render': e['render'], 'copy': e['copy'],
             'display': e['display'], 'engine_round_trip': 1000 * c1['engine'] / 100}
    print('cost ms/frame (one fly alone, K=10, 100 steps, 960x540 is not used: 640x360 frames to /dev/null): ' + ', '.join(f'{k2} {v:.2f}' for k2, v in cost1.items())
          + f"; with 5 engines concurrently: engine round trip {per(cost['engine']):.1f}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'receipt.json').write_text(json.dumps({'forward_mm': f, 'last_half_forward_mm': last, 'yaw_deg': yaw, 'tilt_max': {n: res[n]['tilt_max'] for n in names},
                                                  'z0': z0, 'cost_ms_one_fly': cost1, 'cost_ms_five_engines': {k2: per(v) for k2, v in cost.items()}, 'engine_cost': engine_cost,
                                                  'dn_ref_rate': np.asarray(rate_ref).tolist(), 'drive_mean': {n: drives[n].mean(0).tolist() for n in names}}, indent=1))
    if not ok:
        sys.exit('FAIL fly_link')
    print('PASS fly_link: brain-driven fly in live (host transport)')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'check'
    if cmd == 'build':
        from agent.shiu import worker_binary
        d, m, cam = prepare()
        sc.build(d, 'fly', [], K, brain=1)
        b = brain_setup(start=False)[0]
        print('built', worker_binary(b.n, len(b.col), b.flies, *b.source))
    else:
        check()
