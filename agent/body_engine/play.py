"""Play the fly by firing its neurons (docs/plans/PLAY.md): keys inject Poisson input into named neurons of the resident
Shiu brain (agent/body_engine/keymap.toml), the brain's descending neurons drive the body through fly_link's readout and
gait, and the brain's own drives can outvote the player.

  agent/body_engine/play.sh [world [SEED]]          (interactive: owns the terminal, runs the job through gpu-wait)
  gpu-wait run --priority 60 --vram 4 --timeout 1200 -- .venv/bin/python -m agent.body_engine.play check               (check world [SEED]: on the seeded terrain)

Input: the terminal in kitty keyboard mode (press / release events; ghostty, kitty, foot) and SGR mouse motion; frames go
to mpv. Every key and the mouse are stim entries registered once at start (rate 0); per control step only the probability
row changes, written to the worker's stim_p.bin, which it rereads on every advance (no new kernel, nothing when no key
changed). Walking keys are scaled by the engine's stamina (-DGAME=1 game rows, read from the sensor packet). The world
fires a looming threat (LC4 + LPLC2, both eyes) on `t`; PLAY_EYES=1 adds the world's looming per eye (class Eyes). HUD (terminal line): the window's DN
spikes split into the player's DNs (key targets) and the brain's own DNs; OVERRIDE when own > player while a key is held.
Engine HUD bars show energy / health / stamina in the frame.

check (headless, one engine F = 5, one brain with 5 columns, 1 s each): idle stands upright (tilt < 90); W walks > 1 body length, tilt < 20;
A and D turn opposite ways beyond 10 deg; W with the threat from 0.5 s: own share rises over player share; stamina under W
below idle; neural cost per step with and without a stim write.
"""
import json
import os
import re
import select
import subprocess
import sys
import termios
import time
import tomllib
import tty
from collections import deque
from pathlib import Path

import numpy as np

os.environ.setdefault('MUJOCO_GL', 'egl')  # eye cameras render headless (before mujoco loads)
from agent.body_engine import fly_link as fl, game_check as gc, live, spectate_check as sc

KEYMAP = Path(__file__).with_name('keymap.toml')
PRESETS = fl.OUT.parents[2] / 'shiu/presets'
OUT = fl.OUT.parent / 'play'
RENDER = int(os.environ.get('PLAY_RENDER', 24))  # control steps per frame: 24 = 41.7 fps of sim time; world terrain 0.89x real time vs 0.85x at 16 (62.5 fps), 0.31x at 4
CPU = os.environ.get('PLAY_CPU', '1') == '1'  # physics on the host (smooth.mojo -DCPU=1): one fly, 53 us per 0.1 ms step vs 4.4 ms on one GPU wave
WORK_RATE = float(os.environ.get("PLAY_WORK_RATE", 2e-6))  # stamina + energy per unit of actuator work: W walking does about 7.3e3 per sim second (check 2), so about 70 s to exhaustion; see docs/plans/PLAY.md
EYES = os.environ.get('PLAY_EYES', '0') == '1'  # the world's proximity reaches the brain's looming cells (class Eyes, keymap [loom]); off: walls gate FAILs
EYE_EVERY = 20  # control steps per eye frame: 50 Hz
CAL = 250  # control steps (0.25 s) of W held after fl.WINDOW: the drive reference is their mean DN rate per side; the first 50 ms
# alone (still ramping: 244 vs 524 Hz L/R) left W decoding 0.84 vs 0.67 and the fly turning -34 deg
HOLD = fl.WINDOW + 2 * CAL  # two calibration windows: W alone (-> HEADROOM), then W + symmetric loom (-> 1)
HEADROOM = float(os.environ.get('PLAY_HEADROOM', 0.5))  # W decodes to this drive; the rest is the loom's (0.8 left 0.2: probe R +5.5, 2026-10-07)
NAMES = ('idle', 'w', 'a', 'd', 'w+threat')


def resolve(target, c):
    """Graph indices of a keymap target; a preset name must be a verified preset."""
    from agent import presets
    if target == 'sugar-both':
        return presets.sugar()
    for f in PRESETS.glob('*.json'):
        p = json.loads(f.read_text()).get(target)
        if isinstance(p, dict) and not p.get('verified', True):
            raise SystemExit(f'FAIL keymap target {target}: preset in {f.name} is not verified')
    t, side = (target.rsplit('-', 1) + [''])[:2] if target.endswith(('-left', '-right')) else (target, '')
    ids = np.flatnonzero((c.type == t) & ((c.side == side) if side else True))
    if not len(ids):
        raise SystemExit(f'FAIL keymap target {target}: no cells of that type')
    return ids


class Brain:
    """Resident brain, `cols` columns; every input of keymap.toml a stim entry; readout = DNs per side per column."""
    def __init__(self, cols, km=None, seed=71):
        from agent import shiu_g1
        from agent.experiment import cells
        from agent.shiu import ShiuLive
        km = km or tomllib.loads(KEYMAP.read_text())
        c = cells()
        data = shiu_g1.flywire()
        self.n, self.cols = len(data[3]), cols
        self.inputs = {k: (np.concatenate([resolve(t, c) for t in v['targets']]), v['hz'], v.get('walk', False)) for k, v in km['keys'].items()}
        ms = km['mouse']
        self.inputs |= {'mouse-' + s: (resolve(ms[s], c), 0, True) for s in ('left', 'right')}
        self.inputs |= {'loom-' + s: (np.concatenate([resolve(f'{t}-{s}', c) for t in km['loom']['targets']]), km['loom']['hz'], False)
                        for s in ('left', 'right')}
        self.inputs['threat'] = (np.concatenate([resolve(t, c) for t in km['threat']['targets']]), km['threat']['hz'], False)
        self.km, self.names = km, list(self.inputs)
        self.brain = ShiuLive(*next(shiu_g1.arms(data, 'flywire'))[1], n=self.n, flies=cols, seed=seed)
        for col in range(cols):
            for ids, _, _ in self.inputs.values():
                self.brain.stimulate(ids, 0, fly=col)
        pos = {e: i for i, e in enumerate(self.brain.stim)}
        self.M = np.zeros((cols, len(self.names), len(pos)), np.float32)  # which entries each input of each column drives
        for col in range(cols):
            for j, (ids, _, _) in enumerate(self.inputs.values()):
                self.M[col, j, [pos[col * self.n + i] for i in ids]] = 1
        self.hz = np.array([v[1] for v in self.inputs.values()], np.float32)
        self.walk = np.array([v[2] for v in self.inputs.values()])
        dn = {s: np.flatnonzero((c.super == 'descending') & (c.side == s)) for s in ('left', 'right')}
        player = np.unique(np.concatenate([v[0] for k, v in self.inputs.items() if k != 'threat']))
        self.readout = np.concatenate([col * self.n + dn[s] for col in range(cols) for s in ('left', 'right')]).astype(np.int32)
        self.side = np.concatenate([np.full(len(dn[s]), si) for col in range(cols) for si, s in enumerate(('left', 'right'))])
        self.colof = np.concatenate([np.full(len(dn['left']) + len(dn['right']), col) for col in range(cols)])
        self.mine = np.isin(self.readout % self.n, player)  # DNs the player stimulates directly
        self.brain.start([1], self.readout)
        self.p = None

    def set(self, rates):
        """rates (cols, inputs) in Hz -> the worker's per-entry probabilities, written only when they changed."""
        p = np.clip(np.einsum('cn,cne->e', rates, self.M) * fl.DT, 0, 1).astype(np.float32)
        if self.p is None or not np.array_equal(p, self.p):
            p.tofile(self.brain.work / 'stim_p.bin')
            self.p = p

    def advance(self, k):
        """Cumulative spikes: (cols, 2 sides) all DNs, (cols, 2) player DNs / own DNs."""
        self.brain.advance_start(k)
        return self.collect()

    def collect(self):
        """advance's result once the worker answers (after brain.advance_start)."""
        cum = self.brain.advance_wait().astype(np.int64)
        side = np.zeros((self.cols, 2), np.int64)
        np.add.at(side, (self.colof, self.side), cum)
        split = np.zeros((self.cols, 2), np.int64)
        np.add.at(split, (self.colof, (~self.mine).astype(int)), cum)
        return side, split


class Eyes:
    """Engineered PROXIMITY sensor per eye, not visual looming (body.xml l/r_eye_cam, fovy 157): a 32x24 depth render with
    the fly's own body and the ground hidden; the share of each eye nearer than km[loom].near mm, above `base`, over `span`.
    The drive is the share above its own low-pass over `tau` s: a surface closing in drives it, one that stays put fades
    (a sustained drive held a fly pressed into a wall, own DNs flooding, legs shaking). Still proximity change, not
    image expansion. Output (2,) in [0, 1], left eye first, drives that side's [loom] targets (LC4 + LPLC2)."""
    def __init__(self, m, loom):
        import copy
        import mujoco
        m = copy.deepcopy(m)
        hide = (m.geom_bodyid != 0) | np.isin(m.geom_type, (mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_HFIELD))
        m.geom_matid[hide], m.geom_rgba[hide, 3] = -1, 0  # alpha 0 is not drawn: no own body, no ground, obstacles only
        self.mj, self.m, self.d, self.loom = mujoco, m, mujoco.MjData(m), loom
        self.cams = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, f'fly/{s}_eye_cam_camera') for s in 'lr']
        self.r = mujoco.Renderer(m, 24, 32)
        self.r.enable_depth_rendering()
        self.slow = None  # near share low-passed over tau: a surface that stays put stops driving (an adapting sensor)

    def __call__(self, q):
        self.d.qpos[:] = q
        self.mj.mj_kinematics(self.m, self.d)
        self.mj.mj_camlight(self.m, self.d)
        near = np.empty(2)
        for e, c in enumerate(self.cams):
            self.r.update_scene(self.d, camera=c)
            near[e] = (self.r.render() < self.loom['near']).mean()
        self.prev = near  # last frame's shares, for inspection
        a = EYE_EVERY * fl.K * fl.DT / self.loom['tau']
        self.slow = near.copy() if self.slow is None else self.slow + a * (near - self.slow)
        return np.clip((near - self.slow - self.loom['base']) / self.loom['span'], 0, 1)


def setup(F, render=0, out='/dev/null', seed=None, q0=None):
    """seed: the fly on live.py's seeded terrain (live.WORLD, 0.1 mm bumps and cones), dropped from the flat standing pose."""
    flags = ['-DGAME=1', '-DHUD=1', '-DFOLLOW=1', '-DORBIT=1'] + (['-DCPU=1'] if CPU else []) + (['-DPROFILE=1'] if os.environ.get('PLAY_PROFILE') == '1' else []) + (['-DGUI=1'] if os.environ.get('PLAY_GUI') == '1' else [])  # Kinestride's window (live.mojo GUI)
    if seed is None:
        d, m, _ = fl.prepare(OUT / 'fly')
    else:  # terrain + the world's walls, stones, grass, flowers and two plume sources (agent/body_engine/assets.py), smelled at the antennae
        from agent.body_engine import assets
        from agent.body_engine.world_check import sites_of
        params = live.WORLD[:5] + (seed,) + live.WORLD[6:]
        mjcf = live.terrain(fl.MJCF, params)[0]
        w = live.load(mjcf)
        stones = [(*w.geom_pos[g][:2], float(w.geom_size[g][:2].max())) for g in range(w.ngeom) if w.geom_type[g] == 4 and w.geom_bodyid[g] == 0]
        parts, src = assets.scene(params, seed, stones)
        assets_on = os.environ.get('PLAY_ASSETS', '1') == '1'
        d, m, _ = fl.prepare(OUT / (f'fly-world{seed}' + ('' if assets_on else '-bare')), mjcf=mjcf, flat=fl.MJCF, world=parts if assets_on else None)
        src.tofile(d / 'src.bin')
        np.array([[b, *p] for b, p in sites_of(m)], np.float32).tofile(d / 'sites.bin')
        flags += ['-DWORLD=1', '-DNSITE=2', '-DSOIL=1']
    (d / 'mf.bin').replace(d / 'mf.base')  # prepare re-exports every run: the fresh export is the base (a kept base went stale when the export grew)
    gc.set_rules(d, 0.0, WORK_RATE)
    eng = fl.Engine(d, fl.K, 'play', 0, F=F, flags=tuple(flags), out=out, render=render, q0=q0(d) if q0 else None)
    return d, m, eng


def loop(brain, eng, gaits, rates_at, steps, on_step=None):
    """Brain + body for `steps` control steps (0 = until rates_at returns None). The first HOLD steps hold the body and
    fire `w` in every column; the last 2 CAL of them calibrate the decoder per side: W alone -> HEADROOM (ref_w), then W + symmetric loom -> 1
    (ref_l, fly_link's recipe: the symmetric eye drive is the top of the scale). The map is piecewise linear through (0, 0), (ref_w, HEADROOM),
    (ref_l, 1), so a one-sided loom leaves the other side near HEADROOM instead of clipping both at 1 (2026-10-07: with W alone as the only
    reference a forced one-sided loom was swap-invariant; with the loom reference alone W decoded to 0.1, docs/plans/PLAY.md); then rates_at(i, game)."""
    j, lm = brain.names.index('w'), [brain.names.index('loom-' + s) for s in ('left', 'right')]
    zero = np.zeros((brain.cols, 2), np.int64)
    cum, split, held = deque([zero], fl.WINDOW + 1), deque([zero], fl.WINDOW + 1), gaits[0].held  # one window of history, fixed size
    ref, tr, cost = None, [], dict.fromkeys(('neural', 'write', 'rates', 'gait', 'recv', 'send', 'on_step'), 0.0)  # s per stage
    game = np.ones((brain.cols, 4), np.float32)
    i, pend = 0, None

    def take(k, win, drive):  # step k's body state; game rows feed rates_at one step late (stamina, slow)
        nonlocal game
        t = time.perf_counter()
        q = eng.q = eng.recv()
        cost['recv'] += time.perf_counter() - t
        game = eng.game
        if k >= HOLD:
            if steps:  # play (steps 0) keeps no trajectory: 600 B per 1 ms step grew 2.1 GB per hour
                tr.append(q if brain.cols > 1 else q[None])
            if on_step:
                t = time.perf_counter()
                on_step(k - HOLD, win, game, drive)
                cost['on_step'] += time.perf_counter() - t

    def rates(i):
        if i < HOLD:
            r = np.zeros((brain.cols, len(brain.names)), np.float32)
            r[:, j] = brain.hz[j]
            if i >= HOLD - CAL:  # second window: W + symmetric loom
                r[:, lm] = brain.hz[lm]
            return r
        return rates_at(i - HOLD, game)

    def start(i):  # brain step i on the worker, running while the host does step i-1's gait and body
        t = time.perf_counter()
        r = rates(i)
        cost['rates'] += time.perf_counter() - t
        if r is None:
            return False
        t = time.perf_counter()
        brain.set(r)
        cost['write'] += time.perf_counter() - t
        brain.brain.advance_start(fl.K)
        return True

    going = start(0)
    while going:
        t = time.perf_counter()
        s, sp = brain.collect()
        cost['neural'] += time.perf_counter() - t
        going = not (steps and i + 1 >= steps + HOLD) and start(i + 1)  # next brain step runs during this step's gait + body
        cum.append(s)
        split.append(sp)
        w = min(i + 1, fl.WINDOW)
        rate = (cum[-1] - cum[-1 - w]) / (w * fl.K * fl.DT)
        if i + 1 == HOLD - 2 * CAL:  # start of the W window: W held, body held, brain settled
            cal0 = s
        if i + 1 == HOLD - CAL:  # end of the W window, start of the W + loom window
            ref_w, cal0 = np.maximum(((s - cal0) / (CAL * fl.K * fl.DT)).mean(0), 1.0), s
        if i + 1 == HOLD:
            ref = np.maximum(((s - cal0) / (CAL * fl.K * fl.DT)).mean(0), ref_w + 1.0)
        drive = np.where(rate <= ref_w, HEADROOM * rate / ref_w, HEADROOM + (1 - HEADROOM) * (rate - ref_w) / (ref - ref_w)).clip(0, 1) if i >= HOLD else np.zeros_like(rate, float)
        t = time.perf_counter()
        ctrl = np.stack([g.ctrl(dv) for g, dv in zip(gaits, drive)]) if i >= HOLD else np.tile(held, (brain.cols, 1))
        cost['gait'] += time.perf_counter() - t
        if pend is not None:  # the body's previous step ran while the brain advanced (the brain never reads the body)
            take(*pend)
        t = time.perf_counter()
        eng.send(ctrl)
        cost['send'] += time.perf_counter() - t
        pend = (i, split[-1] - split[0], drive)
        i += 1
    if pend is not None:
        take(*pend)
    cost['steps'] = i
    return np.array(tr), np.stack([ref_w, ref]), cost


def check(*args):
    """check [world [SEED]]: the same gates on the flat floor or the seeded terrain."""
    seed = next((int(x) for x in args[1:] if x.isdigit()), 1) if args[:1] == ('world',) else None
    km = tomllib.loads(KEYMAP.read_text())
    brain = Brain(len(NAMES), km)
    d, m, eng = setup(len(NAMES), seed=seed)
    gaits = [fl.Gait(m, d, fl.K) for _ in NAMES]
    ix = {k: brain.names.index(k) for k in ('w', 'a', 'd', 'threat')}
    half, wins, drives = fl.STEPS // 2, [], []

    def rates_at(i, game):
        r = np.zeros((len(NAMES), len(brain.names)), np.float32)
        for col, keys in ((1, 'w'), (2, 'a'), (3, 'd'), (4, 'w')):
            r[col, ix[keys]] = brain.hz[ix[keys]]
        if i >= half:
            r[4, ix['threat']] = brain.hz[ix['threat']]
        return r

    tr, ref, cost = loop(brain, eng, gaits, rates_at, fl.STEPS, lambda i, win, game, drive: (wins.append(win.copy()), drives.append(drive.copy())))
    brain.brain.close()
    engine_cost = eng.close()
    root = live.roots(m)[0]
    res = {n: fl.metrics(m, root, np.r_[tr[:1, col], tr[:, col]]) for col, n in enumerate(NAMES)}
    game = eng.game
    wins = np.array(wins, float)  # (steps, cols, [player, own])
    before = wins[half - fl.WINDOW:half, 4].sum(0)
    during = wins[-fl.WINDOW:, 4].sum(0)
    fwd = {n: float(res[n]['forward'][-1]) / fl.BODY_LENGTH for n in NAMES}
    yaw = {n: float(res[n]['yaw_deg'][-1]) for n in NAMES}
    n = fl.STEPS + HOLD
    gates = {
        'idle stands': abs(fwd['idle']) < 0.25 and res['idle']['tilt_max'] < 90,  # upright: an upside-down fly (172 deg) also stays put
        'w walks': fwd['w'] > 1 and res['w']['tilt_max'] < 20,
        'w straight': abs(yaw['w']) < 15,  # a -48 deg W used to pass 'w walks'
        'a turns left, d right': yaw['a'] > 10 and yaw['d'] < -10,
        'a/d comparable': 0.5 < abs(yaw['a']) / max(abs(yaw['d']), 1e-9) < 2,
        'threat overrides': during[1] > during[0] and during[1] / max(during.sum(), 1) > before[1] / max(before.sum(), 1),
        'stamina drains under w': game[1, 2] < game[0, 2],
    }
    out = {'forward_body_lengths': fwd, 'yaw_deg': yaw, 'tilt_max': {k: res[k]['tilt_max'] for k in NAMES},
           'w+threat [player, own] DN spikes, last window before / during': [before.tolist(), during.tolist()],
           'game rows end (energy, health, stamina, dead)': {k: game[c].tolist() for c, k in enumerate(NAMES)},
           'drive ref (DN Hz per side: W held, W + symmetric loom)': ref.tolist(),
           'drive [L, R] mean / clipped at 1 share': {k: [np.mean(drives, 0)[c].tolist(), np.mean(np.array(drives)[:, c] >= 1, 0).tolist()] for c, k in enumerate(NAMES)}, 'ms per control step': {k: 1e3 * v / n for k, v in cost.items() if k != 'steps'},
           'engine COST ms/frame': engine_cost, 'work_rate': WORK_RATE, 'gates': gates}
    (OUT / ('check.json' if seed is None else f'check-world{seed}.json')).write_text(json.dumps(out, indent=1, default=float) + '\n')
    for k, v in out.items():
        print(k, json.dumps(v, default=float))
    print(f'A turns {"left" if yaw["a"] > 0 else "right"} (yaw {yaw["a"]:+.1f} deg), D {yaw["d"]:+.1f} deg')
    bad = [k for k, v in gates.items() if not v]
    print('\n'.join(f"{'FAIL' if k in bad else 'PASS'} {k}" for k in gates))
    if bad:
        sys.exit(f'FAIL {len(bad)}/{len(gates)} {bad}')


def walls(*args):
    """walls [SEED]: W held 1 s, the fly 6 mm from the west wall at 45 deg, wall ahead-left (cols 0, 2) or ahead-right (1, 3);
    cols 0-1 see (Eyes -> loom), 2-3 blind twins. PASS: each seeing fly turns away from its wall more than its blind twin
    (10 deg) and keeps farther from it. PLAY_LOOM_SWAP=1 feeds each eye's looming to the other side: must FAIL."""
    seed = next((int(x) for x in args if x.isdigit()), 1)
    km = tomllib.loads(KEYMAP.read_text())
    swap, force = os.environ.get('PLAY_LOOM_SWAP') == '1', os.environ.get('PLAY_LOOM_FORCE') == '1'
    # PLAY_LOOM_FORCE=1: the probe. No eyes, flat floor, no wall: cols 0/1 get a full left/right loom from HOLD on, 2/3 none.
    # Same gate. PASS = play's brain + decoder turn away from a one-sided loom; a FAIL here is the brain, a FAIL only in walls is the sensor.
    # SEED in force mode seeds the brain (71 + SEED): the blind twins differed by 20 deg under W alone, so the probe is paired over seeds
    # (tools/play_probe.sh: 3 seeds, PASS = every seed's see - blind delta has the right sign and the median is beyond 10 deg).
    yaws = (3 * np.pi / 4, 5 * np.pi / 4) * 2

    def q0(d):
        q = np.tile(np.fromfile(d / 'qpos.bin', np.float32), (4, 1))
        q[:, 0], q[:, 1] = (0.0 if force else -14.0), 0.0
        q[:, 3:7] = [(np.cos(y / 2), 0, 0, np.sin(y / 2)) for y in yaws]
        return q

    brain = Brain(4, km, seed=71 + seed if force else 71)
    d, m, eng = setup(4, seed=None if force else seed, q0=q0)
    gaits = [fl.Gait(m, d, fl.K) for _ in range(4)]
    eyes = [] if force else [Eyes(m, km['loom']) for _ in range(2)]
    w, lm = brain.names.index('w'), [brain.names.index('loom-' + s) for s in ('left', 'right')]
    loom, hist = np.zeros((2, 2)), []

    def rates_at(i, game):
        r = np.zeros((4, len(brain.names)), np.float32)
        r[:, w] = brain.hz[w]
        if force:
            loom[:] = np.eye(2)[:, ::-1 if swap else 1] if i >= HOLD else 0.0
        elif i % EYE_EVERY == 0 and getattr(eng, 'q', None) is not None:
            for col in range(2):
                loom[col] = eyes[col](eng.q[col])[::-1 if swap else 1]
        r[:2, lm] = brain.hz[lm] * loom
        if i % EYE_EVERY == 0:
            hist.append(loom.copy())
        return r

    tr, _, cost = loop(brain, eng, gaits, rates_at, fl.STEPS)
    brain.brain.close()
    eng.close()
    root = live.roots(m)[0]
    yaw = [float(fl.metrics(m, root, np.r_[tr[:1, c], tr[:, c]])['yaw_deg'][-1]) for c in range(4)]
    gap = [float(tr[:, c, root].min() + 20) for c in range(4)]  # nearest root x to the wall face (x = -20), mm
    gates = {'turns away (left wall: right turn)': yaw[0] - yaw[2] < -10 and yaw[1] - yaw[3] > 10,
             'keeps clear': gap[0] > gap[2] and gap[1] > gap[3]}
    if force:
        gates.pop('keeps clear')  # no wall on the flat floor
    h = np.array(hist)  # (frames, see col, eye)
    print(json.dumps({'yaw_deg [see L, see R, blind L, blind R]': yaw, 'min gap mm': gap, 'swap': swap, 'force': force,
                      'loom [col][eye] mean, by quarter': np.round([q.mean(0) for q in np.array_split(h, 4)], 2).tolist(),
                      'ms per control step': {k: 1e3 * v / max(cost['steps'], 1) for k, v in cost.items() if k != 'steps'}}, default=float))
    bad = [k for k, v in gates.items() if not v]
    print('\n'.join(f"{'FAIL' if k in bad else 'PASS'} {k}" for k in gates))
    if bad:
        sys.exit(f'FAIL {len(bad)}/{len(gates)} {bad}')


KEY = re.compile(rb'\x1b\[(\d+)(?::\d+)*(?:;\d+(?::(\d))?)?u|\x1b\[<(\d+);(\d+);(\d+)[Mm]|([ -~])')


def play(*args):
    """play [world [SEED]]: flat floor, or the seeded terrain (seed 1 by default)."""
    seed = next((int(x) for x in args[1:] if x.isdigit()), 1) if args[:1] == ('world',) else None
    km = tomllib.loads(KEYMAP.read_text())
    brain = Brain(1, km)
    fifo = OUT / 'frames.fifo'
    OUT.mkdir(parents=True, exist_ok=True)
    fifo.unlink(missing_ok=True)
    os.mkfifo(fifo)
    mpv = subprocess.Popen(['mpv', '--demuxer=rawvideo', f'--demuxer-rawvideo-w={sc.W}', f'--demuxer-rawvideo-h={sc.H}',
                            '--demuxer-rawvideo-mp-format=rgba', '--profile=low-latency', '--untimed', '--title=fly', str(fifo)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    d, m, eng = setup(1, render=RENDER, out=fifo, seed=seed)
    gaits = [fl.Gait(m, d, fl.K)]
    eyes, loom = (Eyes(m, km['loom']) if EYES else None), np.zeros(2)
    th = km['threat']
    dur = round(th['dur_s'] / (fl.K * fl.DT))
    names = {'space': ' '} | {k: k for k in km['keys']}
    held, tap, mouse, lastx, lasty, threat_until = set(), {}, 0.0, None, None, -1
    orbit, ob_tmp = np.array([1, -np.pi / 4, 0.35, 1], np.float64), d / 'orbit.tmp'  # on, azimuth, elevation (rad), zoom
    orbit.astype(np.float32).tofile(ob_tmp)
    os.replace(ob_tmp, d / 'orbit.bin')
    io = os.environ.get('PLAY_IO')  # gpu-wait jobs have no terminal: play.sh owns it and relays keys / HUD through two fifos here
    if io:
        fd, old, out = os.open(Path(io) / 'keys', os.O_RDWR), None, open(Path(io) / 'hud', 'w')
    else:
        fd, old, out = sys.stdin.fileno(), termios.tcgetattr(sys.stdin.fileno()), sys.stdout
        tty.setraw(fd)
    out.write('\x1b[>11u\x1b[?1003h\x1b[?1006h')  # kitty keys with press/release, SGR mouse motion
    out.flush()

    def rates_at(i, game):
        nonlocal mouse, lastx, lasty, threat_until, loom
        while select.select([fd], [], [], 0)[0]:
            for g in KEY.finditer(os.read(fd, 4096)):
                if g.group(1):
                    ch, ev = chr(int(g.group(1))), g.group(2) or b'1'
                    if ch in 'q\x1b':
                        return None
                    if ch == 'c' and ev == b'1':  # camera back to the default follow view
                        orbit[:] = (1, -np.pi / 4, 0.35, 1)
                        orbit.astype(np.float32).tofile(ob_tmp)
                        os.replace(ob_tmp, d / 'orbit.bin')
                    (held.discard if ev == b'3' else held.add)(ch)
                elif g.group(3):  # SGR mouse: plain motion steers, left drag orbits the camera, wheel zooms
                    b, x, y = int(g.group(3)), int(g.group(4)), int(g.group(5))
                    dx, dy = (0, 0) if lastx is None else (x - lastx, y - lasty)
                    lastx, lasty = x, y
                    if b == 32:
                        orbit[1] -= 0.05 * dx
                        orbit[2] = min(max(orbit[2] + 0.05 * dy, -0.2), 1.5)
                    elif b in (64, 65):
                        orbit[3] = min(max(orbit[3] * (0.9 if b == 64 else 1.1), 0.2), 8.0)
                    elif b in (35, 43):
                        mouse += dx
                    else:
                        continue
                    if b != 35:
                        orbit.astype(np.float32).tofile(ob_tmp)
                        os.replace(ob_tmp, d / 'orbit.bin')
                elif g.group(6):  # a terminal without kitty events: a keypress holds for 0.3 s
                    ch = g.group(6).decode()
                    if ch == 'q':
                        return None
                    tap[ch] = i + round(0.3 / (fl.K * fl.DT))
        keys = held | {k for k, t in tap.items() if t > i}
        if th['key'] in keys:
            threat_until = i + dur
        r = np.zeros((1, len(brain.names)), np.float32)
        for k, ch in names.items():
            r[0, brain.names.index(k)] = brain.hz[brain.names.index(k)] * (ch in keys)
        mouse *= 0.9
        hz = min(abs(mouse) * km['mouse']['hz_per_px'], km['mouse']['max_hz'])
        r[0, brain.names.index('mouse-left' if mouse < 0 else 'mouse-right')] = hz
        r[0, brain.walk] *= 0.2 + 0.8 * game[0, 2]  # tired: the player's walking drive weakens, the brain's own does not
        r[0, brain.names.index('threat')] = brain.hz[brain.names.index('threat')] * (i < threat_until)
        if eyes and i % EYE_EVERY == 0 and getattr(eng, 'q', None) is not None:
            loom = eyes(eng.q)
        for e, side in enumerate(('left', 'right')):
            r[0, brain.names.index('loom-' + side)] = brain.hz[brain.names.index('loom-' + side)] * loom[e]
        return r

    tick = fl.K * fl.DT  # one control step of sim time; the clock holds play to it, a slow step is not made up (Minecraft's tick)
    clock, start = [None], [time.perf_counter()]

    def hud(i, win, game, drive):
        now = time.perf_counter()
        if i == 0:
            start[0] = now
        if clock[0] is None or now - (clock[0] + i * tick) > 0.1:  # start, or more than 0.1 s behind: drop the debt
            clock[0] = now - i * tick
        wait = clock[0] + (i + 1) * tick - now
        if wait > 0:
            time.sleep(wait)
        if i % 20:
            return
        p, o = win[0]
        keys = held | {k for k, t in tap.items() if t > i}
        flag = ' OVERRIDE' if keys and o > p else ''
        speed = (i + 1) * tick / max(time.perf_counter() - start[0], 1e-9)
        out.write(f'\r\x1b[2K{speed:4.2f}x real time | player {p:5d} own {o:5d} | drive L {drive[0, 0]:.2f} R {drive[0, 1]:.2f} | E {game[0, 0]:.2f} '
                         f'S {game[0, 2]:.2f}{" DEAD" if game[0, 3] else ""} | {"".join(sorted(keys))}{" THREAT" if i < threat_until else ""}{flag}')
        out.flush()

    try:
        _, _, cost = loop(brain, eng, gaits, rates_at, 0, hud)
    finally:
        out.write('\x1b[<u\x1b[?1003l\x1b[?1006l\r\n')
        out.flush()
        if old is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        brain.brain.close()
        ec = eng.close()
        mpv.terminate()
    import resource  # host ms per control step by stage, the engine's own COST line (ms per frame), peak RSS
    (OUT / 'cost.json').write_text(json.dumps({k: 1e3 * v / max(cost['steps'], 1) for k, v in cost.items() if k != 'steps'}
                                              | {'steps': cost['steps'], 'engine ms/frame': ec,
                                                 'maxrss_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}, indent=1) + '\n')


if __name__ == '__main__':
    a = sys.argv[1:] or ['play']
    check(*a[1:]) if a[0] == 'check' else walls(*a[1:]) if a[0] == 'walls' else play(*a[a[0] == 'play':])
