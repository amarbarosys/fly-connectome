"""Spider orientation (docs/plans/SPIDER_BRAIN.md U2b): the brain's eyes lead, the body follows.

The closed eye loop of agent/spider_eye_loop.py, now with the legs: every 10 ms the gait controller (spider_gait.gait,
FREQ tetrapod, the stance-to-stride mapping the link was gated with) gets speed 0.6 and turn = trim + gaze / (TAU x yaw-per-turn), both measured by gait_rate with no brain,
the eye-in-head angle the brain's eye motors have built up (a fixed eye-to-body mapping, as the fly's gait mapping; the
brain itself is the trained scaffold pathway). The body rotating moves both eyes, so the prey slips on the retina and the
brain keeps it foveated through the body's own motion. The prey crosses at 6-9 deg/s (held-out speeds), 191 mm out.

check (5 s, the last 2 s scored; truth from the body packet: head pose, prey position):
  G1 body arm: median eye-on-prey error < 1 deg
  G2 body arm: |eye-in-head gaze| < 1/3 of the no-turn arm's (the body took the turn over)
  G3 body arm: body yaw change has the prey bearing change's sign and >= 50 % of its size
  M1 mutant, turn sign flipped: eye error >= 3 x the body arm's or the body turns the wrong way
  C1 silenced brain (no eye motor output): eye error >= 5 x the body arm's
usage: gpu-wait run --shared --vram 4 --priority 60 --timeout 3600 --env PYTHONPATH=$PWD -- .venv/bin/python -m agent.spider_body_loop check
"""
import json
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor

from pathlib import Path

import numpy as np

from agent import spider_eye_loop as L
from agent.body_engine.spider_gait import gait

DEBUG = os.environ.get('SPIDER_TIER') == 'debug'  # debug tier: 200 steps (32 animals as the claim), debug seeds, no seed claim
T, SCORE, SPEED = (200, 80, 0.3) if DEBUG else (500, 200, 0.3)  # 0.3: the hand-wired eye holds 0.61 deg walking, 3.3-3.9 at 0.6 (stride wobble)
PROF = bool(os.environ.get('SPIDER_PYPROF'))  # Python-side split per step: engine exchange, brain step, bookkeeping
START = threading.Lock()  # parallel arms: model files and engine builds are made one at a time
YAW_PER_TURN, TRIM, TAU = -44.0, 0.0, 1.0  # at SPEED 0.3 (gait_rate: -40.9 to +40.9 deg/s over turn -1..1, trim yaw +1.1); was -140 at 0.6.
# gait_rate (no brain): yaw deg/s per unit turn (sign: + turn yaws right), turn for zero yaw; body closes the gaze in TAU s
RAMP = 30  # steps (0.3 s) over which the gait speed rises from 0, as fly_link ramps its servos: full speed at step 0 jerked the head 2.0 deg
GAIN_FILE = L.G.ROOT / '.work/agent/spider/gait-rate.json'  # gait_rate's measured yaw deg/s per unit turn at SPEED
POS = 5.0  # 1/s, the teacher's centring term (0.2 s); set before the run, not tuned
REFLEX = False  # 95fa88a added it; with the position-trained pathway the reflex-off arm held 0.54 / 0.60 deg, the reflex arm lost the prey (38d70f0)
SEEDS = (5, 7) if DEBUG else (47, 53)  # prey speeds, trial stream; claim seeds pass agent.seeds.claim (docs/seeds_used.json)
DEAD = 1.0  # deg: the body turns only on the gaze beyond this (salticid-like pivots); with TAU 1 s (was 0.25: 6f8a87b, the body-turn loop ran away). Set before the run


def yaw(q):
    """heading deg of a (7+, ...) qpos: one animal (vector) or all at once (nq, B)."""
    w, x, y, z = q[3:7]
    return np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def wrap(a):
    return (a + 180) % 360 - 180


GAIN = {'body': 1.0, 'noturn': 0.0, 'flip': -1.0, 'silenced': 1.0, 'stand': 0.0, 'novor': 1.0, 'hand': 1.0}


def hand_net():
    """the hand-wired circuit, fresh per run: its rng draws every input spike, so a shared one made identical arms differ ~2x."""
    n = L.P.Net(np.random.default_rng(1), hand=True)
    return (lambda: n.trial_reset(L.G.B)), n.step


def ci(x, q=(5, 95)):
    """bootstrap 90 % interval of the median over animals."""
    m = np.median(np.random.default_rng(0).choice(x, (2000, len(x))), 1)
    return tuple(np.percentile(m, q))


def turn_law(gaze, arm):
    """the gait turn command from the eye-in-head angle: the body closes the gaze beyond DEAD in TAU s."""
    return TRIM + GAIN[arm] * np.clip(np.sign(gaze) * np.maximum(np.abs(gaze) - DEAD, 0) / TAU / YAW_PER_TURN, -1, 1)


def pursue(net, rec, sp, rng, arm, view=None, log=None):
    """arm: body, noturn, flip, silenced. Returns per-spider (eye error, |gaze|, body yaw change, prey bearing change), last SCORE steps."""
    reset, step = net
    reset()
    B = len(sp)
    tgt0 = rng.uniform(-0.3, 0.3, B)
    with START:
        eyes = L.Eyes(tgt0, sp, T, view=view, row=view is None)  # row: AM_L's darkest pixel per column only, byte-identical (spider_eyerow_test)
    gaze, eye_v, prev, cost = np.zeros(B), np.zeros(B), None, np.zeros(3)
    err, gz, y0, b0, yaws, bears = [], [], None, None, [], []
    for k in range(T):
        t0 = time.perf_counter()
        turn = turn_law(gaze, arm)
        ctrl = None if arm == 'stand' else gait(k * L.P.DT, np.full(B, SPEED * min(1.0, k / RAMP) if RAMP else SPEED), turn).T
        now = L.receptors(rec, eyes.sense(gaze, ctrl))
        t1 = time.perf_counter()
        q = eyes.q
        head_yaw = yaw(q)
        bear = np.degrees(np.arctan2(q[-6] - q[1], q[-7] - q[0]))  # prey from the head, world frame
        if y0 is None:
            y0, b0, last_yaw = head_yaw.copy(), bear.copy(), head_yaw.copy()
        if REFLEX and arm != 'novor':  # counter-rotation reflex (off: it fights the position-trained pathway, 38d70f0)
            gaze = gaze - wrap(head_yaw - last_yaw)
        last_yaw = head_yaw.copy()
        err.append(wrap(bear - head_yaw - gaze))  # signed: split into slow (lag) and fast (wobble) parts below
        gz.append(np.abs(gaze))
        yaws.append(wrap(head_yaw - y0)); bears.append(wrap(bear - b0))
        t2 = time.perf_counter()
        d = step(L.P.motion(prev, now).astype(np.float32)) if prev is not None and arm != 'silenced' else np.zeros(B)
        cost += (t1 - t0, time.perf_counter() - t2, t2 - t1)
        if log is not None:  # per step, all animals: bearing, head yaw, gaze, eye command, turn (deg / units)
            log.append(np.stack([wrap(bear - b0), wrap(head_yaw - y0), gaze, d, turn]))
        prev = now
        eye_v = eye_v + L.P.KA * d
        gaze = gaze + eye_v * L.P.DT
    eyes.close()
    if PROF:
        print(f'PROFILE {arm}: s/step engine exchange {cost[0] / T:.4f}, brain {cost[1] / T:.4f}, bookkeeping {cost[2] / T:.4f}', flush=True)
    last = slice(T - SCORE, T)
    e = np.array(err)
    slow = np.apply_along_axis(lambda x: np.convolve(x, np.ones(25) / 25, 'same'), 0, e)  # 0.25 s mean: lag; the rest: stride wobble
    e, fast, slow = np.abs(e), np.abs(e - slow), np.abs(slow)
    return {'lost': np.where((e > 2.5).any(0), (e > 2.5).argmax(0), T), 'trace': np.median(e, 1),'err': np.median(e[last], 0), 'slow': np.median(slow[last], 0), 'fast': np.median(fast[last], 0), 'gaze': np.median(np.array(gz)[last], 0),
            'yaw': np.array(yaws)[-1], 'bearing': np.array(bears)[-1]}


def trace():
    """body arm, per-step log: where the eye's world direction leaves the prey, and what the eye command and body did."""
    rec = L.P.retina(np.random.default_rng(0))
    _, net = L.G.train('learn', 300, rec, 2, keep=True, pos=POS)
    sp = L.P.speeds(np.random.default_rng(31), 6, 9, L.G.B)
    log = []
    pursue(L.pathway(net), rec, sp, np.random.default_rng(37), 'body', log=log)
    net.close()
    a = np.array(log)  # (T, 5, B)
    np.save(L.G.ROOT / '.work/agent/spider/trace.npy', a)
    eye = a[:, 1] + a[:, 2]
    err = wrap(a[:, 0] - eye)
    print('step  bearing  yaw  gaze  eye_world  err  d  turn  (median over animals; err also p90)')
    for k in list(range(0, 60, 5)) + list(range(60, T, 40)):
        m = lambda x: float(np.median(x[k]))
        print(f'{k:4d} {m(a[:, 0]):8.2f} {m(a[:, 1]):6.2f} {m(a[:, 2]):6.2f} {m(eye):8.2f} {m(np.abs(err)):6.2f}/{np.percentile(np.abs(err[k]), 90):6.2f} {m(a[:, 3]):6.2f} {m(a[:, 4]):+.3f}')


def gait_rate():
    """no brain: body yaw rate (deg/s) per turn command, 2 s walk, 0.5-2 s scored; B spiders over turn -1..1."""
    B = L.G.B
    turn = np.linspace(-1, 1, B)
    eyes = L.Eyes(np.zeros(B), np.zeros(B), 200)
    ys = []
    for k in range(200):
        eyes.sense(np.zeros(B), gait(k * L.P.DT, np.full(B, SPEED), turn).T)
        ys.append([yaw(eyes.q[:, i]) for i in range(B)])
    eyes.close()
    y = np.unwrap(np.radians(np.array(ys)), axis=0)
    rate = np.degrees(y[-1] - y[50]) / 1.5
    GAIN_FILE.write_text(json.dumps({'speed': SPEED, 'yaw_per_turn': float(np.polyfit(turn, rate, 1)[0])}) + '\n')
    yd = np.degrees(y)
    np.save(L.G.ROOT / '.work/agent/spider/head-yaw.npy', yd)  # (steps, B) head yaw deg, 10 ms steps
    i0 = int(np.argmin(np.abs(turn - TRIM)))
    v = np.diff(yd[50:, i0]) / L.P.DT
    print(f'turn {turn[i0]:+.3f} (trim): yaw rate mean {v.mean():+.1f} deg/s, std {v.std():.1f} deg/s, stride wobble p-p {np.ptp(yd[50:, i0] - np.polyval(np.polyfit(np.arange(150), yd[50:, i0], 1), np.arange(150))):.2f} deg')
    for t, r in zip(turn[::4], rate[::4]):
        print(f'turn {t:+.2f} yaw rate {r:+.1f} deg/s')


def walk_wobble():
    """gait_rate's recorded head yaw (after 0.5 s), the near-straight animals only (|turn| <= 0.25): the stride wobble."""
    y = np.load(L.G.ROOT / '.work/agent/spider/head-yaw.npy')[50:]
    return y[:, np.abs(np.linspace(-1, 1, y.shape[1])) <= 0.25]


def diag():
    """stand vs walk-no-turn: step at which the prey leaves the 2.5 deg retina, median error trace every 0.5 s."""
    rec = L.P.retina(np.random.default_rng(0))
    _, net = L.G.train('learn', 300, rec, 2, keep=True)
    sp = L.P.speeds(np.random.default_rng(31), 6, 9, L.G.B)
    for arm in ('stand', 'noturn'):
        r = pursue(L.pathway(net), rec, sp, np.random.default_rng(37), arm)
        print(arm, 'lost at step median', np.median(r['lost']), 'trace', np.round(r['trace'][::50], 2).tolist(), 'yaw', np.round(np.median(np.abs(r['yaw'])), 1), flush=True)
    net.close()


def preflight():
    """CPU, no engine: a stub body (yaw rate = YAW_PER_TURN x turn) under turn_law, prey bearing moving 6-9 deg/s, a perfect
    eye (gaze = bearing - yaw). The body arm must turn with the bearing, the flip arm against it; all finite."""
    rng = np.random.default_rng(0)
    v = rng.uniform(6, 9, 64) * rng.choice([-1, 1], 64)
    out = {}
    for arm in ('body', 'flip'):
        y = np.zeros(64)
        for k in range(T):
            y = y + YAW_PER_TURN * turn_law(v * k * L.P.DT - y, arm) * L.P.DT
        out[arm] = float(np.mean(np.sign(y) == np.sign(v)))
    ok = out['body'] >= 0.9 and out['flip'] < 0.5 and np.isfinite(y).all()
    print(f"{'PASS' if ok else 'FAIL'} preflight turn law: body follows the bearing sign in {out['body']:.0%} (>= 90 %), flip {out['flip']:.0%} (< 50 %)")
    if not ok:
        sys.exit('FAIL preflight')


def gain_ok():
    g = json.loads(GAIN_FILE.read_text()) if GAIN_FILE.is_file() else {}
    if g.get('speed') != SPEED or abs(g['yaw_per_turn'] - YAW_PER_TURN) > 0.1 * abs(YAW_PER_TURN):
        sys.exit(f'FAIL turn gain: YAW_PER_TURN {YAW_PER_TURN} at speed {SPEED}, measured {g} ({GAIN_FILE}); run gait_rate and set it')


def check():
    preflight()
    gain_ok()
    if not DEBUG:
        from agent import seeds
        seeds.claim('spider_body_loop', SEEDS)
    rec = L.P.retina(np.random.default_rng(0))
    src = [f.read_bytes() for f in (Path(__file__), Path(L.__file__), Path(L.G.__file__), L.G.ROOT / 'agent/spider_pool.mojo', *sorted((L.G.ROOT / 'agent/body_engine').glob('*.mojo')))]
    key = lambda arm, *w: src + [repr((arm, SEEDS, T, SCORE, L.G.B, DEAD, TAU, POS, REFLEX)).encode(), *w]
    sp = L.P.speeds(np.random.default_rng(SEEDS[0]), 6, 9, L.G.B)
    h = L.G.cached('hand', key('hand'), lambda: pursue(hand_net(), rec, sp, np.random.default_rng(SEEDS[1]), 'hand'))
    hf = float(np.mean(np.sign(h['yaw']) == np.sign(h['bearing'])))
    lo, hi = ci(h['err'])
    print(f"hand-wired body arm: eye error {np.median(h['err']):.2f} deg (90 % CI {lo:.2f}-{hi:.2f}; slow {np.median(h['slow']):.2f}, fast {np.median(h['fast']):.2f}), "
          f"body follows the bearing in {hf:.0%}", flush=True)
    if lo < 1 <= hi:
        sys.exit(f'FAIL undecided: the hand arm CI {lo:.2f}-{hi:.2f} straddles 1 deg; more animals, not a verdict')
    if not (hi < 1 and hf >= 0.9):
        sys.exit('FAIL body cannot pursue: the hand-wired circuit misses G1/G3, a learner run would test the body, not the learning')
    _, net = L.G.train_cached('learn', 300, rec, 2, pos=POS)  # POS: it also learns to centre
    wkey = b''.join(np.ascontiguousarray(x).tobytes() for x in net.weights())
    arms, ref = ('body', 'noturn', 'flip', 'silenced'), ('noturn', 'silenced')  # ref: not the arm under test, cached
    nets = {a: L.G.clone(net) for a in arms}  # one resident pathway and one engine per arm, all arms at once
    one = lambda a: pursue(L.pathway(nets[a]), rec, sp, np.random.default_rng(SEEDS[1]), a)
    with ThreadPoolExecutor(len(arms)) as ex:
        r = dict(zip(arms, ex.map(lambda a: L.G.cached(a, key(a, wkey), lambda: one(a)) if a in ref else one(a), arms)))
    for n in [net, *nets.values()]:
        n.close()
    med = {a: {k: float(np.median(v)) if k in ('lost', 'trace') else float(np.median(np.abs(v)) if k != 'err' else np.median(v)) for k, v in x.items()} for a, x in r.items()}
    b = r['body']
    follow = np.sign(b['yaw']) == np.sign(b['bearing'])
    frac = np.abs(b['yaw']) / np.maximum(np.abs(b['bearing']), 1e-6)
    print(json.dumps(med))
    ok = [(f"G1 eye on prey while the body moves: median error {med['body']['err']:.2f} deg < 1", med['body']['err'] < 1),
          (f"G2 body took the turn: |gaze| {med['body']['gaze']:.2f} < 1/3 of no-turn {med['noturn']['gaze']:.2f}", med['body']['gaze'] < med['noturn']['gaze'] / 3),
          (f"G3 body yaw follows the prey bearing: sign in {follow.mean():.0%}, size median {np.median(frac):.0%} (>= 50 %)", follow.mean() >= 0.9 and np.median(frac) >= 0.5),
          (f"M1 mutant flipped turn: error {med['flip']['err']:.2f} vs {med['body']['err']:.2f} deg, or wrong-way body", med['flip']['err'] >= 3 * med['body']['err'] or
           np.mean(np.sign(r['flip']['yaw']) == np.sign(r['flip']['bearing'])) < 0.5),
          (f"C1 silenced brain: error {med['silenced']['err']:.2f} >= 5 x {med['body']['err']:.2f}", med['silenced']['err'] >= 5 * med['body']['err'])]
    for k, v in ok:
        print(('PASS ' if v else 'FAIL ') + k)
    (L.G.ROOT / f".work/agent/spider/body-loop{'-debug' if DEBUG else ''}.json").write_text(json.dumps(med, indent=2) + '\n')
    if not all(v for _, v in ok):
        sys.exit('FAIL spider orientation')
    print('PASS spider orientation: the eyes lead, the body follows')


if __name__ == '__main__':
    {'check': check, 'diag': diag, 'gait_rate': gait_rate, 'trace': trace, 'preflight': preflight}[sys.argv[1]]()
