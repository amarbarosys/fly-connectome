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
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from agent import spider_eye_loop as L
from agent.body_engine.spider_gait import gait

T, SCORE, SPEED = 500, 200, 0.6
START = threading.Lock()  # parallel arms: model files and engine builds are made one at a time
YAW_PER_TURN, TRIM, TAU = -140.0, 0.0, 0.25
POS = 5.0  # 1/s, the teacher's centring term (0.2 s); set before the run, not tuned
REFLEX = False  # 95fa88a added it; with the position-trained pathway the reflex-off arm held 0.54 / 0.60 deg, the reflex arm lost the prey (38d70f0)
SEEDS = (41, 43)  # prey speeds, trial stream: new for the reflex-free claim (31, 37 were used while choosing the design)  # gait_rate (no brain): yaw deg/s per unit turn (sign: + turn yaws right), turn for zero yaw; body closes the gaze in TAU s


def yaw(q):
    w, x, y, z = q[3:7]
    return np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def wrap(a):
    return (a + 180) % 360 - 180


def pursue(net, rec, sp, rng, arm, view=None, log=None):
    """arm: body, noturn, flip, silenced. Returns per-spider (eye error, |gaze|, body yaw change, prey bearing change), last SCORE steps."""
    reset, step = net
    reset()
    B = len(sp)
    tgt0 = rng.uniform(-0.3, 0.3, B)
    with START:
        eyes = L.Eyes(tgt0, sp, T, view=view)
    gaze, eye_v, prev = np.zeros(B), np.zeros(B), None
    err, gz, y0, b0, yaws, bears = [], [], None, None, [], []
    for k in range(T):
        turn = TRIM + {'body': 1.0, 'noturn': 0.0, 'flip': -1.0, 'silenced': 1.0, 'stand': 0.0, 'novor': 1.0}[arm] * np.clip(gaze / TAU / YAW_PER_TURN, -1, 1)
        ctrl = None if arm == 'stand' else gait(k * L.P.DT, np.full(B, SPEED), turn).T
        now = L.receptors(rec, eyes.sense(gaze, ctrl))
        q = eyes.q
        head_yaw = np.array([yaw(q[:, i]) for i in range(B)])
        bear = np.degrees(np.arctan2(q[-6] - q[1], q[-7] - q[0]))  # prey from the head, world frame
        if y0 is None:
            y0, b0, last_yaw = head_yaw.copy(), bear.copy(), head_yaw.copy()
        if REFLEX and arm != 'novor':  # counter-rotation reflex (off: it fights the position-trained pathway, 38d70f0)
            gaze = gaze - wrap(head_yaw - last_yaw)
        last_yaw = head_yaw.copy()
        err.append(np.abs(wrap(bear - head_yaw - gaze)))
        gz.append(np.abs(gaze))
        yaws.append(wrap(head_yaw - y0)); bears.append(wrap(bear - b0))
        d = step(L.P.motion(prev, now).astype(np.float32)) if prev is not None and arm != 'silenced' else np.zeros(B)
        if log is not None:  # per step, all animals: bearing, head yaw, gaze, eye command, turn (deg / units)
            log.append(np.stack([wrap(bear - b0), wrap(head_yaw - y0), gaze, d, turn]))
        prev = now
        eye_v = eye_v + L.P.KA * d
        gaze = gaze + eye_v * L.P.DT
    eyes.close()
    last = slice(T - SCORE, T)
    e = np.array(err)
    return {'lost': np.where((e > 2.5).any(0), (e > 2.5).argmax(0), T), 'trace': np.median(e, 1),'err': np.median(np.array(err)[last], 0), 'gaze': np.median(np.array(gz)[last], 0),
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


def check():
    rec = L.P.retina(np.random.default_rng(0))
    _, net = L.G.train_cached('learn', 300, rec, 2, pos=POS)  # POS: it also learns to centre
    sp = L.P.speeds(np.random.default_rng(SEEDS[0]), 6, 9, L.G.B)
    arms = ('body', 'noturn', 'flip', 'silenced')
    nets = {a: L.G.clone(net) for a in arms}  # one resident pathway and one engine per arm, all arms at once
    with ThreadPoolExecutor(len(arms)) as ex:
        r = dict(zip(arms, ex.map(lambda a: pursue(L.pathway(nets[a]), rec, sp, np.random.default_rng(SEEDS[1]), a), arms)))
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
    (L.G.ROOT / '.work/agent/spider/body-loop.json').write_text(json.dumps(med, indent=2) + '\n')
    if not all(v for _, v in ok):
        sys.exit('FAIL spider orientation')
    print('PASS spider orientation: the eyes lead, the body follows')


if __name__ == '__main__':
    {'check': check, 'diag': diag, 'gait_rate': gait_rate, 'trace': trace}[sys.argv[1]]()
