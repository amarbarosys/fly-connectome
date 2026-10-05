"""The first real closed loop for the spider (docs/plans/SPIDER_BRAIN.md U2): rendered principal eye -> self-wiring brain ->
eye-muscle motor neurons -> gaze -> the next rendered frame.

Per 10 ms control step (live.mojo -DBRAIN=1 -DEYES=1 -DGAZE=1, K = 100 physics steps, the spider standing on its stance
servos): the engine moves the prey (prey_set, the ENVIRONMENT), turns both AM eyes by each animal's gaze yaw and renders
them; the host reads AM_L's darkest pixel per column (a vertical strip, like the real principal retina), turns each of the pathway's 1,200 receptors into an activation by blurring the
row's darkness with the receptor's own 0.15 deg tuning at its angle (no prey coordinates: pixels only), and the pathway
(agent/spider_pool_gpu.py, the whole 102,512-neuron scaffold) steps on the Reichardt motion of two frames; its eye motor
counts move the gaze (KA x d, as in training) for the next frame.

check  (a) rendered activation vs the angle model P.activation(tgt - gaze) at 32 prey angles, eyes turned +-1 deg, prey still:
           correlation >= 0.9 and the peak receptor's angle within one pixel (0.055 deg) of the model's in every animal (foveal
           receptors are 0.005 deg apart, 10 per pixel, so a receptor-count bar would ask for sub-pixel rendering)
       (b) closed loop in the engine on held-out speeds 6-9 deg/s, 3 seeds: the GPU-trained pathway's foveal error
           < 1.5 x the hand-wired circuit's in the same loop and >= 2x below the random-sign-trained control; mutant:
           the flipped-teacher pathway >= 2 x learned
usage: gpu-wait run --shared --vram 4 --priority 60 --timeout 7200 --env PYTHONPATH=$PWD -- .venv/bin/python -m agent.spider_eye_loop check [ITERS]
"""
import json
import subprocess
import sys

import mujoco
import numpy as np

from agent import spider_pool_gpu as G
from agent.body_engine import spider_eyes as E
from agent.body_engine.spider_link import CAM, build, files, sizes
from agent.body_engine.spider_gait import NU

P = G.P
W, H, K = 256, 128, 100  # AM_L: 256 px over 14 deg (0.055 deg/px, receptor tuning 0.15 deg), 128 rows (+-3.5 deg: the standing head pitches ~1 deg); K x 0.1 ms = the pathway's 10 ms step
DIST = E.PREY_R / np.tan(np.radians(0.15))  # prey 191 mm out: its disc subtends the receptor tuning width
BG = 0.6 * 255


def env_rows(head, quat, tgt0, sp):
    """per animal (x0, y0, z0, vx, vy, t_on): the prey at azimuth tgt0 deg of AM_L's axis, moving at sp deg/s across it."""
    rows = []
    for a, s in zip(tgt0, sp):
        p0 = E.place(head, quat, 'AM_L', a, 0, DIST)
        v = (E.place(head, quat, 'AM_L', a + 1e-3, 0, DIST) - p0) / 1e-3 * s  # mm per deg x deg/s
        rows.append((*p0, v[0], v[1], 0.0))
    return rows


class Eyes:
    """One engine run of `steps` frames for B spiders; sense(gaze) sends this frame's ctrl + gaze and returns AM_L's row."""
    def __init__(self, tgt0, sp, steps):
        self.d = files(G.B, 'eyeloop', prey=True)
        q0 = np.fromfile(self.d / 'qpos.bin', np.float32).reshape(-1, G.B)
        head, quat = q0[:3, 0].astype(float), q0[3:7, 0].astype(float)
        E.write_tables(self.d, env_rows(head, quat, tgt0, sp), npix=W)
        self.ctrl = np.fromfile(self.d / 'ctrl.bin', np.float32).reshape(NU, G.B)  # the stance, held
        self.nq, self.nv, self.ns = sizes(self.d)
        b = build(self.d, G.B, K, 1, w=W, h=H, eyes=True, gaze=True)
        self.p = subprocess.Popen([str(b), str(self.d) + '/', str(steps), *CAM, '0', '1'], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def sense(self, gaze):
        self.p.stdin.write(np.ascontiguousarray(self.ctrl).tobytes() + np.asarray(gaze, np.float32).tobytes())
        self.p.stdin.flush()
        self.p.stdout.read(4 * (NU + self.nq + self.nv + self.ns) * G.B)  # the body packet: unused here
        ret = np.frombuffer(self.p.stdout.read(G.B * E.NE * H * W), np.uint8).reshape(G.B, E.NE, H, W)
        return ret[:, 0].min(1).astype(np.float32)  # AM_L darkest pixel per column (a vertical strip retina), left edge = + azimuth

    def close(self):
        self.p.stdin.close()
        if self.p.wait() != 0:
            raise SystemExit(f'FAIL eye loop: engine exited {self.p.returncode}')


def receptors(rec, row):
    """(B, 1200) activations in the brain's wired order from AM_L's column darkness: darkness blurred by each receptor's tuning."""
    true, wired = rec
    f = W / 2 / np.tan(np.radians(14.0) / 2)
    ang = -np.degrees(np.arctan((np.arange(W) + 0.5 - W / 2) / f))  # pixel column -> azimuth (+ left)
    dark = np.clip((BG - row) / BG, 0, 1)
    g = np.exp(-0.5 * ((true[:, None] - ang[None]) / 0.15) ** 2).astype(np.float32)  # (1200, W)
    a = dark @ g.T
    a /= np.maximum(a.max(1, keepdims=True), 1e-6)  # peak 1 where the prey is, like the angle model
    a *= (a.max(1, keepdims=True) > 1e-3)
    return a[:, np.argsort(wired)].astype(np.float32)


def loop(net, rec, sp, rng):
    """the run() loop of spider_pool_gpu with the engine's eyes; net = (reset(), step(motion) -> d per animal). Returns the median foveal error."""
    reset, step = net
    reset()
    tgt0 = rng.uniform(-0.3, 0.3, len(sp))
    eyes = Eyes(tgt0, sp, P.STEPS)
    gaze, eye_v, prev, errs = np.zeros(len(sp)), np.zeros(len(sp)), None, []
    for k in range(P.STEPS):
        now = receptors(rec, eyes.sense(gaze))
        d = step(P.motion(prev, now).astype(np.float32)) if prev is not None else np.zeros(len(sp))
        prev = now
        eye_v = eye_v + P.KA * d
        gaze = gaze + eye_v * P.DT
        errs.append(np.abs(tgt0 + sp * (k + 1) * P.DT - gaze))  # truth for scoring only, never to the brain
    eyes.close()
    return np.median(np.array(errs)[15:], 0)


def pathway(net):
    def step(m):
        c = net.step(m).astype(np.float64)
        return (c[:, :6].sum(1) - c[:, 6:].sum(1)) / 6
    return net.reset, step


def test(net, rec, seed):
    return float(np.median(np.concatenate([loop(net, rec, P.speeds(np.random.default_rng(seed + q), 6, 9, G.B), np.random.default_rng(seed + q)) for q in range(2)])))


def render_gate(rec):
    """(a): still prey at 32 angles, the eyes turned +-1 deg (gaze sign checked), one engine run of 2 frames."""
    tgt = np.linspace(-1.2, 1.2, G.B)
    gaze = np.where(np.arange(G.B) % 2, 1.0, -1.0)  # the eyes turned +-1 deg: the prey must land at tgt - gaze
    eyes = Eyes(tgt, np.zeros(G.B), 2)
    got = receptors(rec, eyes.sense(gaze))
    eyes.sense(gaze)
    eyes.close()
    want = P.activation(rec, tgt - gaze)
    corr = np.array([np.corrcoef(g, w)[0, 1] for g, w in zip(got, want)])
    ang = rec[0]  # unshuffled retina: wired order = angle order
    peak = np.abs(ang[np.argmax(got, 1)] - ang[np.argmax(want, 1)])
    px = 14.0 / W
    ok = corr.min() >= 0.9 and peak.max() <= px
    print(f"{'PASS' if ok else 'FAIL'} (a) rendered AM_L vs angle model at 32 angles: corr min {corr.min():.3f} median {np.median(corr):.3f}, "
          f"peak angle off by max {peak.max():.4f} deg (bar one pixel {px:.4f})")
    return ok


def check(iters='300'):
    rec = P.retina(np.random.default_rng(0))
    bad = int(not render_gate(rec))
    hand = P.Net(np.random.default_rng(1), hand=True)
    e_hand = test((lambda: hand.trial_reset(G.B), hand.step), rec, 9)
    res = {'hand_engine': e_hand}
    for seed in (2, 3, 4):
        r = {}
        for arm in ('learn', 'shuffle', 'flip'):
            sim, net = G.train(arm, int(iters), rec, seed, keep=True)
            r[arm] = {'sim_after': sim['after'], 'engine': test(pathway(net), rec, 9)}
            net.close()
        L = r['learn']['engine']
        checks = {'B1 learned < 1.5 hand (engine)': L < 1.5 * e_hand, 'B2 random-sign >= 2 learned (engine)': r['shuffle']['engine'] >= 2 * L,
                  'M1 mutant flipped teacher >= 2 learned (engine)': r['flip']['engine'] >= 2 * L}
        print(f"seed {seed}: engine learned {L:.3f} random-sign {r['shuffle']['engine']:.3f} flipped {r['flip']['engine']:.3f} hand {e_hand:.3f} deg; "
              f"angle model learned {r['learn']['sim_after']:.3f}", flush=True)
        for k, v in checks.items():
            print(f"{'PASS' if v else 'FAIL'} seed {seed} {k}")
        bad += sum(not v for v in checks.values())
        res[seed] = r
    (G.ROOT / '.work/agent/spider/eye-loop.json').write_text(json.dumps(res, indent=2) + '\n')
    if bad:
        sys.exit(f'FAIL {bad} checks')
    print('PASS spider closed loop: rendered eyes -> brain -> eye muscles -> next frame')


if __name__ == '__main__':
    {'check': check}[sys.argv[1]](*sys.argv[2:])
