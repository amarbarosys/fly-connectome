"""Step 3, slice 1: the principal-eye pathway of the designed spider scaffold (agent/spider_scaffold.py) learns pursuit
from empty synapses. One eye: AM_retina (1200) -> AM1 (2000) -> AM2 (2000) -> 12 eye-muscle motor neurons (6 pull right,
6 pull left; winsor2023: six muscles move each principal eye tube; the muscle split and the AM2 -> muscle tract are
inferred). Every synapse slot comes from the scaffold's tract fan-in (20 per AM1/AM2 neuron; 200 per motor neuron,
inferred) and starts at weight 0.

Same task, plant, neurons and gates as tools/spider_blank_pool_test.py (pursuit onset, eye moves only by the network,
speeds train 2-5, test 6-9 deg/s). Retina neurons: receptor k (in its 12-patch, 100-receptor layout) fires Poisson at
GAIN x the patch's Reichardt motion, receptors 0-49 of a patch for rightward, 50-99 for leftward motion (motion computed
at the retina is a model choice). Learning: each motor neuron gets the slip direction as its teaching signal; AM2 and
AM1 neurons get it through fixed random feedback (feedback alignment), gated by their own spikes; prune |w| < THETA and
regrow at a new random pre of the same tract at a fixed 30 % slot budget.
Gates: P1 learned < 1.5 x the hand-built pool-test circuit on held-out speeds; P2 random-sign feedback control >= 2x
learned; P3 before training as bad as silenced (> 2 x hand).
usage: .venv/bin/python tools/spider_scaffold_learn_test.py -> .work/agent/spider/scaffold-learn.json
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import spider_blank_pool_test as P  # noqa: E402
from spider_blank_pool_test import DT, GAIN, KA, KICK, NOISE_HZ, S, STEPS, SUB, T_MBR, T_RFC, TAU, V_TH, activation, motion, retina, speeds  # noqa: E402

N_RET, N1, N2, N_MN, F12, F_MN = 1200, 2000, 2000, 12, 20, 200
THETA, WMAX, ETA, FRAC, B = 0.02, 0.3, 0.01, 0.3, 32
ROOT = Path(__file__).resolve().parents[1]
CH = np.repeat(np.arange(12), 100), np.tile(np.r_[np.zeros(50, int), np.ones(50, int)], 12)  # receptor -> (patch, dir)


class Layer:
    """Sparse fan-in layer: post j reads pre idx[j, :] with weights w[j, :] (slots of the scaffold tract)."""

    def __init__(self, rng, n_post, n_pre, fan):
        self.rng, self.n_pre = rng, n_pre
        self.idx = np.stack([rng.choice(n_pre, fan, replace=False) for _ in range(n_post)]).astype(np.int32)
        self.w = np.zeros((n_post, fan), np.float32)
        self.fb = rng.choice([-1.0, 1.0], (n_post, 2)).astype(np.float32)

    def drive(self, pre_spikes):  # (B, n_pre) -> (B, n_post)
        return (pre_spikes[:, self.idx] * self.w).sum(-1) * KICK

    def rewire(self):
        np.clip(self.w, -WMAX, WMAX, out=self.w)
        dead = np.abs(self.w) < THETA
        self.w[dead] = 0
        need = int(FRAC * self.w.size) - int((~dead).sum())
        if need > 0:
            k = self.rng.choice(np.flatnonzero(dead), need, replace=False)
            self.idx.flat[k] = self.rng.integers(0, self.n_pre, need)  # regrow at a new random pre of the tract
            self.w.flat[k] = self.rng.choice([-1, 1], need) * 1.5 * THETA


class Pathway:
    def __init__(self, rng):
        self.rng = rng
        self.l1, self.l2, self.lm = Layer(rng, N1, N_RET, F12), Layer(rng, N2, N1, F12), Layer(rng, N_MN, N2, F_MN)
        self.lm.fb[:] = np.r_[np.tile([1, -1], (6, 1)), np.tile([-1, 1], (6, 1))]  # motor: its own direction, not random

    def reset(self, n):
        z = lambda k: np.zeros((n, k), np.float32)
        self.u = [z(N1), z(N2), z(N_MN)]; self.r = [z(N1), z(N2), z(N_MN)]
        self.tr = [z(N_RET), z(N1), z(N2)]

    def step(self, m, silence=False, learn=False):
        n, f = len(m), np.float32
        rate = GAIN * SUB * np.where(CH[1] == 0, np.clip(m, 0, None)[:, CH[0]], np.clip(-m, 0, None)[:, CH[0]]).astype(f)
        decay, dk = f(np.exp(-SUB / T_MBR)), f(np.exp(-SUB / TAU))
        pre_sum = [np.zeros((n, N_RET), f), np.zeros((n, N1), f), np.zeros((n, N2), f)]
        act = [np.zeros((n, N1), f), np.zeros((n, N2), f), np.zeros((n, N_MN), f)]
        for _ in range(S):
            x = (self.rng.random((n, N_RET), f) < rate).astype(f)
            spikes_in = x
            for li, layer in enumerate((self.l1, self.l2, self.lm)):
                self.tr[li] *= dk; self.tr[li] += spikes_in; pre_sum[li] += self.tr[li]
                noise = (self.rng.random(self.u[li].shape, f) < NOISE_HZ * SUB) * KICK
                free = self.r[li] <= 0
                self.u[li] *= decay; self.u[li] += layer.drive(spikes_in) + noise; self.u[li] *= free
                sp = self.u[li] > V_TH
                self.u[li][sp] = 0; self.r[li] -= f(SUB); self.r[li][sp] = f(T_RFC)
                act[li] += sp
                spikes_in = sp.astype(f)
        self.pre_sum, self.act = pre_sum, act
        c = act[2]
        d = c[:, :6].sum(1) - c[:, 6:].sum(1)
        return np.zeros(n) if silence else d.astype(np.float64)

    def teach(self, slip):
        err = np.stack([slip, -slip], 1).astype(np.float32) / 10
        for li, layer in enumerate((self.l1, self.l2, self.lm)):
            h = (err @ layer.fb.T) * (self.act[li] > 0)  # (n, post)
            pre = self.pre_sum[li][:, layer.idx]  # (n, post, fan)
            layer.w += ETA * np.einsum('np,npk->pk', h, pre) / len(slip)
            layer.rewire()


def run(net, rec, sp, rng, silence=False, learn=False, shuffle=False):
    n = len(sp)
    net.reset(n)
    tgt0, gaze, eye_v, prev, errs = rng.uniform(-0.3, 0.3, n), np.zeros(n), np.zeros(n), None, []
    for k in range(STEPS):
        tgt = tgt0 + sp * k * DT
        now = activation(rec, tgt - gaze)
        d = net.step(motion(prev, now), silence) if prev is not None else np.zeros(n)
        prev = now
        eye_v = eye_v + KA * d
        gaze = gaze + eye_v * DT
        if learn and k > 1:
            slip = sp - eye_v
            net.teach(slip * rng.choice([-1, 1], n) if shuffle else slip)
        errs.append(np.abs(tgt + sp * DT - gaze))
    return np.median(np.array(errs)[15:], 0)


def test(net, rec, seed=9):
    return float(np.median(run(net, rec, speeds(np.random.default_rng(seed), 6, 9, 128), np.random.default_rng(seed))))


def train(arm, iters=300):
    rec = retina(np.random.default_rng(0))
    net, rng = Pathway(np.random.default_rng(2)), np.random.default_rng(3)
    before, curve = test(net, rec), []
    for it in range(iters):
        run(net, rec, speeds(rng, 2, 5, B), rng, learn=True, shuffle=arm == 'shuffle')
        if it % 25 == 24:
            curve.append(round(test(net, rec, 11), 2)); print(arm, it + 1, curve[-1], flush=True)
    silenced = float(np.median(run(net, rec, speeds(np.random.default_rng(9), 6, 9, 128), np.random.default_rng(9), silence=True)))
    return {'before': before, 'after': test(net, rec), 'silenced': silenced, 'curve': curve}


def main():
    rec = retina(np.random.default_rng(0))
    hand = P.Net(np.random.default_rng(1), hand=True)
    e_hand = float(np.median(P.run(hand, rec, speeds(np.random.default_rng(9), 6, 9, 128), np.random.default_rng(9))))
    with ProcessPoolExecutor(2) as ex:
        out = dict(zip(('learn', 'shuffle'), ex.map(train, ('learn', 'shuffle'))))
    out['hand'] = e_hand
    p1 = out['learn']['after'] < 1.5 * e_hand
    p2 = out['shuffle']['after'] >= 2 * out['learn']['after']
    p3 = out['learn']['before'] > 2 * e_hand
    out['gates'] = {'P1': p1, 'P2': p2, 'P3': p3}
    (ROOT / '.work/agent/spider').mkdir(parents=True, exist_ok=True)
    (ROOT / '.work/agent/spider/scaffold-learn.json').write_text(json.dumps(out, indent=2) + '\n')
    print(f"P1 scaffold AM pathway learned (held-out 6-9 deg/s): {out['learn']['after']:.2f} vs hand {e_hand:.2f} -> {'PASS' if p1 else 'FAIL'}")
    print(f"P2 random-sign feedback control {out['shuffle']['after']:.2f} -> {'PASS' if p2 else 'FAIL'}")
    print(f"P3 before training {out['learn']['before']:.2f} (silenced {out['learn']['silenced']:.2f}) -> {'PASS' if p3 else 'FAIL'}")
    bad = 3 - p1 - p2 - p3
    if bad:
        print(f'FAIL {bad}/3 checks'); sys.exit(1)
    print('PASS scaffold slice: the empty principal-eye pathway wired itself for pursuit')


if __name__ == '__main__':
    main()
