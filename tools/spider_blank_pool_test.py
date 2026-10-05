"""Blank neuron pool that wires itself for spider pursuit (Brain idea 2026-10-04, empty-neuron-pool-idea).

Task that NEEDS the neurons (the prerequisite the E2 test lacked): pursuit onset. The target appears on the fovea moving
10-40 deg/s; the eye moves ONLY by the network's motor command (no centroid position fallback, no efference copy),
for 200 ms. Retina = tools/spider_principal_eye_test.retina (1200 receptors over 5 deg). Input: 12 local Reichardt
patches (100 receptors each, wired order), each split into R and L Poisson populations (rate = GAIN x motion+/-).
Eye plant: the two motor neurons' (R - L) spikes per 10 ms change eye VELOCITY by KA deg/s each (velocity
integrator in the oculomotor plant), so pursuit holds when retinal slip reaches zero. Speeds stay <= 10 deg/s, the
Reichardt linear range at 10 ms (tools/spider_eye_neurons_test.py). Neurons: Shiu 2024 LIF (agent/shiu.py values,
no spider recordings exist), 0.5 ms substep.

Arms (same pool size 80, same inputs, same motor readout):
  hand    hand-built circuit: every patch's R population -> motor R, L -> motor L (no pool), fixed
  blank   empty pool: no synapses at start; teaching signal per output neuron = retinal slip direction (climbing-
          fibre-like: motor R learns pre-trace x slip, motor L x -slip), pool neurons get it through fixed random
          feedback (feedback alignment), gated by their own spikes; plus structural
          plasticity at a fixed synapse budget; reward every 10 ms = -|target speed - eye speed| after the command,
          advantage = reward minus the batch mean over its std; weights bounded |w| <= WMAX: |w| < THETA pruned and regrown at a random empty slot
          (DeepST / DeepR rewiring, corpus 2307.11735, 1912.12047)
  shuffle the same pool and rule, but each trial's slip sign is randomised every step (same magnitude, direction
          decoupled), the control. (Rolling feedback across trials was tried first and is NOT a control: it still
          learned to 0.15 deg. Measured: true 0.14, zero error 1.24, random sign 0.44, flipped 5.96 deg.)
Gates (median foveal error at 150-200 ms):
  P0 the task needs neurons: hand intact < 1 deg and hand silenced >= 3x hand
  P1 blank after training < 1.5 x hand on HELD-OUT speeds (train 2-5, test 6-9 deg/s, both directions)
  P2 blank beats the random-sign feedback control by >= 2x on held-out speeds
  P3 before training the blank pool is as bad as silenced (> 2 x hand): learning, not luck
A scalar reward (three-factor, Net.learn with a reward) did NOT learn at pool sizes 2, 8 or 80 (1.32-1.40 deg after
150 batches), so it is kept only as the reward path of Net.learn.
Claim if it passes: a plastic net learns spider-like pursuit. Never spider anatomy.
usage: .venv/bin/python tools/spider_blank_pool_test.py  -> .work/agent/spider/blank-pool.json
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from spider_principal_eye_test import DT, WIN, retina  # noqa: E402

V_TH, T_MBR, TAU, T_RFC, W_SYN, F_POI = 7.0, 20e-3, 5e-3, 2.2e-3, 0.275, 250  # agent/shiu.py
SUB, PATCH, GAIN, KA = 5e-4, 12, 400.0, 1.5
POOL, BUDGET, THETA, ETA, NOISE_HZ, WMAX = 80, 600, 0.02, 0.01, 30.0, 0.3  # WMAX: 0.3 x 68.75 = 21 mV per spike
STEPS, B = 20, 64  # 200 ms trials, 64 trials per batch
ROOT = Path(__file__).resolve().parents[1]


def activation(rec, off):
    """Receptor activations (n, 1200) in the brain's wired order, float32."""
    true, wired = rec
    a = np.exp(-0.5 * ((true[None].astype(np.float32) - off[:, None].astype(np.float32)) / np.float32(.15)) ** 2)
    a *= (np.abs(off) <= WIN / 2)[:, None]
    return a[:, np.argsort(wired)]


def motion(a, b):
    """Per-patch Reichardt output (n, PATCH) from activations at two times (prev a, now b)."""
    m = a[:, :-1] * b[:, 1:] - a[:, 1:] * b[:, :-1]
    return m[:, :PATCH * (m.shape[1] // PATCH)].reshape(len(a), PATCH, -1).sum(2)


S = int(round(DT / SUB))  # substeps per 10 ms
KICK = np.float32(W_SYN * F_POI)


class Net:
    """Inputs (2 x PATCH Poisson populations, one unit each) -> pool -> 2 motor LIF. W_in (POOL, 2P), W_out (2, POOL).
    hand=True: inputs straight to motor (R patches -> motor 0, L -> motor 1). float32; per 10 ms one draw of every
    random number and one eligibility contraction (einsum over the substeps), not one outer product per substep."""

    def __init__(self, rng, hand=False, pool=None, teach=False):
        global POOL
        POOL = pool or POOL
        self.rng, self.hand, self.teach = rng, hand, teach
        self.fb = rng.choice([-1.0, 1.0], (POOL, 2)).astype(np.float32)  # fixed random feedback: motor error -> pool
        self.w_in = np.zeros((POOL, 2 * PATCH), np.float32); self.w_out = np.zeros((2, POOL), np.float32)
        self.base = 0.0

    def trial_reset(self, n):
        f = np.float32
        self.up, self.um = np.zeros((n, POOL), f), np.zeros((n, 2), f)
        self.rp, self.rm = np.zeros((n, POOL), f), np.zeros((n, 2), f)
        self.tr_in, self.tr_pool = np.zeros((n, 2 * PATCH), f), np.zeros((n, POOL), f)
        self.e_in, self.e_out = np.zeros((n, POOL, 2 * PATCH), f), np.zeros((n, 2, POOL), f)
        self.rate_p, self.rate_m = np.full(POOL, NOISE_HZ * SUB, f), np.full(2, NOISE_HZ * SUB, f)
        self.h_in, self.h_pool = np.empty((S, n, 2 * PATCH), f), np.empty((S, n, POOL), f)  # trace history
        self.h_sp, self.h_sm = np.empty((S, n, POOL), f), np.empty((S, n, 2), f)  # spike history

    def step(self, m, silence=False, learn=False):
        """10 ms; m (n, PATCH) motion -> (n,) R - L motor spikes."""
        n, f = len(m), np.float32
        rates = (GAIN * SUB * np.concatenate([np.clip(m, 0, None), np.clip(-m, 0, None)], 1)).astype(f)
        decay, dk = f(np.exp(-SUB / T_MBR)), f(np.exp(-SUB / TAU))
        X = (self.rng.random((S, n, 2 * PATCH), f) < rates).astype(f)  # every input spike of the 10 ms
        NM = (self.rng.random((S, n, 2), f) < NOISE_HZ * SUB) * KICK
        if not self.hand:
            NP = (self.rng.random((S, n, POOL), f) < NOISE_HZ * SUB) * KICK  # spontaneous pool spikes: exploration
            DRIVE = (X.reshape(S * n, -1) @ (self.w_in.T * KICK)).reshape(S, n, POOL) + NP  # input -> pool, all substeps
        for t in range(S):
            x = X[t]
            self.tr_in *= dk; self.tr_in += x; self.h_in[t] = self.tr_in
            if self.hand:
                drive_m = KICK * np.stack([x[:, :PATCH].sum(1), x[:, PATCH:].sum(1)], 1)
            else:
                free = self.rp <= 0
                self.up *= decay; self.up += DRIVE[t]; self.up *= free
                sp = self.up > V_TH
                self.up[sp] = 0; self.rp -= f(SUB); self.rp[sp] = f(T_RFC)
                self.h_sp[t] = sp
                self.tr_pool *= dk; self.tr_pool += sp; self.h_pool[t] = self.tr_pool
                drive_m = self.h_sp[t] @ (self.w_out.T * KICK)
            drive_m = drive_m + NM[t]
            free = self.rm <= 0
            self.um *= decay; self.um += drive_m; self.um *= free
            sm = self.um > V_TH
            self.um[sm] = 0; self.rm -= f(SUB); self.rm[sm] = f(T_RFC)
            self.h_sm[t] = sm
        if learn and not self.hand and self.teach:  # teaching mode: keep this step's summed traces and pool activity
            self.s_in, self.s_pool, self.s_sp = self.h_in.sum(0), self.h_pool.sum(0), self.h_sp.sum(0)
        elif learn and not self.hand:  # eligibility: pre trace x (post spike - mean rate), summed over the substeps
            self.e_in += np.einsum('snp,sni->npi', self.h_sp - self.rate_p, self.h_in, optimize=True)
            self.e_out += np.einsum('snm,snp->nmp', self.h_sm - self.rate_m, self.h_pool, optimize=True)
            self.rate_p = f(0.98) * self.rate_p + f(0.02) * self.h_sp.mean((0, 1))
            self.rate_m = f(0.98) * self.rate_m + f(0.02) * self.h_sm.mean((0, 1))
        count = self.h_sm.sum(0)
        return np.zeros(n) if silence else (count[:, 0] - count[:, 1]).astype(np.float64)

    def teach_step(self, slip):
        """Per-direction error (retinal slip, deg/s, + = target faster rightward): motor R should fire more when
        slip > 0, motor L when slip < 0 (a climbing-fibre-like signal per output neuron). Pool neurons get the motor
        error through fixed random feedback (feedback alignment), gated by their own activity. Local and blank."""
        err = np.stack([slip, -slip], 1).astype(np.float32) / 10  # (n, 2)
        self.w_out += ETA * np.einsum('nm,np->mp', err, self.s_pool) / len(slip)
        h = (err @ self.fb.T) * (self.s_sp > 0)  # (n, POOL)
        self.w_in += ETA * np.einsum('np,ni->pi', h, self.s_in) / len(slip)
        self.learn(None)

    def learn(self, reward):
        """Three-factor update over the batch, then structural rewiring at a fixed budget."""
        if reward is not None:
            adv = reward - self.base
            self.base = 0.9 * self.base + 0.1 * reward.mean()
            a = ((adv - adv.mean()) / (adv.std() + 1e-9)).astype(np.float32)  # normalized advantage
            self.w_in += ETA * np.einsum('n,npi->pi', a, self.e_in) / len(a)
            self.w_out += ETA * np.einsum('n,nmp->mp', a, self.e_out) / len(a)
            self.e_in[:] = 0; self.e_out[:] = 0  # eligibility of this step consumed
        for w in (self.w_in, self.w_out):
            np.clip(w, -WMAX, WMAX, out=w)  # bounded synapses (weights exploded to |w| 472 without it)
            alive = np.abs(w) >= THETA
            w[~alive] = 0.0
            free = np.flatnonzero(~alive)
            need = int(BUDGET * w.size / (self.w_in.size + self.w_out.size)) - int(alive.sum())
            if need > 0 and len(free):  # regrow weak new synapses at random empty slots
                k = self.rng.choice(free, min(need, len(free)), replace=False)
                w.flat[k] = self.rng.choice([-1, 1], len(k)) * 1.5 * THETA


def run(net, rec, speeds, rng, silence=False, learn=False, shuffle=False):
    n = len(speeds)
    net.trial_reset(n)
    tgt0, gaze, eye_v, prev = rng.uniform(-0.3, 0.3, n), np.zeros(n), np.zeros(n), None
    errs, rew = [], np.zeros(n)
    for k in range(STEPS):
        tgt = tgt0 + speeds * k * DT
        off = tgt - gaze
        now = activation(rec, off)
        d = net.step(motion(prev, now), silence, learn) if prev is not None else np.zeros(n)
        prev = now  # reused as the next step's previous frame
        eye_v = eye_v + KA * d
        gaze = gaze + eye_v * DT
        if learn and prev is not None and k > 1:  # reward every 10 ms: retinal slip left after this step's command
            if net.teach:
                slip = speeds - eye_v
                net.teach_step(slip * rng.choice([-1, 1], n) if shuffle else slip)  # control: sign decoupled per trial
            else:
                r = -np.abs(speeds - eye_v)
                net.learn(np.roll(r, 1) if shuffle else r)
        e = np.abs(tgt + speeds * DT - gaze)
        errs.append(e); rew -= e
    return np.median(np.array(errs)[15:], 0)


def speeds(rng, lo, hi, n=B):
    return rng.uniform(lo, hi, n) * rng.choice([-1, 1], n)


def train(arm, iters=400, pool=None, teach=True):
    rec = retina(np.random.default_rng(0))
    test = lambda net, seed=9: float(np.median(run(net, rec, speeds(np.random.default_rng(seed), 6, 9, 256), np.random.default_rng(seed))))
    net, rng = Net(np.random.default_rng(2), pool=pool, teach=teach), np.random.default_rng(3)
    before, curve = test(net), []
    for it in range(iters):
        run(net, rec, speeds(rng, 2, 5), rng, learn=True, shuffle=arm == 'shuffle')
        if it % 50 == 49:
            curve.append(round(test(net, 11), 2)); print(arm, it + 1, curve[-1], flush=True)
    return {'before': before, 'after': test(net), 'curve': curve, 'synapses': int((net.w_in != 0).sum() + (net.w_out != 0).sum())}


def main():
    rec = retina(np.random.default_rng(0))
    test = lambda net, seed=9: float(np.median(run(net, rec, speeds(np.random.default_rng(seed), 6, 9, 256), np.random.default_rng(seed))))
    hand = Net(np.random.default_rng(1), hand=True)
    e_hand = test(hand)
    e_sil = float(np.median(run(hand, rec, speeds(np.random.default_rng(9), 6, 9, 256), np.random.default_rng(9), silence=True)))
    out = {'hand': e_hand, 'silenced': e_sil}
    with ProcessPoolExecutor(2) as ex:  # the two arms are independent: one core each
        out |= dict(zip(('blank', 'shuffle'), ex.map(train, ('blank', 'shuffle'))))
    p0 = e_hand < 1 and e_sil >= 3 * e_hand
    p1 = out['blank']['after'] < 1.5 * e_hand
    p2 = out['shuffle']['after'] >= 2 * out['blank']['after']
    p3 = out['blank']['before'] > 2 * e_hand
    out['gates'] = {'P0': p0, 'P1': p1, 'P2': p2, 'P3': p3}
    (ROOT / '.work/agent/spider').mkdir(parents=True, exist_ok=True)
    (ROOT / '.work/agent/spider/blank-pool.json').write_text(json.dumps(out, indent=2) + '\n')
    print(f"P0 task needs neurons: hand {e_hand:.2f} deg, silenced {e_sil:.2f} -> {'PASS' if p0 else 'FAIL'}")
    print(f"P1 blank pool after training (held-out 6-9 deg/s): {out['blank']['after']:.2f} vs hand {e_hand:.2f} -> {'PASS' if p1 else 'FAIL'}")
    print(f"P2 vs shuffled-feedback control {out['shuffle']['after']:.2f} -> {'PASS' if p2 else 'FAIL'}")
    print(f"P3 blank before training {out['blank']['before']:.2f} -> {'PASS' if p3 else 'FAIL'}")
    bad = 4 - p0 - p1 - p2 - p3
    if bad:
        print(f"FAIL {bad}/4 checks"); sys.exit(1)


if __name__ == '__main__':
    main()
