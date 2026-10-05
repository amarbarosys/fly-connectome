"""GPU self-wiring spider pathway (agent/spider_pool.mojo) and its numpy oracle (HashPathway: same counter-hash random
numbers, integer refractory, kernel order). Scaffold slice of tools/spider_scaffold_learn_test.py: one principal eye,
retina 1200 -> AM1 2000 -> AM2 2000 -> 12 eye-muscle motor neurons, fan-in 20 / 20 / 200, every slot empty at start;
teaching signal per motor neuron (slip direction, motor layer ungated), fixed random feedback to AM1/AM2 (gated by
their own spikes), prune/regrow on the device after every teach step (30 % slot budget). Fixes from astra's review
(.work/team/astra-spider-pool.md): ungated motor layer, 5-substep refractory, per-step rewiring, paired controls.

check  (1) parity: oracle vs GPU from the same nonzero weights, 3 steps x 32 columns: motor spike counts equal, layer
           activity >= 99.9 % equal (float32 sums in another order may flip a threshold), and one TEACH: weights within
           1e-5 relative; (2) learning on the GPU (pursuit, held-out 6-9 deg/s): P1 < 1.5 x the hand-built pool-test
           circuit, P2 random-sign feedback control >= 2 x learned, P3 before training > 2 x hand
usage: gpu-wait run --priority 70 --timeout 3600 -- .venv/bin/python -m agent.spider_pool_gpu check [ITERS]
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from agent.shiu import uniform

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import spider_blank_pool_test as P  # noqa: E402

B, NR, N1, N2, NM, F12, FM = 32, 1200, 2000, 2000, 12, 20, 200
NT = NR + N1 + N2 + NM
POST0, NPOST, FAN = (NR, NR + N1, NR + N1 + N2), (N1, N2, NM), (F12, F12, FM)
SBASE = (0, N1 * F12, 2 * N1 * F12)
NS = N1 * F12 + N2 * F12 + NM * FM
S, REFR, SUB, V_TH, KICK, GAIN, P_NOISE, ETA = 20, 5, 5e-4, 7.0, np.float32(0.275 * 250), 400.0, np.float32(30 * 5e-4), 0.01
THETA, WMAX, FRAC = 0.02, 0.3, 0.3
DIR = ROOT / '.work/agent/spider/pool-gpu'
MOJO = Path.home() / 'iTools/bin/mojo-cached'


def init(seed=2):
    """Slots (global pre ids), empty weights, feedback, receptor channels."""
    rng = np.random.default_rng(seed)
    pre_lo = (0, NR, NR + N1)
    pre_n = (NR, N1, N2)
    idx = np.concatenate([np.stack([rng.choice(pre_n[l], FAN[l], replace=False) for _ in range(NPOST[l])]).ravel() + pre_lo[l] for l in range(3)]).astype(np.int32)
    fb = np.zeros((NT, 2), np.float32)
    fb[NR:NR + N1 + N2] = rng.choice([-1.0, 1.0], (N1 + N2, 2))
    fb[POST0[2]:] = np.r_[np.tile([1, -1], (6, 1)), np.tile([-1, 1], (6, 1))]
    ch = (np.arange(NR) // 100 * 2 + (np.arange(NR) % 100 >= 50)).astype(np.int32)
    return idx, np.zeros(NS, np.float32), fb, ch


class HashPathway:
    """numpy twin of spider_pool.mojo, column-major state (B, NT)."""

    def __init__(self, idx, w, fb, ch, seed=7):
        self.idx, self.w, self.fb, self.ch, self.seed, self.t, self.tstep = idx.copy(), w.copy(), fb, ch, seed, 0, 0
        self.reset()

    def reset(self):
        f = np.float32
        self.u, self.tr = np.zeros((B, NT), f), np.zeros((B, NT), f)
        self.refr = np.zeros((B, NT), np.int32)

    def step(self, m):
        f = np.float32
        decay, dk = f(np.exp(-SUB / 20e-3)), f(np.exp(-SUB / 5e-3))
        self.ps, self.act = np.zeros((B, NT), f), np.zeros((B, NT), np.int32)
        spk = np.zeros((B, NT), f)
        e_all = (np.arange(B)[:, None] * NT + np.arange(NT)[None]).astype(np.uint32)
        v = m[:, self.ch // 2]
        rate = (f(GAIN * SUB) * np.where(self.ch % 2 == 0, np.maximum(v, 0), np.maximum(-v, 0))).astype(f)
        for _ in range(S):
            s = (uniform(self.seed, self.t, e_all[:, :NR]) < rate).astype(f)
            spk[:, :NR] = s
            self.tr[:, :NR] = self.tr[:, :NR] * dk + s; self.ps[:, :NR] += self.tr[:, :NR]
            for l in range(3):
                a, n, fan, sb = POST0[l], NPOST[l], FAN[l], SBASE[l]
                ii = self.idx[sb:sb + n * fan].reshape(n, fan); ww = self.w[sb:sb + n * fan].reshape(n, fan)
                drive = (ww[None] * spk[:, ii]).sum(-1, dtype=f)
                noise = np.where(uniform(self.seed + 1, self.t, e_all[:, a:a + n]) < P_NOISE, KICK, f(0))
                r = self.refr[:, a:a + n]
                x = (self.u[:, a:a + n] * decay + drive * KICK + noise).astype(f)
                busy = r > 0
                sp = ~busy & (x > V_TH)
                self.u[:, a:a + n] = np.where(busy | sp, f(0), x)
                self.refr[:, a:a + n] = np.where(busy, r - 1, np.where(sp, REFR, 0))
                spk[:, a:a + n] = sp
                self.act[:, a:a + n] += sp
                self.tr[:, a:a + n] = self.tr[:, a:a + n] * dk + sp; self.ps[:, a:a + n] += self.tr[:, a:a + n]
            self.t += 1
        return self.act[:, POST0[2]:].astype(np.uint16)

    def teach(self, err):
        for l in range(3):
            a, n, fan, sb = POST0[l], NPOST[l], FAN[l], SBASE[l]
            ii = self.idx[sb:sb + n * fan].reshape(n, fan)
            h = (err @ self.fb[a:a + n].T) * ((self.act[:, a:a + n] > 0) if l < 2 else 1)  # (B, n); motor layer ungated
            self.w[sb:sb + n * fan] += (np.float32(ETA) * np.float32(F12) / np.float32(fan) * np.einsum('bj,bjk->jk', h, self.ps[:, ii]) / B).ravel()
            np.clip(self.w[sb:sb + n * fan], -WMAX, WMAX, out=self.w[sb:sb + n * fan])  # the kernel clamps every step
        alive = []
        for l in range(3):  # prune kernel
            ww = self.w[SBASE[l]:SBASE[l] + NPOST[l] * FAN[l]]
            ww[np.abs(ww) < np.float32(THETA)] = 0
            alive.append(int((ww != 0).sum()))
        pre_lo, pre_n = (0, NR, NR + N1), (NR, N1, N2)
        for l in range(3):  # regrow kernel
            n, sb = NPOST[l] * FAN[l], SBASE[l]
            ww, ii = self.w[sb:sb + n], self.idx[sb:sb + n]
            sl = np.arange(sb, sb + n).astype(np.uint32)
            q = np.float32((np.float32(0.3) * np.float32(n) - np.float32(alive[l])) / np.float32(max(n - alive[l], 1)))
            grow = (ww == 0) & (uniform(self.seed + 2, self.tstep, sl) < q)
            pick = uniform(self.seed + 3, self.tstep, sl)
            ii[grow] = (pre_lo[l] + np.minimum((pick * np.float32(pre_n[l])).astype(np.int32), pre_n[l] - 1))[grow]
            ww[grow] = np.where(uniform(self.seed + 4, self.tstep, sl) < 0.5, np.float32(1.5 * THETA), np.float32(-1.5 * THETA))[grow]
        self.tstep += 1


class GPUPathway:
    def __init__(self, idx, w, fb, ch, seed=7):
        DIR.mkdir(parents=True, exist_ok=True)
        idx.tofile(DIR / 'idx.bin'); w.tofile(DIR / 'w.bin'); fb.tofile(DIR / 'fb.bin'); ch.tofile(DIR / 'ch.bin')
        src = ROOT / 'agent/spider_pool.mojo'
        binary = ROOT / f'.work/agent/spider/build/spider_pool_b{B}'
        if not binary.is_file() or binary.stat().st_mtime < src.stat().st_mtime:
            subprocess.run([str(MOJO), 'build', f'-DB={B}', str(src), '-o', str(binary)], check=True, capture_output=True)
        self.p = subprocess.Popen([str(binary), str(DIR) + '/', str(seed)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def _cmd(self, code, *payload):
        self.p.stdin.write(np.int32(code).tobytes() + b''.join(np.ascontiguousarray(x).tobytes() for x in payload)); self.p.stdin.flush()

    def reset(self):
        self._cmd(3)

    def step(self, m):
        self._cmd(1, m.astype(np.float32))
        return np.frombuffer(self.p.stdout.read(B * NM * 2), np.uint16).reshape(B, NM)

    def teach(self, err):
        self._cmd(2, err.astype(np.float32))

    def weights(self):
        self._cmd(4)
        return np.frombuffer(self.p.stdout.read(NS * 4), np.float32).copy(), np.frombuffer(self.p.stdout.read(NS * 4), np.int32).copy()

    def set_weights(self, w, idx):
        self._cmd(5, w.astype(np.float32), idx.astype(np.int32))

    def close(self):
        self._cmd(0); self.p.stdin.close(); self.p.wait()


def run(net, rec, sp, rng, silence=False, learn=False, shuffle=None, flip=False):
    """shuffle: a separate generator for the control's random signs, so both arms see the same task stream."""
    n = len(sp)
    net.reset()
    tgt0, gaze, eye_v, prev, errs = rng.uniform(-0.3, 0.3, n), np.zeros(n), np.zeros(n), None, []
    for k in range(P.STEPS):
        tgt = tgt0 + sp * k * P.DT
        now = P.activation(rec, tgt - gaze)
        if prev is not None:
            c = net.step(P.motion(prev, now).astype(np.float32)).astype(np.float64)
            d = np.zeros(n) if silence else (c[:, :6].sum(1) - c[:, 6:].sum(1)) / 6  # per-direction mean: 2-motor scale (astra: 12 motors at KA 1.5 gave 2.85 deg/s noise)
        else:
            d = np.zeros(n)
        prev = now
        seen = sp - eye_v  # the slip this step's neurons responded to, before their own command moves the eye
        eye_v = eye_v + P.KA * d
        gaze = gaze + eye_v * P.DT
        if learn and k > 1:
            slip = seen.copy()
            slip = slip * shuffle.choice([-1, 1], n) if shuffle is not None else -slip if flip else slip
            net.teach((np.stack([slip, -slip], 1) / 10).astype(np.float32))
        errs.append(np.abs(tgt + sp * P.DT - gaze))
    return np.median(np.array(errs)[15:], 0)


def parity():
    idx, w, fb, ch = init()
    rng = np.random.default_rng(4)
    w[:] = rng.normal(0, 0.15, NS).astype(np.float32)  # nonzero, so activity reaches the motor layer
    cpu, gpu = HashPathway(idx, w, fb, ch), GPUPathway(idx, w, fb, ch)
    m = np.random.default_rng(1).normal(0, 0.02, (B, 12)).astype(np.float32)
    same_m, layer_match = True, []
    for _ in range(3):
        a, g = cpu.step(m), gpu.step(m)
        same_m &= bool((a == g).all())
    gpu.teach(np.random.default_rng(2).normal(0, 1, (B, 2)).astype(np.float32))
    cpu.teach(np.random.default_rng(2).normal(0, 1, (B, 2)).astype(np.float32))
    wg, ig = gpu.weights()
    gpu.close()
    werr = float(np.abs(wg - cpu.w).max() / np.abs(cpu.w).max())
    ok = same_m and werr <= 1e-5 and (ig == cpu.idx).all()
    print(f"{'PASS' if ok else 'FAIL'} (1) parity: motor spikes equal over 3 steps x {B} columns: {same_m}; "
          f"motor spikes {int(a.sum())}; weights after TEACH rel err {werr:.2e} (bar 1e-5)")
    return ok


def train(arm, iters, rec, seed=2, stop_above=None):
    """arm: learn, shuffle (random-sign teacher), frozen (no teaching), flip (mutant: teacher sign reversed).
    frozen never changes a weight, so it is not trained (its after = its before, exactly). stop_above: a control arm ends
    once its held-out error has been >= stop_above at 4 successive test points from batch 100 on (decided, not run out)."""
    idx, w, fb, ch = init(seed)
    net, rng, sign = GPUPathway(idx, w, fb, ch, seed=7), np.random.default_rng(seed + 1), np.random.default_rng(seed + 11)
    test = lambda s: float(np.median(np.concatenate([run(net, rec, P.speeds(np.random.default_rng(s + q), 6, 9, B), np.random.default_rng(s + q)) for q in range(4)])))
    before, curve, t0 = test(9), [], time.perf_counter()
    n = 0
    for it in range(0 if arm == 'frozen' else iters):
        n = it + 1
        run(net, rec, P.speeds(rng, 2, 5, B), rng, learn=arm != 'frozen', shuffle=sign if arm == 'shuffle' else None, flip=arm == 'flip')
        if it % 25 == 24:
            curve.append(round(test(11), 2)); print(arm, it + 1, curve[-1], f'{(time.perf_counter() - t0) / (it + 1):.2f} s/batch', flush=True)
            if stop_above and it >= 99 and len(curve) >= 4 and min(curve[-4:]) >= stop_above:
                break
    silenced = float(np.median(np.concatenate([run(net, rec, P.speeds(np.random.default_rng(9 + q), 6, 9, B), np.random.default_rng(9 + q), silence=True) for q in range(4)])))
    out = {'before': before, 'after': test(9), 'silenced': silenced, 'curve': curve, 's_per_batch': (time.perf_counter() - t0) / max(n, 1), 'batches': n}
    net.close()
    return out


def main(cmd='check', iters='300'):
    ok = parity()
    rec = P.retina(np.random.default_rng(0))
    hand = P.Net(np.random.default_rng(1), hand=True)
    e_hand = float(np.median(P.run(hand, rec, P.speeds(np.random.default_rng(9), 6, 9, 128), np.random.default_rng(9))))
    res, bad = {'hand': e_hand}, int(not ok)
    for seed in (2, 3, 4):  # paired arms: same init, task tape and noise per seed
        r = {'learn': train('learn', int(iters), rec, seed)}
        r |= {arm: train(arm, int(iters), rec, seed, stop_above=3 * r['learn']['after']) for arm in ('shuffle', 'frozen', 'flip')}
        res[seed] = r
        L = r['learn']['after']
        checks = {'P1 learned < 1.5 hand': L < 1.5 * e_hand, 'P2 random-sign >= 2 learned': r['shuffle']['after'] >= 2 * L,
                  'P3 before > 2 hand': r['learn']['before'] > 2 * e_hand, 'P4 frozen >= 2 learned': r['frozen']['after'] >= 2 * L,
                  'M1 mutant flipped teacher >= 2 learned': r['flip']['after'] >= 2 * L}
        print(f"seed {seed}: before {r['learn']['before']:.2f} learn {L:.2f} random-sign {r['shuffle']['after']:.2f} "
              f"frozen {r['frozen']['after']:.2f} flipped {r['flip']['after']:.2f} hand {e_hand:.2f}; {r['learn']['s_per_batch']:.2f} s/batch")
        for k, v in checks.items():
            print(f"{'PASS' if v else 'FAIL'} seed {seed} {k}")
        bad += sum(not v for v in checks.values())
    (ROOT / '.work/agent/spider/pool-gpu.json').write_text(json.dumps(res | {'parity': bool(ok)}, indent=2) + '\n')
    if bad:
        sys.exit(f'FAIL {bad} checks')
    print('PASS spider pool on the GPU')


if __name__ == '__main__':
    main(*sys.argv[1:])
