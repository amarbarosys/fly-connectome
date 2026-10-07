"""First spider neurons: spiking motion and velocity neurons drive principal-eye tracking (E2).

Replaces the velocity estimate inside spider_principal_eye_test.py's tracking mode (one line of
arithmetic) with spiking neurons. Lock-on (E1) and scanning (E3) stay arithmetic for now.

Circuit (horizontal plane):
  retina receptors (1200, in the order the brain assumes they are wired)
  -> Reichardt pairs on wired neighbours: delayed(left) x now(right) - delayed(right) x now(left)
  -> R and L motion populations, 20 LIF neurons each, Poisson drive at rate = GAIN x slip+ / slip-
  -> velocity neuron (LIF) integrating R - L spikes through excitatory / inhibitory synapses
  -> slip velocity = K x (R - L spike count) per 10 ms; target velocity = slip + eye velocity
     (efference copy of the eye command). K is fitted once on calibration runs, then frozen.
Neuron equations and every parameter are Shiu et al. 2024 fly values (agent/shiu.py), borrowed:
no spider recordings exist. At Shiu's stimulus weight (0.275 x 250 = 68.75 mV per event vs a
7 mV threshold) each input event fires a motion neuron, so they relay Poisson input capped by
the 2.2 ms refractory period.

Gate (2026-10-07 redesign, registered before its first run). The first gate could not fail:
the foveal centroid re-placed the eye on the target every 10 ms step, the efference copy
(eye velocity) already carried target velocity, and error was scored against the target one
step before the gaze command applied, so intact, silenced and swapped all read |v| x DT = 0.25
deg. Now the retina's position (centroid) is sampled every KC = 10 steps (100 ms, model
choice); between samples only the pursuit velocity moves the eye, and pursuit velocity
integrates the neurons' slip (vel += 0.3 x slip; efference copy = the pursuit command, never
the position correction). Error = |gaze - target| at the time the gaze command applies.
E2 positive control: intact < 1 deg and shuffled retina > 2 deg (bar unchanged).
N2 velocity neuron silenced: error >= 3x intact.
N3 R and L populations swapped: error >= 3x intact.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from spider_principal_eye_test import DT, N_REC, TUBE, SACC, WIN, coarse, retina  # noqa: E402

V_TH, T_MBR, TAU, T_RFC, W_SYN, F_POI = 7.0, 20e-3, 5e-3, 2.2e-3, 0.275, 250  # agent/shiu.py
SUB, N_POP, GAIN = 1e-4, 20, 400.0  # 0.1 ms neuron step; GAIN: slip units -> Hz (model choice)
KC = 10  # retinal position (centroid) sample period in 10 ms steps: model choice


class EyeNeurons:
    """R/L motion populations + velocity neuron, batched over trials. Shiu LIF, exact decay. In place."""

    def __init__(self, n, rng, swap=False, silence=False):
        self.n, self.rng, self.swap, self.silence = n, rng, swap, silence
        shape = (n, 2, N_POP)
        self.u, self.ref, self.ev = np.zeros(shape), np.zeros(shape), np.empty(shape)
        self.free, self.spk = np.empty(shape, bool), np.empty(shape, bool)
        self.p, self.counts, self.now = np.empty((n, 2, 1)), np.empty((n, 2)), np.empty((n, 2))
        self.vu, self.vg, self.d = np.zeros(n), np.zeros(n), np.empty(n)  # velocity neuron state (kept for inspection)

    def step(self, slip):
        """10 ms of neuron time; returns R - L spike counts per trial (a buffer reused next step)."""
        r, l = (1, 0) if self.swap else (0, 1)
        np.clip(slip, 0, None, out=self.p[:, r, 0]); np.negative(slip, out=self.p[:, l, 0])
        np.clip(self.p[:, l, 0], 0, None, out=self.p[:, l, 0])
        self.p *= GAIN * SUB  # Poisson event probability per 0.1 ms
        self.counts[:] = 0
        decay = np.exp(-SUB / T_MBR)
        for _ in range(int(round(DT / SUB))):
            np.less_equal(self.ref, 0, out=self.free)
            self.rng.random(out=self.ev); np.less(self.ev, self.p, out=self.spk)
            self.u *= decay; np.add(self.u, W_SYN * F_POI, out=self.u, where=self.spk); self.u *= self.free
            np.greater(self.u, V_TH, out=self.spk)
            self.spk.sum(-1, out=self.now); self.counts += self.now
            np.copyto(self.u, 0.0, where=self.spk)
            self.ref -= SUB; np.copyto(self.ref, T_RFC, where=self.spk)
            np.subtract(self.now[:, 0], self.now[:, 1], out=self.d); self.d *= W_SYN
            self.vg *= np.exp(-SUB / TAU); self.vg += self.d
            np.multiply(self.vg, SUB / T_MBR, out=self.d); self.vu *= decay; self.vu += self.d
        np.subtract(self.counts[:, 0], self.counts[:, 1], out=self.d)
        if self.silence: self.d[:] = 0
        return self.d


class Retina:
    """Receptor image, Reichardt motion and centroid for n trials, in preallocated buffers."""

    def __init__(self, rec, n):
        true, wired = rec
        order = np.argsort(wired)  # columns in the order the brain assumes the retina is laid out
        self.true, self.wired = true[order], wired[order]
        self.a, self.prev = np.zeros((n, N_REC)), np.zeros((n, N_REC))
        self.mag, self.far = np.empty(n), np.empty(n, bool)
        self.motion, self.sum, self.pos = np.empty(n), np.empty(n), np.empty(n)

    def see(self, off):
        """New image of targets at offset off; motion = Reichardt pairs on wired neighbours vs the previous image."""
        self.a, self.prev = self.prev, self.a
        a = self.a
        np.subtract(self.true, off[:, None], out=a); a /= 0.15; np.square(a, out=a); a *= -0.5; np.exp(a, out=a)
        np.abs(off, out=self.mag); np.greater(self.mag, WIN / 2, out=self.far); np.copyto(a, 0.0, where=self.far[:, None])
        np.einsum('ij,ij->i', self.prev[:, :-1], a[:, 1:], out=self.motion)  # row dot products, no n x 1199 temporary
        np.einsum('ij,ij->i', self.prev[:, 1:], a[:, :-1], out=self.mag); self.motion -= self.mag
        a.sum(1, out=self.sum); np.dot(a, self.wired, out=self.pos)


def calibrate(rec, rng):
    """Fit K: slip velocity per (R - L) spike count, gaze held still, known velocities."""
    xs, ys = [], []
    for v in (-10, -6, -3, 3, 6, 10):  # Reichardt is linear only to ~10 deg/s at 10 ms delay
        neu, ret = EyeNeurons(50, rng), Retina(rec, 50)
        off0 = rng.uniform(-1.5, 1.5, 50)
        for k in range(15):
            ret.see(np.clip(off0 + v * k * DT, -2.4, 2.4))
            if k:
                d = neu.step(ret.motion)
                if k > 3: xs += list(d); ys += [v] * 50
    return float(np.polyfit(xs, ys, 1)[0])


def track(rec, K, rng, swap=False, silence=False, n=100, steps=100, seed=7):
    trng = np.random.default_rng(seed)
    v = trng.uniform(10, 40, n) * trng.choice([-1, 1], n); x0 = trng.uniform(-20, 20, n) - v * 0.5
    lo, hi = -TUBE + 1, TUBE - 1
    bear = [coarse(lambda s, i=i: float(np.clip(x0[i] + v[i] * s, lo, hi)), trng) for i in range(n)]
    neu, ret = EyeNeurons(n, rng, swap, silence), Retina(rec, n)
    gaze, vel, tg, off, b, est, step = (np.zeros(n) for _ in range(7))
    seen, unseen, err = np.empty(n, bool), np.empty(n, bool), np.empty((steps, n))
    for k in range(steps):
        t = k * DT
        np.multiply(v, t, out=tg); tg += x0; np.clip(tg, lo, hi, out=tg)
        np.subtract(tg, gaze, out=off); ret.see(off)
        np.greater(ret.sum, 1e-6, out=seen); np.logical_not(seen, out=unseen)
        slip = neu.step(ret.motion); slip *= 0.3 * K
        vel += slip; np.copyto(vel, 0.0, where=unseen)  # pursuit integrates retinal slip while the target is seen
        if k % KC == 0:  # retinal position sample: the eye jumps onto the centroid
            np.divide(ret.pos, ret.sum, out=est, where=seen); est += gaze; np.copyto(gaze, est, where=seen)
        np.multiply(vel, DT, out=step); gaze += step
        for i, f in enumerate(bear): b[i] = f(t)
        np.subtract(b, gaze, out=step); np.clip(step, -SACC * DT, SACC * DT, out=step)
        np.add(gaze, step, out=gaze, where=unseen)  # off the retina: saccade toward the secondary-eye bearing
        np.clip(gaze, -TUBE, TUBE, out=gaze)
        np.multiply(v, t + DT, out=step); step += x0; np.clip(step, lo, hi, out=step)  # target when the command applies
        np.subtract(gaze, step, out=err[k]); np.abs(err[k], out=err[k])
    return float(np.median(np.median(err[30:], 0)))


def main():
    rng = np.random.default_rng(0)
    rec, bad = retina(np.random.default_rng(0)), retina(np.random.default_rng(0), shuffle=True)
    K = calibrate(rec, np.random.default_rng(1))
    print(f"calibration: K = {K:.3f} deg/s per (R-L) spike per 10 ms")
    intact, shuf = track(rec, K, rng), track(bad, K, rng)
    silent, swp = track(rec, K, rng, silence=True), track(rec, K, rng, swap=True)
    ok2, okn2, okn3 = intact < 1 and shuf > 2, silent >= 3 * intact, swp >= 3 * intact
    print(f"E2 with neurons, position every {KC} steps: intact {intact:.2f} deg, shuffled retina {shuf:.2f} deg "
          f"(bars < 1, > 2) -> {'PASS' if ok2 else 'FAIL'}")
    print(f"N2 velocity neuron silenced: {silent:.2f} deg ({silent / intact:.1f}x, bar 3x) -> {'PASS' if okn2 else 'FAIL'}")
    print(f"N3 R/L swapped: {swp:.2f} deg ({swp / intact:.1f}x, bar 3x) -> {'PASS' if okn3 else 'FAIL'}")
    bad_n = 3 - ok2 - okn2 - okn3
    if bad_n:
        print(f"FAIL {bad_n}/3 checks"); sys.exit(1)


if __name__ == "__main__":
    main()
