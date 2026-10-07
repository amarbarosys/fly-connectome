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

N1 equivalence: E2 intact tracking error within 0.1 deg of the arithmetic controller's 0.23.
N2 velocity neuron silenced: tracking error rises >= 3x.
N3 R and L populations swapped: error grows over the trial (late > 2x early) and >= 3x intact.
E2 rerun with neurons, bar unchanged: intact < 1 deg and shuffled retina > 2 deg.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from spider_principal_eye_test import DT, TUBE, SACC, WIN, coarse, retina  # noqa: E402

V_TH, T_MBR, TAU, T_RFC, W_SYN, F_POI = 7.0, 20e-3, 5e-3, 2.2e-3, 0.275, 250  # agent/shiu.py
SUB, N_POP, GAIN = 1e-4, 20, 400.0  # 0.1 ms neuron step; GAIN: slip units -> Hz (model choice)


class EyeNeurons:
    """R/L motion populations + velocity neuron, batched over trials. Shiu LIF, exact decay."""

    def __init__(self, n, rng, swap=False, silence=False):
        self.n, self.rng, self.swap, self.silence = n, rng, swap, silence
        self.u = np.zeros((n, 2, N_POP)); self.ref = np.zeros((n, 2, N_POP))
        self.vu, self.vg = np.zeros(n), np.zeros(n)  # velocity neuron state (kept for inspection)

    def step(self, slip):
        """10 ms of neuron time; returns R - L spike counts per trial."""
        rates = GAIN * np.stack([np.clip(slip, 0, None), np.clip(-slip, 0, None)], 1)  # n x 2
        if self.swap: rates = rates[:, ::-1]
        counts = np.zeros((self.n, 2))
        decay = np.exp(-SUB / T_MBR)
        for _ in range(int(round(DT / SUB))):
            free = self.ref <= 0
            ev = self.rng.random(self.u.shape) < (rates[:, :, None] * SUB)
            self.u = np.where(free, self.u * decay + ev * W_SYN * F_POI, 0.0)
            spk = self.u > V_TH
            counts += spk.sum(-1)
            self.u[spk] = 0.0; self.ref = np.where(spk, T_RFC, self.ref - SUB)
            self.vg = self.vg * np.exp(-SUB / TAU) + W_SYN * (spk[:, 0].sum(-1) - spk[:, 1].sum(-1))
            self.vu = self.vu * decay + self.vg * SUB / T_MBR
        d = counts[:, 0] - counts[:, 1]
        return np.zeros(self.n) if self.silence else d


def receptors(rec, off):
    true, wired = rec
    a = np.exp(-0.5 * ((true[None] - off[:, None]) / 0.15) ** 2) * (np.abs(off) <= WIN / 2)[:, None]
    return a[:, np.argsort(wired)]  # ordered as the brain assumes the retina is laid out


def reichardt(prev, now):
    return (prev[:, :-1] * now[:, 1:] - prev[:, 1:] * now[:, :-1]).sum(1)


def centroid(rec, a):
    wired_sorted = np.sort(rec[1])
    s = a.sum(1)
    return np.where(s > 1e-6, (a * wired_sorted).sum(1) / np.maximum(s, 1e-9), np.nan)


def calibrate(rec, rng):
    """Fit K: slip velocity per (R - L) spike count, gaze held still, known velocities."""
    xs, ys = [], []
    for v in (-10, -6, -3, 3, 6, 10):  # Reichardt is linear only to ~10 deg/s at 10 ms delay
        neu, prev = EyeNeurons(50, rng), None
        off0 = rng.uniform(-1.5, 1.5, 50)
        for k in range(15):
            off = np.clip(off0 + v * k * DT, -2.4, 2.4)
            a = receptors(rec, off)
            if prev is not None:
                d = neu.step(reichardt(prev, a))
                if k > 3: xs += list(d); ys += [v] * 50
            prev = a
    return float(np.polyfit(xs, ys, 1)[0])


def track(rec, K, rng, swap=False, silence=False, n=100, steps=100, seed=7):
    trng = np.random.default_rng(seed)
    v = trng.uniform(10, 40, n) * trng.choice([-1, 1], n); x0 = trng.uniform(-20, 20, n) - v * 0.5
    tgt = lambda t: np.clip(x0 + v * t, -TUBE + 1, TUBE - 1)
    bear = [coarse(lambda s, i=i: float(tgt(s)[i]), trng) for i in range(n)]
    neu = EyeNeurons(n, rng, swap, silence)
    gaze, prev_a, prev_gaze, errs, was_seen = np.zeros(n), None, np.zeros(n), [], np.zeros(n, bool)
    vel = np.zeros(n)
    for k in range(steps):
        t = k * DT; tg = tgt(t)
        a = receptors(rec, tg - gaze)
        est = gaze + centroid(rec, a)
        seen = ~np.isnan(est)
        slip = K * neu.step(reichardt(prev_a, a)) if prev_a is not None else np.zeros(n)
        eye_v = np.where(was_seen, (gaze - prev_gaze) / DT, 0.0)  # efference copy: pursuit only, not saccades
        vel = np.where(was_seen, 0.7 * vel + 0.3 * (slip + eye_v), 0.0)  # same smoothing as arithmetic E2
        prev_gaze, prev_a, was_seen = gaze.copy(), a, seen
        b = np.array([f(t) for f in bear])
        new = np.where(seen, est + vel * DT, gaze + np.clip(b - gaze, -SACC * DT, SACC * DT))
        gaze = np.clip(new, -TUBE, TUBE)
        errs.append(np.abs(gaze - tg))
    e = np.array(errs)
    return np.median(np.median(e[30:], 0)), np.median(e[30:60]), np.median(e[70:])


def main():
    rng = np.random.default_rng(0)
    rec, bad = retina(np.random.default_rng(0)), retina(np.random.default_rng(0), shuffle=True)
    K = calibrate(rec, np.random.default_rng(1))
    print(f"calibration: K = {K:.3f} deg/s per (R-L) spike per 10 ms")
    intact, _, _ = track(rec, K, rng)
    shuf, _, _ = track(bad, K, rng)
    silent, _, _ = track(rec, K, rng, silence=True)
    swp, early, late = track(rec, K, rng, swap=True)
    ok1 = abs(intact - 0.23) <= 0.1
    ok2 = silent >= 3 * intact
    ok3 = late > 2 * early and swp >= 3 * intact
    ok4 = intact < 1 and shuf > 2
    print(f"N1 equivalence: intact {intact:.2f} deg vs arithmetic 0.23 -> {'PASS' if ok1 else 'FAIL'}")
    print(f"N2 velocity neuron silenced: {silent:.2f} deg ({silent / intact:.1f}x) -> {'PASS' if ok2 else 'FAIL'}")
    print(f"N3 R/L swapped: {swp:.2f} deg ({swp / intact:.1f}x), early {early:.2f} late {late:.2f} -> {'PASS' if ok3 else 'FAIL'}")
    print(f"E2 with neurons: intact {intact:.2f} deg, shuffled retina {shuf:.2f} deg (bar > 2) -> {'PASS' if ok4 else 'FAIL'}")
    bad_n = 4 - ok1 - ok2 - ok3 - ok4
    if bad_n:
        print(f"FAIL {bad_n}/4 checks"); sys.exit(1)


if __name__ == "__main__":
    main()
