"""Spider feedback rule: does prey vibration fall before the spider shakes?

Hung et al. 2026: "decreases in prey vibrations predicted subsequent crouching and shaking".
Data: their S4 supplementary video, decoded with plot-scrape into
.work/spider-docs/hung-videos/s4_timeline.csv (2 episodes, 0.1 s steps, manual state labels
cyan/blue=static, straw/green=crouch, orange=high-frequency shaking, inferred from the Fig 5C
legend; pure-fly vibration z-score).

Fixed before the first run (2026-10-03):
F1 (primary): mean change in fly vibration over the 1 s before each manual high-frequency
   onset (state changes into orange from any labelled state) is negative, p < 0.05 one-sided,
   against 1000 circular time-shifts of the vibration trace within each episode (keeps the
   trace's autocorrelation, breaks its timing relative to the spider).
F2 (reported, not gated): same for crouch onsets.
A first look before this test (26 onsets, no null) saw -0.84 for shaking and -0.49 for crouch.
"""
import csv
import sys
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[1] / ".work/spider-docs/hung-videos/s4_timeline.csv"
WIN, N_SHIFT, MIN_SHIFT = 10, 1000, 50  # 1 s window, shifts of at least 5 s


def episodes():
    rows = list(csv.DictReader(open(DATA)))
    for ep in sorted({r["episode"] for r in rows}):
        e = [r for r in rows if r["episode"] == ep]
        yield [r["manual"] for r in e], np.array([float(r["fly_pure_z"]) for r in e])


def onsets(lab, code):
    return [i for i in range(WIN, len(lab)) if lab[i] == code and lab[i - 1] not in (code, ".", "")]


def stat(eps, code, shifts=None):
    d = []
    for k, (lab, z) in enumerate(eps):
        zz = np.roll(z, shifts[k]) if shifts is not None else z
        d += [zz[i] - zz[i - WIN] for i in onsets(lab, code)]
    return np.mean(d), len(d)


def main():
    eps = list(episodes())
    rng = np.random.default_rng(0)
    ok = True
    for code, name, gated in (("o", "high-frequency", True), ("g", "crouch", False)):
        real, n = stat(eps, code)
        null = np.array([stat(eps, code, [rng.integers(MIN_SHIFT, len(z) - MIN_SHIFT) for _, z in eps])[0]
                         for _ in range(N_SHIFT)])
        p = (np.sum(null <= real) + 1) / (N_SHIFT + 1)
        verdict = ("PASS" if p < 0.05 and real < 0 else "FAIL") if gated else "reported"
        ok &= verdict != "FAIL"
        print(f"{'F1' if gated else 'F2'} {name} onsets: n {n}, fly vibration change over prior 1 s "
              f"{real:+.2f} z, shifted null {null.mean():+.2f}+-{null.std():.2f}, p {p:.3f} -> {verdict}")
    if not ok:
        print("FAIL 1/1 gated checks")
        sys.exit(1)


if __name__ == "__main__":
    main()
