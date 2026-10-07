"""Spider hunting-state controller on Hung 2026 leg-joint spectra, label-free checks.

Data: GordusLab/spider_prey_capture_paper 2_wavelet_hmm/data, one side-camera video (100 Hz,
4769 frames), 20 joints in 4 files of 5 joints x 50 wavelet frequencies (0.1-50 Hz, arr_0).
Hung et al. 2026 define three states from these spectra: static (low power), crouching (joints
3-10 Hz) and high-frequency shaking (peak ~10 Hz, the web resonance).

Controller: per frame, summed joint power in movement (3-50 Hz) and shake (9-12 Hz) bands,
smoothed 0.2 s. Moving = movement power above the median; shake = moving and shake band carries
> SHAKE_FRAC of movement power; else crouch. A state change needs MIN_DWELL frames (hysteresis).
Thresholds were fixed before the first run (median, 0.35, 0.2 s), not tuned on the result.

The paper's hand labels (83.4% HMM accuracy) are not public, so these checks are weaker:
B1 signatures: static has the lowest movement power; crouch peak frequency in 3-10 Hz; shake
   peak in 9-12 Hz. Partly circular: the controller is built from these bands.
B2 persistence: bouts on the real frame order must be longer (median) than on time-shuffled
   frames in 95% of 200 shuffles: states last because behaviour lasts. Not circular.
B3 agreement with Hung's method: k-means (k=3) on log spectra, clusters matched to our states;
   agreement must beat 95% of 200 runs where our controller sees a scrambled frequency axis.
"""
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

DATA = Path(__file__).resolve().parents[1] / ".work/spider-docs/hung-repo/2_wavelet_hmm/data"
FPS, SMOOTH, MIN_DWELL, SHAKE_FRAC, N_SHUF = 100, 20, 20, 0.35, 200
STATIC, CROUCH, SHAKE = 0, 1, 2


def load():
    parts = [np.load(f)["arr_0"].reshape(-1, 5, 50) for f in sorted(DATA.glob("*_wavelet.npz"))]
    freqs = np.load(next(DATA.glob("*_wavelet.npz")))["arr_2"]
    return np.concatenate(parts, 1), freqs  # frames x 20 joints x 50 freqs


def controller(spec, freqs):
    p = spec.sum(1)  # frames x freqs
    k = np.ones(SMOOTH) / SMOOTH
    move = np.convolve(p[:, (freqs >= 3)].sum(1), k, "same")
    shake = np.convolve(p[:, (freqs >= 9) & (freqs <= 12)].sum(1), k, "same")
    raw = np.where(move <= np.median(move), STATIC, np.where(shake > SHAKE_FRAC * move, SHAKE, CROUCH))
    out, cur, run = np.empty_like(raw), raw[0], 0
    for t, s in enumerate(raw):  # hysteresis: switch only after MIN_DWELL frames of a new state
        run = run + 1 if s != cur else 0
        if run >= MIN_DWELL:
            cur, run = s, 0
        out[t] = cur
    return out


def bout_median(states):
    edges = np.flatnonzero(np.diff(states)) + 1
    return np.median(np.diff(np.r_[0, edges, len(states)]))


def match(a, b):
    return max((np.array(m)[b] == a).mean() for m in permutations(range(3)))


def main():
    spec, freqs = load()
    st = controller(spec, freqs)
    frac = np.bincount(st, minlength=3) / len(st)
    p = spec.sum(1)
    move = p[:, freqs >= 3].sum(1)
    peak = {s: freqs[freqs >= 3][p[st == s][:, freqs >= 3].mean(0).argmax()] for s in (CROUCH, SHAKE)}
    pow_ = {s: move[st == s].mean() for s in range(3)}
    ok1 = (pow_[STATIC] < min(pow_[CROUCH], pow_[SHAKE]) and 3 <= peak[CROUCH] <= 10 and 9 <= peak[SHAKE] <= 12)
    print(f"states static/crouch/shake {frac[0]:.0%}/{frac[1]:.0%}/{frac[2]:.0%} of {len(st)} frames")
    print(f"B1 signatures: movement power {pow_[0]:.1f}/{pow_[1]:.1f}/{pow_[2]:.1f}, peak crouch "
          f"{peak[CROUCH]:.1f} Hz, shake {peak[SHAKE]:.1f} Hz -> {'PASS' if ok1 else 'FAIL'}")

    rng = np.random.default_rng(0)
    real = bout_median(st)
    shuf = np.array([bout_median(controller(spec[rng.permutation(len(spec))], freqs)) for _ in range(N_SHUF)])
    pct = (shuf < real).mean()
    ok2 = pct >= 0.95
    print(f"B2 persistence: median bout {real / FPS:.2f} s vs time-shuffled {shuf.mean() / FPS:.2f} s, "
          f"beats {pct:.1%} -> {'PASS' if ok2 else 'FAIL'}")

    km = KMeans(3, random_state=0, n_init="auto").fit_predict(np.log1p(spec.sum(1)))
    agree = match(st, km)
    scram = np.array([match(controller(spec[:, :, rng.permutation(50)], freqs), km) for _ in range(N_SHUF)])
    pct3 = (scram < agree).mean()
    ok3 = pct3 >= 0.95
    print(f"B3 vs k-means: agreement {agree:.1%}, scrambled-frequency controller {scram.mean():.1%}"
          f"+-{scram.std():.1%}, beats {pct3:.1%} -> {'PASS' if ok3 else 'FAIL'}")
    if not (ok1 and ok2 and ok3):
        print(f"FAIL {3 - ok1 - ok2 - ok3}/3 checks")
        sys.exit(1)


if __name__ == "__main__":
    main()
