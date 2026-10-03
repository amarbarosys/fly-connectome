"""Spider turning test: can an 8-leg vibration circuit find the loudest web radius?

Data: Hung et al. 2026 (PLoS Comp Biol, PMC13592698), GordusLab/spider_prey_capture_paper,
5_localization/processed. One video, 26 radii x 7150 frames of pixel-fluctuation signal,
plus 10 turning events (amplitude-predicted vs actual turn angle).

Check A: reproduce the paper's turn regression from the 10 events (r = 0.9289 reported).
Check B: 8 legs, each cos^p-tuned over all radii with leaky integration, turn toward the
winning leg (population vector scored 39 deg, worse). Score = circular error vs the loudest radius. Leg-direction wiring shuffled 200x is the
control; real wiring must beat 95% of shuffles. Not a test against the spider's own turns:
per-radius input for the 10 events is not published. Test frames cover 3 radii in one ~50 deg
arc of one video; readout chosen after trying 3 variants on this same data.
Data fetch (sparse, skips 1.5 GB of weights): git clone --depth 1 --filter=blob:none --sparse
https://github.com/GordusLab/spider_prey_capture_paper .work/spider-docs/hung-repo, then
git sparse-checkout set 5_localization
"""
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1] / ".work/spider-docs/hung-repo/5_localization/processed"
STEM = "082923_spider_prey_C001H001S0001_signal_map_radii"
TAU = 50      # leg integration, frames
SMOOTH = 100  # truth window, frames
N_SHUF = 200
TUNE_P = 4    # leg angular tuning sharpness, cos^p


def circ_err(a, b):
    return np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)


def leaky(x, tau):
    y = np.empty_like(x)
    acc = x[:, 0]
    for t in range(x.shape[1]):
        acc = acc + (x[:, t] - acc) / tau
        y[:, t] = acc
    return y


def steer(leg_amp, leg_ang):
    return leg_ang[leg_amp.argmax(0)]  # winner-take-all: turn to the leg feeling most


def main():
    import pandas as pd
    df = pd.read_csv(ROOT / "singal_data.csv", encoding="utf-8-sig")
    r = stats.linregress(df.actual_turn0, df.predict_turn0)
    ok_a = abs(r.rvalue - 0.9289) < 0.01
    print(f"A paper regression: r={r.rvalue:.4f} slope={r.slope:.4f} p={r.pvalue:.2e} -> {'PASS' if ok_a else 'FAIL'}")

    sig = np.load(ROOT / f"{STEM}_data.npz")["arr_0"]
    xy = np.load(ROOT / f"{STEM}_coordinate_982.npz")["arr_0"]
    mid = xy.mean(1)
    rad_ang = np.arctan2(*(mid - mid.mean(0)).T[::-1])

    k = np.ones(SMOOTH) / SMOOTH
    sm = np.array([np.convolve(s, k, "same") for s in sig])
    top = np.sort(sm, 0)
    clear = top[-1] > 1.1 * top[-2]  # frames with one clearly loudest radius
    truth = rad_ang[sm.argmax(0)]
    n_dirs = len(set(sm.argmax(0)[clear]))
    if n_dirs < 3:
        print(f"FAIL B: only {n_dirs} distinct loudest radii in test frames, cannot test steering")
        sys.exit(1)

    leg_ang = np.arange(8) * np.pi / 4
    tuning = np.clip(np.cos(leg_ang[:, None] - rad_ang[None]), 0, None) ** TUNE_P  # leg x radius
    legs = leaky(tuning @ sig, TAU)

    err = np.degrees(circ_err(steer(legs, leg_ang), truth)[clear]).mean()
    rng = np.random.default_rng(0)
    shuf = np.array([np.degrees(circ_err(steer(legs, rng.permutation(leg_ang)), truth)[clear]).mean()
                     for _ in range(N_SHUF)])
    pct = (shuf > err).mean()
    ok_b = pct >= 0.95
    print(f"B 8-leg circuit: {clear.sum()} clear frames, {n_dirs} directions, error {err:.1f} deg, "
          f"shuffled {shuf.mean():.1f}+-{shuf.std():.1f} deg, beats {pct:.1%} -> {'PASS' if ok_b else 'FAIL'}")
    if not (ok_a and ok_b):
        print(f"FAIL {2 - ok_a - ok_b}/2 checks")
        sys.exit(1)


if __name__ == "__main__":
    main()
