"""Spider turning test: can an 8-leg vibration circuit find the loudest web radius?

Data: Hung et al. 2026 (PLoS Comp Biol, PMC13592698), GordusLab/spider_prey_capture_paper,
5_localization/processed. One video, 26 radii x 7150 frames of pixel-fluctuation signal,
plus 10 turning events (amplitude-predicted vs actual turn angle).

Check A: reproduce the paper's turn regression from the 10 events (r = 0.9289 reported).

Check B: 8 legs, each cos^p-tuned over all radii with leaky integration, turn toward the
winning leg (population vector scored 39 deg, worse). Score = circular error vs the loudest
radius. Leg-direction wiring shuffled 200x is the control; real wiring must beat 95% of
shuffles. Not a test against the spider's own turns: per-radius input for the 10 events is not
published. Test frames cover 3 radii in one ~50 deg arc of one video; readout chosen after
trying 3 variants on this same data.

Check C: Mortimer et al. 2019 (J R Soc Interface, Dryad doi:10.5061/dryad.7rm572m) finite-element
Araneus web, 8 legs at the hub, source on each of 14 radii, with and without spider mass (28
cases). Winning leg = largest first transverse peak. Leg angles are not in the data: each leg's
angle is the circular mean of sources it wins in the OTHER radii (leave-one-out). Must beat 95%
of 200 shuffles of winner vs source. Radius angles read from the label positions in
AraneusWebandSpider.pdf.

Data: Hung, sparse clone (skips 1.5 GB of weights): git clone --depth 1 --filter=blob:none
--sparse https://github.com/GordusLab/spider_prey_capture_paper .work/spider-docs/hung-repo,
then git sparse-checkout set 5_localization. Mortimer: Dryad zip (browser download, API needs
a token) unzipped into .work/spider-docs/mortimer2019/.
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


MORT = Path(__file__).resolve().parents[1] / ".work/spider-docs/mortimer2019/Data/A6/Reference amplitude"
# radius label centres (pt) from pdftotext -bbox AraneusWebandSpider.pdf, small font, y down
RADII = {1: (320, 165), 2: (304, 113), 3: (273, 74), 4: (234, 55), 5: (169, 52), 6: (111, 80), 7: (83, 121),
         8: (74, 173), 9: (87, 232), 10: (124, 272), 11: (171, 294), 12: (238, 290), 13: (277, 262), 14: (313, 210)}
NOISE = 1e-8  # web noise floor, AraneusCode.m


def fem_winners():
    xy = np.array(list(RADII.values()), float)
    d = xy - xy.mean(0)
    src_ang = dict(zip(RADII, np.arctan2(-d[:, 1], d[:, 0])))
    cases = []
    for mass in ("mass", "no mass"):
        for r in RADII:
            x = np.genfromtxt(MORT / mass / f"A6R{r}{mass.replace(' ', '')}.txt", delimiter=";", skip_header=2)
            tr = x[:, 3::3]
            first = np.argmax(np.abs(tr) > 5 * NOISE, 0)
            peak = [np.abs(tr[i:i + 40, k]).max() for k, i in enumerate(first)]
            cases.append((r, src_ang[r], int(np.argmax(peak))))
    return cases


def loo_error(cases, wins):
    errs = []
    for (r, ang, _), win in zip(cases, wins):
        others = [a for (rr, a, _), w in zip(cases, wins) if rr != r and w == win]
        if not others:
            errs.append(np.pi / 2)  # leg never wins elsewhere: chance-level error
            continue
        errs.append(circ_err(np.angle(np.exp(1j * np.array(others)).mean()), ang))
    return np.degrees(np.mean(errs))


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

    cases = fem_winners()
    wins = np.array([w for _, _, w in cases])
    err_c = loo_error(cases, wins)
    rng = np.random.default_rng(1)
    shuf_c = np.array([loo_error(cases, rng.permutation(wins)) for _ in range(N_SHUF)])
    pct_c = (shuf_c > err_c).mean()
    ok_c = pct_c >= 0.95
    print(f"C FEM web, 8 legs: {len(cases)} cases, winners {''.join(str(w + 1) for _, _, w in cases[:14])}, "
          f"error {err_c:.1f} deg, shuffled {shuf_c.mean():.1f}+-{shuf_c.std():.1f} deg, beats {pct_c:.1%} "
          f"-> {'PASS' if ok_c else 'FAIL'}")
    if not (ok_a and ok_b and ok_c):
        print(f"FAIL {3 - ok_a - ok_b - ok_c}/3 checks")
        sys.exit(1)


if __name__ == "__main__":
    main()
