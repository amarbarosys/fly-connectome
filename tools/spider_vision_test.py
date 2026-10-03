"""Jumping-spider secondary-eye sector circuit: motion anywhere around -> turn to face it.

Spec: Winsor, Morehouse & Jakob 2023 (Distributed Vision ch. 10) summarising Land 1971 and Zurek &
Nelson 2012: secondary eyes (AL, PL) detect motion over ~360 deg and turn the body to bring the
principal eyes onto the target, even with other eyes masked; masking an eye removes turns to motion
in its field. Horizontal plane only (PM eyes look up/back, left out). Field limits are model
choices, not measurements: AL 0-70 deg each side (overlapping in front), PL 60-170 deg each side,
leaving a 20 deg rear blind zone.

Circuit: each eye = row of motion detectors (2 deg spacing, |change| of a moving dot's Gaussian
image between two time steps); global winner-take-all over all detectors; turn = winning
detector's azimuth. No response if the winner is below threshold. Same conventions as
spider_turn_test.py (200 shuffles, must beat 95%).

V1: unmasked, 200 random dots: mean turn error must beat 95% of 200 shuffles of detector azimuths.
V2: mask left PL: dots in its exclusive zone must get <5% responses, dots elsewhere in view >95%.
Zone = from 3 blur widths past AL's edge (79 deg) to 165 deg left. First run used a hand-picked 75
deg edge and failed at 5.1%: all 3 responses were dots at 75-78 deg blurring onto AL's last
detector, so the boundary now follows the image blur. The literature predicts the pattern, not
these exact rates.
Checks the circuit against published behaviour, not against recorded data.
"""
import sys

import numpy as np

EYES = {"AL_L": (0, 70), "AL_R": (-70, 0), "PL_L": (60, 170), "PL_R": (-170, -60)}  # deg, + = left
SPACING, BLUR, THRESH, N_TRIALS, N_SHUF = 2.0, 3.0, 0.05, 200, 200


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def build(eyes):
    az = [np.arange(lo, hi + 1e-9, SPACING) for lo, hi in eyes.values()]
    eye = np.concatenate([[name] * len(a) for name, a in zip(eyes, az)])
    return np.radians(np.concatenate(az)), eye


def respond(det_az, live, dot0, dot1):
    image = lambda d: np.exp(-0.5 * (np.degrees(wrap(det_az - d)) / BLUR) ** 2)
    motion = np.abs(image(dot1) - image(dot0)) * live
    k = motion.argmax()
    return k if motion[k] > THRESH else None


def turn_error(det_az, turn_az, live, dot0, dot1):
    err = [abs(wrap(turn_az[k] - b)) for a, b in zip(dot0, dot1)
           if (k := respond(det_az, live, a, b)) is not None]
    return np.degrees(np.mean(err)), len(err)


def main():
    rng = np.random.default_rng(0)
    det_az, eye = build(EYES)
    live = np.ones(len(det_az))
    dot0 = rng.uniform(-np.pi, np.pi, N_TRIALS)
    dot1 = wrap(dot0 + rng.choice([-1, 1], N_TRIALS) * np.radians(2.0))

    err, n = turn_error(det_az, det_az, live, dot0, dot1)
    shuf = np.array([turn_error(det_az, rng.permutation(det_az), live, dot0, dot1)[0] for _ in range(N_SHUF)])
    pct = (shuf > err).mean()
    ok1 = pct >= 0.95
    print(f"V1 unmasked: {n}/{N_TRIALS} responded, turn error {err:.1f} deg, shuffled "
          f"{shuf.mean():.1f}+-{shuf.std():.1f}, beats {pct:.1%} -> {'PASS' if ok1 else 'FAIL'}")

    masked = live * (eye != "PL_L")
    deg = np.degrees(dot1)
    excl = (deg > EYES["AL_L"][1] + 3 * BLUR) & (deg < 165)
    seen = (np.abs(deg) < 165) & ~excl & ~((deg > EYES["AL_L"][1]) & (deg < 165))  # edge band excluded
    resp = np.array([respond(det_az, masked, a, b) is not None for a, b in zip(dot0, dot1)])
    r_ex, r_ok = resp[excl].mean(), resp[seen].mean()
    ok2 = r_ex < 0.05 and r_ok > 0.95
    print(f"V2 mask PL_L: responses in its zone {r_ex:.1%} ({excl.sum()} dots), elsewhere "
          f"{r_ok:.1%} ({seen.sum()} dots) -> {'PASS' if ok2 else 'FAIL'}")
    if not (ok1 and ok2):
        print(f"FAIL {2 - ok1 - ok2}/2 checks")
        sys.exit(1)


if __name__ == "__main__":
    main()
