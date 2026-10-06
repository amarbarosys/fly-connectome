"""Hand-wired circuit with the walking body, debug tier: eye error and |gaze| per body time constant TAU, deadband and the counter-rotation reflex.
The hand arm missed G1 at TAU 1 s (3.32 deg, gaze 5-11 deg: the body lags v x TAU behind the prey). Tuning on debug seeds only.
usage: SPIDER_TIER=debug gpu-wait run --shared --vram 4 --priority 60 --timeout 1200 --env PYTHONPATH=$PWD --env SPIDER_TIER=debug -- .venv/bin/python tools/spider_tau_sweep.py
"""
import numpy as np

from agent import spider_body_loop as S

L = S.L
rec = L.P.retina(np.random.default_rng(0))
hand = L.P.Net(np.random.default_rng(1), hand=True)
sp = L.P.speeds(np.random.default_rng(S.SEEDS[0]), 6, 9, L.G.B)
for rep in range(2):  # the defaults (SPEED, RAMP, YAW_PER_TURN, TAU, DEAD), twice: run-to-run noise
    for arm in ('noturn', 'hand'):
        h = S.pursue((lambda: hand.trial_reset(L.G.B), hand.step), rec, sp, np.random.default_rng(S.SEEDS[1]), arm)
        f = np.mean(np.sign(h['yaw']) == np.sign(h['bearing']))
        print(f"rep {rep} {arm}: eye error {np.median(h['err']):.2f} deg, |gaze| {np.median(h['gaze']):.2f}, follows {f:.0%}, size {np.median(np.abs(h['yaw']) / np.maximum(np.abs(h['bearing']), 1e-6)):.0%}", flush=True)
