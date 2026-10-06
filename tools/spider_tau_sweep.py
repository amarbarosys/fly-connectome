"""Hand-wired circuit with the walking body, debug tier: eye error and |gaze| per body time constant TAU, deadband and the counter-rotation reflex.
Hand arm at the defaults, repeated: repeatability (identical numbers) and the error split slow (lag) vs fast (wobble), 32 animals.
usage: SPIDER_TIER=debug gpu-wait run --shared --vram 4 --priority 60 --timeout 1200 --env PYTHONPATH=$PWD --env SPIDER_TIER=debug -- .venv/bin/python tools/spider_tau_sweep.py
"""
import numpy as np

from agent import spider_body_loop as S

L = S.L
rec = L.P.retina(np.random.default_rng(0))
sp = L.P.speeds(np.random.default_rng(S.SEEDS[0]), 6, 9, L.G.B)
for rep in range(2):  # the defaults, twice: with hand_net() fresh per run the two must be identical
    for arm in ('noturn', 'hand'):
        h = S.pursue(S.hand_net(), rec, sp, np.random.default_rng(S.SEEDS[1]), arm)
        f = np.mean(np.sign(h['yaw']) == np.sign(h['bearing']))
        lo, hi = S.ci(h['err'])
        print(f"rep {rep} {arm}: eye error {np.median(h['err']):.3f} deg (CI {lo:.2f}-{hi:.2f}), slow {np.median(h['slow']):.2f}, fast {np.median(h['fast']):.2f}, |gaze| {np.median(h['gaze']):.2f}, follows {f:.0%}", flush=True)
