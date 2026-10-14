"""Evolve the blank pool instead of teaching it (vero self-replication study 2026-10-14, soup lesson: selection on a
scalar works where a gradient-like rule on it fails). spider_blank_pool_test: the scalar reward -|slip| through the
three-factor rule did NOT learn (1.32-1.40 deg). Here the same kind of scalar, the median foveal error on training
speeds 2-5 deg/s, only selects: POP sparse genotypes (the pool's w_in and w_out at the 600-synapse budget, started
like the regrow: random +-1.5 THETA), the ELITE best copied with copy errors as the only variation (SeRANN): gaussian
SIG on every live synapse plus MOVE synapses pruned and regrown at random. Same Net, task, plant and test set.
Arms: evolve | shuffle (fitness permuted across the population every generation: selection on noise, the control).
Gates, frozen before the claim run (held-out 6-9 deg/s, test seed 9 as spider_blank_pool_test, hand there 0.21):
  E1 evolve best after GEN generations < 1.5 x hand
  E2 shuffle best >= 2 x evolve best
  E3 evolve generation-0 best > 2 x hand (selection did the work, not a lucky start)
Seeds claimed in docs/seeds_used.json (spider_evolve); --seeds E S replicates with fresh seeds (receipt evolve-E.json).
--smoke: 2 generations of 4, no seed claim, no gates.
usage: .venv/bin/python tools/spider_evolve_test.py  -> .work/agent/spider/evolve.json
"""
import copy
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import spider_blank_pool_test as bp  # noqa: E402
from agent.seeds import claim  # noqa: E402

SMOKE = '--smoke' in sys.argv
POP, ELITE, GEN, SIG, MOVE = (4, 2, 2, 0.05, 10) if SMOKE else (32, 8, 150, 0.05, 10)
ARG = sys.argv[sys.argv.index('--seeds') + 1:][:2] if '--seeds' in sys.argv else []
SEEDS = {'evolve': int(ARG[0]), 'shuffle': int(ARG[1])} if ARG else {'evolve': 501, 'shuffle': 502}
REC = []


def error(args):
    """Median foveal error of one genotype on (lo, hi) deg/s speeds drawn from seed."""
    net, seed, lo, hi, n = args
    REC or REC.append(bp.retina(np.random.default_rng(0)))
    rng = np.random.default_rng(seed)
    return float(np.median(bp.run(net, REC[0], bp.speeds(rng, lo, hi, n), rng)))


def child(parent, rng):
    c = copy.deepcopy(parent)
    c.rng = np.random.default_rng(rng.integers(1 << 62))
    for w in (c.w_in, c.w_out):
        live = np.flatnonzero(w)
        w.flat[live] += rng.normal(0, SIG, len(live)).astype(np.float32)
    c.w_in.flat[rng.choice(np.flatnonzero(c.w_in), MOVE, replace=False)] = 0  # moved: regrown at random below
    c.learn(None)  # clip, prune |w| < THETA, regrow to the budget
    return c


def evolve(arm, ex):
    rng = np.random.default_rng(SEEDS[arm])
    pop = [bp.Net(np.random.default_rng(rng.integers(1 << 62))) for _ in range(POP)]
    for n in pop:
        n.learn(None)
    curve, gen0 = [], None
    for g in range(GEN):
        s = int(rng.integers(1 << 62))
        fit = np.array(list(ex.map(error, [(n, s, 2, 5, 32) for n in pop])))
        if g == 0:
            gen0 = error((pop[int(np.argmin(fit))], 9, 6, 9, 256))
        curve.append(round(float(fit.min()), 3))
        order = np.argsort(rng.permutation(fit) if arm == 'shuffle' else fit)
        elite = [pop[i] for i in order[:ELITE]]
        pop = elite + [child(elite[i % ELITE], rng) for i in range(POP - ELITE)]
        if g % 10 == 9:
            print(arm, g + 1, curve[-1], flush=True)
    return {'gen0': gen0, 'after': error((pop[0], 9, 6, 9, 256)), 'curve': curve}


def main():
    if not SMOKE:
        claim('spider_evolve', list(SEEDS.values()))
    hand = error((bp.Net(np.random.default_rng(1), hand=True), 9, 6, 9, 256))
    with ProcessPoolExecutor(16) as ex:  # resident workers for every generation of both arms
        out = {'hand': hand} | {arm: evolve(arm, ex) for arm in SEEDS}
    if SMOKE:
        print(json.dumps(out)); return
    e, sh = out['evolve'], out['shuffle']
    gates = {'E1': e['after'] < 1.5 * hand, 'E2': sh['after'] >= 2 * e['after'], 'E3': e['gen0'] > 2 * hand}
    out['gates'] = gates
    (bp.ROOT / '.work/agent/spider').mkdir(parents=True, exist_ok=True)
    (bp.ROOT / f".work/agent/spider/evolve{'-' + ARG[0] if ARG else ''}.json").write_text(json.dumps(out, indent=2) + '\n')
    print(f"E1 evolved {e['after']:.2f} deg vs hand {hand:.2f} -> {'PASS' if gates['E1'] else 'FAIL'}")
    print(f"E2 shuffled-fitness control {sh['after']:.2f} -> {'PASS' if gates['E2'] else 'FAIL'}")
    print(f"E3 generation-0 best {e['gen0']:.2f} -> {'PASS' if gates['E3'] else 'FAIL'}")
    bad = 3 - sum(gates.values())
    if bad:
        print(f"FAIL {bad}/3 checks"); sys.exit(1)


if __name__ == '__main__':
    main()
