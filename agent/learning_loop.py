"""A Shiu fly that learns where food is: reward dopamine on every bite, in the closed loop.

Graph: ALLN inhibition x LN (no flood), KC -> MBON x GAIN, KC -> avoid-MBON edges plastic (one
factor per KC per fly, agent/verify_plastic.py). Every step the fly sniffs (food and decoy ORNs at
HZ x the map's concentration, strongest = HZ) and reads the valence index (approach - avoid) /
(approach + avoid) of the labelled MBONs (agent/mb_valence.py) at 100 ms. The fly sniffs each odour
alone (odour_valence) at the start of a trial and at most every RESNIFF steps after its factors
changed (learning itself applies at once; the walker sees it up to RESNIFF steps late): the valence of
a mixture is taken as the concentration-weighted sum. Re-sniffing after every learning step cost 150
to 160 sniffs per trial in the pain loop (some fly is punished nearly every step), more than the 100
of the old walker. It steers by tropotaxis (maps.steer: two antennae,
turn toward the side whose odours are worth more), with a small random turn per step. The first
walker (keep heading while the mixture's valence rises, else tumble) let learning change valence
(pure-odour probes, agent/pain_loop.py, c869304) without changing behaviour. A bite releases dopamine LR x hunger;
hunger rises 1/STEPS per step and drops BITE per bite. Learners and twins (dopamine 0) share
starts, mirrors and tumbles; factors persist over TRIALS, positions reset. Food VM2+DM1+VA2 vs decoy
DA1+VA1d+DL3: the naive fly prefers the decoy (valence +0.26 vs +0.55 in mbon_floor).
Pass: in the last third of trials learners out-score their twins, exact sign-flip over pairs.
"""
import json
import sys
import time

import numpy as np

from agent import maps
from agent.brain_bench import ODORS6
from agent.experiment import cells
from agent.mb_valence import mbons
from agent.neural_connectome import ROOT
from agent.paired_significance import _exact_sign_flip
from agent.shiu import ShiuLive, graph_dir
from agent.verify_shiu import _reference_graph

FOOD, DECOY = ODORS6[1], ODORS6[0]
LN, GAIN, HZ, SEED = 12, 4, 200., 91
PAIRS, STEPS, TRIALS, STEP, LR, BITE, WINDOW, NOISE, RESNIFF = 48, 100, 6, .15, .5, .3, 1000, .2, 10


def setup():
    pre, post, w, _ = _reference_graph()
    c = cells()
    names, n = c.klass, len(c.klass)
    approach, avoid = mbons(c)
    kc = np.flatnonzero(names == 'Kenyon_Cell')
    graph_dir(pre, post, w, n)  # pre is sorted, so edge groups follow CSR order
    is_kc, is_app, is_avo = (np.isin(np.arange(n), x) for x in (kc, approach, avoid))
    grp = np.zeros(len(pre), np.uint8)
    grp[(names[pre] == 'ALLN') & (w < 0)] = 1
    grp[is_kc[pre] & is_app[post]] = 2
    grp[is_kc[pre] & is_avo[post]] = 3
    odour = [np.flatnonzero((names == 'olfactory') & np.isin(c.type, ['ORN_' + g for g in o])) for o in (FOOD, DECOY)]
    return pre, post, w, n, grp, kc, approach, avoid, odour


def brain(engine, pre, post, w, n, grp, kc, odour, flies):
    b = engine(pre, post, w, n=n, flies=flies, seed=SEED)
    for f in range(flies):
        for ids in odour:
            b.stimulate(ids, HZ, fly=f)
    b.edge_groups(grp, np.tile(np.float32([1, LN, GAIN, GAIN]), (flies, 1)))
    b.plastic(3, kc)
    return b


def smell(live, m, pos, sources, sign, column, kind):
    """Sniff the mixture each fly smells where it stands (strongest odour at HZ), so that reward() credits
    the KCs of the moment of learning. reward() acts on the KCs of the LAST sniff: after a valence probe
    that is odour B alone, and crediting B whatever the fly smelled sank the reward loop (p=0.86)."""
    conc = maps.concentration(m, pos, sources, sign)
    strongest = conc.max(1, keepdims=True)
    conc = np.where(strongest > 1e-3, conc / np.maximum(strongest, 1e-12), 0)
    live.sniff((HZ * live.dt * conc[column, kind]).astype(np.float32))


def odour_valence(live, only, n_approach, flies):
    """(F, 2) valence of each odour sniffed alone; `only` = per-odour Poisson probabilities."""
    out = []
    for p in only:
        x = live.sniff(p)[-1].reshape(-1, flies).astype(float)
        app, avo = x[:n_approach].sum(0), x[n_approach:].sum(0)
        out.append(np.where(app + avo > 0, (app - avo) / np.maximum(app + avo, 1), 0.))
    return np.stack(out, 1)


def main(map_name='tmaze'):
    pre, post, w, n, grp, kc, approach, avoid, odour = setup()
    flies, m, rng = 2 * PAIRS, maps.build(map_name), np.random.default_rng(20261003)
    live = brain(ShiuLive, pre, post, w, n, grp, kc, odour, flies)
    mbon = np.concatenate([approach, avoid])
    live.start([WINDOW], (np.arange(flies)[None] * n + mbon[:, None]).reshape(-1))
    entries, p0 = live._stim_arrays()
    column, kind = entries // n, np.where(np.isin(entries % n, odour[0]), 0, 1)
    only = [np.where(kind == k, p0, 0).astype(np.float32) for k in (0, 1)]
    sign = np.tile(np.where(np.arange(PAIRS) % 2, 1., -1.), 2)
    learner = np.arange(flies) < PAIRS
    hunger, scores, started = np.ones(flies), [], time.perf_counter()
    for trial in range(TRIALS):
        walls, sources = maps.mirrored(m, sign)
        pos, head = maps.starts(m, sign[:PAIRS], rng)
        pos, head = np.tile(pos, (2, 1)), np.tile(head, 2)
        tumble = np.tile(rng.uniform(-np.pi, np.pi, (STEPS + 1, PAIRS)), (1, 2))
        bites, sniffs, stale = np.zeros((flies, 2)), 2, False
        valence = odour_valence(live, only, len(approach), flies)
        for t in range(STEPS):
            head = maps.steer(m, pos, head, sources, sign, valence) + NOISE * tumble[t]
            pos, blocked = maps.move(pos, head, STEP, walls)
            head = np.where(blocked, head + tumble[t + 1], head)
            bite = maps.eat(pos, sources)
            bites += bite
            hunger = np.minimum(1, hunger + 1 / STEPS)
            fed = bite[:, 0] > 0
            if (fed & learner).any():
                smell(live, m, pos, sources, sign, column, kind)
                sniffs += 1
                live.reward(np.where(fed & learner, LR * hunger, 0).astype(np.float32))
                stale = True
            if stale and t % RESNIFF == RESNIFF - 1:
                valence, sniffs, stale = odour_valence(live, only, len(approach), flies), sniffs + 2, False
            hunger = np.where(fed, np.maximum(0, hunger - BITE), hunger)
        scores.append(bites[:, 0] - bites[:, 1])
        print(f"trial {trial}: learners {scores[-1][learner].mean():+.2f}  twins {scores[-1][~learner].mean():+.2f}  "
              f"food bites {bites[learner, 0].sum():.0f}/{bites[~learner, 0].sum():.0f}  sniffs {sniffs}  {time.perf_counter() - started:.0f} s", flush=True)
    live.close()
    scores = np.array(scores)  # (trial, fly)
    late = scores[-max(1, TRIALS // 3):].mean(0)
    diff = late[:PAIRS] - late[PAIRS:]
    p, _ = _exact_sign_flip(list(diff), scale=1000)
    out = {'map': map_name, 'food': list(FOOD), 'decoy': list(DECOY), 'pairs': PAIRS, 'steps': STEPS, 'trials': TRIALS,
           'lr': LR, 'curve_learners': scores[:, :PAIRS].mean(1).tolist(), 'curve_twins': scores[:, PAIRS:].mean(1).tolist(),
           'late_learners': float(late[:PAIRS].mean()), 'late_twins': float(late[PAIRS:].mean()),
           'pairs_learner_higher': int((diff > 0).sum()), 'exact_two_sided_sign_flip_p': p,
           'wall_s': time.perf_counter() - started}
    out['pass'] = bool(out['late_learners'] > out['late_twins'] and p < .05)
    print(f"late learners {out['late_learners']:+.2f} twins {out['late_twins']:+.2f}  higher {out['pairs_learner_higher']}/{PAIRS}  "
          f"p={p:.4g}  {'PASS' if out['pass'] else 'FAIL'}", flush=True)
    (ROOT / f'.work/agent/shiu/learning-loop-{map_name}.json').write_text(json.dumps(out, indent=2) + '\n')
    return out['pass']


if __name__ == '__main__':
    sys.exit(0 if main(*sys.argv[1:2]) else 1)
