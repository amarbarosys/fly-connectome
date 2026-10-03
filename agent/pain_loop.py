"""A Shiu fly that learns to stay out of a trap: punishment on trap contact, in the closed loop.

Graph, walker (tropotaxis on per-odour valence, re-sniffed only after learning) and map as agent/learning_loop.py, with the decoy (odour B) as the trap: it is never
used up, and each step a fly is within EAT of it is a contact. Three arms of PAIRS flies share starts,
mirrors and tumbles: pain (punishment PUN on contact, the KC -> approach plane, agent/verify_pain.py),
relief (pain plus reward RELIEF on each step that takes it further from the trap within ESCAPE steps
of a contact, the KC -> avoid plane: whatever it smells while escaping gains value, trap odour too)
and twins (nothing). Food bites still score but release no dopamine, so only pain teaches.
After each trial every fly sniffs A alone and B alone (the parked diagnostic: does learning reach the
odour valence, and does the walker use it).
Pass: in the last third of trials pain learners touch the trap less than their twins, exact sign-flip
over pairs. The pain learner with the fewest late contacts is packed as a preset
(agent/fly_checkpoint.py) to .work/agent/shiu/flies/.

usage: python -m agent.pain_loop [MAP]
"""
import json
import sys
import time

import numpy as np

from agent import maps
from agent.fly_checkpoint import pack
from agent.learning_loop import NOISE, RESNIFF, STEP, STEPS, TRIALS, WINDOW, odour_valence, setup, smell
from agent.neural_connectome import ROOT
from agent.paired_significance import _exact_sign_flip
from agent.shiu import ShiuLive
from agent.verify_pain import brain

PAIRS, PUN, RELIEF, ESCAPE = 32, .5, .2, 5


def main(map_name='tmaze'):
    world = setup()
    pre, post, w, n, grp, kc, approach, avoid, odour = world
    flies, m, rng = 3 * PAIRS, maps.build(map_name), np.random.default_rng(20261003)
    live = brain(ShiuLive, pre, post, w, n, grp, kc, odour, flies)
    mbon = np.concatenate([approach, avoid])
    live.start([WINDOW], (np.arange(flies)[None] * n + mbon[:, None]).reshape(-1))
    entries, p0 = live._stim_arrays()
    column, kind = entries // n, np.where(np.isin(entries % n, odour[0]), 0, 1)
    only = [np.where(kind == k, p0, 0).astype(np.float32) for k in (0, 1)]
    sign = np.tile(np.where(np.arange(PAIRS) % 2, 1., -1.), 3)
    arm = np.repeat(np.arange(3), PAIRS)  # 0 pain, 1 relief, 2 twin
    learner = arm < 2

    contacts, food, probes, started = [], [], [], time.perf_counter()
    for trial in range(TRIALS):
        walls, sources = maps.mirrored(m, sign)
        trap = sources[..., 2] == 1
        pos, head = maps.starts(m, sign[:PAIRS], rng)
        pos, head = np.tile(pos, (3, 1)), np.tile(head, 3)
        tumble = np.tile(rng.uniform(-np.pi, np.pi, (STEPS + 1, PAIRS)), (1, 3))
        touch, bites, since, sniffs, stale = np.zeros(flies), np.zeros(flies), np.full(flies, ESCAPE + 1), 2, False
        valence = odour_valence(live, only, len(approach), flies)
        gap = lambda: np.where(trap, np.linalg.norm(pos[:, None] - sources[..., :2], axis=2), np.inf).min(1)
        for t in range(STEPS):
            head = maps.steer(m, pos, head, sources, sign, valence) + NOISE * tumble[t]
            before = gap()
            pos, blocked = maps.move(pos, head, STEP, walls)
            head = np.where(blocked, head + tumble[t + 1], head)
            amount = sources[..., 3].copy()
            bite = maps.eat(pos, sources)
            sources[..., 3] = np.where(trap, amount, sources[..., 3])  # the trap is never used up
            hit = bite[:, 1] > 0
            touch += hit
            bites += bite[:, 0]
            since = np.where(hit, 0, since + 1)
            pun = np.where(hit & learner, PUN, 0).astype(np.float32)
            rel = np.where((arm == 1) & ~hit & (since <= ESCAPE) & (gap() > before), RELIEF, 0).astype(np.float32)
            if pun.any() or rel.any():
                smell(live, m, pos, sources, sign, column, kind)  # credit the KCs of what it smells now
                sniffs += 1
                live.reward(rel, pun)
                stale = True
            if stale and t % RESNIFF == RESNIFF - 1:
                valence, sniffs, stale = odour_valence(live, only, len(approach), flies), sniffs + 2, False
        probes.append(odour_valence(live, only, len(approach), flies).T)
        contacts.append(touch)
        food.append(bites)
        print(f"trial {trial}: trap contacts pain {touch[arm == 0].mean():.2f} relief {touch[arm == 1].mean():.2f} twin {touch[arm == 2].mean():.2f}  "
              f"valence B pain {probes[-1][1][arm == 0].mean():+.2f} relief {probes[-1][1][arm == 1].mean():+.2f} twin {probes[-1][1][arm == 2].mean():+.2f}  "
              f"sniffs {sniffs}  {time.perf_counter() - started:.0f} s", flush=True)
    factor = live.factors()
    live.close()
    contacts, food, probes = np.array(contacts), np.array(food), np.array(probes)  # (trial, fly), (trial, odour, fly)
    late = contacts[-max(1, TRIALS // 3):].mean(0)
    out = {'map': map_name, 'pairs': PAIRS, 'steps': STEPS, 'trials': TRIALS, 'pun': PUN, 'relief': RELIEF, 'escape': ESCAPE, 'arms': {}}
    for a, name in ((0, 'pain'), (1, 'relief')):
        diff = late[arm == 2] - late[arm == a]  # positive: learner touches less
        p, _ = _exact_sign_flip(list(diff), scale=1000)
        out['arms'][name] = {'late_contacts': float(late[arm == a].mean()), 'late_contacts_twin': float(late[arm == 2].mean()),
                             'pairs_fewer': int((diff > 0).sum()), 'exact_two_sided_sign_flip_p': p,
                             'curve_contacts': contacts[:, arm == a].mean(1).tolist(), 'curve_food': food[:, arm == a].mean(1).tolist(),
                             'curve_valence_A': probes[:, 0, arm == a].mean(1).tolist(), 'curve_valence_B': probes[:, 1, arm == a].mean(1).tolist()}
        print(f"{name}: late contacts {late[arm == a].mean():.2f} vs twins {late[arm == 2].mean():.2f}  fewer {(diff > 0).sum()}/{PAIRS}  p={p:.4g}", flush=True)
    out['arms']['twin'] = {'curve_contacts': contacts[:, arm == 2].mean(1).tolist(), 'curve_food': food[:, arm == 2].mean(1).tolist(),
                           'curve_valence_A': probes[:, 0, arm == 2].mean(1).tolist(), 'curve_valence_B': probes[:, 1, arm == 2].mean(1).tolist()}
    pain = out['arms']['pain']
    out['pass'] = bool(pain['late_contacts'] < pain['late_contacts_twin'] and pain['exact_two_sided_sign_flip_p'] < .05)
    (ROOT / f'.work/agent/shiu/pain-loop-{map_name}.json').write_text(json.dumps(out, indent=2) + '\n')
    best = int(np.flatnonzero(arm == 0)[np.argmin(late[arm == 0])])
    flies_dir = ROOT / '.work/agent/shiu/flies'
    flies_dir.mkdir(parents=True, exist_ok=True)
    pack(flies_dir / f'trap-avoider-{map_name}.npz', world, factor[best], f'trap-avoider {map_name} column {best}')
    print('PASS' if out['pass'] else 'FAIL pain loop: pain learners do not touch the trap less than twins', flush=True)
    return out['pass']


if __name__ == '__main__':
    sys.exit(0 if main(*sys.argv[1:2]) else 1)
