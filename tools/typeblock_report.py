"""Score the preregistered typeblock arms (docs/plans/TYPEBLOCK_PREREG.md) with the G4 metric of
tools/learning_specificity_report.py: A valence gap after the A reward, per seed, real vs each arm.

    .venv/bin/python tools/typeblock_report.py   -> table, verdict, writes specificity/typeblock-report.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from learning_specificity_report import D, load, gaps, ci, sign_p  # noqa: E402

RECOVERS, FAILS = .37, .10  # fixed in the preregistration


def latest(name):
    runs = sorted(D.glob(f'runs/*/{name}-0.json'))
    if not runs: raise SystemExit(f'FAIL missing receipt runs/*/{name}-0.json')
    return json.loads(runs[-1].read_text())['records'], str(runs[-1].relative_to(D))


def main():
    real = gaps(load('run-real'), 'after_A', 'o1')
    rows, means = {}, []
    for k in range(5):
        recs, path = latest(f'run-typeblock{k}')
        if len(recs) != len(real): raise SystemExit(f'FAIL typeblock{k}: {len(recs)} seeds, real has {len(real)}')
        g = gaps(recs, 'after_A', 'o1')
        w, l, p = sign_p(real, g)
        rows[f'typeblock{k}'] = {'gap_ci': ci(g), 'real_won': w, 'real_lost': l, 'sign_p': p, 'receipt': path}
        means.append(g.mean())
    block = [gaps(load(f'run-block{k}'), 'after_A', 'o1').mean() for k in range(5)]
    up, down = sum(m >= RECOVERS for m in means), sum(m <= FAILS for m in means)
    verdict = 'RECOVERS' if up >= 4 else 'FAILS' if down >= 4 else 'PARTIAL'
    out = {'real_gap_ci': ci(real), 'class_block_means': [round(float(m), 4) for m in block],
           'typeblock': rows, 'verdict': verdict, 'thresholds': {'recovers': RECOVERS, 'fails': FAILS}}
    (D/'typeblock-report.json').write_text(json.dumps(out, indent=2))
    print(f"real A gap {ci(real)}   class-block means {out['class_block_means']}")
    for name, r in rows.items():
        print(f"{name}: gap {r['gap_ci']}  real won {r['real_won']}/{r['real_won'] + r['real_lost']}  sign p {r['sign_p']:.4f}")
    print(f'VERDICT {verdict} (>= {RECOVERS} in {up}/5, <= {FAILS} in {down}/5)')


if __name__ == '__main__':
    main()
