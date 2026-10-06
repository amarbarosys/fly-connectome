"""Gates G0-G7 of docs/plans/LEARNING_SPECIFICITY.md from the .work/agent/shiu/specificity receipts. Thresholds are the
frozen ones (c273b7d + amendment 87bfbd4); nothing here is tuned to the data.

    .venv/bin/python tools/learning_specificity_report.py   -> prints every gate, writes report.json, exits 1 on a missing arm
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

D = Path(__file__).resolve().parents[1] / '.work/agent/shiu/specificity'
NOISE_VAL_SD = {0: .081, 1: .014, 2: .109, 3: .031, 4: .023, 5: .061}
DN_THRESHOLD = 17.8
NULLS = [f'shuffle{k}' for k in range(5)] + [f'block{k}' for k in range(5)] + ['random0']


def load(name):
    f = D / f'{name}-0.json'
    if not f.exists(): raise SystemExit(f'FAIL missing receipt {f}')
    return json.loads(f.read_text())['records']


def gap(rec, phase, odour, field='valence'):
    x = np.array(rec[phase][odour][field])
    h = len(x) // 2
    return x[:h].mean() - x[h:].mean()


def gaps(recs, phase, odour, field='valence'):
    return np.array([gap(r, phase, odour, field) for r in recs])


def ci(x):
    m, se = x.mean(), x.std(ddof=1) / math.sqrt(len(x))
    h = stats.t.ppf(.975, len(x) - 1) * se
    return [round(float(m), 4), round(float(m - h), 4), round(float(m + h), 4)]


def sign_p(a, b):
    w, l = int((a > b).sum()), int((a < b).sum())
    return w, l, float(stats.binomtest(w, w + l, .5).pvalue) if w + l else 1.0


def main():
    out, gates = {}, {}
    noise = load('noise-real')
    g0 = {f'o{o}': ci(gaps(noise, 'after_A', f'o{o}')) for o in range(6)}
    gates['G0 instrument'] = all(abs(g0[f'o{o}'][0]) <= 3 * NOISE_VAL_SD[o] / math.sqrt(len(noise)) for o in range(6))
    out['G0'] = g0
    real = load('run-real')
    a = gaps(real, 'after_A', 'o1')
    out['G1 A gap'] = ci(a)
    gates['G1 learning'] = int((a > .3).sum()) >= 7
    g2 = {}
    for o in range(2, 6):
        x = np.abs(gaps(real, 'after_A', f'o{o}'))
        g2[f'o{o}'] = {'median_abs': round(float(np.median(x)), 4), 'ci': ci(gaps(real, 'after_A', f'o{o}')),
                       'pass': bool(np.median(x) < .5 * np.median(a) and np.median(x) < 3 * NOISE_VAL_SD[o] + .1)}
    out['G2'] = g2
    gates['G2 specificity'] = all(v['pass'] for v in g2.values())
    b = gaps(real, 'after_reversal', 'o0')
    out['G3 B gap after reversal'] = ci(b)
    out['G3 A gap after reversal (reported)'] = ci(gaps(real, 'after_reversal', 'o1'))
    gates['G3 reversal'] = int((b > .3).sum()) >= 7
    g4 = {}
    for name in NULLS:
        n = gaps(load(f'run-{name}'), 'after_A', 'o1')
        w, l, p = sign_p(a, n)
        g4[name] = {'null_A_gap': ci(n), 'real_higher': w, 'null_higher': l, 'p': round(p, 5), 'pass': bool(w == len(a) and p < .05)}
    out['G4'] = g4
    gates['G4 wiring'] = all(v['pass'] for v in g4.values())
    dn = gaps(real, 'after_A', 'o1', 'dn')
    held = np.mean([np.abs(gaps(real, 'after_A', f'o{o}', 'dn')).mean() for o in range(2, 6)])
    same = max(int((dn > 0).sum()), int((dn < 0).sum()))
    out['G5 A DN gap'] = ci(dn)
    out['G5 held-out mean |DN gap|'] = round(float(held), 2)
    gates['G5 DN reach'] = bool(abs(dn.mean()) > DN_THRESHOLD and same >= 7 and abs(dn.mean()) > held)
    kc = gaps(load('run-cutkc_mbon'), 'after_A', 'o1')
    mo = gaps(load('run-cutmbon_out'), 'after_A', 'o1', 'dn')
    out['G6 cut:kc_mbon A gap'] = ci(kc)
    out['G6 cut:mbon_out A DN gap'] = ci(mo)
    gates['G6 causal cuts'] = bool(np.median(kc) < .1 and abs(mo.mean()) < DN_THRESHOLD and abs(mo.mean()) < abs(dn.mean()))
    g7 = {}
    for s in (0.5, 2.0):
        ai = gaps(real, 'after_A_intensity', f'o1x{s}')
        held_ok = {f'o{o}': bool(np.median(np.abs(gaps(real, 'after_A_intensity', f'o{o}x{s}'))) < .5 * np.median(ai)) for o in range(2, 6)}
        g7[f'x{s}'] = {'A_gap': ci(ai), 'A_7of8': int((ai > .15).sum()) >= 7, 'held_out_spared': held_ok}
    out['G7'] = g7
    gates['G7 intensity'] = all(v['A_7of8'] and all(v['held_out_spared'].values()) for v in g7.values())
    out['gates'] = gates
    for k, v in out.items():
        print(k, json.dumps(v))
    (D / 'report.json').write_text(json.dumps(out, indent=2) + '\n')
    print('\n'.join(f"{'PASS' if v else 'FAIL'} {k}" for k, v in gates.items()))


if __name__ == '__main__':
    main()
