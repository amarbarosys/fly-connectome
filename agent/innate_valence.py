"""Innate odour valence in Shiu descending neurons: does the real wiring carry a like/dislike axis?

Five glomeruli with published innate valence, each stimulated alone (its ORNs at HZ):
attractive DM1, VA2 (Semmelhack and Wang 2009, Nature 459:218); aversive DM5 (same paper, high
vinegar), DA2 (geosmin, Stensmyr et al. 2012, Cell 151:1345), DL4 (parasitoid odours, Ebrahim et al.
2015, PLoS Biol 13:e1002317). For each held-out glomerulus, the valence axis is the mean DN response
to the other aversive glomeruli minus the other attractive ones (unit vectors, 0-200 ms); a repeat
of the held-out glomerulus scores 1 when its projection has the published sign. All 11 arms (real,
5 shuffled, 5 block) x two graphs (authors'; antennal-lobe inhibition x12, agent/shiu_apl.py), as
columns. Inference: exact sign-flip over the 5 x K held-out repeats of real minus the family mean.
"""
import json

import numpy as np

from agent import shiu_g1
from agent.experiment import Experiment, cells
from agent.neural_connectome import ROOT
from agent.paired_significance import _exact_sign_flip

VALENCE = {'DM1': +1, 'VA2': +1, 'DM5': -1, 'DA2': -1, 'DL4': -1}  # +1 attractive
K, HZ, WINDOW = 8, 200., 2000


def score(x):
    """x (dn, glomerulus, repeat) -> (glomerulus, repeat): 1 where the held-out projection has the right sign,
    0 where the wrong one, 0.5 where it is zero (silent DNs make no call, which is chance, not an error)."""
    unit = x / (np.linalg.norm(x, axis=0, keepdims=True) + 1e-9)
    mean = unit.mean(2)
    sign = np.array(list(VALENCE.values()))
    out = np.zeros(x.shape[1:])
    for g in range(len(sign)):
        rest = np.arange(len(sign)) != g
        axis = mean[:, rest & (sign > 0)].mean(1) - mean[:, rest & (sign < 0)].mean(1)
        call = np.sign(axis @ unit[:, g])
        out[g] = np.where(call == 0, .5, call == sign[g])
    return out


def main():
    c = cells()
    names = c.klass
    dn = np.flatnonzero(c.super == 'descending')
    stimuli = {g: (np.flatnonzero((names == 'olfactory') & (c.type == 'ORN_' + g)), HZ) for g in VALENCE}
    arms, graphs = zip(*[(name, g) for name, g, _ in shiu_g1.arms(shiu_g1.flywire(), 'flywire')])
    res = Experiment(list(graphs), len(names), stimuli, (WINDOW,), {'dn': dn},
                     edges={'ln': lambda p, q, w: (names[p] == 'ALLN') & (w < 0)},
                     variants=[{}, {'ln': 12}], repeats=K, per_cell=('dn',), seed=81).run()
    x = res['cells']['dn'][-1].astype(float)  # (dn, arm, variant, glomerulus, repeat)
    out = {'valence': VALENCE, 'dns': len(dn), 'K': K, 'graphs': {}}
    for v, graph in enumerate(('authors', 'alln_x12')):
        hits = {arm: score(x[:, a, v]) for a, arm in enumerate(arms)}
        spikes = {arm: float(x[:, a, v].sum(0).mean()) for a, arm in enumerate(arms)}
        row = {'accuracy': {arm: float(h.mean()) for arm, h in hits.items()}, 'dn_spikes': spikes,
               'per_glomerulus_real': dict(zip(VALENCE, hits['real'].mean(1).tolist())), 'tests': {}}
        for family in ('shuffled', 'block'):
            control = np.mean([hits[a] for a in arms if a.startswith(family)], axis=0)
            p, _ = _exact_sign_flip(list((hits['real'] - control).ravel()), scale=10000)
            row['tests'][family] = {'real': float(hits['real'].mean()), 'control': float(control.mean()), 'p': p}
        out['graphs'][graph] = row
        print(graph, 'real', f"{row['accuracy']['real']:.2f}", 'per glomerulus', {g: round(s, 2) for g, s in row['per_glomerulus_real'].items()},
              'DN', f"{spikes['real']:.0f}", flush=True)
        for family, t in row['tests'].items():
            print(f"  {family:8s} control {t['control']:.2f}  p={t['p']:.4g}   arms " +
                  ' '.join(f"{row['accuracy'][a]:.2f}/{spikes[a]:.0f}DN" for a in arms if a.startswith(family)), flush=True)
    (ROOT / '.work/agent/shiu/innate-valence.json').write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()
