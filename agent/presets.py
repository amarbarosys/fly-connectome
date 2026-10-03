"""Skill presets per sense: drive each sensory cell type on one side, record which output neurons
fire, keep the stimuli that fire any as presets, and check a preset by replaying it.

A sense is (stimulus cells, readout cells, rate). Stimuli are every type of that sense with at least
MIN_CELLS cells on each side, each side alone, on the authors' graph (agent/verify_shiu), K repeats,
counts over 0-200 ms. Per stimulus: the mean readout vector, cells with z > 4 against silence (noise
floor one spike per K repeats), and the side contrast (cells with |z| > 4 against the mirror
stimulus). A preset = a stimulus that fires at least one readout cell at z > 4: {cells, ids hash,
spikes, signature (mean counts of its z > 4 cells, strongest first), side cells}. check replays every
preset with a fresh seed: PASS when cosine >= 0.9 to the stored signature and the strongest cell is
the same; check writes each preset's result into the file ('verified'), and only verified presets
are skills to use: a preset that does not reproduce under a fresh seed is noise, not a skill.

  vision  visual_projection types -> descending neurons
  taste   gustatory types, plus the authors' 21 sugar GRNs (both sides, 'sugar-both') ->
          descending neurons, brain motor neurons (MN9 proboscis among them) and PAM/PPL dopamine
          neurons (the reward and punishment teachers of the mushroom body)
  sensory mechanosensory, hygrosensory, thermosensory, unknown sensory and ocellar types -> the
          taste readout
  motion  visual_centrifugal types (HS/VS among them) -> descending neurons
  ascending  ascending neuron types (the nerve cord reporting to the brain) -> the taste readout
  trace   CPU only: hops over excitatory edges from the sugar GRNs to every PAM type, and the
          synapses into that type from cells at the previous hop

usage: python -m agent.presets SENSE map     -> .work/agent/shiu/presets/SENSE.json
       python -m agent.presets SENSE check   PASS/FAIL per preset, FAIL k/N non-zero
"""
import hashlib
import json
import re
import sys
from collections import Counter

import numpy as np

from agent.experiment import Experiment, cells
from agent.neural_connectome import ROOT
from agent.verify_shiu import REF, _reference_graph

K, HZ, WINDOWS, MIN_CELLS = 8, 150., (2000,), 8


def sugar():
    """The authors' sugar GRNs (code/benchmark.py), graph indices."""
    index, source = _reference_graph()[3], (REF / 'code/benchmark.py').read_text()
    return np.array([index[int(x)] for x in re.findall(r'7205\d{14}', source[source.index("'sugar'"):source.index("'p9'")])])


def by_type(c, mask, min_cells):
    mask = mask & np.isin(c.side, ('left', 'right')) & (c.type != '')
    count = Counter(zip(c.type[mask], c.side[mask]))
    types = sorted({t for (t, s), v in count.items() if v >= min_cells and count[(t, 'left' if s == 'right' else 'right')] >= min_cells})
    return {f'{t}-{s}': np.flatnonzero(mask & (c.type == t) & (c.side == s)) for t in types for s in ('left', 'right')}


def sense(name, c):
    """(stimuli: name -> cell ids, readout cell ids)."""
    lr = np.isin(c.side, ('left', 'right'))
    dn = (c.super == 'descending') & lr
    if name == 'vision':
        return by_type(c, c.super == 'visual_projection', MIN_CELLS), np.flatnonzero(dn)
    dan = np.char.startswith(c.type.astype(str), 'PAM') | np.char.startswith(c.type.astype(str), 'PPL')
    wide = np.flatnonzero(dn | (c.super == 'motor') | dan)
    if name == 'taste':
        stim = by_type(c, c.klass == 'gustatory', 3)
        stim['sugar-both'] = sugar()
        return stim, wide
    if name == 'sensory':
        return by_type(c, np.isin(c.klass, ('mechanosensory', 'hygrosensory', 'thermosensory', 'unknown_sensory', 'ocellar')), 3), wide
    if name == 'motion':
        return by_type(c, c.super == 'visual_centrifugal', 3), np.flatnonzero(dn)
    if name == 'ascending':
        return by_type(c, c.super == 'ascending', 3), wide
    raise SystemExit(f'unknown sense {name}')


def counts(stim, readout, seed):
    """Mean and variance over K repeats of readout counts at 0-200 ms: (S, R) each, and cell names."""
    c = cells()
    pre, post, w, _ = _reference_graph()
    res = Experiment([(pre, post, w)], len(c.klass), {k: (ids, HZ) for k, ids in stim.items()}, WINDOWS,
                     {'out': readout}, repeats=K, per_cell=('out',), seed=seed).run()
    x = res['cells']['out'][-1, :, 0, 0].reshape(len(readout), len(stim), K).astype(float)
    side = np.array([s[:1] or '-' for s in c.side[readout]])
    names = np.array([f'{t or k}-{s}' for t, k, s in zip(c.type[readout], c.klass[readout], side)])
    return x.mean(2).T, x.var(2, ddof=1).T, names


def ids_hash(ids):
    return hashlib.sha256(np.asarray(ids, np.int64).tobytes()).hexdigest()[:12]


def path(name):
    return ROOT / f'.work/agent/shiu/presets/{name}.json'


def map_presets(name):
    stim, readout = sense(name, cells())
    mean, var, names = counts(stim, readout, seed=71)
    keys = list(stim)
    out = {'sense': name, 'hz': HZ, 'K': K, 'window_ms': 200, 'stimuli': len(stim), 'readout': len(readout), 'presets': {}, 'silent': []}
    for i, k in enumerate(keys):
        z = mean[i] / np.sqrt(var[i] / K + 1 / K)
        base = k.rsplit('-', 1)
        mirror = keys.index(base[0] + ('-right' if base[1] == 'left' else '-left')) if base[1] in ('left', 'right') else i
        zs = (mean[i] - mean[mirror]) / np.sqrt(var[i] / K + var[mirror] / K + 1 / K)
        hit = np.flatnonzero(z > 4)
        if not len(hit):
            out['silent'].append(k)
            continue
        hit = hit[np.argsort(-mean[i, hit])]
        out['presets'][k] = {'cells': len(stim[k]), 'ids_sha': ids_hash(stim[k]), 'spikes': round(float(mean[i].sum()), 1),
                             'signature': {str(names[d]): round(float(mean[i, d]), 2) for d in hit},
                             'side_cells': int((np.abs(zs) > 4).sum())}
    path(name).parent.mkdir(parents=True, exist_ok=True)
    path(name).write_text(json.dumps(out, indent=1) + '\n')
    print(f"{len(out['presets'])} presets, {len(out['silent'])} silent of {len(keys)} stimuli -> {path(name)}")
    for k, p in sorted(out['presets'].items(), key=lambda x: -x[1]['spikes']):
        print(f"  {k:16s} spikes {p['spikes']:8.1f} side {p['side_cells']:4d}  {list(p['signature'].items())[:6]}")


def check(name):
    saved = json.loads(path(name).read_text())
    stim, readout = sense(name, cells())
    stim = {k: v for k, v in stim.items() if k in saved['presets']}
    mean, _, names = counts(stim, readout, seed=72)
    bad = []
    for i, k in enumerate(stim):
        p = saved['presets'][k]
        ref = np.array([p['signature'].get(str(n), 0.) for n in names])
        cos = float(mean[i] @ ref / (np.linalg.norm(mean[i]) * np.linalg.norm(ref) + 1e-9))
        top, stored = str(names[np.argmax(mean[i])]), next(iter(p['signature']))
        ok = ids_hash(stim[k]) == p['ids_sha'] and cos >= .9 and top == stored
        p['verified'], p['replay_cos'] = ok, round(cos, 3)
        print('PASS' if ok else 'FAIL', f'{k:16s} cos {cos:.3f} top {top} (stored {stored})')
        bad += [] if ok else [k]
    path(name).write_text(json.dumps(saved, indent=1) + '\n')
    print(f'{len(stim) - len(bad)}/{len(stim)} presets verified -> {path(name)}')
    if bad:
        print(f'FAIL {len(bad)}/{len(stim)} {" ".join(bad)}')
        sys.exit(1)


def trace():
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    c = cells()
    pre, post, w, _ = _reference_graph()
    n, exc = len(c.klass), np.asarray(w) > 0
    g = csr_matrix((np.ones(exc.sum()), (np.asarray(pre)[exc], np.asarray(post)[exc])), (n, n))
    src = sugar()
    hops = shortest_path(g, unweighted=True, indices=src).min(0)
    syn = csr_matrix((np.asarray(w)[exc], (np.asarray(pre)[exc], np.asarray(post)[exc])), (n, n)).tocsc()
    out = {}
    for t in sorted({t for t in c.type if t.startswith('PAM')}):
        ids = np.flatnonzero(c.type == t)
        h = hops[ids].min()
        col = syn[:, ids]
        prev = np.flatnonzero(hops == h - 1) if np.isfinite(h) else []
        into = float(col[prev].sum()) if len(prev) else 0.
        out[t] = {'cells': len(ids), 'min_hops': None if not np.isfinite(h) else int(h),
                  'synapses_from_previous_hop': into, 'synapses_total_in': float(col.sum())}
        print(f"{t:6s} cells {len(ids):3d} hops {out[t]['min_hops']}  from hop-1 {into:7.0f} of {out[t]['synapses_total_in']:7.0f} excitatory in")
    path('trace-sugar-pam').write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    trace() if sys.argv[1] == 'trace' else {'map': map_presets, 'check': check}[sys.argv[2]](sys.argv[1])
