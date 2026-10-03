"""Brain settings (docs/plans/BRAIN_CONTROLS.md): every hand intervention as one declarative setting.

Two device primitives carry everything, so a setting adds no kernel, no launch and no per-cell state:
  scale rules  silence = (cells, any, 0), cut = (A, B, k), gain = (A, any, k): edge groups (1 B per edge) and the
               per-column scale row; a rule is split by modulation plane so a cut dopamine path stays modulatory
  drive        stimulate (Poisson, rate) and inject (constant current) as kick-table entries with amount + window
plus the existing switches: modulation (ShiuBrain.modulators) and record (a path for ShiuLive.start(record=)).
Later rules override earlier ones on the edges they share. Times in ms; dt from the brain.

    s = {'stimulate': [{'cells': {'type': 'LC4', 'side': 'right'}, 'hz': 200, 'from_ms': 100, 'to_ms': 300}],
         'silence': [{'cells': {'type': 'DNp01'}}],
         'cut': [{'pre': {'class': 'ALLN'}, 'post': {'class': 'ALPN'}, 'scale': 0}],
         'gain': [{'pre': {'class': 'Kenyon_Cell'}, 'scale': 4}],
         'inject': [{'cells': {'type': 'MN9'}, 'current': 2.0}],
         'modulation': {'octopamine': {'tau': 1.0, 'b': 0.1}}}
    digest = apply(brain, s)          # or apply(brain, tomllib.load(f))

Selectors: {'ids': [...]} or any of type / class / side / nt / super (FlyWire columns), all must match.
"""
import hashlib

import numpy as np

PLANES = ('dopamine', 'octopamine', 'serotonin')


def select(sel, cells, n):
    """Boolean mask over the n neurons for one selector; None or {} selects every neuron."""
    mask = np.ones(n, bool)
    for key, value in (sel or {}).items():
        if key == 'ids':
            m = np.zeros(n, bool)
            m[np.asarray(value, np.int64)] = True
        else:
            m = getattr(cells, {'class': 'klass'}.get(key, key)) == value
        mask &= m
    return mask


def apply(brain, settings, cells=None):
    """Apply a setting to a ShiuBrain / ShiuGPU / ShiuLive (before start or run); returns the hex digest of the
    exact tables the device gets, which every result should carry."""
    if cells is None:
        from agent.experiment import cells as load
        cells = load()
    n, dt = brain.n, brain.dt
    steps = lambda ms: int(round(ms * 1e-3 / dt))
    if 'modulation' in settings:
        m = settings['modulation']
        brain.modulators(cells.nt, **{k: tuple(m.get(p, {}).get(k, d) for p in PLANES)
                                      for k, d in (('tau', 1.0), ('k', 0.0), ('b', 0.0))})
    rules = [(r['cells'], None, 0.0) for r in settings.get('silence', [])]
    rules += [(r['pre'], r.get('post'), float(r.get('scale', 0.0))) for r in settings.get('cut', [])]
    rules += [(r['pre'], None, float(r['scale'])) for r in settings.get('gain', [])]
    if rules:
        pre = np.repeat(np.arange(n), np.diff(brain.row))
        grp = np.zeros(len(brain.col), np.uint8) if brain.grp is None else brain.grp.copy()
        scale = np.ones((brain.flies, 1), np.float32) if brain.scale is None else brain.scale
        plane = np.full(scale.shape[1], -1, np.int32) if brain.modp is None else brain.modp.copy()
        cols = [scale]
        for p_sel, q_sel, k in rules:
            hit = select(p_sel, cells, n)[pre] & select(q_sel, cells, n)[brain.col]
            for q in np.unique(plane[grp[hit]]):  # one group per (rule, plane): modulatory edges stay modulatory
                g = len(plane)
                if g > 255:
                    raise SystemExit('FAIL settings: more than 256 edge groups')
                grp[hit & (plane[grp] == q)] = g
                plane = np.r_[plane, q]
                cols.append(np.full((brain.flies, 1), k, np.float32))
        brain.edge_groups(grp, np.hstack(cols))
        if brain.modp is not None:  # the new groups' planes, same taus and gains
            brain.modulate(plane, _taus(brain), brain.mk, brain.mb)
    for r in settings.get('stimulate', []):
        brain.stimulate(np.flatnonzero(select(r['cells'], cells, n)), float(r['hz']),
                        window=(steps(r.get('from_ms', 0)), steps(r['to_ms']) if 'to_ms' in r else 2**31 - 1))
    for r in settings.get('inject', []):
        brain.inject(np.flatnonzero(select(r['cells'], cells, n)), float(r['current']),
                     window=(steps(r.get('from_ms', 0)), steps(r['to_ms']) if 'to_ms' in r else 2**31 - 1))
    return digest(brain)


def _taus(brain):
    """tau per existing plane, recovered from the stored decay factors (modulate takes taus)."""
    return -brain.dt / np.log(brain.mdecay.astype(np.float64))


def digest(brain):
    """sha256 over everything the device reads that a setting can change."""
    h = hashlib.sha256()
    entries, p = brain._stim_arrays()
    amount, window = brain._drive_arrays(entries)
    for a in (brain.grp, brain.scale, brain.modp, getattr(brain, 'mk', None), getattr(brain, 'mb', None),
              getattr(brain, 'mdecay', None), entries, p, amount, window):
        h.update(b'-' if a is None else np.ascontiguousarray(a).tobytes())
    return h.hexdigest()
