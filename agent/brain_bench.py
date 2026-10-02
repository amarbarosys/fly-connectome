"""Body-free CPU sweep for olfactory population dynamics and adaptation."""
from __future__ import annotations

import csv
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

from agent.neural_connectome import ANNOTATIONS, N, ROOT, CPUBrain, normalize_weights_inplace
from agent.mb_plasticity import mushroom_body_cells, odor_drive, sparsen

OUT = ROOT / '.work/agent/brain-bench'
ODORS = (('DA1', 'VA1d', 'DL3'), ('VM2', 'DM1', 'VA2'))


def _populations():
    roots = np.load(ROOT / 'data/proofread_root_ids_783.npy')
    by_root = {int(root): i for i, root in enumerate(roots)}
    pn = []
    orn_by_glomerulus = {g: [] for odor in ODORS for g in odor}
    with Path(ANNOTATIONS).open() as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            node = by_root.get(int(row['root_id']))
            if node is None:
                continue
            if row['cell_class'] == 'ALPN':
                pn.append(node)
            if row['cell_class'] == 'olfactory':
                for glomerulus in orn_by_glomerulus:
                    if row['cell_type'] == f'ORN_{glomerulus}':
                        orn_by_glomerulus[glomerulus].append(node)
    drives = np.zeros((N, 2), np.float32)
    for column, odor in enumerate(ODORS):
        nodes = np.unique(np.concatenate([np.asarray(orn_by_glomerulus[g], dtype=np.int32) for g in odor]))
        if nodes.size == 0:
            raise ValueError(f'no annotated ORNs for {odor}')
        drives[nodes, column] = 1
    cells = mushroom_body_cells()
    return drives, np.unique(np.asarray(pn, dtype=np.int32)), cells['KC'], cells['MBON']


def _jaccard(left, right):
    union = np.count_nonzero(left | right)
    return float(np.count_nonzero(left & right) / union) if union else 0.0


def _run_config(brain, drive, pn, kc, mbon, da, ka):
    brain.adaptation_da, brain.adaptation_ka = np.float32(da), np.float32(ka)
    brain.reset()
    per_exchange = np.zeros((10, 2), np.float32)
    mbon_trace = np.zeros((10, 2), np.float32)
    dn = None
    zeros = np.zeros((2, 8), np.float32)
    for exchange in range(10):
        result = brain.step(zeros, ticks=4, extra_drive=drive)
        per_exchange[exchange] = brain.ever.mean(axis=0)
        mbon_trace[exchange] = brain.trace[mbon].sum(axis=0)
        dn = np.asarray(result['features'], np.float32)
    pn_sets = brain.ever[pn].copy()
    kc_sets = brain.ever[kc].copy()
    dn_variance = float(np.var(dn, axis=0).mean())
    row = {
        'da': da, 'ka': ka,
        'brain_active_fraction': brain.ever.mean(axis=0).tolist(),
        'pn_active_fraction': (pn_sets.mean(axis=0)).tolist(),
        'pn_jaccard': _jaccard(pn_sets[:, 0], pn_sets[:, 1]),
        'kc_active_fraction': (kc_sets.mean(axis=0)).tolist(),
        'kc_jaccard': _jaccard(kc_sets[:, 0], kc_sets[:, 1]),
        'mbon_trace_final': mbon_trace[-1].tolist(),
        'dn_feature_variance': dn_variance,
        'pass': bool(brain.ever.mean() < .05 and _jaccard(pn_sets[:, 0], pn_sets[:, 1]) < .6
                     and np.all(kc_sets.mean(axis=0) < .10)
                     and _jaccard(kc_sets[:, 0], kc_sets[:, 1]) < .3 and dn_variance > 0),
    }
    return row, per_exchange, mbon_trace, pn_sets, kc_sets


def sweep():
    OUT.mkdir(parents=True, exist_ok=True)
    drive, pn, kc, mbon = _populations()
    brain = CPUBrain(flies=2)
    configs = [(da, ka) for da in (.8, .9, .95, .98) for ka in (.05, .1, .2, .5, 1.)]
    rows, series, pn_sets, kc_sets = [], [], [], []
    for da, ka in configs:
        started = time.perf_counter()
        row, active, mbon_trace, pn_active, kc_active = _run_config(brain, drive, pn, kc, mbon, da, ka)
        row['seconds_per_config'] = time.perf_counter() - started
        rows.append(row); series.append((active, mbon_trace)); pn_sets.append(pn_active); kc_sets.append(kc_active)
        print(f"da={da:.2f} ka={ka:.2f} pass={row['pass']} {row['seconds_per_config']:.3f}s", flush=True)
    np.savez_compressed(OUT / 'sweep.npz',
        active_fraction=np.stack([x[0] for x in series]), mbon_trace=np.stack([x[1] for x in series]),
        pn_sets=np.stack(pn_sets), kc_sets=np.stack(kc_sets), config=np.asarray(configs, np.float32))
    summary = {'status': 'PASS' if any(r['pass'] for r in rows) else 'NO_CONFIG_PASSED',
        'conditions': [list(x) for x in ODORS], 'exchanges': 10, 'ticks_per_exchange': 4,
        'flies': 2, 'configuration_count': len(rows), 'rows': rows,
        'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'state_bytes': sum(x.nbytes for x in (brain.v, brain.spikes, brain.trace, brain.a, brain.ever)),
        'graph_bytes': brain.graph.data.nbytes + brain.graph.indices.nbytes + brain.graph.indptr.nbytes,
        'memory_rule': 'single CPUBrain reused across condition columns; cached ORN node indices; float32 adaptation state'}
    (OUT / 'sweep.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def _normalize_graph_inplace(brain):
    """Normalize the existing CSR weights in place after the raw gain sweep."""
    normalize_weights_inplace(brain.graph.indptr, brain.graph.data)


def sweep_gain():
    """P1b: raw global and row-normalized recurrent gains, each with adaptation off/on."""
    OUT.mkdir(parents=True, exist_ok=True)
    drive, pn, kc, mbon = _populations()
    brain = CPUBrain(flies=2)
    configs = [('global', g, adapt) for g in (.02, .03, .05, .08, .12) for adapt in (False, True)]
    configs += [('normalized', g, adapt) for g in (20., 30., 50., 80., 120., 200.) for adapt in (False, True)]
    rows, active, instant, mbon_traces, pn_sets, kc_sets = [], [], [], [], [], []
    zeros = np.zeros((2, 8), np.float32)
    for mode, gain, adapt in configs:
        if mode == 'normalized' and brain.gain_mode != 'normalized':
            _normalize_graph_inplace(brain)
        brain.gain_mode, brain.gain = mode, np.float32(gain)
        brain.adaptation_da, brain.adaptation_ka = np.float32(.98), np.float32(1.0 if adapt else 0.)
        brain.reset()
        mbon_trace = np.zeros((10, 2), np.float32)
        started = time.perf_counter()
        for exchange in range(10):
            result = brain.step(zeros, ticks=4, extra_drive=drive, measure_activity=True)
            instant_row = result['tick_active_fraction']
            mbon_trace[exchange] = brain.trace[mbon].sum(axis=0)
        elapsed = time.perf_counter() - started
        pset, kset = brain.ever[pn].copy(), brain.ever[kc].copy()
        pj, kj = _jaccard(pset[:, 0], pset[:, 1]), _jaccard(kset[:, 0], kset[:, 1])
        brain_ever = brain.ever.mean(axis=0)
        kc_active = kset.mean(axis=0)
        dn_variance = float(np.var(np.asarray(result['features'], np.float32), axis=0).mean())
        row = {'mode': mode, 'gain': gain, 'adaptation': adapt,
            'adaptation_da': .98 if adapt else .98, 'adaptation_ka': 1. if adapt else 0.,
            'brain_ever': brain_ever.tolist(), 'brain_instant_last_exchange': instant_row,
            'pn_ever': pset.mean(axis=0).tolist(), 'pn_jaccard': pj,
            'kc_ever': kc_active.tolist(), 'kc_jaccard': kj,
            'mbon_trace_final': mbon_trace[-1].tolist(), 'dn_feature_variance': dn_variance,
            'seconds_per_config': elapsed,
            'pass': bool(brain_ever.mean() < .05 and pj < .6 and np.all(kc_active < .10)
                and np.all(kc_active > .01) and kj < .3 and dn_variance > 0)}
        rows.append(row)
        active.append(brain_ever); instant.append(instant_row); mbon_traces.append(mbon_trace)
        pn_sets.append(pset); kc_sets.append(kset)
        print(f"{mode} gain={gain:g} adapt={adapt} pass={row['pass']} {elapsed:.3f}s", flush=True)
    np.savez_compressed(OUT / 'sweep-p1b.npz', brain_ever=np.asarray(active, np.float32),
        brain_instant_last_exchange=np.asarray(instant, np.float32), mbon_trace=np.asarray(mbon_traces, np.float32),
        pn_sets=np.asarray(pn_sets, np.bool_), kc_sets=np.asarray(kc_sets, np.bool_),
        mode=np.asarray([0 if r[0] == 'global' else 1 for r in configs], np.int8),
        gain=np.asarray([r[1] for r in configs], np.float32), adaptation=np.asarray([r[2] for r in configs], np.bool_))
    summary = {'status': 'PASS' if any(r['pass'] for r in rows) else 'NO_CONFIG_PASSED',
        'conditions': [list(x) for x in ODORS], 'exchanges': 10, 'ticks_per_exchange': 4,
        'flies': 2, 'configuration_count': len(rows), 'rows': rows,
        'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'state_bytes': sum(x.nbytes for x in (brain.v, brain.spikes, brain.trace, brain.a, brain.ever)),
        'csr_bytes': brain.graph.data.nbytes + brain.graph.indices.nbytes + brain.graph.indptr.nbytes,
        'graph_weight_bytes': brain.graph.data.nbytes,
        'csr_structure_bytes': brain.graph.indices.nbytes + brain.graph.indptr.nbytes,
        'normalized_edge_count': int(brain.graph.data.size),
        'raw_edge_count': int(brain.graph.data.size),
        'normalization': 'each CSR post row divided by sum(abs(incoming weights)); zero rows remain zero'}
    (OUT / 'sweep-p1b.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def sweep_power(powers=(.5, .6, .7, .8, .9, 1.), base_gains=(25., 30., 35., 40., 50.), caps=(None,), name='p1c'):
    """P1c: rows divided by (sum |w|)**p; gain scaled so the median row matches p = 1.

    `caps` optionally limits KC spiking per tick to that fraction (`sparsen`, APL stand-in)."""
    from agent.verify_sparse_learning import ODORS
    OUT.mkdir(parents=True, exist_ok=True)
    cells, drive = mushroom_body_cells(), odor_drive(*ODORS)
    kc, mbon = cells['KC'], cells['MBON']
    brain = CPUBrain(flies=len(ODORS), gain_mode='global')
    raw, indptr = brain.graph.data.copy(), brain.graph.indptr
    totals = np.add.reduceat(np.abs(raw), indptr[:-1])[np.diff(indptr) > 0]
    median = float(np.median(totals[totals > 0]))
    zeros, rows, sets = np.zeros((len(ODORS), 8), np.float32), [], []
    pairs = ((0, 1), (0, 2), (1, 2))
    for power, cap in ((p, c) for p in powers for c in caps):
        np.copyto(brain.graph.data, raw)
        normalize_weights_inplace(indptr, brain.graph.data, power)
        brain.sparse = None
        if cap is not None:
            sparsen(brain, cells, cap)
        for base in base_gains:
            brain.gain = np.float32(base * median ** (power - 1))
            brain.reset()
            started = time.perf_counter()
            for _ in range(10):
                brain.step(zeros, ticks=4, extra_drive=drive)
            ks, ms, ds = brain.ever[kc], brain.ever[mbon], brain.ever[brain.descending]
            kc_active, kj = ks.mean(axis=0), [_jaccard(ks[:, i], ks[:, j]) for i, j in pairs]
            mj = [_jaccard(ms[:, i], ms[:, j]) for i, j in pairs]
            brain_ever = brain.ever.mean(axis=0)
            row = {'power': power, 'kc_cap': cap, 'base_gain': base, 'gain': float(brain.gain),
                   'brain_ever': brain_ever.tolist(), 'kc_ever': kc_active.tolist(), 'kc_jaccard': kj,
                   'mbon_count': ms.sum(axis=0).tolist(), 'mbon_jaccard': mj,
                   'dn_count': ds.sum(axis=0).tolist(), 'seconds': time.perf_counter() - started}
            row['pass'] = bool(np.all((kc_active > .01) & (kc_active < .10)) and max(kj) < .3
                               and np.all(ms.any(axis=0)) and max(mj) < .8
                               and ds.any() and brain_ever.max() < .05)
            rows.append(row); sets.append(np.concatenate([ks, ms, ds]))
            print(f"p={power:g} cap={cap} g0={base:g} KC={np.round(kc_active, 3).tolist()} "
                  f"MBON={row['mbon_count']} DN={row['dn_count']} pass={row['pass']}", flush=True)
    np.savez_compressed(OUT / f'sweep-{name}.npz', sets=np.packbits(np.asarray(sets), axis=1),
                        power=np.asarray([r['power'] for r in rows], np.float32),
                        gain=np.asarray([r['gain'] for r in rows], np.float32))
    summary = {'status': 'PASS' if any(r['pass'] for r in rows) else 'NO_CONFIG_PASSED',
               'odors': [list(o) for o in ODORS], 'exchanges': 10, 'median_row_abs_total': median,
               'rows': rows, 'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (OUT / f'sweep-{name}.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


ODORS6 = (('DA1', 'VA1d', 'DL3'), ('VM2', 'DM1', 'VA2'), ('DM2', 'VC1', 'DA2'),
          ('DM3', 'DM4', 'VA6'), ('DL5', 'DC1', 'VA7l'), ('VM7d', 'VC3', 'DL1'))
APL = (31044,)  # FlyWire APL with KC edges; its partner row 31045 has none


def sweep_class_gain(seeds=3, keep=.8):
    """P1d: per-class gains folded into weights at the sparse point (normalized gain 35).

    Grid ranges come from agent.operating_point (APL ~7x, MBON 5-11x, DN 23-83x short of
    threshold). Each odour runs as `seeds` columns, each driving a random `keep` subset of its
    ORNs, so every criterion is checked across odours and input jitter, not one deterministic run.
    """
    sys.path.insert(0, str(Path.home() / 'iTools/science/regime-sweep'))
    from regime_sweep import sweep
    cells = mushroom_body_cells()
    kc, mbon = cells['KC'], cells['MBON']
    rng = np.random.default_rng(20261002)
    base = odor_drive(*ODORS6)
    drive = np.repeat(base, seeds, axis=1) * (rng.random((N, base.shape[1] * seeds)) < keep)
    brain = CPUBrain(flies=drive.shape[1], gain_mode='normalized', gain=35)
    normalized, zeros = brain.graph.data.copy(), np.zeros((drive.shape[1], 8), np.float32)
    odour = np.repeat(np.arange(len(ODORS6)), seeds)
    across = [(i, j) for i in range(len(odour)) for j in range(i) if odour[i] != odour[j] and i % seeds == j % seeds]
    within = [(i, j) for i in range(len(odour)) for j in range(i) if odour[i] == odour[j]]

    def run(gain, apl_in, apl_out, mbon_gain, dn_gain):
        np.copyto(brain.graph.data, normalized)
        brain.scale_edges(apl_in, post=APL)
        brain.scale_edges(apl_out, pre=APL)
        brain.scale_edges(mbon_gain, post=mbon)
        brain.scale_edges(dn_gain, post=brain.descending)
        brain.gain = np.float32(gain)
        brain.reset()
        for _ in range(10):
            brain.step(zeros, ticks=4, extra_drive=drive)
        ks, ms, ds = brain.ever[kc], brain.ever[mbon], brain.ever[brain.descending]
        return ({'kc_active': ks.mean(axis=0).tolist(),
                 'kc_jaccard_across_odours': max(_jaccard(ks[:, i], ks[:, j]) for i, j in across),
                 'kc_jaccard_same_odour_min': min(_jaccard(ks[:, i], ks[:, j]) for i, j in within),
                 'mbon_count': ms.sum(axis=0).tolist(), 'dn_count_max': int(ds.sum(axis=0).max()),
                 'apl_fired': bool(brain.ever[list(APL)].any()), 'brain_active': brain.ever.mean(axis=0).tolist()},
                {'sets': np.packbits(np.concatenate([ks, ms, ds]), axis=0)})

    grid = {'gain': [35., 50., 80.], 'apl_in': [1., 8.], 'apl_out': [1., 8., 32.],
            'mbon_gain': [1., 5., 11.], 'dn_gain': [1., 25., 80.]}
    criteria = {'kc_floor': ('kc_active', '>', .01), 'kc_ceiling': ('kc_active', '<', .10),
                'kc_distinct': ('kc_jaccard_across_odours', '<', .3),
                'mbon_responds_every_odour': ('mbon_count', '>=', 1),
                'dn_fires': ('dn_count_max', '>=', 1), 'brain_bounded': ('brain_active', '<', .05)}
    return sweep(grid, run, criteria, OUT / 'sweep-p1d')


if __name__ == '__main__':
    if '--p1d' in sys.argv:
        result = sweep_class_gain()
        result['peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    elif '--p1c-cap' in sys.argv:
        result = sweep_power((1.,), (45., 50., 55., 60., 65., 70.), (.005, .01, .015, .02), 'p1c-cap')
    else:
        result = sweep_power() if '--p1c' in sys.argv else sweep_gain()
    print(json.dumps({'status': result['status'], 'peak_rss_kib': result['peak_rss_kib']}))
