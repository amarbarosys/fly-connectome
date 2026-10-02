"""Body-free CPU sweep for olfactory population dynamics and adaptation."""
from __future__ import annotations

import csv
import json
import resource
import time
from pathlib import Path

import numpy as np

from agent.neural_connectome import ANNOTATIONS, N, ROOT, CPUBrain, normalize_weights_inplace
from agent.mb_plasticity import mushroom_body_cells

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


if __name__ == '__main__':
    result = sweep_gain()
    print(json.dumps({'status': result['status'], 'peak_rss_kib': result['peak_rss_kib']}))
