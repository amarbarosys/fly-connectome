"""ShiuBrain against the closed-form EPSP and against Brian2's published sugar-GRN runs."""
import csv
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from agent.neural_connectome import ROOT
from agent.shiu import T_DLY, T_MBR, TAU, W_SYN, ShiuBrain

REF = Path.home() / 'Projects/imports/FlyAssets/eonsystemspbc_fly-brain'


def epsp_check():
    """One presynaptic spike of 10 synapses: post u(t) = A k (e^-t/tau - e^-t/t_mbr) after the delay."""
    brain = ShiuBrain([0], [1], [10], n=2)
    brain.u[0, 0] = 100.0  # cell 0 crosses threshold on step 0 (it is not refractory yet)
    trace = []
    for _ in range(300):
        brain.step(1)
        trace.append(float(brain.u[1, 0]))
    # spike on step 0 is delivered on step 18 and integrates from step 19
    t = (np.arange(300) - round(T_DLY / 1e-4)) * 1e-4
    a, k = 10 * W_SYN, TAU / (TAU - T_MBR)
    tt = np.maximum(t, 0)
    expected = np.where(t >= 0, a * k * (np.exp(-tt / TAU) - np.exp(-tt / T_MBR)), 0)
    error = float(np.abs(np.asarray(trace) - expected).max())
    return {'max_abs_error_mV': error, 'peak_mV': float(max(trace)), 'pass': error < 1e-4}


def brian2_reference(seconds):
    rows = [r for r in csv.DictReader(open(REF / 'data/results/nature_2026_07/manifest.csv'))
            if r['backend_key'] == 'brian2cpp' and r['experiment_key'] == 'sugar'
            and float(r['t_run']) == seconds and r['n_run'] == '1']
    return [int(r['spikes']) for r in rows], [int(r['active_neurons']) for r in rows]


def within(value, reference):
    """Mean of our trials inside the reference range widened by 3 SD of the reference rounds."""
    return bool(min(reference) - 3 * np.std(reference) <= value <= max(reference) + 3 * np.std(reference))


def sugar_check(seconds=1.0, trials=5):
    con = pd.read_parquet(REF / 'data/2025_Connectivity_783.parquet')
    ids = pd.read_csv(REF / 'data/2025_Completeness_783.csv', index_col=0).index.to_numpy(np.int64)
    source = (REF / 'code/benchmark.py').read_text()
    sugar = [int(x) for x in re.findall(r'7205\d{14}', source[source.index("'sugar'"):source.index("'p9'")])]
    index = {int(root): i for i, root in enumerate(ids)}
    brain = ShiuBrain(con['Presynaptic_Index'].to_numpy(), con['Postsynaptic_Index'].to_numpy(),
                      con['Excitatory x Connectivity'].to_numpy(), n=len(ids), flies=trials, seed=20261002)
    brain.stimulate([index[s] for s in sugar], 200.0)
    started = time.perf_counter()
    brain.step(round(seconds / brain.dt))
    spikes, active = brain.spikes.sum(axis=0), brain.ever.sum(axis=0)
    ref_spikes, ref_active = brian2_reference(seconds)
    return {'sugar_cells': len(sugar), 'seconds': seconds, 'trials': trials,
            'spikes': spikes.tolist(), 'active_neurons': active.tolist(),
            'brian2_spikes': ref_spikes, 'brian2_active_neurons': ref_active,
            'wall_s': time.perf_counter() - started,
            'pass': within(spikes.mean(), ref_spikes) and within(active.mean(), ref_active)}


def _reference_graph():
    con = pd.read_parquet(REF / 'data/2025_Connectivity_783.parquet')
    ids = pd.read_csv(REF / 'data/2025_Completeness_783.csv', index_col=0).index.to_numpy(np.int64)
    return (con['Presynaptic_Index'].to_numpy(), con['Postsynaptic_Index'].to_numpy(),
            con['Excitatory x Connectivity'].to_numpy(), {int(r): i for i, r in enumerate(ids)})


def mushroom_body_probe(rates=(5., 20., 100.), seconds=.2):
    """Not a gate: KC, MBON and APL activity for the three verify_sparse_learning odours on the
    authors' graph, ORNs as Poisson input, with and without KC->KC synapses."""
    from agent.neural_connectome import ANNOTATIONS
    from agent.verify_sparse_learning import ODORS
    pre, post, w, index = _reference_graph()
    cells, orn = {'KC': [], 'MBON': [], 'APL': []}, {}
    for row in csv.DictReader(open(ANNOTATIONS), delimiter='\t'):
        i = index.get(int(row['root_id']))
        if i is None:
            continue
        name = {'Kenyon_Cell': 'KC', 'MBON': 'MBON'}.get(row['cell_class']) or ('APL' if row['cell_type'] == 'APL' else None)
        if name:
            cells[name].append(i)
        if row['cell_class'] == 'olfactory' and row['cell_type'].startswith('ORN_'):
            orn.setdefault(row['cell_type'][4:], []).append(i)
    is_kc = np.zeros(len(index), bool)
    is_kc[cells['KC']] = True
    kc_kc = is_kc[pre] & is_kc[post]
    rows = []
    for label, keep in (('full', np.ones_like(kc_kc)), ('no_kc_kc', ~kc_kc)):
        for rate in rates if label == 'full' else rates[-1:]:
            brain = ShiuBrain(pre[keep], post[keep], w[keep], n=len(index), flies=len(ODORS), seed=3)
            for column, odor in enumerate(ODORS):
                brain.stimulate(sum((orn[g] for g in odor), []), rate, fly=column)
            brain.step(round(seconds / brain.dt))
            kcs = brain.ever[cells['KC']]
            union = lambda a, b: max(int((a | b).sum()), 1)
            rows.append({'graph': label, 'orn_hz': rate, 'kc_active': kcs.mean(axis=0).tolist(),
                         'kc_jaccard': [float((kcs[:, i] & kcs[:, j]).sum() / union(kcs[:, i], kcs[:, j]))
                                        for i, j in ((0, 1), (0, 2), (1, 2))],
                         'mbon_active': brain.ever[cells['MBON']].sum(axis=0).tolist(),
                         'apl_hz': (brain.spikes[cells['APL']].sum(axis=0) / seconds).tolist(),
                         'brain_active': brain.ever.mean(axis=0).tolist()})
    out = {'kc_kc_edges': int(kc_kc.sum()), 'kc_kc_synapses': int(w[kc_kc].sum()),
           'kc_kc_share_of_kc_excitation': float(w[kc_kc].sum() / w[is_kc[post] & (w > 0)].sum()), 'rows': rows}
    path = ROOT / '.work/agent/shiu/mushroom-body-probe.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + '\n')
    return out


def verify():
    result = {'epsp': epsp_check(), 'sugar': sugar_check()}
    result['pass'] = result['epsp']['pass'] and result['sugar']['pass']
    out = ROOT / '.work/agent/shiu/verification.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    return result


if __name__ == '__main__':
    import sys
    if '--mb' in sys.argv:
        print(json.dumps(mushroom_body_probe()))
    else:
        raise SystemExit(0 if verify()['pass'] else 1)
