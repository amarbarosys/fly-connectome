"""ShiuBrain against the closed-form EPSP and against Brian2's published sugar-GRN runs."""
import csv
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from agent.neural_connectome import ROOT
from agent.shiu import T_DLY, T_MBR, TAU, W_SYN, ShiuBrain, ShiuGPU

REF = Path.home() / 'Projects/imports/FlyAssets/eonsystemspbc_fly-brain'


def epsp_check():
    """One presynaptic spike of 10 synapses: post u(t) = A k (e^-t/tau - e^-t/t_mbr) after the delay."""
    brain = ShiuBrain([0], [1], [10], n=2)
    brain.u[0] = 100.0  # cell 0 crosses threshold on step 0 (it is not refractory yet)
    trace = []
    for _ in range(300):
        brain.step(1)
        trace.append(float(brain.u[1]))
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


def sugar_check(seconds=1.0, trials=5, engine=ShiuBrain):
    con = pd.read_parquet(REF / 'data/2025_Connectivity_783.parquet')
    ids = pd.read_csv(REF / 'data/2025_Completeness_783.csv', index_col=0).index.to_numpy(np.int64)
    source = (REF / 'code/benchmark.py').read_text()
    sugar = [int(x) for x in re.findall(r'7205\d{14}', source[source.index("'sugar'"):source.index("'p9'")])]
    index = {int(root): i for i, root in enumerate(ids)}
    brain = engine(con['Presynaptic_Index'].to_numpy(), con['Postsynaptic_Index'].to_numpy(),
                   con['Excitatory x Connectivity'].to_numpy(), n=len(ids), flies=trials, seed=20261002)
    brain.stimulate([index[s] for s in sugar], 200.0)
    started = time.perf_counter()
    counts = brain.run([round(seconds / brain.dt)])[0]
    spikes, active = counts.sum(axis=0), (counts > 0).sum(axis=0)
    ref_spikes, ref_active = brian2_reference(seconds)
    return {'engine': engine.__name__, 'sugar_cells': len(sugar), 'seconds': seconds, 'trials': trials,
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


def flood_code_probe(seeds=3, keep=.8, rate=100., seconds=.2, windows_ms=(10, 15, 20, 30, 50), engine=ShiuBrain, arm=None, odours=None):
    """Not a gate: does odour identity survive the flood, (1) in the first ms after onset,
    (2) in KC firing rates during the flood? 3 odours x `seeds` ORN-jitter columns."""
    from agent.neural_connectome import ANNOTATIONS
    from agent.verify_sparse_learning import ODORS
    odours = odours or ODORS
    pre, post, w, index = _reference_graph()
    if arm is not None:
        pre, post, w = arm
    kc, orn = [], {}
    for row in csv.DictReader(open(ANNOTATIONS), delimiter='\t'):
        i = index.get(int(row['root_id']))
        if i is not None and row['cell_class'] == 'Kenyon_Cell':
            kc.append(i)
        if i is not None and row['cell_class'] == 'olfactory' and row['cell_type'].startswith('ORN_'):
            orn.setdefault(row['cell_type'][4:], []).append(i)
    rng = np.random.default_rng(20261002)
    odour = np.repeat(np.arange(len(odours)), seeds)
    brain = engine(pre, post, w, n=len(index), flies=len(odour), seed=11)
    for column, o in enumerate(odour):
        cells = np.asarray(sum((orn[g] for g in odours[o]), []))
        brain.stimulate(cells[rng.random(len(cells)) < keep], rate, fly=column)
    same = [(i, j) for i in range(len(odour)) for j in range(i) if odour[i] == odour[j]]
    diff = [(i, j) for i in range(len(odour)) for j in range(i) if odour[i] != odour[j]]

    def jaccard(sets):
        jac = lambda i, j: float((sets[:, i] & sets[:, j]).sum() / max(int((sets[:, i] | sets[:, j]).sum()), 1))
        return float(np.mean([jac(*p) for p in same])), float(np.mean([jac(*p) for p in diff]))

    def correlation(x):
        c = np.corrcoef(x.T)
        return float(np.mean([c[p] for p in same])), float(np.mean([c[p] for p in diff]))

    def decode(x):
        """Leave-one-out nearest centroid (correlation) over the columns; chance = 1/odours."""
        hits = 0
        for i in range(len(odour)):
            others = [j for j in range(len(odour)) if j != i]
            centroids = [x[:, [j for j in others if odour[j] == o]].mean(axis=1) for o in range(len(odours))]
            with np.errstate(invalid='ignore', divide='ignore'):
                scores = np.nan_to_num([np.corrcoef(x[:, i], c)[0, 1] for c in centroids], nan=-1.0)
            hits += int(np.argmax(scores) == odour[i] and scores.max() > -1)  # silent column = miss
        return hits / len(odour)

    steps = [round(ms * 1e-3 / brain.dt) for ms in windows_ms] + [round(seconds / brain.dt)]
    counts_at = [c[kc].astype(np.float64) for c in brain.run(steps)]
    onset = []
    for ms, counts in zip(windows_ms, counts_at):
        sets = counts > 0
        js, jd = jaccard(sets)
        onset.append({'ms': ms, 'kc_active': float(sets.mean()), 'kc_cells_per_column': sets.sum(axis=0).tolist(),
                      'jaccard_same_odour': js, 'jaccard_diff_odour': jd,
                      'decode_accuracy': decode(counts) if counts.any() else None})
    flood = counts_at[-1] - counts_at[-2]
    cs, cd = correlation(flood)
    js, jd = jaccard(counts_at[-1] > 0)
    out = {'engine': engine.__name__, 'odours': [list(o) for o in odours], 'columns': len(odour), 'orn_hz': rate, 'chance': 1 / len(odours),
           'onset': onset,
           'flood': {'window_ms': [windows_ms[-1], seconds * 1e3], 'kc_active': float((counts_at[-1] > 0).mean()),
                     'ever_jaccard_same': js, 'ever_jaccard_diff': jd,
                     'rate_corr_same_odour': cs, 'rate_corr_diff_odour': cd, 'decode_accuracy': decode(flood),
                     'kc_mean_hz': float(flood.mean() / (seconds - windows_ms[-1] * 1e-3))}}
    def distinct(similarity):
        """Per column: mean similarity to same-odour columns minus to other-odour columns."""
        same_mask = odour[:, None] == odour[None, :]
        np.fill_diagonal(same_mask, False)
        return [float(similarity[i][same_mask[i]].mean() - similarity[i][odour != odour[i]].mean())
                for i in range(len(odour))]

    def jaccard_matrix(sets):
        sets = sets.astype(np.float64)
        inter = sets.T @ sets
        size = sets.sum(axis=0)
        return inter / np.maximum(size[:, None] + size[None, :] - inter, 1)

    with np.errstate(invalid='ignore', divide='ignore'):
        out['per_column'] = {**{f'onset_{ms}ms': distinct(jaccard_matrix(c > 0)) for ms, c in zip(windows_ms, counts_at)},
                             'flood_rate': distinct(np.nan_to_num(np.corrcoef(flood.T)))}
    if arm is not None:
        return out
    path = ROOT / f'.work/agent/shiu/flood-code-probe-{engine.__name__}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + '\n')
    return out


def exact_check(steps=(50, 100, 200), columns=2):
    """GPU spike counts per cell against the oracle with identical Poisson draws, sugar input."""
    pre, post, w, index = _reference_graph()
    source = (REF / 'code/benchmark.py').read_text()
    sugar = [index[int(x)] for x in re.findall(r'7205\d{14}', source[source.index("'sugar'"):source.index("'p9'")])]
    runs = {}
    for engine in (ShiuBrain, ShiuGPU):
        brain = engine(pre, post, w, n=len(index), flies=columns, seed=7)
        brain.stimulate(sugar, 200.0)
        started = time.perf_counter()
        runs[engine.__name__] = (brain.run(list(steps)), time.perf_counter() - started)
    rows = [{'steps': s, 'cells_differing': int((a != b).any(axis=1).sum()), 'spikes_cpu': int(a.sum()), 'spikes_gpu': int(b.sum())}
            for s, a, b in zip(steps, runs['ShiuBrain'][0], runs['ShiuGPU'][0])]
    return {'rows': rows, 'cpu_s': runs['ShiuBrain'][1], 'gpu_s': runs['ShiuGPU'][1],
            'first_divergent_steps': next((r['steps'] for r in rows if r['cells_differing']), None),
            'pass': rows[0]['cells_differing'] == 0}


def verify_gpu():
    started = time.perf_counter()
    result = {'exact': exact_check(), 'sugar': sugar_check(engine=ShiuGPU)}
    cpu = json.loads((ROOT / '.work/agent/shiu/flood-code-probe-ShiuBrain.json').read_text())
    gpu = flood_code_probe(engine=ShiuGPU)
    result['flood'] = {'cpu': cpu['flood'], 'gpu': gpu['flood'],
                       'onset_cpu': cpu['onset'], 'onset_gpu': gpu['onset'],
                       'pass': gpu['flood']['decode_accuracy'] == cpu['flood']['decode_accuracy']
                       and abs(gpu['flood']['kc_active'] - cpu['flood']['kc_active']) < .02}
    result['wall_s'] = time.perf_counter() - started
    result['pass'] = result['exact']['pass'] and result['sugar']['pass'] and result['flood']['pass']
    (ROOT / '.work/agent/shiu/verification-gpu.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('exact', 'wall_s', 'pass')}))
    print(json.dumps({'sugar': {k: result['sugar'][k] for k in ('spikes', 'active_neurons', 'brian2_spikes', 'brian2_active_neurons', 'wall_s', 'pass')}}))
    return result


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
    if '--gpu' in sys.argv:
        raise SystemExit(0 if verify_gpu()['pass'] else 1)
    if '--flood-gpu' in sys.argv:  # 6 odours x 10 jitter columns (60), GPU
        from agent.brain_bench import ODORS6
        print(json.dumps(flood_code_probe(engine=ShiuGPU, odours=ODORS6, seeds=10, windows_ms=(15, 20, 25, 30, 40, 50))))
    elif '--flood' in sys.argv:
        print(json.dumps(flood_code_probe()))
    elif '--mb' in sys.argv:
        print(json.dumps(mushroom_body_probe()))
    else:
        raise SystemExit(0 if verify()['pass'] else 1)
