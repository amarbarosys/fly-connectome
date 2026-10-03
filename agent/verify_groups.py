"""Edge groups in the Shiu engines: does a weight variant run as columns equal its own graph?

Three variants of the authors' graph (as is; ALLN inhibition x16; APL -> KC x4 with ALLN x4), powers
of two so the scaled float32 weight is exact either way. Grouped: one graph, 3 x C columns, grp/scale.
Separate: each variant's own graph with the same columns stimulated, so every Poisson entry
(column * n + neuron) matches. Pass when GPU counts agree column for column up to the worker's run-to-run noise (see same()), and the CPU
oracle with groups matches the GPU for the first 20 ms. Cell groups: in one run, per-group sums
made on the device equal the host sum of the same run's per-cell readout of every grouped cell.
Keys: columns keyed to other columns with identical weights reproduce them exactly.
"""
import json
import sys

import numpy as np

from agent.brain_bench import ODORS6
from agent.neural_connectome import ROOT
from agent.shiu import ShiuBrain, ShiuGPU, graph_dir
from agent.verify_shiu import _reference_graph
from agent.experiment import cells

VARIANTS = ((1, 1), (1, 16), (4, 4))  # (APL -> KC gain, ALLN inhibition gain)
COLS, HZ = 8, 200.  # per variant: 2 odours x 4 columns


def same(a, b, n):
    """Equal up to the worker's float-atomic run-to-run noise (measured 200 of 4.58M spikes): spike
    totals within 1e-4 and at most 1e-3 of cells differing. Unshared inputs differ in most cells."""
    return abs(int(a.sum()) - int(b.sum())) <= 1e-4 * max(int(a.sum()), 1) and (a != b).any(axis=1).sum() <= 1e-3 * n


def run(engine, pre, post, w, n, orn, windows, groups=None, only=None):
    flies = len(VARIANTS) * COLS
    brain = engine(pre, post, w, n=n, flies=flies, seed=71)
    for c in range(flies):
        if only is None or c // COLS == only:
            brain.stimulate([i for g in ODORS6[c % COLS // 4] for i in orn[g]], HZ, fly=c)
    if groups is not None:
        brain.edge_groups(*groups)
    return brain.run(windows)[-1]


def main():
    pre, post, w, _ = _reference_graph()
    table = cells()
    names, n = table.klass, len(table.klass)
    is_kc, apl = names == "Kenyon_Cell", table.type == "APL"
    orn = {t[4:]: np.flatnonzero((names == 'olfactory') & (table.type == t)) for t in np.unique(table.type) if t.startswith('ORN_')}
    path = graph_dir(pre, post, w, n)  # CSR order: groups must follow it
    row, col = np.fromfile(path / 'row_ptr.bin', np.int32), np.fromfile(path / 'col_idx.bin', np.int32)
    src = np.repeat(np.arange(n), np.diff(row))
    sign = np.fromfile(path / 'weights.bin', np.float32) < 0
    grp = np.zeros(len(col), np.uint8)
    grp[(names[src] == 'ALLN') & sign] = 1
    grp[apl[src] & is_kc[col]] = 2
    scale = np.ones((len(VARIANTS) * COLS, 3), np.float32)
    for v, (out, ln) in enumerate(VARIANTS):
        scale[v * COLS:(v + 1) * COLS, 1:] = ln, out
    grouped = run(ShiuGPU, pre, post, w, n, orn, [2000], (grp, scale))
    rows = []
    for v, variant in enumerate(VARIANTS):
        scaled = w.copy()  # the variant as its own graph: integer synapses scaled by powers of two
        scaled[(names[pre] == 'ALLN') & (w < 0)] *= variant[1]
        scaled[apl[pre] & is_kc[post]] *= variant[0]
        alone = run(ShiuGPU, pre, post, scaled, n, orn, [2000], only=v)
        cs = slice(v * COLS, (v + 1) * COLS)
        rows.append({'variant': variant, 'spikes_grouped': int(grouped[:, cs].sum()), 'spikes_separate': int(alone[:, cs].sum()),
                     'cells_differing': int((grouped[:, cs] != alone[:, cs]).any(axis=1).sum()), 'same': bool(same(grouped[:, cs], alone[:, cs], n))})
        print(rows[-1], flush=True)
    cpu = run(ShiuBrain, pre, post, w, n, orn, [200], (grp, scale))
    gpu = run(ShiuGPU, pre, post, w, n, orn, [200], (grp, scale))
    oracle = {'spikes_cpu': int(cpu.sum()), 'spikes_gpu': int(gpu.sum()), 'cells_differing': int((cpu != gpu).any(axis=1).sum()),
              'same': bool(same(cpu, gpu, n))}
    print('cpu vs gpu 20 ms', oracle, flush=True)
    # common random numbers: two blocks of identical weights, the second keyed to the first's columns
    twin = ShiuGPU(pre, post, w, n=n, flies=2 * COLS, seed=71)
    for c in range(2 * COLS):
        twin.stimulate([i for g in ODORS6[c % COLS // 4] for i in orn[g]], HZ, fly=c, key=c % COLS)
    pair = twin.run([2000])[-1]
    keyed = {'spikes_first': int(pair[:, :COLS].sum()), 'spikes_keyed_copy': int(pair[:, COLS:].sum()),
             'cells_differing': int((pair[:, :COLS] != pair[:, COLS:]).any(axis=1).sum()), 'same': bool(same(pair[:, :COLS], pair[:, COLS:], n))}
    print('keyed copy', keyed, flush=True)
    members = [np.flatnonzero(m) for m in (is_kc, names == 'MBON', table.super == "descending", apl)]
    cgrp = np.full(n, -1, np.int32)
    for k, m in enumerate(members):
        cgrp[m] = k
    flies, windows = len(VARIANTS) * COLS, [500, 2000]
    cell = np.concatenate(members)
    readout = (np.arange(flies)[:, None] * n + cell[None]).reshape(-1)
    brain = ShiuGPU(pre, post, w, n=n, flies=flies, seed=71)
    for c in range(flies):
        brain.stimulate([i for g in ODORS6[c % COLS // 4] for i in orn[g]], HZ, fly=c)
    brain.edge_groups(grp, scale)
    sums, got = brain.run_groups(windows, cgrp, readout)
    host = np.zeros_like(sums)
    np.add.at(host, (np.arange(len(windows))[:, None], cgrp[readout % n][None], (readout // n)[None]), got)
    summed = {'groups_kc_mbon_dn_apl_last_window': sums[-1].sum(1).tolist(), 'equal_to_host_sum': bool(np.array_equal(sums, host)),
              'bytes_sums': int(sums.nbytes), 'bytes_full_counts': len(windows) * flies * n * 4}
    print('group sums', summed, flush=True)
    ok = (all(r['same'] and r['spikes_grouped'] > 0 for r in rows) and oracle['same']
          and summed['equal_to_host_sum'] and sums[-1, 0].sum() > 0 and keyed['same'])
    out = {'status': 'PASS' if ok else 'FAIL', 'variants': rows, 'cpu_gpu_20ms': oracle, 'group_sums': summed, 'keyed_copy': keyed}
    (ROOT / '.work/agent/shiu/verify-groups.json').write_text(json.dumps(out, indent=2) + '\n')
    print(out['status'])
    return ok


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
