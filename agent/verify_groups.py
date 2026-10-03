"""Edge groups in the Shiu engines: does a weight variant run as columns equal its own graph?

Three variants of the authors' graph (as is; ALLN inhibition x16; APL -> KC x4 with ALLN x4), powers
of two so the scaled float32 weight is exact either way. Grouped: one graph, 3 x C columns, grp/scale.
Separate: each variant's own graph with the same columns stimulated, so every Poisson entry
(column * n + neuron) matches. Pass when GPU counts are identical column for column, and the CPU
oracle with groups matches the GPU for the first 20 ms.
"""
import json
import sys

import numpy as np

from agent import shiu_g1
from agent.brain_bench import ODORS6
from agent.neural_connectome import ROOT
from agent.shiu import ShiuBrain, ShiuGPU, graph_dir
from agent.shiu_apl import masks, mb_variant
from agent.shiu_loop import cells

VARIANTS = ((1, 1, 1, 1, 1), (1, 1, 1, 1, 16), (1, 1, 1, 4, 4))  # mb_variant: kckc, dan, apl in, apl out, alln
COLS, HZ = 8, 200.  # per variant: 2 odours x 4 columns


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
    pre, post, w, names, _ = shiu_g1.flywire()
    n = len(names)
    orn, _ = cells('flywire')
    is_kc, apl = masks(n)[:2]
    path = graph_dir(pre, post, w, n)  # CSR order: groups must follow it
    row, col = np.fromfile(path / 'row_ptr.bin', np.int32), np.fromfile(path / 'col_idx.bin', np.int32)
    src = np.repeat(np.arange(n), np.diff(row))
    sign = np.fromfile(path / 'weights.bin', np.float32) < 0
    grp = np.zeros(len(col), np.uint8)
    grp[(names[src] == 'ALLN') & sign] = 1
    grp[apl[src] & is_kc[col]] = 2
    scale = np.ones((len(VARIANTS) * COLS, 3), np.float32)
    for v, (_, _, _, out, ln) in enumerate(VARIANTS):
        scale[v * COLS:(v + 1) * COLS, 1:] = ln, out
    grouped = run(ShiuGPU, pre, post, w, n, orn, [2000], (grp, scale))
    rows = []
    for v, variant in enumerate(VARIANTS):
        alone = run(ShiuGPU, pre, post, mb_variant(pre, post, w, names, is_kc, apl, *variant[:4], g_ln=variant[4]),
                    n, orn, [2000], only=v)
        cs = slice(v * COLS, (v + 1) * COLS)
        rows.append({'variant': variant, 'spikes_grouped': int(grouped[:, cs].sum()), 'spikes_separate': int(alone[:, cs].sum()),
                     'cells_differing': int((grouped[:, cs] != alone[:, cs]).any(axis=1).sum())})
        print(rows[-1], flush=True)
    cpu = run(ShiuBrain, pre, post, w, n, orn, [200], (grp, scale))
    gpu = run(ShiuGPU, pre, post, w, n, orn, [200], (grp, scale))
    oracle = {'spikes_cpu': int(cpu.sum()), 'spikes_gpu': int(gpu.sum()), 'cells_differing': int((cpu != gpu).any(axis=1).sum())}
    print('cpu vs gpu 20 ms', oracle, flush=True)
    ok = all(r['cells_differing'] == 0 and r['spikes_grouped'] > 0 for r in rows) and oracle['cells_differing'] == 0
    out = {'status': 'PASS' if ok else 'FAIL', 'variants': rows, 'cpu_gpu_20ms': oracle}
    (ROOT / '.work/agent/shiu/verify-groups.json').write_text(json.dumps(out, indent=2) + '\n')
    print(out['status'])
    return ok


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
