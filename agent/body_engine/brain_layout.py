"""The brain's layout for the engine's live brain view (kinestride brain_view.mojo): every neuron of the simulated graph
at a 3D point, its class, and named groups whose activity the trigger strip reads. One directory per dataset; the
view knows nothing of FlyWire. Graph order = the Shiu worker's neuron index (agent/experiment.cells()).

  xyz.bin     float32 N x 3, view axes (x lateral, y up, z front-back), centred, max |coord| = 1
              point = soma (FlyWire soma_x/y/z) when annotated, else pos_x/y/z (a point on the neuron); voxels 4 x 4 x 40 nm
  class.bin   uint8 N, index into classes.txt (super_class); 255 = no annotation row, not drawn (14 of 138,639)
  groups.bin  int32: G, then per group: count, ids...      groups.txt: name, effect, kind (input|output) per line

  .venv/bin/python -m agent.body_engine.brain_layout       -> .work/agent/body_engine/brain-layout/flywire/
"""
import csv
import sys
import tomllib

import numpy as np

from agent.body_engine.play import KEYMAP, resolve
from agent.experiment import cells
from agent.neural_connectome import ANNOTATIONS, ROOT
from agent.verify_shiu import _reference_graph

OUT = ROOT / '.work/agent/body_engine/brain-layout/flywire'
CLASSES = ('optic', 'central', 'sensory', 'visual_projection', 'ascending', 'descending', 'sensory_ascending', 'visual_centrifugal',
           'motor', 'endocrine', 'unknown')
VOXEL = np.array([4.0, 4.0, 40.0])  # nm per FlyWire voxel (x, y, z)
OUTPUTS = (('DNp09', 'walk forward'), ('DNa02-right', 'turn left'), ('DNa02-left', 'turn right'), ('DNp01', 'escape jump'), ('MDN', 'back up'))


def groups(c, km):
    """(name, effect, kind, ids): the descending neurons that drive the body, then every input the keymap fires."""
    out = [(t, e, 'output', resolve(t, c)) for t, e in OUTPUTS]
    out += [(s, f'drive {s[3]}', 'output', np.flatnonzero((c.super == 'descending') & (c.side == side))) for s, side in (('DN-L', 'left'), ('DN-R', 'right'))]
    for key, v in km['keys'].items():
        try:
            ids = np.concatenate([resolve(t, c) for t in v['targets']])
        except SystemExit as e:
            print(f'skip key {key}: {e}')
            continue
        if any(np.array_equal(np.sort(ids), np.sort(o[3])) for o in out):
            continue  # the key fires an output group's own neurons (w, a, d, space): its activity is that group's, not an input
        out.append((f'key {key}', ' + '.join(v['targets']), 'input', ids))
    out += [(f'loom {s[0].upper()}', f'{"+".join(km["loom"]["targets"])} {s} eye', 'input', np.concatenate([resolve(f'{t}-{s}', c) for t in km['loom']['targets']]))
            for s in ('left', 'right')]
    return out


def main():
    index = _reference_graph()[3]
    n = len(index)
    c = cells()
    soma, pos = np.full((n, 3), np.nan), np.full((n, 3), np.nan)
    for r in csv.DictReader(open(ANNOTATIONS), delimiter='\t'):
        i = index.get(int(r['root_id']))
        if i is not None:
            if r['soma_x']:
                soma[i] = [float(r[k]) for k in ('soma_x', 'soma_y', 'soma_z')]
            if r['pos_x']:
                pos[i] = [float(r[k]) for k in ('pos_x', 'pos_y', 'pos_z')]
    both = ~np.isnan(soma[:, 0]) & ~np.isnan(pos[:, 0])
    p = np.where(np.isnan(soma[:, :1]), pos, soma) * VOXEL
    placed = ~np.isnan(p[:, 0])
    print(f'neurons {n}: soma {int((~np.isnan(soma[:, 0])).sum())}, pos only {int((placed & np.isnan(soma[:, 0])).sum())}, unplaced {int((~placed).sum())}; '
          f'median soma-pos distance {np.median(np.linalg.norm(soma[both] - pos[both], axis=1)):.0f} (voxels), extent um {np.round((np.nanmax(p, 0) - np.nanmin(p, 0)) / 1000)}')
    if (~placed).sum() > n // 1000:
        sys.exit(f'FAIL brain_layout: {int((~placed).sum())} neurons have no position (> 0.1 %)')
    v = np.stack([p[:, 0], -p[:, 1], p[:, 2]], 1)  # FlyWire y points down
    v -= (np.nanmin(v, 0) + np.nanmax(v, 0)) / 2
    v /= np.nanmax(np.abs(v))
    v[~placed] = 0
    klass = np.array([CLASSES.index(s) if s in CLASSES else len(CLASSES) - 1 for s in c.super], np.uint8)
    klass[~placed] = 255
    gs = groups(c, tomllib.loads(KEYMAP.read_text()))
    OUT.mkdir(parents=True, exist_ok=True)
    v.astype(np.float32).tofile(OUT / 'xyz.bin')
    klass.tofile(OUT / 'class.bin')
    np.concatenate([[len(gs)], *[[len(ids), *ids] for *_, ids in gs]]).astype(np.int32).tofile(OUT / 'groups.bin')
    (OUT / 'groups.txt').write_text(''.join(f'{name}\t{effect}\t{kind}\n' for name, effect, kind, _ in gs))
    (OUT / 'classes.txt').write_text(''.join(f'{k}\n' for k in CLASSES))
    for name, effect, kind, ids in gs:
        print(f'  {kind:6} {name:12} {len(ids):5} neurons  {effect}')
    empty = [name for name, *_, ids in gs if not (klass[ids] != 255).any()]
    if empty:
        sys.exit(f'FAIL brain_layout: groups with no drawn neuron: {empty}')
    print(f'PASS brain_layout: {n} neurons, {len(gs)} groups, {sum(f.stat().st_size for f in OUT.iterdir())} B in {OUT}')


if __name__ == '__main__':
    main()
