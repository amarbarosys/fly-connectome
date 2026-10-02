"""Test the optic-column map against anatomy, using the linked skeleton morphology."""
import json
from pathlib import Path
import time
import numpy as np
from agent.neural_connectome import ROOT, ASSETS, digest

MAP = ASSETS/'spatial-vision.npz'
MORPHOLOGY = ROOT/'.work/agent/skeleton-link/morphology.npz'
PAIRS = 20000
SEED = 20261002


def correlate(lattice, centroid, generator, count=PAIRS):
    """Correlate lattice distance with arbor centroid distance over random neuron pairs."""
    size = len(lattice)
    first, second = generator.integers(0, size, count), generator.integers(0, size, count)
    keep = first != second
    first, second = first[keep], second[keep]
    lattice_gap = np.linalg.norm(lattice[first]-lattice[second], axis=1)
    centroid_gap = np.linalg.norm(centroid[first]-centroid[second], axis=1)
    return float(np.corrcoef(lattice_gap, centroid_gap)[0, 1]), int(len(lattice_gap))


def verify():
    column = np.load(MAP, allow_pickle=False)
    morphology = np.load(MORPHOLOGY)
    # exact integer keys: root IDs exceed 2**53, so any float64 path corrupts them
    row_of = {int(root): row for row, root in enumerate(morphology['root_id'].astype(np.int64))}
    centroids = np.stack([morphology[f'centroid_{axis}_nm'] for axis in 'xyz'], axis=1)
    results, missing = {}, 0
    # per eye: the two optic lobes are mirrored lattices, and pooling them could
    # manufacture a correlation out of the left/right split alone
    for eye in sorted(set(column['eye'].tolist())):
        select = column['eye'] == eye
        rows = np.array([row_of.get(int(r), -1) for r in column['root_ids'][select].astype(np.int64)])
        found = rows >= 0
        missing += int((~found).sum())
        lattice = np.stack([column['p'][select][found], column['q'][select][found]], axis=1).astype(np.float64)
        centroid = centroids[rows[found]]
        assert np.isfinite(centroid).all(), 'a mapped neuron has no finite centroid'
        measured, pairs = correlate(lattice, centroid, np.random.default_rng(SEED+eye))
        shuffled = []
        for repeat in range(5):
            control = np.random.default_rng(SEED+100*repeat+eye)
            shuffled.append(correlate(lattice[control.permutation(len(lattice))], centroid, control)[0])
        results[f'eye_{eye}'] = {'neurons': int(found.sum()), 'pairs_sampled': pairs,
            'measured_correlation': measured, 'shuffled_control_mean': float(np.mean(shuffled)),
            'shuffled_control_max_abs': float(np.max(np.abs(shuffled))),
            'beats_control': measured > float(np.max(np.abs(shuffled)))}
    report = {'status': 'PASS' if all(r['beats_control'] for r in results.values()) else 'FAIL',
        'question': 'do neighbouring optic columns hold neurons with neighbouring arbors',
        'mapped_neurons_total': int(len(column['root_ids'])),
        'mapped_neurons_without_morphology': missing, 'per_eye': results,
        'control': 'lattice coordinates permuted among same-eye neurons, 5 repeats',
        'scope': 'correlation between column-lattice distance and arbor centroid distance on the '
                 'LOD1 ds2 downsampling; tests the column assignment against anatomy and does not '
                 'calibrate retinal angles or validate image registration',
        'sources_sha256': {str(p.relative_to(ROOT)): digest(p) for p in (MAP, MORPHOLOGY, Path(__file__))},
        'recorded_unix': time.time()}
    out = ROOT/'.work/agent/column-geometry'
    out.mkdir(parents=True, exist_ok=True)
    (out/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    for name, row in results.items():
        print(f"{name}: {row['neurons']:4d} neurons  r={row['measured_correlation']:+.4f}  "
              f"shuffled {row['shuffled_control_mean']:+.4f} (max abs {row['shuffled_control_max_abs']:.4f})  "
              f"{'beats control' if row['beats_control'] else 'DOES NOT beat control'}")
    print('status', report['status'], '| mapped neurons without morphology:', missing)
    return report

if __name__ == '__main__':verify()
