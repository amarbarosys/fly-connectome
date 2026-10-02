"""CPUBrain.load_columns swaps a control arm's wiring in place, equal to a fresh build."""
import numpy as np
import pytest

from agent.neural_connectome import ROOT, CPUBrain, graph_arrays

ARM = ROOT / '.work/agent/matched-baseline/block-s7'


def features(brain, drive):
    brain.reset()
    return np.stack([brain.step(np.zeros((1, 8), np.float32), extra_drive=drive) for _ in range(5)])


@pytest.mark.skipif(not ARM.is_dir(), reason='control arm graphs are .work artifacts')
def test_load_columns_matches_fresh_build(monkeypatch):
    row, _, weights = graph_arrays(ROOT / 'data')
    arm_row, arm_col, arm_weights = graph_arrays(ARM)
    assert np.array_equal(row, arm_row) and np.array_equal(weights, arm_weights)
    drive = np.zeros((139255, 1), np.float32)
    drive[:2000] = 1
    brain = CPUBrain(flies=1)
    real = features(brain, drive)
    brain.load_columns(ARM)
    assert np.array_equal(brain.graph.indices, arm_col)
    swapped = features(brain, drive)
    monkeypatch.setattr('agent.neural_connectome.graph_arrays', lambda: graph_arrays(ARM))
    assert np.array_equal(swapped, features(CPUBrain(flies=1), drive))
    assert not np.array_equal(swapped, real)
    brain.load_columns(ROOT / 'data')
    assert np.array_equal(features(brain, drive), real)
