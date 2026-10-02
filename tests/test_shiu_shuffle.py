"""Properties of the degree-preserving shuffle, after Kisame76/drosophila-brain-mlx tests/test_shuffle_pack.py."""
import os, sys, unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.shiu_shuffle import shuffle_edges

N = 300


def graph(seed=0, edges=900):
    """Random graph without parallel edges; out-degrees skewed so repairs are needed."""
    rng = np.random.default_rng(seed)
    pre = np.minimum(rng.geometric(.02, edges * 3) - 1, N - 1)
    post = rng.integers(0, N, pre.size)
    keep = pre != post
    pre, post = pre[keep], post[keep]
    _, first = np.unique(pre * N + post, return_index=True)
    first = np.sort(first)[:edges]
    return pre[first], post[first], rng.choice([-3, -1, 1, 2, 5], first.size)


class TestShuffleEdges(unittest.TestCase):
    def test_every_cell_keeps_its_in_degree(self):
        # out-degree, sign and out-weight hold by construction: pre and weights are never touched
        pre, post, _ = graph()
        new, _ = shuffle_edges(pre, post, seed=0)
        self.assertEqual(new.shape, post.shape)
        np.testing.assert_array_equal(np.bincount(new, minlength=N), np.bincount(post, minlength=N))

    def test_input_mix_changes_but_total_input_is_conserved(self):
        pre, post, w = graph()
        new, _ = shuffle_edges(pre, post, seed=0)
        before, after = np.bincount(post, w, N), np.bincount(new, w, N)
        self.assertFalse(np.allclose(before, after))
        self.assertAlmostEqual(before.sum(), after.sum())

    def test_no_parallel_edges_after_repair(self):
        pre, post, _ = graph()
        new, stats = shuffle_edges(pre, post, seed=0)
        self.assertGreaterEqual(stats['repair_rounds'], 1)
        self.assertEqual(np.unique(pre * N + new).size, pre.size)

    def test_the_wiring_changes_and_the_seed_decides_how(self):
        pre, post, _ = graph()
        a, stats = shuffle_edges(pre, post, seed=0)
        np.testing.assert_array_equal(a, shuffle_edges(pre, post, seed=0)[0])
        self.assertFalse(np.array_equal(a, shuffle_edges(pre, post, seed=1)[0]))
        self.assertEqual(stats['edges_kept'], np.intersect1d(pre * N + a, pre * N + post).size)
        self.assertLess(stats['edges_kept'], post.size)
        self.assertEqual(stats['self_loops'], 0)

    def test_repeats_left_after_the_last_round_are_refused(self):
        pre, post, _ = graph()
        refused = 0
        for seed in range(5):
            try:
                shuffle_edges(pre, post, seed=seed, max_rounds=0)
            except RuntimeError:
                refused += 1
        self.assertEqual(refused, 5)

    def test_bad_input_is_refused(self):
        with self.assertRaises(ValueError):  # parallel edge in the input
            shuffle_edges([0, 0, 1], [1, 1, 2], seed=0)
        with self.assertRaises(ValueError):  # length mismatch
            shuffle_edges([0, 0], [1, 2, 3], seed=0)


if __name__ == '__main__':
    unittest.main()
