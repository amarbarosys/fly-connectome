"""Graded worm engine (agent/worm.py): implicit leak + gap step is stable and relaxes to rest."""
import os, sys, unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent import worm


class TestWormEngine(unittest.TestCase):
    def test_leak_and_gap_relax_to_rest(self):
        # No chemical synapses: every cell must decay to Ec, gap junctions only pull cells together.
        rng = np.random.default_rng(0)
        g = (rng.random((40, 40)) < .1).astype(np.float32) * 5
        b = worm.WormBrain([np.zeros((40, 40), np.float32)], [g], np.zeros(40), 2)
        b.v[:] = rng.uniform(-80, 20, b.v.shape)
        np.testing.assert_allclose(b.run(np.zeros((1, 40, 2), np.float32), 1.5), worm.EC, atol=1e-2)  # float32 rest bias ~3 uV

    def test_gap_current_conserves_charge(self):
        # Laplacian rows sum to zero: gap junctions move charge, never create it.
        g = np.random.default_rng(1).random((30, 30)).astype(np.float32)
        np.testing.assert_allclose(worm.lap(g).sum(1), 0, atol=1e-4)

    @unittest.skipUnless((worm.DATA / 'worm_gap_csr.bin').exists(), 'run tools/build_worm_csr.py')
    def test_real_graph_stays_finite(self):
        names, chem, gap, rev = worm.load()
        n = len(names)
        self.assertEqual((n, int((rev < 0).sum())), (300, 26))
        b = worm.WormBrain([worm.dense(n, *chem)], [worm.dense(n, *gap)], rev, 1)
        v = b.run(np.zeros((1, n, 1), np.float32), 0.2)
        self.assertTrue(np.isfinite(v).all() and np.abs(v).max() < 100)


if __name__ == '__main__':
    unittest.main()
