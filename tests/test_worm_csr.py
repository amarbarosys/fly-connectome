"""C. elegans CSR (tools/build_worm_csr.py) matches the Cook 2019 sheets edge for edge."""
import os, sys, unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
import build_worm_csr as bw


@unittest.skipUnless(os.path.exists(bw.XLSX), f'fetch {bw.URL} to {bw.XLSX}')
class TestWormCsr(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bw.main()
        cls.names = open(f'{bw.DIR}/worm_nodes.txt').read().split()
        cls.edges = bw.read_all()

    def test_csr_equals_sheet(self):
        for kind in bw.SHEETS:
            magic, row_ptr, col, w = bw.read_csr(f'{bw.DIR}/worm_{kind}_csr.bin')
            self.assertEqual(magic, bw.SHEETS[kind][1])
            self.assertEqual((row_ptr[0], row_ptr[-1]), (0, len(col)))
            self.assertTrue(np.all(np.diff(row_ptr) >= 0) and np.isfinite(w).all())
            self.assertTrue(0 <= col.min() and col.max() < len(self.names))
            got = {(self.names[col[k]], self.names[i]): float(w[k])
                   for i in range(len(self.names)) for k in range(row_ptr[i], row_ptr[i + 1])}
            self.assertEqual(got, self.edges[kind])

    def test_gap_junctions_are_symmetric(self):
        gap = self.edges['gap']
        self.assertTrue(all(gap.get((b, a)) == v for (a, b), v in gap.items()))

    def test_touch_and_command_neurons_present(self):
        for n in ['ALML', 'ALMR', 'AVM', 'PLML', 'PLMR', 'AVAL', 'AVAR', 'AVBL', 'AVBR', 'AVDL', 'PVCL']:
            self.assertIn(n, self.names)


if __name__ == '__main__':
    unittest.main()
