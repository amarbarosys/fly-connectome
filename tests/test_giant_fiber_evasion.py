import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestGiantFiberEvasion(unittest.TestCase):
    def test_evasion_reflex(self):
        from giant_fiber_evasion import GiantFiberEvasionSimulation
        sim = GiantFiberEvasionSimulation()
        res = sim.run_evasion_trial(enable_giant_fiber=True)
        self.assertTrue(res["evaded"], "Fly failed to evade")
        self.assertLessEqual(res["reaction_latency_ms"], 25.0)


if __name__ == "__main__":
    unittest.main()
