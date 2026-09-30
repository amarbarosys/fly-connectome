#!/usr/bin/env python3
"""
tests/test_biophysical_kinetics.py
Unit tests for biophysical Hill-Langmuir dose-response kinetics.
"""

import unittest
from biophysical_kinetics import compute_hill_gain, compute_compound_gains

class TestBiophysicalKinetics(unittest.TestCase):

    def test_zero_dose_neutral(self):
        """Zero concentration must return exact neutral baseline 1.0."""
        g = compute_hill_gain(affinity_nm=100.0, dose_nm=0.0, action_mode="antagonist")
        self.assertEqual(g, 1.0)
        g_ag = compute_hill_gain(affinity_nm=50.0, dose_nm=0.0, action_mode="agonist")
        self.assertEqual(g_ag, 1.0)

    def test_antagonist_monotonic_drop(self):
        """Increasing antagonist concentration must monotonically suppress gain towards 0."""
        ki = 200.0
        g_low = compute_hill_gain(affinity_nm=ki, dose_nm=20.0, action_mode="antagonist")   # 0.1x Ki -> ~0.91
        g_mid = compute_hill_gain(affinity_nm=ki, dose_nm=200.0, action_mode="antagonist")  # 1.0x Ki -> 0.50
        g_high = compute_hill_gain(affinity_nm=ki, dose_nm=2000.0, action_mode="antagonist")# 10x Ki -> ~0.09
        
        self.assertAlmostEqual(g_mid, 0.5, places=2)
        self.assertTrue(1.0 > g_low > g_mid > g_high > 0.0)

    def test_agonist_monotonic_rise(self):
        """Increasing agonist concentration must increase gain above 1.0."""
        ec50 = 100.0
        g_mid = compute_hill_gain(affinity_nm=ec50, dose_nm=100.0, action_mode="agonist", emax=2.0)
        g_high = compute_hill_gain(affinity_nm=ec50, dose_nm=1000.0, action_mode="agonist", emax=2.0)
        
        self.assertAlmostEqual(g_mid, 2.0, places=2) # 1.0 + 2.0*0.5 = 2.0
        self.assertTrue(g_high > g_mid > 1.0)

    def test_null_control_no_affinity(self):
        """Negative control with missing or null affinity remains exactly 1.0 across all channels."""
        ctrl = {
            "name": "Saline",
            "primary_target": None,
            "affinity_val_nm": None,
            "action_mode": "neutral"
        }
        gains = compute_compound_gains(ctrl, dose_nm=1000.0)
        for nt, g in gains.items():
            self.assertEqual(g, 1.0, f"{nt} must be 1.0 for null control")

    def test_picrotoxin_gaba_selective(self):
        """Picrotoxin (Rdl channel blocker) selectively reduces GABA without altering other transmitters."""
        ptx = {
            "name": "Picrotoxin",
            "primary_target": "Rdl (GABA-A)",
            "affinity_val_nm": 300.0,
            "action_mode": "channel_blocker"
        }
        gains = compute_compound_gains(ptx, dose_nm=300.0)
        self.assertAlmostEqual(gains["gaba"], 0.5, places=2)
        for nt in ["ach", "glu", "dop", "ser", "oct"]:
            self.assertEqual(gains[nt], 1.0, f"{nt} should remain unaffected")

if __name__ == "__main__":
    unittest.main()
