#!/usr/bin/env python3
"""
sun_compass_verifier.py
Polarized Sun Compass & Ring Attractor Neural Compass.
Simulates the Central Complex EPG ring attractor dynamics and validates
against real in-vivo 2-photon calcium imaging recordings (Layer 5 DANDI 000289).
"""

import h5py
import numpy as np

class RingAttractorCompass:
    def __init__(self, n_wedges: int = 42):
        self.n_wedges = n_wedges
        self.angles = np.linspace(0, 2 * np.pi, n_wedges, endpoint=False)
        
        # Biologically-grounded Mexican-hat recurrent synaptic kernel
        # Recurrent excitation (sigma = 0.28 rad) + broad lateral inhibition
        diff = np.zeros((n_wedges, n_wedges), dtype=np.float64)
        for i in range(n_wedges):
            for j in range(n_wedges):
                d = abs(self.angles[i] - self.angles[j])
                diff[i, j] = min(d, 2 * np.pi - d)

        self.sigma_exc = 0.28
        self.diff = diff
        self.W_rec = 2.8 * np.exp(-0.5 * (diff / self.sigma_exc)**2) - 0.85

        # Initial persistent bump at center index 21
        self.v = np.exp(-0.5 * (diff[21] / self.sigma_exc)**2) * 2.4

    def step(self, angular_velocity: float = 0.0) -> np.ndarray:
        """Executes one continuous-attractor update step with saturating dynamics."""
        # Shift drive for turn velocity
        shift = angular_velocity * np.roll(self.v, 1) - angular_velocity * np.roll(self.v, -1)
        syn_in = np.dot(self.W_rec, self.v) + 0.3 * shift
        
        # Saturating non-linearity + supralinear spiking threshold
        v_next = np.maximum(0.0, np.tanh(0.50 * self.v + 0.50 * syn_in) * 2.4)
        if v_next.max() > 0:
            self.v = (v_next / v_next.max())**1.5 * 2.39
        return self.v.copy()

    def get_aligned_profile(self) -> np.ndarray:
        peak_idx = np.argmax(self.v)
        mid_idx = self.n_wedges // 2
        return np.roll(self.v, mid_idx - peak_idx)

def verify_against_layer5():
    nwb_path = "data/sub-13_ses-20201201_behavior_ophys.nwb"
    print("================ SUN COMPASS / RING ATTRACTOR VERIFICATION ================")
    print(f"Loading in-vivo 2-photon calcium recordings from Layer 5: {nwb_path}")
    
    with h5py.File(nwb_path, "r") as f:
        rois = f["processing/ophys/Fluorescence/RoiResponseSeries/data"][:] # 11,020 x 42 ROIs
        
    N_frames = 500
    print(f"Extracting empirical calcium activity bump profile across {N_frames} frames...")
    
    aligned_bio = []
    for t in range(N_frames):
        frame = np.maximum(0.0, rois[t])
        if frame.max() > 0.5:
            p_idx = np.argmax(frame)
            rolled = np.roll(frame, 21 - p_idx)
            aligned_bio.append(rolled)
            
    bio_profile = np.mean(aligned_bio, axis=0)
    
    # Calculate Biological Full-Width at Half-Maximum (FWHM)
    bio_half = (bio_profile.max() + bio_profile.min()) / 2.0
    bio_fwhm_wedges = np.sum(bio_profile > bio_half)
    bio_fwhm_deg = bio_fwhm_wedges * (360.0 / 42.0)

    print("\n[Stage 1] Biological Ground Truth (Layer 5 Empirical Baseline):")
    print(f"  - Peak ΔF/F Amplitude:      {bio_profile.max():.2f}")
    print(f"  - Baseline Inactive Level:  {bio_profile.min():.2f}")
    print(f"  - Contrast Ratio:           {(bio_profile.max() / (bio_profile.min() + 1e-4)):.1f}x")
    print(f"  - Compass Bump FWHM:        {bio_fwhm_deg:.1f}° ({bio_fwhm_wedges} wedges)")

    print("\n[Stage 2] Simulating Ring Attractor Neural Dynamics...")
    compass = RingAttractorCompass(n_wedges=42)
    
    # Run 50 steps in darkness (testing working memory bump persistence)
    for _ in range(50):
        compass.step()
        
    sim_profile = compass.get_aligned_profile()
    
    # Calculate Simulated FWHM
    sim_half = (sim_profile.max() + sim_profile.min()) / 2.0
    sim_fwhm_wedges = np.sum(sim_profile > sim_half)
    sim_fwhm_deg = sim_fwhm_wedges * (360.0 / 42.0)

    # Focus on the heading bump aperture (±90 degrees from peak: indices 11 to 31)
    aperture = slice(11, 32)
    bio_aperture = bio_profile[aperture]
    sim_aperture = sim_profile[aperture]
    corr = np.corrcoef(bio_aperture, sim_aperture)[0, 1]
    
    print("\n[Stage 3] Attractor Persistence & Shape Matching:")
    print(f"  - Simulated Bump Amplitude: {sim_profile.max():.2f}")
    print(f"  - Simulated Bump FWHM:      {sim_fwhm_deg:.1f}° ({sim_fwhm_wedges} wedges)")
    print(f"  - FWHM Delta:               {abs(sim_fwhm_deg - bio_fwhm_deg):.1f}°")
    print(f"  - Bump Aperture Match:      {corr * 100:.1f}% (Pearson r = {corr:.4f})")

    # Render ASCII profile comparison
    print("\nHeading Bump Aperture (±90° Arc, 21 Wedges):")
    print("  Arc:        -90°              0°(PEAK)            +90°")
    
    def render_spark(arr, max_val):
        ticks = [" ", " ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
        return "".join([ticks[min(len(ticks)-1, int((v / max_val) * (len(ticks)-1)))] for v in arr])

    print(f"  Bio Calcium: [{render_spark(bio_aperture, bio_aperture.max())}]")
    print(f"  Sim Attract: [{render_spark(sim_aperture, sim_aperture.max())}]")

    assert corr > 0.80, f"Shape correlation below threshold: {corr:.4f}"
    assert abs(sim_fwhm_deg - bio_fwhm_deg) <= 20.0, f"FWHM error too large: {abs(sim_fwhm_deg - bio_fwhm_deg)}°"
    print("\nSun Compass Ring Attractor verified against Layer 5 empirical data: PASS")
    print("=========================================================================")

if __name__ == "__main__":
    verify_against_layer5()
