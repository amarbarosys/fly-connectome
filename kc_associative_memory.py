#!/usr/bin/env python3
"""
kc_associative_memory.py
One-Shot Associative Memory based on the 5,177 Drosophila Mushroom Body Kenyon Cells (KCs)
and 96 Output Neurons (MBONs).
Implements biological high-dimensional sparse projection (5% winner-take-all) with Hebbian plasticity.
"""

import numpy as np
import polars as pl
import scipy.sparse as sp

class MushroomBodyAssociativeMemory:
    def __init__(self, n_inputs: int = 64, n_classes: int = 10, sparsity: float = 0.05):
        self.n_inputs = n_inputs
        self.n_kcs = 5177 # Real Drosophila KC population
        self.n_classes = n_classes
        self.k_winners = int(self.n_kcs * sparsity) # 5% sparse code (~258 active KCs)

        # Biological PN -> KC projection matrix: each KC samples 6 random input PNs
        np.random.seed(42)
        pn_to_kc = np.zeros((self.n_kcs, self.n_inputs), dtype=np.float32)
        for i in range(self.n_kcs):
            chosen_pns = np.random.choice(self.n_inputs, size=6, replace=False)
            pn_to_kc[i, chosen_pns] = np.random.uniform(0.5, 1.5, size=6)
        self.W_pn_kc = pn_to_kc

        # KC -> MBON plasticity matrix (learned via one-shot dopamine-gated Hebbian learning)
        self.W_kc_mbon = np.zeros((self.n_classes, self.n_kcs), dtype=np.float32)

    def encode_kcs(self, x: np.ndarray) -> np.ndarray:
        """Projects input into high-dimensional space and applies 5% winner-take-all inhibition."""
        # Linear dendritic summation
        kc_drive = np.dot(self.W_pn_kc, x)
        
        # APL lateral inhibition: Top-k winner-take-all
        kc_spikes = np.zeros(self.n_kcs, dtype=np.float32)
        top_k_indices = np.argpartition(kc_drive, -self.k_winners)[-self.k_winners:]
        kc_spikes[top_k_indices] = 1.0 # Pure sparse binary code
        return kc_spikes

    def train_one_shot(self, x: np.ndarray, target_class: int):
        """Single-shot dopamine-gated Hebbian update."""
        kc_spikes = self.encode_kcs(x)
        # Potentiate target class, depress non-target
        self.W_kc_mbon[target_class] += 1.0 * kc_spikes
        # Normalize weights
        norm = np.linalg.norm(self.W_kc_mbon[target_class])
        if norm > 0:
            self.W_kc_mbon[target_class] /= norm

    def predict(self, x: np.ndarray) -> int:
        """Readout from MBONs."""
        kc_spikes = self.encode_kcs(x)
        mbon_activations = np.dot(self.W_kc_mbon, kc_spikes)
        return int(np.argmax(mbon_activations))

def run_one_shot_benchmark():
    print("================ KENYON CELL ONE-SHOT ASSOCIATIVE MEMORY ================")
    print("Architecture: 64 Sensory Inputs -> 5,177 Kenyon Cells (5% Sparsity) -> 10 MBONs")
    
    mem = MushroomBodyAssociativeMemory(n_inputs=64, n_classes=10, sparsity=0.05)
    
    # Generate 10 distinct random prototype patterns (single training sample each)
    np.random.seed(1337)
    prototypes = np.random.randn(10, 64).astype(np.float32)
    # Normalize prototypes
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True)

    print("\n[Phase 1] Training 10 classes with EXACTLY 1 presentation each (One-Shot)...")
    for c in range(10):
        mem.train_one_shot(prototypes[c], target_class=c)
    print("Training complete in 1 pass (Zero backpropagation, Zero epochs).")

    print("\n[Phase 2] Evaluating Noise Robustness on Corrupted Memory Retrieval:")
    print("Noise Level | Correct / Total | Accuracy (%) | Status")
    print("------------|-----------------|--------------|-------")
    
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    n_trials = 100
    
    for noise in noise_levels:
        correct = 0
        for _ in range(n_trials):
            c_true = np.random.randint(0, 10)
            # Add Gaussian noise
            noisy_input = prototypes[c_true] + np.random.normal(0, noise, size=64).astype(np.float32)
            c_pred = mem.predict(noisy_input)
            if c_pred == c_true:
                correct += 1
                
        acc = (correct / n_trials) * 100.0
        status = "PERFECT" if acc == 100.0 else ("EXCELLENT" if acc >= 90.0 else ("ROBUST" if acc >= 70.0 else "DEGRADED"))
        print(f" {noise*100:5.0f}% noise |   {correct:3d} / {n_trials:3d}   |   {acc:5.1f}%    | {status}")

    print("=========================================================================")

if __name__ == "__main__":
    run_one_shot_benchmark()
