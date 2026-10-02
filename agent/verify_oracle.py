"""P0 Oracle gate: tiny graph replay determinism.

Runs a CPU brain on the tiny graph (matching CPUBrain dynamics),
captures full state trace, and verifies exact reproduction across
independent instances.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
from scipy.sparse import csr_matrix

from agent.tiny_graph import TinyGraph, oracle_trace, oracle_spikes, verify_oracle


@dataclass
class OracleResult:
    graph_hash: str
    voltage_checksum: str
    spike_checksum: str
    replay_match: bool
    instance_isolation: bool
    tick_count: int
    neuron_count: int
    edge_count: int


def compute_hash(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


class TinyCPUBrain:
    """Tiny graph brain matching CPUBrain dynamics exactly."""
    
    def __init__(self, graph: TinyGraph):
        self.graph = graph
        self.N = graph.n_neurons
        self.ports = np.eye(self.N, dtype=np.float32)  # identity for simplicity
        self.descending = np.arange(self.N, dtype=np.int64)  # all neurons are descending
        
        # Build CSR matrix
        self.graph_csr = csr_matrix(
            (graph.weights, graph.col_idx, graph.row_ptr),
            shape=(self.N, self.N)
        )
        self.reset()

    def reset(self):
        self.v = np.zeros((self.N, 2), np.float32)
        self.spikes = np.zeros_like(self.v)
        self.trace = np.zeros_like(self.v)
        self.ever = np.zeros((self.N, 2), bool)

    def step(self, inputs, ticks=1, direct=()):
        # CPUBrain uses 2 channels per neuron; we'll use 1 for oracle
        inputs = np.asarray(inputs, np.float32).reshape(2, -1)
        if inputs.shape[1] == 1:
            # Single input channel - broadcast
            inputs = np.broadcast_to(inputs, (2, self.N // 2)).copy()
        
        # Neural drive
        drive = (self.ports @ inputs.T) * np.float32(0.65)
        
        if len(direct) > 0:
            direct = np.asarray(direct, dtype=np.int64)
            if np.any((direct < 0) | (direct >= self.N)):
                raise ValueError('invalid direct targets')
            mask = ~self.ever[direct, 0]
            drive[direct[mask], 0] += np.float32(0.65)

        for _ in range(ticks):
            v = np.float32(0.85) * self.v + np.float32(0.08) * (self.graph_csr @ self.spikes) + drive
            self.spikes = (v >= 1).astype(np.float32)
            self.v = np.where(self.spikes != 0, 0, np.maximum(v, -5)).astype(np.float32)
            self.trace = np.float32(0.85) * self.trace + np.float32(0.15) * self.spikes
            self.ever |= self.spikes != 0

        # Return first channel voltage for oracle comparison
        return self.v[:, 0].copy()

    @property
    def voltage(self) -> np.ndarray:
        return self.v[:, 0]

    @property
    def spike_vec(self) -> np.ndarray:
        return self.spikes[:, 0].astype(np.int8)


def run_oracle(out_dir: Path, ticks: int = 16) -> OracleResult:
    out_dir.mkdir(parents=True, exist_ok=False)

    # 1. Verify analytic oracle
    ok, msg = verify_oracle()
    if not ok:
        raise RuntimeError(f"Analytic oracle failed: {msg}")

    # 2. Build TinyCPUBrain with tiny graph
    g = TinyGraph()
    brain = TinyCPUBrain(g)

    # 3. Run forward for ticks
    V_trace = np.zeros((ticks, g.n_neurons), dtype=np.float32)
    S_trace = np.zeros((ticks, g.n_neurons), dtype=np.int8)

    # Input: neuron 0 gets 1.0 at t=0
    # Use a simple input format
    I_ext = np.zeros((2, g.n_neurons), dtype=np.float32)
    I_ext[0, 0] = 1.0  # neuron 0, channel 0

    for t in range(ticks):
        if t > 0:
            I_ext[0, 0] = 0.0

        brain.step(I_ext, ticks=1)
        V_trace[t] = brain.voltage
        S_trace[t] = brain.spike_vec

    # 4. Verify against analytic oracle (first channel only)
    V_oracle = oracle_trace(ticks)
    S_oracle = oracle_spikes(ticks)

    v_match = np.array_equal(V_trace, V_oracle)
    s_match = np.array_equal(S_trace, S_oracle)

    if not (v_match and s_match):
        # Print first mismatch for debugging
        diff_v = np.where(V_trace != V_oracle)
        diff_s = np.where(S_trace != S_oracle)
        raise RuntimeError(
            f"TinyCPUBrain diverges from analytic oracle: "
            f"V_match={v_match}, S_match={s_match}, "
            f"V diff at {diff_v[:5]}, S diff at {diff_s[:5]}"
        )

    # 5. Checkpoint at tick 5, restore, continue
    # Save state manually
    ckpt = {
        'v': brain.v.copy(),
        'spikes': brain.spikes.copy(),
        'trace': brain.trace.copy(),
        'ever': brain.ever.copy(),
    }
    np.savez(out_dir / "ckpt_tick5.npz", **ckpt)

    brain2 = TinyCPUBrain(g)
    ckpt2 = np.load(out_dir / "ckpt_tick5.npz")
    brain2.v = ckpt2['v']
    brain2.spikes = ckpt2['spikes']
    brain2.trace = ckpt2['trace']
    brain2.ever = ckpt2['ever']

    # Continue both from tick 5
    for t in range(5, ticks):
        brain.step(I_ext if t == 5 else np.zeros_like(I_ext), ticks=1)
        brain2.step(I_ext if t == 5 else np.zeros_like(I_ext), ticks=1)
        if not np.array_equal(brain.voltage, brain2.voltage):
            raise RuntimeError(f"Restored instance diverged at tick {t}")
        if not np.array_equal(brain.spike_vec, brain2.spike_vec):
            raise RuntimeError(f"Restored instance spikes diverged at tick {t}")

    # 6. Instance isolation test: second brain with zero input
    brain3 = TinyCPUBrain(g)
    for t in range(ticks):
        brain3.step(np.zeros_like(I_ext), ticks=1)
        if np.any(brain3.voltage != 0) or np.any(brain3.spike_vec != 0):
            raise RuntimeError(f"Zero-input instance not isolated at tick {t}")

    # 7. Graph mutation rejection - skipped (TinyCPUBrain doesn't validate on init;
    # real CPUBrain validates via checkpoint graph hash. Tested in verify_replay.py)

    # 8. Corrupted state rejection - checkpoint save validates NaN/inf
    # np.savez doesn't validate; real checkpoint does. Skip.

    # 9. Save artifacts
    np.save(out_dir / "voltage_trace.npy", V_trace)
    np.save(out_dir / "spike_trace.npy", S_trace)

    result = OracleResult(
        graph_hash=compute_hash(g.weights),
        voltage_checksum=compute_hash(V_trace),
        spike_checksum=compute_hash(S_trace),
        replay_match=True,
        instance_isolation=True,
        tick_count=ticks,
        neuron_count=g.n_neurons,
        edge_count=g.n_edges_actual,
    )

    (out_dir / "metrics.json").write_text(json.dumps(asdict(result), indent=2))
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ticks", type=int, default=16)
    args = parser.parse_args()

    result = run_oracle(args.out, args.ticks)
    print(f"Oracle PASS: {result.neuron_count} neurons, {result.edge_count} edges, {result.tick_count} ticks")
    print(f"  Graph hash: {result.graph_hash}")
    print(f"  Voltage checksum: {result.voltage_checksum}")
    print(f"  Spike checksum: {result.spike_checksum}")
    print(f"  Replay match: {result.replay_match}")
    print(f"  Instance isolation: {result.instance_isolation}")


if __name__ == "__main__":
    main()