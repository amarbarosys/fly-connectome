"""Tiny deterministic graph for P0 oracle gate.

100 neurons, fixed topology, known analytic trace. Used to verify replay determinism
independent of full connectome data.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TinyGraph:
    """Immutable tiny graph with known oracle trace."""
    n_neurons: int = 100
    n_edges: int = 347
    seed: int = 0xFEEDFACE

    def __post_init__(self):
        # Ring topology: i -> (i+1) % N, plus skip connections
        rows = []
        cols = []
        weights = []

        for i in range(self.n_neurons):
            # Ring edge
            rows.append(i)
            cols.append((i + 1) % self.n_neurons)
            weights.append(0.5)

            # Skip-2 edge (every 3rd neuron)
            if i % 3 == 0:
                rows.append(i)
                cols.append((i + 2) % self.n_neurons)
                weights.append(0.3)

            # Skip-5 edge (every 7th neuron)
            if i % 7 == 0:
                rows.append(i)
                cols.append((i + 5) % self.n_neurons)
                weights.append(0.2)

            # Inhibitory back-edge (every 11th)
            if i % 11 == 0:
                rows.append(i)
                cols.append((i - 1) % self.n_neurons)
                weights.append(-0.4)

        rows_arr = np.array(rows, dtype=np.int32)
        cols_arr = np.array(cols, dtype=np.int32)
        weights_arr = np.array(weights, dtype=np.float32)

        # CSR
        row_ptr = np.zeros(self.n_neurons + 1, dtype=np.int32)
        for r in rows_arr:
            row_ptr[r + 1] += 1
        np.cumsum(row_ptr, out=row_ptr)

        # Sort by row for CSR
        order = np.argsort(rows_arr)
        rows_arr = rows_arr[order]
        cols_arr = cols_arr[order]
        weights_arr = weights_arr[order]

        object.__setattr__(self, '_rows', rows_arr)
        object.__setattr__(self, '_cols', cols_arr)
        object.__setattr__(self, '_weights', weights_arr)
        object.__setattr__(self, '_row_ptr', row_ptr)

    @property
    def row_ptr(self) -> np.ndarray:
        return self._row_ptr.copy()

    @property
    def col_idx(self) -> np.ndarray:
        return self._cols.copy()

    @property
    def weights(self) -> np.ndarray:
        return self._weights.copy()

    @property
    def n_edges_actual(self) -> int:
        return len(self._rows)


# Oracle trace: deterministic voltage trajectory matching CPUBrain dynamics
# CPUBrain: leak=0.85, synaptic=0.08, threshold=1.0, reset=0, clamp=-5
# Input: neuron 0 receives 1.0 at t=0, then 0 (scaled by drive factor 0.65)
def oracle_trace(ticks: int = 16) -> np.ndarray:
    """Return (ticks, n_neurons) voltage trace for tiny graph matching CPUBrain."""
    g = TinyGraph()
    n = g.n_neurons
    V = np.zeros((ticks, n), dtype=np.float32)
    spikes = np.zeros((n, 2), dtype=np.float32)  # 2 channels like CPUBrain
    trace = np.zeros((n, 2), dtype=np.float32)
    ever = np.zeros((n, 2), dtype=bool)

    # Build CSR matrix like CPUBrain
    from scipy.sparse import csr_matrix
    graph_csr = csr_matrix((g.weights, g.col_idx, g.row_ptr), shape=(n, n))

    # CPUBrain uses ports @ inputs * 0.65
    # For oracle: ports = identity, inputs[0, 0] = 1.0 at t=0
    # So drive[0, 0] = 0.65 at t=0, then 0
    drive = np.zeros((n, 2), dtype=np.float32)
    drive[0, 0] = 0.65

    for t in range(ticks):
        if t > 0:
            drive[0, 0] = 0.0

        # CPUBrain dynamics: v = 0.85*v + 0.08*(graph @ spikes) + drive
        if t == 0:
            V[t] = drive[:, 0]
        else:
            synaptic = np.float32(0.08) * (graph_csr @ spikes[:, 0])
            V[t] = (np.float32(0.85) * V[t - 1] + synaptic + drive[:, 0])

        # Threshold & spike
        spikes[:, 0] = (V[t] >= 1.0).astype(np.float32)
        V[t] = np.where(spikes[:, 0] != 0, 0, np.maximum(V[t], -5)).astype(np.float32)

        # Trace update
        trace[:, 0] = np.float32(0.85) * trace[:, 0] + np.float32(0.15) * spikes[:, 0]
        ever[:, 0] |= spikes[:, 0] != 0

    return V


def oracle_spikes(ticks: int = 16) -> np.ndarray:
    """Return (ticks, n_neurons) binary spike trace."""
    V = oracle_trace(ticks)
    return (V >= 1.0).astype(np.int8)


# Expected checksums for verification (computed on first run)
ORACLE_VOLTAGE_CHECKSUM = 0x5FA8_A5C5
ORACLE_SPIKE_CHECKSUM = 0xC511_7D35


def verify_oracle() -> tuple[bool, str]:
    """Run oracle and verify checksums."""
    import zlib
    V = oracle_trace(16)
    S = oracle_spikes(16)

    v_crc = zlib.crc32(V.tobytes()) & 0xFFFFFFFF
    s_crc = zlib.crc32(S.tobytes()) & 0xFFFFFFFF

    ok = (v_crc == ORACLE_VOLTAGE_CHECKSUM) and (s_crc == ORACLE_SPIKE_CHECKSUM)
    msg = f"Voltage CRC: {v_crc:08X} (exp {ORACLE_VOLTAGE_CHECKSUM:08X}), Spike CRC: {s_crc:08X} (exp {ORACLE_SPIKE_CHECKSUM:08X})"
    return ok, msg


if __name__ == "__main__":
    # First run: print checksums to update constants
    V = oracle_trace(16)
    S = oracle_spikes(16)
    import zlib
    v_crc = zlib.crc32(V.tobytes()) & 0xFFFFFFFF
    s_crc = zlib.crc32(S.tobytes()) & 0xFFFFFFFF
    print(f"ORACLE_VOLTAGE_CHECKSUM = 0x{v_crc:08X}")
    print(f"ORACLE_SPIKE_CHECKSUM = 0x{s_crc:08X}")

    ok, msg = verify_oracle()
    print(msg)
    print("PASS" if ok else "FAIL")
    exit(0 if ok else 1)