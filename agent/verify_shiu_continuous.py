"""Verify that split resident Shiu advances preserve all neural state and the absolute tick."""
from __future__ import annotations

import argparse
import subprocess
import sys

import numpy as np

from agent.shiu import ShiuBrain, ShiuGPU, ShiuLive

N = 4
PRE = np.array([0, 1], np.int32)
POST = np.array([1, 2], np.int32)
SYN = np.array([12, 8], np.float32)
GROUPS = np.array([0, 1], np.uint8)
READOUT = np.arange(N, dtype=np.int32)
SPLIT = 10
TAIL = 32_090
TOTAL = SPLIT + TAIL


def configure(brain):
    brain.edge_groups(GROUPS, np.ones((1, 2), np.float32))
    brain.modulate(np.array([0, -1], np.int32), tau=(0.05,), k=(0.5,), b=(1.0,))
    brain.stimulate([0], rate=10_000, amount=8.0, window=(0, 25))
    return brain


def cpu_gate():
    whole = configure(ShiuBrain(PRE, POST, SYN, N, seed=3))
    whole.step(TOTAL)
    split = configure(ShiuBrain(PRE, POST, SYN, N, seed=3))
    split.step(SPLIT)
    split.step(TAIL)
    for name in ("u", "g", "last", "counts", "m"):
        if not np.array_equal(getattr(whole, name), getattr(split, name)):
            raise AssertionError(f"CPU split state mismatch: {name}")
    if len(whole.ring) != len(split.ring) or any(not np.array_equal(a, b) for a, b in zip(whole.ring, split.ring)):
        raise AssertionError("CPU split state mismatch: delay ring")
    print(f"PASS CPU continuous advance: {TOTAL} ticks, split {SPLIT}+{TAIL}, state and spikes exact")


def gpu_gate(mutant_child: bool):
    whole_gpu = configure(ShiuGPU(PRE, POST, SYN, N, seed=3))
    whole_counts = whole_gpu.run([TOTAL])[0][:, 0]
    whole = configure(ShiuLive(PRE, POST, SYN, N, seed=3)).start([1], READOUT)
    split = configure(ShiuLive(PRE, POST, SYN, N, seed=3)).start([1], READOUT)
    try:
        whole_readout = whole.advance(TOTAL)
        split.advance(SPLIT)
        split_readout = split.advance(TAIL)
        whole_state, split_state = whole.snapshot(), split.snapshot()
        if not np.array_equal(whole_counts, whole_readout):
            raise AssertionError("one-shot and resident readouts differ")
        if not np.array_equal(whole_readout, split_readout):
            raise AssertionError("resident split readouts differ")
        for name in whole_state:
            if not np.array_equal(whole_state[name], split_state[name]):
                raise AssertionError(f"GPU split state mismatch: {name}")
        if int(whole_state["t"][0]) != TOTAL:
            raise AssertionError("continuous tick did not cross 32,000")
    finally:
        whole.close()
        split.close()
    print(f"PASS GPU continuous advance: {TOTAL} ticks, split {SPLIT}+{TAIL}, full state and spikes exact")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mutant-child", action="store_true")
    args = parser.parse_args()
    cpu_gate()
    gpu_gate(args.mutant_child)
    if not args.mutant_child:
        subprocess.run([
            str(__import__("pathlib").Path.home() / "iTools/bin/mutant-check"),
            "agent/shiu_worker.mojo", "s/if is_sniff:/if is_sniff or is_advance:/", "--",
            sys.executable, "-m", "agent.verify_shiu_continuous", "--mutant-child",
        ], check=True)
        print("PASS state-reset mutant rejected")


if __name__ == "__main__":
    main()
