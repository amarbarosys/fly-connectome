"""Evaluate the task policy in the deterministic kinematic harness."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from arena import BeaconArena
from policy import BeaconAvoidPolicy


def run_episode(seed: int, max_steps: int = 600) -> dict:
    arena = BeaconArena(seed, max_steps)
    policy = BeaconAvoidPolicy()
    policy.reset(seed)
    done = False
    while not done:
        done = arena.step(policy.act(arena.observe()))
    return asdict(arena.result())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument(
        "--out", type=Path, default=Path("../.work/agent/runs/baseline.json")
    )
    args = parser.parse_args()
    rows = [run_episode(args.seed + i, args.max_steps) for i in range(args.episodes)]
    report = {
        "evidence": "kinematic_policy_only",
        "seed_start": args.seed,
        "episodes": rows,
        "summary": {
            "success_rate": sum(r["success"] for r in rows) / len(rows),
            "collision_rate": sum(r["collision"] for r in rows) / len(rows),
            "mean_steps": sum(r["steps"] for r in rows) / len(rows),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], sort_keys=True))
    print(f"receipt: {args.out.resolve()}")


if __name__ == "__main__":
    main()
