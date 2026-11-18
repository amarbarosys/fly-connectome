"""Inspectable task policy; inputs/actions are defined in private note loop."""

from __future__ import annotations

import numpy as np


class BeaconAvoidPolicy:
    """Turn toward a beacon unless an obstacle is close and in the way."""

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, observation: np.ndarray) -> np.ndarray:
        goal_bearing, _, obstacle_bearing, obstacle_proximity = np.asarray(
            observation, dtype=np.float32
        )
        # Positive error means steer toward the goal. Under threat, bias away
        # from the obstacle. Side semantics are intentionally exposed for the
        # live FlyGym sign check before interpreting trajectory outcomes.
        steer = float(np.clip(goal_bearing, -1.0, 1.0))
        if obstacle_proximity > 0.05 and abs(obstacle_bearing) < 0.65:
            # Turn decisively away as soon as the obstacle enters the forward
            # corridor; a gentle late correction is too slow at this speed.
            side = obstacle_bearing if abs(obstacle_bearing) > 0.015 else 0.1
            avoid = -np.sign(side)
            steer = 0.95 * avoid + 0.05 * steer
        base = 0.55
        turn = steer * 0.20
        return np.clip(np.array([base - turn, base + turn]), -1, 1)
