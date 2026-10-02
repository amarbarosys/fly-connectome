"""Seeded planar task geometry and deterministic episode scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    success: bool
    collision: bool
    steps: int
    final_distance: float


class BeaconArena:
    """Kinematic task harness for policy evaluation, not fly physics."""

    def __init__(self, seed: int, max_steps: int = 600):
        self.seed = seed
        self.max_steps = max_steps
        rng = np.random.default_rng(seed)
        self.position = np.zeros(2, dtype=np.float32)
        self.heading = 0.0
        self.goal = np.array([2.0, 0.0], dtype=np.float32)
        self.obstacle = np.array([0.8, rng.uniform(-0.16, 0.16)], dtype=np.float32)
        self.step_count = 0
        self.collision = False

    def observe(self) -> np.ndarray:
        goal_vec = self.goal - self.position
        obstacle_vec = self.obstacle - self.position
        goal_bearing = _wrap(np.arctan2(goal_vec[1], goal_vec[0]) - self.heading)
        obstacle_bearing = _wrap(
            np.arctan2(obstacle_vec[1], obstacle_vec[0]) - self.heading
        )
        goal_distance = np.linalg.norm(goal_vec)
        obstacle_distance = np.linalg.norm(obstacle_vec)
        proximity = np.clip((0.45 - obstacle_distance) / 0.35, 0, 1)
        return np.array(
            [goal_bearing / np.pi, np.clip(goal_distance / 2.0, 0, 1),
             obstacle_bearing / np.pi, proximity], dtype=np.float32
        )

    def step(self, action: np.ndarray) -> bool:
        left, right = np.clip(np.asarray(action), -1, 1)
        self.heading = _wrap(self.heading + float(right - left) * 0.08)
        speed = max(0.0, float(left + right) * 0.025)
        self.position += speed * np.array(
            [np.cos(self.heading), np.sin(self.heading)], dtype=np.float32
        )
        self.step_count += 1
        self.collision = np.linalg.norm(self.position - self.obstacle) < 0.12
        reached = np.linalg.norm(self.position - self.goal) < 0.12
        return bool(reached or self.collision or self.step_count >= self.max_steps)

    def result(self) -> EpisodeResult:
        return EpisodeResult(
            seed=self.seed,
            success=bool(np.linalg.norm(self.position - self.goal) < 0.12),
            collision=bool(self.collision),
            steps=self.step_count,
            final_distance=float(np.linalg.norm(self.position - self.goal)),
        )


def _wrap(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)
