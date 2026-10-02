"""Task-action to FlyGym low-level locomotion adapter."""

from __future__ import annotations

import numpy as np


def make_turning_controller(sim, fly):
    """Construct FlyGym's existing locomotion controller for a running sim."""
    from flygym_demo.complex_terrain import (
        HybridTurningController,
        PreprogrammedSteps,
    )

    steps = PreprogrammedSteps()
    return HybridTurningController(
        timestep=sim.timestep,
        preprogrammed_steps=steps,
        output_dof_order=fly.get_actuated_jointdofs_order("position"),
    )


def action_to_descending_signal(action: np.ndarray) -> np.ndarray:
    """Map normalized left/right drive to the tutorial's descending signal."""
    signal = np.asarray(action, dtype=np.float32)
    if signal.shape != (2,):
        raise ValueError("action must contain [left_drive, right_drive]")
    return np.clip(signal, -1, 1)
