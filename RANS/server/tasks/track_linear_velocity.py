# Copyright (c) Space Robotics Lab, SnT, University of Luxembourg, SpaceR
# RANS: arXiv:2310.07393

"""
TrackLinearVelocity Task
========================
The spacecraft must maintain a randomly sampled target linear velocity (vx_t, vy_t).

Observation (6 values):
    [Δvx, Δvy, cos(θ), sin(θ), vx, vy]
    where Δv = v_target − v_current.

Reward:
    r = exp(-‖v_error‖² / (2·σ_v²))

Episode terminates when ‖v_error‖ < tolerance.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np

from .base import BaseTask


class TrackLinearVelocityTask(BaseTask):
    """Track a target 2-D linear velocity in the world frame."""

    _DEFAULTS: Dict[str, Any] = {
        "tolerance": 0.05,           # success threshold (m/s)
        "reward_sigma": 0.50,        # velocity reward width
        "max_target_speed": 1.00,    # maximum sampled target speed (m/s)
    }

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        super().__init__(config)
        cfg = {**self._DEFAULTS, **(config or {})}
        self.tolerance: float = cfg["tolerance"]
        self.reward_sigma: float = cfg["reward_sigma"]
        self.max_target_speed: float = cfg["max_target_speed"]

        self._target_vel = np.zeros(2, dtype=np.float64)

    # ------------------------------------------------------------------
    # BaseTask interface
    # ------------------------------------------------------------------

    def reset(self, spacecraft_state: np.ndarray) -> Dict[str, Any]:
        speed = np.random.uniform(0.0, self.max_target_speed)
        direction = np.random.uniform(0.0, 2.0 * math.pi)
        self._target_vel = np.array(
            [speed * math.cos(direction), speed * math.sin(direction)]
        )
        return {"target_linear_velocity": self._target_vel.tolist()}

    def get_observation(self, spacecraft_state: np.ndarray) -> np.ndarray:
        _, _, theta, vx, vy, _ = spacecraft_state
        dvx = self._target_vel[0] - vx
        dvy = self._target_vel[1] - vy
        return np.array(
            [dvx, dvy, math.cos(theta), math.sin(theta), vx, vy],
            dtype=np.float32,
        )

    def compute_reward(
        self, spacecraft_state: np.ndarray
    ) -> Tuple[float, bool, Dict[str, Any]]:
        vx, vy = spacecraft_state[3], spacecraft_state[4]
        vel_error = float(np.linalg.norm(self._target_vel - np.array([vx, vy])))
        reward = self._reward_exponential(vel_error, self.reward_sigma)
        goal_reached = vel_error < self.tolerance
        info = {
            "velocity_error_ms": vel_error,
            "goal_reached": goal_reached,
            "target_linear_velocity": self._target_vel.tolist(),
        }
        return reward, goal_reached, info

    @property
    def num_observations(self) -> int:
        return 6
