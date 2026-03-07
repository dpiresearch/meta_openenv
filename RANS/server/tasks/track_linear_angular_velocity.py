# Copyright (c) Space Robotics Lab, SnT, University of Luxembourg, SpaceR
# RANS: arXiv:2310.07393

"""
TrackLinearAngularVelocity Task
================================
The spacecraft must simultaneously track a target linear velocity (vx_t, vy_t)
AND a target angular velocity (ω_t).

Observation (8 values):
    [Δvx, Δvy, Δω, cos(θ), sin(θ), vx, vy, ω]

Reward:
    r = w_l · exp(-‖Δv‖² / (2·σ_l²))
      + w_a · exp(-|Δω|² / (2·σ_a²))

This matches the TrackLinearAngularVelocityTask in the RANS codebase
(omniisaacgymenvs/tasks/MFP2D/track_linear_angular_velocity.py).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np

from .base import BaseTask


class TrackLinearAngularVelocityTask(BaseTask):
    """Track target linear AND angular velocity simultaneously."""

    _DEFAULTS: Dict[str, Any] = {
        "tolerance_linear": 0.05,         # (m/s)
        "tolerance_angular": 0.10,        # (rad/s)
        "reward_sigma_linear": 0.50,
        "reward_sigma_angular": 0.50,
        "linear_weight": 0.70,            # w_l
        "angular_weight": 0.30,           # w_a
        "max_target_speed": 1.00,         # (m/s)
        "max_target_angular_speed": 1.00, # (rad/s)
    }

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        super().__init__(config)
        cfg = {**self._DEFAULTS, **(config or {})}
        self.tolerance_linear: float = cfg["tolerance_linear"]
        self.tolerance_angular: float = cfg["tolerance_angular"]
        self.reward_sigma_linear: float = cfg["reward_sigma_linear"]
        self.reward_sigma_angular: float = cfg["reward_sigma_angular"]
        self.linear_weight: float = cfg["linear_weight"]
        self.angular_weight: float = cfg["angular_weight"]
        self.max_target_speed: float = cfg["max_target_speed"]
        self.max_target_angular_speed: float = cfg["max_target_angular_speed"]

        self._target_linear_vel = np.zeros(2, dtype=np.float64)
        self._target_angular_vel: float = 0.0

    # ------------------------------------------------------------------
    # BaseTask interface
    # ------------------------------------------------------------------

    def reset(self, spacecraft_state: np.ndarray) -> Dict[str, Any]:
        speed = np.random.uniform(0.0, self.max_target_speed)
        direction = np.random.uniform(0.0, 2.0 * math.pi)
        self._target_linear_vel = np.array(
            [speed * math.cos(direction), speed * math.sin(direction)]
        )
        self._target_angular_vel = np.random.uniform(
            -self.max_target_angular_speed, self.max_target_angular_speed
        )
        return {
            "target_linear_velocity": self._target_linear_vel.tolist(),
            "target_angular_velocity": self._target_angular_vel,
        }

    def get_observation(self, spacecraft_state: np.ndarray) -> np.ndarray:
        _, _, theta, vx, vy, omega = spacecraft_state
        dvx = self._target_linear_vel[0] - vx
        dvy = self._target_linear_vel[1] - vy
        domega = self._target_angular_vel - omega
        return np.array(
            [dvx, dvy, domega, math.cos(theta), math.sin(theta), vx, vy, omega],
            dtype=np.float32,
        )

    def compute_reward(
        self, spacecraft_state: np.ndarray
    ) -> Tuple[float, bool, Dict[str, Any]]:
        vx, vy, omega = spacecraft_state[3], spacecraft_state[4], spacecraft_state[5]
        linear_error = float(
            np.linalg.norm(self._target_linear_vel - np.array([vx, vy]))
        )
        angular_error = abs(self._target_angular_vel - omega)

        r_linear = self._reward_exponential(linear_error, self.reward_sigma_linear)
        r_angular = self._reward_exponential(angular_error, self.reward_sigma_angular)
        reward = self.linear_weight * r_linear + self.angular_weight * r_angular

        goal_reached = (
            linear_error < self.tolerance_linear
            and angular_error < self.tolerance_angular
        )
        info = {
            "linear_velocity_error_ms": linear_error,
            "angular_velocity_error_rads": angular_error,
            "goal_reached": goal_reached,
        }
        return reward, goal_reached, info

    @property
    def num_observations(self) -> int:
        return 8
