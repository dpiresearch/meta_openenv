# Copyright (c) Space Robotics Lab, SnT, University of Luxembourg, SpaceR
# RANS: arXiv:2310.07393

"""
GoToPosition Task
=================
The spacecraft must reach a randomly sampled target position (x_t, y_t).

Observation (6 values):
    [Δx_body, Δy_body, cos(θ), sin(θ), vx, vy]
    where Δ values are relative-to-target in the spacecraft's body frame.

Reward (RANS paper, exponential mode):
    r = exp(-‖p_error‖² / (2·σ_p²))

Episode terminates when ‖p_error‖ < tolerance  OR  step limit reached.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np

from .base import BaseTask


class GoToPositionTask(BaseTask):
    """Navigate spacecraft to a target 2-D position."""

    # Default hyper-parameters (can be overridden via config dict)
    _DEFAULTS: Dict[str, Any] = {
        "tolerance": 0.10,          # success threshold (m)
        "reward_sigma": 1.00,       # width of Gaussian reward
        "reward_mode": "exponential",  # "exponential" | "inverse"
        "spawn_min_radius": 0.50,   # minimum target distance from origin (m)
        "spawn_max_radius": 3.00,   # maximum target distance from origin (m)
    }

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        super().__init__(config)
        cfg = {**self._DEFAULTS, **(config or {})}
        self.tolerance: float = cfg["tolerance"]
        self.reward_sigma: float = cfg["reward_sigma"]
        self.reward_mode: str = cfg["reward_mode"]
        self.spawn_min_radius: float = cfg["spawn_min_radius"]
        self.spawn_max_radius: float = cfg["spawn_max_radius"]

        self._target = np.zeros(2, dtype=np.float64)

    # ------------------------------------------------------------------
    # BaseTask interface
    # ------------------------------------------------------------------

    def reset(self, spacecraft_state: np.ndarray) -> Dict[str, Any]:
        r = np.random.uniform(self.spawn_min_radius, self.spawn_max_radius)
        angle = np.random.uniform(0.0, 2.0 * math.pi)
        self._target = np.array([r * math.cos(angle), r * math.sin(angle)])
        return {"target_position": self._target.tolist()}

    def get_observation(self, spacecraft_state: np.ndarray) -> np.ndarray:
        x, y, theta, vx, vy, _ = spacecraft_state
        dx, dy = self._target[0] - x, self._target[1] - y
        dx_b, dy_b = self._world_to_body(dx, dy, theta)
        return np.array(
            [dx_b, dy_b, math.cos(theta), math.sin(theta), vx, vy],
            dtype=np.float32,
        )

    def compute_reward(
        self, spacecraft_state: np.ndarray
    ) -> Tuple[float, bool, Dict[str, Any]]:
        x, y = spacecraft_state[0], spacecraft_state[1]
        pos_error = float(np.linalg.norm(self._target - np.array([x, y])))

        if self.reward_mode == "exponential":
            reward = self._reward_exponential(pos_error, self.reward_sigma)
        else:
            reward = self._reward_inverse(pos_error)

        goal_reached = pos_error < self.tolerance
        info = {
            "position_error_m": pos_error,
            "goal_reached": goal_reached,
            "target_position": self._target.tolist(),
        }
        return reward, goal_reached, info

    @property
    def num_observations(self) -> int:
        return 6
