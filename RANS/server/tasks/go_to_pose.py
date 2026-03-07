# Copyright (c) Space Robotics Lab, SnT, University of Luxembourg, SpaceR
# RANS: arXiv:2310.07393

"""
GoToPose Task
=============
The spacecraft must reach a target position AND heading (x_t, y_t, θ_t).

Observation (7 values):
    [Δx_body, Δy_body, cos(Δθ), sin(Δθ), vx, vy, ω]

Reward:
    r = w_p · exp(-‖p_error‖² / (2·σ_p²))
      + w_h · exp(-|heading_error|² / (2·σ_h²))

Episode terminates when ‖p_error‖ < tol_p AND |heading_error| < tol_h.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np

from .base import BaseTask


class GoToPoseTask(BaseTask):
    """Navigate spacecraft to a target 2-D pose (position + heading)."""

    _DEFAULTS: Dict[str, Any] = {
        "tolerance_pos": 0.10,          # position success threshold (m)
        "tolerance_heading": 0.10,      # heading success threshold (rad)
        "reward_sigma_pos": 1.00,       # position reward width
        "reward_sigma_heading": 0.50,   # heading reward width
        "position_weight": 0.70,        # w_p
        "heading_weight": 0.30,         # w_h
        "spawn_min_radius": 0.50,
        "spawn_max_radius": 3.00,
    }

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        super().__init__(config)
        cfg = {**self._DEFAULTS, **(config or {})}
        self.tolerance_pos: float = cfg["tolerance_pos"]
        self.tolerance_heading: float = cfg["tolerance_heading"]
        self.reward_sigma_pos: float = cfg["reward_sigma_pos"]
        self.reward_sigma_heading: float = cfg["reward_sigma_heading"]
        self.position_weight: float = cfg["position_weight"]
        self.heading_weight: float = cfg["heading_weight"]
        self.spawn_min_radius: float = cfg["spawn_min_radius"]
        self.spawn_max_radius: float = cfg["spawn_max_radius"]

        self._target_pos = np.zeros(2, dtype=np.float64)
        self._target_heading: float = 0.0

    # ------------------------------------------------------------------
    # BaseTask interface
    # ------------------------------------------------------------------

    def reset(self, spacecraft_state: np.ndarray) -> Dict[str, Any]:
        r = np.random.uniform(self.spawn_min_radius, self.spawn_max_radius)
        angle = np.random.uniform(0.0, 2.0 * math.pi)
        self._target_pos = np.array([r * math.cos(angle), r * math.sin(angle)])
        self._target_heading = np.random.uniform(-math.pi, math.pi)
        return {
            "target_position": self._target_pos.tolist(),
            "target_heading_rad": self._target_heading,
        }

    def get_observation(self, spacecraft_state: np.ndarray) -> np.ndarray:
        x, y, theta, vx, vy, omega = spacecraft_state
        dx, dy = self._target_pos[0] - x, self._target_pos[1] - y
        dx_b, dy_b = self._world_to_body(dx, dy, theta)
        d_theta = self._wrap_angle(self._target_heading - theta)
        return np.array(
            [dx_b, dy_b, math.cos(d_theta), math.sin(d_theta), vx, vy, omega],
            dtype=np.float32,
        )

    def compute_reward(
        self, spacecraft_state: np.ndarray
    ) -> Tuple[float, bool, Dict[str, Any]]:
        x, y, theta = spacecraft_state[0], spacecraft_state[1], spacecraft_state[2]
        pos_error = float(np.linalg.norm(self._target_pos - np.array([x, y])))
        heading_error = abs(self._wrap_angle(self._target_heading - theta))

        r_pos = self._reward_exponential(pos_error, self.reward_sigma_pos)
        r_head = self._reward_exponential(heading_error, self.reward_sigma_heading)
        reward = self.position_weight * r_pos + self.heading_weight * r_head

        goal_reached = (
            pos_error < self.tolerance_pos and heading_error < self.tolerance_heading
        )
        info = {
            "position_error_m": pos_error,
            "heading_error_rad": heading_error,
            "goal_reached": goal_reached,
        }
        return reward, goal_reached, info

    @property
    def num_observations(self) -> int:
        return 7
