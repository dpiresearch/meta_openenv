# Copyright (c) Space Robotics Lab, SnT, University of Luxembourg, SpaceR
# RANS: Reinforcement Learning based Autonomous Navigation for Spacecrafts
# arXiv:2310.07393 — El-Hariry, Richard, Olivares-Mendez
#
# OpenEnv-compatible implementation

"""Base class for RANS spacecraft navigation tasks."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np


class BaseTask(ABC):
    """
    Abstract base class for all RANS spacecraft navigation tasks.

    Subclasses define:
      - The task-specific observation vector
      - The reward function (matching the RANS paper's formulations)
      - Target generation and episode reset logic
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config: Dict[str, Any] = config or {}
        self._target: Any = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def reset(self, spacecraft_state: np.ndarray) -> Dict[str, Any]:
        """
        Sample a new target and reset internal episode state.

        Args:
            spacecraft_state: Current state vector [x, y, θ, vx, vy, ω].

        Returns:
            Dictionary with task metadata (target values, etc.).
        """

    @abstractmethod
    def get_observation(self, spacecraft_state: np.ndarray) -> np.ndarray:
        """
        Compute the task-specific observation vector from the spacecraft state.

        Args:
            spacecraft_state: Current state [x, y, θ, vx, vy, ω].

        Returns:
            1-D float32 array of length ``num_observations``.
        """

    @abstractmethod
    def compute_reward(
        self, spacecraft_state: np.ndarray
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Compute reward, done flag, and diagnostic info.

        Args:
            spacecraft_state: Current state [x, y, θ, vx, vy, ω].

        Returns:
            (reward, done, info) tuple.
        """

    # ------------------------------------------------------------------
    # Common helpers
    # ------------------------------------------------------------------

    @property
    def num_observations(self) -> int:
        """Size of the task-specific state observation vector."""
        return 0

    @property
    def name(self) -> str:
        return self.__class__.__name__

    # ------------------------------------------------------------------
    # Shared reward primitives  (from RANS paper Sec. IV-B)
    # ------------------------------------------------------------------

    @staticmethod
    def _reward_exponential(error: float, sigma: float) -> float:
        """exp(-error² / (2·σ²))  — tight peak near zero."""
        return math.exp(-(error**2) / max(2.0 * sigma**2, 1e-9))

    @staticmethod
    def _reward_inverse(error: float) -> float:
        """1 / (1 + error)  — smooth monotone decay."""
        return 1.0 / (1.0 + error)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap angle to (−π, π]."""
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _world_to_body(dx: float, dy: float, theta: float) -> Tuple[float, float]:
        """Rotate world-frame displacement into body frame."""
        c, s = math.cos(theta), math.sin(theta)
        return c * dx + s * dy, -s * dx + c * dy
