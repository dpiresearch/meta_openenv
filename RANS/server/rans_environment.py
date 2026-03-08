# Copyright (c) Space Robotics Lab, SnT, University of Luxembourg, SpaceR
# RANS: arXiv:2310.07393 — OpenEnv-compatible implementation

"""
RANSEnvironment
===============
OpenEnv ``Environment`` subclass that wraps the 2-D spacecraft simulator and
the RANS task suite.

Supported tasks (set via RANS_TASK env-var or constructor argument):
  • GoToPosition            — reach a target (x, y)
  • GoToPose                — reach a target (x, y, θ)
  • TrackLinearVelocity     — maintain (vx_t, vy_t)
  • TrackLinearAngularVelocity — maintain (vx_t, vy_t, ω_t)

The environment follows the RANS paper (arXiv:2310.07393) physics and reward
formulations, adapted to run in CPU-only Docker containers without Isaac Gym.
"""

from __future__ import annotations

import math
import os
import uuid
from typing import Any, Dict, Optional

import numpy as np

try:
    from openenv.core.env_server.interfaces import Action, Environment, Observation
except ImportError:
    from pydantic import BaseModel as Action  # type: ignore[assignment]
    from pydantic import BaseModel as Environment  # type: ignore[assignment]
    from pydantic import BaseModel as Observation  # type: ignore[assignment]

try:
    # Installed package import
    from rans_env.models import SpacecraftAction, SpacecraftObservation, SpacecraftState
    from rans_env.server.spacecraft_physics import Spacecraft2D, SpacecraftConfig
    from rans_env.server.tasks import TASK_REGISTRY
except ImportError:
    # Development / test import (package not yet installed, RANS dir on sys.path)
    import sys, os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.dirname(__file__)))
    from models import SpacecraftAction, SpacecraftObservation, SpacecraftState  # type: ignore[no-redef]
    from server.spacecraft_physics import Spacecraft2D, SpacecraftConfig  # type: ignore[no-redef]
    from server.tasks import TASK_REGISTRY  # type: ignore[no-redef]


class RANSEnvironment(Environment):
    """
    RANS spacecraft navigation environment for OpenEnv.

    References
    ----------
    El-Hariry, Richard, Olivares-Mendez (2023).
    "RANS: Highly-Parallelised Simulator for Reinforcement Learning based
    Autonomous Navigating Spacecrafts."  arXiv:2310.07393.
    """

    def __init__(
        self,
        task: str = "GoToPosition",
        spacecraft_config: Optional[SpacecraftConfig] = None,
        task_config: Optional[Dict[str, Any]] = None,
        max_episode_steps: int = 500,
        initial_pos_range: float = 2.0,
        initial_vel_range: float = 0.1,
    ) -> None:
        """
        Parameters
        ----------
        task:
            One of TASK_REGISTRY keys.  Overridden by RANS_TASK env-var.
        spacecraft_config:
            Physical platform configuration.  Uses 8-thruster MFP2D default.
        task_config:
            Dict of task hyper-parameters forwarded to the task constructor.
        max_episode_steps:
            Hard step limit per episode (overrides RANS_MAX_STEPS env-var).
        initial_pos_range:
            Half-width of the uniform distribution for random initial position.
        initial_vel_range:
            Half-width for random initial velocities.
        """
        # Allow env-var overrides (useful for Docker deployments)
        task = os.environ.get("RANS_TASK", task)
        max_episode_steps = int(
            os.environ.get("RANS_MAX_STEPS", str(max_episode_steps))
        )

        if task not in TASK_REGISTRY:
            raise ValueError(
                f"Unknown task '{task}'. "
                f"Available: {sorted(TASK_REGISTRY.keys())}"
            )

        self._task_name = task
        self._max_episode_steps = max_episode_steps
        self._initial_pos_range = initial_pos_range
        self._initial_vel_range = initial_vel_range

        # Physics simulator — RANS_NUM_THRUSTERS overrides spacecraft_config
        if spacecraft_config is None:
            n_env = os.environ.get("RANS_NUM_THRUSTERS")
            if n_env is not None:
                n = int(n_env)
                spacecraft_config = SpacecraftConfig.from_num_thrusters(n)
            else:
                spacecraft_config = SpacecraftConfig.default_8_thruster()
        self._spacecraft = Spacecraft2D(spacecraft_config)

        # Task
        self._task = TASK_REGISTRY[task](task_config or {})

        # Episode bookkeeping
        self._step_count: int = 0
        self._total_reward: float = 0.0
        self._ep_state = SpacecraftState(task=self._task_name)

    # ------------------------------------------------------------------
    # OpenEnv Environment interface
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Start a new episode with a randomised initial spacecraft state."""
        init_state = self._sample_initial_state()
        self._spacecraft.reset(init_state)

        task_info = self._task.reset(self._spacecraft.state)

        self._step_count = 0
        self._total_reward = 0.0
        self._ep_state = SpacecraftState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            task=self._task_name,
            **self._physical_state_dict(),
        )

        return self._make_observation(reward=0.0, done=False, info=task_info)

    def step(self, action: Action) -> Observation:
        """
        Advance the simulation by one step.

        Control mode is selected automatically based on which fields are set:

        1. **Thrusters** (``action.thrusters`` is not None):
           List of per-thruster activations ∈ [0, 1].
        2. **Force/torque** (``action.fx``, ``action.fy``, or ``action.torque``
           is not None):
           Direct world-frame force/torque — bypasses thruster geometry.
        3. **Velocity target** (``action.vx_target``, ``action.vy_target``, or
           ``action.omega_target`` is not None):
           PD controller drives the spacecraft toward the requested velocities.
        """
        if not hasattr(action, "thrusters"):
            raise ValueError(
                f"Expected SpacecraftAction, received {type(action).__name__}."
            )

        # ── Mode 1: thruster activations ─────────────────────────────────
        if action.thrusters is not None:
            activations = np.array(action.thrusters, dtype=np.float64)
            n = self._spacecraft.n_thrusters
            if len(activations) != n:
                padded = np.zeros(n, dtype=np.float64)
                padded[: min(len(activations), n)] = activations[:n]
                activations = padded
            self._spacecraft.step(activations)

        # ── Mode 2: direct force / torque ────────────────────────────────
        elif any(v is not None for v in [action.fx, action.fy, action.torque]):
            self._spacecraft.step_force_torque(
                fx_world=float(action.fx or 0.0),
                fy_world=float(action.fy or 0.0),
                torque=float(action.torque or 0.0),
            )

        # ── Mode 3: velocity target ───────────────────────────────────────
        elif any(
            v is not None
            for v in [action.vx_target, action.vy_target, action.omega_target]
        ):
            self._spacecraft.step_velocity_target(
                vx_target=float(action.vx_target or 0.0),
                vy_target=float(action.vy_target or 0.0),
                omega_target=float(action.omega_target or 0.0),
            )

        # ── No action — advance with zero forces ──────────────────────────
        else:
            self._spacecraft.step(np.zeros(self._spacecraft.n_thrusters))
        self._step_count += 1

        # Compute task reward
        reward, goal_reached, info = self._task.compute_reward(
            self._spacecraft.state
        )
        self._total_reward += reward

        # Determine episode termination
        done = goal_reached or (self._step_count >= self._max_episode_steps)

        # Rebuild persistent state (Pydantic models are immutable by default)
        self._ep_state = SpacecraftState(
            episode_id=self._ep_state.episode_id,
            step_count=self._step_count,
            task=self._task_name,
            total_reward=self._total_reward,
            goal_reached=goal_reached,
            **self._physical_state_dict(),
        )

        return self._make_observation(reward=reward, done=done, info=info)

    @property
    def state(self) -> SpacecraftState:
        return self._ep_state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sample_initial_state(self) -> np.ndarray:
        """Uniform random initial state (small velocities, random pose)."""
        r = self._initial_pos_range
        v = self._initial_vel_range
        return np.array(
            [
                np.random.uniform(-r, r),               # x
                np.random.uniform(-r, r),               # y
                np.random.uniform(-math.pi, math.pi),   # θ
                np.random.uniform(-v, v),               # vx
                np.random.uniform(-v, v),               # vy
                np.random.uniform(-v, v),               # ω
            ],
            dtype=np.float64,
        )

    def _physical_state_dict(self) -> Dict[str, float]:
        s = self._spacecraft.state
        return {
            "x": float(s[0]),
            "y": float(s[1]),
            "heading_rad": float(s[2]),
            "vx": float(s[3]),
            "vy": float(s[4]),
            "angular_velocity_rads": float(s[5]),
        }

    def _make_observation(
        self, reward: float, done: bool, info: Dict[str, Any]
    ) -> SpacecraftObservation:
        task_obs = self._task.get_observation(self._spacecraft.state)
        return SpacecraftObservation(
            state_obs=task_obs.tolist(),
            thruster_transforms=self._spacecraft.get_thruster_transforms().tolist(),
            thruster_masks=self._spacecraft.get_thruster_masks().tolist(),
            mass=self._spacecraft.config.mass,
            inertia=self._spacecraft.config.inertia,
            task=self._task_name,
            reward=float(reward),
            done=bool(done),
            info={**info, "step": self._step_count},
        )
