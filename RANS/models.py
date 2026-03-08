# Copyright (c) Space Robotics Lab, SnT, University of Luxembourg, SpaceR
# RANS: arXiv:2310.07393 — OpenEnv-compatible implementation

"""
models.py
---------
Action, Observation, and State dataclasses for the RANS spacecraft environment.

These follow the OpenEnv conventions (openenv-core):
  Action      — sent by the RL agent / client to the server
  Observation — returned by the server after reset() / step()
  State       — persistent episode metadata readable via /state
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

try:
    from openenv.core.env_server.interfaces import Action, Observation, State
except ImportError:
    # Fallback for standalone development / testing without openenv-core
    from pydantic import BaseModel as Action  # type: ignore[assignment]
    from pydantic import BaseModel as Observation  # type: ignore[assignment]
    from pydantic import BaseModel as State  # type: ignore[assignment]

from pydantic import field_validator


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class SpacecraftAction(Action):
    """
    Control action for the RANS spacecraft.

    Three mutually-exclusive control modes are supported.  The environment
    picks whichever mode has non-None fields (priority: thrusters > force/torque
    > velocity target).

    **Mode 1 — Thruster activations (default)**
        ``thrusters``: list of N floats, each in [0, 1].  Length must match the
        platform's thruster count (8 for the default MFP2D layout).
        Accepts a comma-separated string from the web UI form.
        Example::

            SpacecraftAction(thrusters=[1, 0, 0, 0, 0, 0, 0, 0])

    **Mode 2 — Direct world-frame force / torque**
        ``fx``, ``fy``: force components in N (world frame, any sign).
        ``torque``: yaw torque in N·m (positive = CCW).
        Bypasses thruster geometry entirely — useful for high-level control
        or when you don't care about actuator layout.
        Example::

            SpacecraftAction(fx=2.0, fy=0.0, torque=0.5)

    **Mode 3 — Target velocity (PD controller)**
        ``vx_target``, ``vy_target``: desired world-frame linear velocities (m/s).
        ``omega_target``: desired yaw rate (rad/s).
        The environment applies a proportional controller each step to drive
        the spacecraft toward the requested velocities.
        Example::

            SpacecraftAction(vx_target=0.5, vy_target=0.0, omega_target=0.0)
    """

    # ── Mode 1: thruster activations ─────────────────────────────────────
    thrusters: Optional[List[float]] = None

    # ── Mode 2: direct world-frame force / torque ────────────────────────
    fx: Optional[float] = None      # N
    fy: Optional[float] = None      # N
    torque: Optional[float] = None  # N·m

    # ── Mode 3: velocity targets (PD controller) ─────────────────────────
    vx_target: Optional[float] = None    # m/s
    vy_target: Optional[float] = None    # m/s
    omega_target: Optional[float] = None  # rad/s

    @field_validator("thrusters", mode="before")
    @classmethod
    def _coerce_thrusters(cls, v: Any) -> Optional[List[float]]:
        """Accept JSON-array string, comma-separated string, or None."""
        if v is None:
            return None
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return None
            if v.startswith("["):
                try:
                    parsed = json.loads(v)
                    return parsed if parsed else None
                except json.JSONDecodeError:
                    pass
            # Comma-separated: "0.5,0.5,..."
            parsed = [float(x.strip()) for x in v.split(",") if x.strip()]
            return parsed if parsed else None
        return v


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class SpacecraftObservation(Observation):
    """
    Full observation returned after each ``reset()`` / ``step()``.

    Fields
    ------
    state_obs : List[float]
        Task-specific state vector (6–8 floats depending on task).
        Content varies per task — see individual task docstrings.

    thruster_transforms : List[List[float]]
        Shape [n_thrusters × 5].  Each row: [px, py, dx, dy, force_max].
        Encodes the physical layout of thrusters on the platform.

    thruster_masks : List[float]
        Binary mask [n_thrusters].  1.0 = thruster slot is occupied.

    mass : float
        Platform mass in kg.

    inertia : float
        Moment of inertia about the yaw axis (kg·m²).

    task : str
        Active task name, e.g. "GoToPosition".

    reward : float
        Scalar reward for the most recent step (0.0 after reset).

    done : bool
        True when the episode has ended (goal reached or step limit).

    info : Dict[str, Any]
        Task-specific diagnostics, e.g. position_error, goal_reached.
    """

    state_obs: List[float] = []
    thruster_transforms: List[List[float]] = []
    thruster_masks: List[float] = []
    mass: float = 10.0
    inertia: float = 0.50
    task: str = "GoToPosition"
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class SpacecraftState(State):
    """
    Persistent episode state (accessible via GET /state).

    Tracks the spacecraft's physical state and current task configuration
    so that observers (dashboards, loggers) can monitor the episode without
    participating in the step loop.

    Note: ``episode_id`` and ``step_count`` are inherited from
    ``openenv.core.env_server.interfaces.State`` when openenv-core is
    installed.  They are also declared here explicitly so the class works
    as a standalone Pydantic model without openenv-core.
    """

    # Fields also present in the openenv-core State base class
    episode_id: str = ""
    step_count: int = 0

    task: str = "GoToPosition"
    # Physical state
    x: float = 0.0
    y: float = 0.0
    heading_rad: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    angular_velocity_rads: float = 0.0
    # Episode metadata
    total_reward: float = 0.0
    goal_reached: bool = False
