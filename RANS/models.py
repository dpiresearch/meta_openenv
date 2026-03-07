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

from typing import Any, Dict, List

try:
    from openenv.core.env_server.interfaces import Action, Observation, State
except ImportError:
    # Fallback for standalone development / testing without openenv-core
    from pydantic import BaseModel as Action  # type: ignore[assignment]
    from pydantic import BaseModel as Observation  # type: ignore[assignment]
    from pydantic import BaseModel as State  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class SpacecraftAction(Action):
    """
    Control action for the RANS spacecraft.

    ``thrusters`` is a list of activations, one per thruster, each in [0, 1].
    For binary (on/off) control pass values of 0.0 or 1.0.
    The list length should match the thruster count of the configured platform
    (8 for the default MFP2D layout).

    Example (8-thruster, fire thruster 0 only)::

        SpacecraftAction(thrusters=[1, 0, 0, 0, 0, 0, 0, 0])
    """

    thrusters: List[float]


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
