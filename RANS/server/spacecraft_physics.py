# Copyright (c) Space Robotics Lab, SnT, University of Luxembourg, SpaceR
# RANS: arXiv:2310.07393

"""
2-D Spacecraft (Modular Floating Platform) Physics Simulation
=============================================================
Pure-NumPy implementation of the rigid-body dynamics described in the RANS
paper, Section III.  This lets the environment run inside Docker containers
without NVIDIA Isaac Gym / Isaac Sim.

State vector: [x, y, θ, vx, vy, ω]
  x, y   — world-frame position  (m)
  θ      — heading / yaw angle   (rad, wrapped to (−π, π])
  vx, vy — world-frame linear velocity   (m/s)
  ω      — angular velocity              (rad/s)

Action: activations ∈ [0, 1]^n_thrusters (continuous) or {0,1}^n (binary).

Dynamics:
  F_body = Σ_i  a_i · F_max_i · d̂_i          (body-frame force vector)
  F_world = R(θ) · F_body
  a_linear  = F_world / m                      (linear acceleration)
  τ_i = (p_i × F_max_i · d̂_i)_z = p_x·d_y − p_y·d_x
  α = Σ_i a_i · τ_i / I                        (angular acceleration)
  Integration: Euler with timestep dt
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Thruster configuration
# ---------------------------------------------------------------------------

@dataclass
class ThrusterConfig:
    """Configuration for a single thruster on the 2-D spacecraft."""

    position: np.ndarray   # [px, py] in body frame (m)
    direction: np.ndarray  # unit-vector [dx, dy] of applied force in body frame
    force_max: float = 1.0  # peak thrust (N)

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64)
        self.direction = np.asarray(self.direction, dtype=np.float64)
        # Normalise direction
        norm = np.linalg.norm(self.direction)
        if norm > 1e-9:
            self.direction = self.direction / norm


# ---------------------------------------------------------------------------
# Spacecraft configuration
# ---------------------------------------------------------------------------

@dataclass
class SpacecraftConfig:
    """
    Physical parameters and thruster layout for the 2-D spacecraft.

    Matches the MFP2D (Modular Floating Platform) configuration used in RANS.
    """

    mass: float = 10.0          # kg
    inertia: float = 0.50       # kg·m²
    dt: float = 0.02            # simulation timestep  (50 Hz)
    max_episode_steps: int = 500
    thrusters: List[ThrusterConfig] = field(default_factory=list)

    @classmethod
    def default_8_thruster(cls) -> "SpacecraftConfig":
        """
        Standard 8-thruster MFP2D layout from the RANS paper.

        Thrusters are arranged in four pairs:
          • Two pairs provide pure translational force (±X, ±Y in body frame)
          • Two pairs create coupled translational + rotational force (diagonal)

        This gives the platform full 3-DoF controllability (x, y, θ).
        """
        r = 0.31  # m — radial distance from CoM to thruster attachment point

        thrusters = [
            # ── Translational thrusters (body ±X) ──────────────────────────
            # Mounted on ±Y edges, thrust along +X body axis
            ThrusterConfig(position=[ 0.0,  r], direction=[1.0, 0.0]),
            ThrusterConfig(position=[ 0.0, -r], direction=[1.0, 0.0]),
            # Mounted on ±Y edges, thrust along −X body axis
            ThrusterConfig(position=[ 0.0,  r], direction=[-1.0, 0.0]),
            ThrusterConfig(position=[ 0.0, -r], direction=[-1.0, 0.0]),

            # ── Rotational / combined thrusters (diagonal) ──────────────────
            # CCW torque: thrusters at corners fire tangentially
            ThrusterConfig(
                position=[ r * 0.707,  r * 0.707],
                direction=[-0.707,  0.707],
            ),
            ThrusterConfig(
                position=[-r * 0.707, -r * 0.707],
                direction=[ 0.707, -0.707],
            ),
            # CW torque
            ThrusterConfig(
                position=[-r * 0.707,  r * 0.707],
                direction=[ 0.707,  0.707],
            ),
            ThrusterConfig(
                position=[ r * 0.707, -r * 0.707],
                direction=[-0.707, -0.707],
            ),
        ]
        return cls(thrusters=thrusters)

    @classmethod
    def default_4_thruster(cls) -> "SpacecraftConfig":
        """
        Minimal 4-thruster layout (under-actuated in rotation).
        Useful for simpler position-tracking experiments.
        """
        r = 0.31
        thrusters = [
            ThrusterConfig(position=[ 0.0,  r], direction=[1.0, 0.0]),   # +X
            ThrusterConfig(position=[ 0.0, -r], direction=[-1.0, 0.0]),  # −X
            ThrusterConfig(position=[ r,  0.0], direction=[0.0, 1.0]),   # +Y
            ThrusterConfig(position=[-r,  0.0], direction=[0.0, -1.0]),  # −Y
        ]
        return cls(thrusters=thrusters)

    @classmethod
    def from_num_thrusters(
        cls,
        n: int,
        radius: float = 0.31,
        force_max: float = 1.0,
    ) -> "SpacecraftConfig":
        """
        Generate a symmetric N-thruster layout around a circle of given radius.

        Thrusters are placed at angles ``2πi/N`` and fire tangentially,
        alternating CCW/CW so the platform retains 3-DoF controllability.

        Args:
            n: Number of thrusters (4 ≤ n ≤ 16, must be even).
            radius: Radial distance from CoM to each thruster (m).
            force_max: Peak thrust per thruster (N).
        """
        if n < 4 or n > 16 or n % 2 != 0:
            raise ValueError(
                f"n must be an even integer in [4, 16], got {n}."
            )
        thrusters = []
        for i in range(n):
            theta = 2.0 * math.pi * i / n
            px = radius * math.cos(theta)
            py = radius * math.sin(theta)
            # Alternate CCW / CW tangential firing direction
            sign = 1.0 if i % 2 == 0 else -1.0
            dx = -sign * math.sin(theta)
            dy =  sign * math.cos(theta)
            thrusters.append(
                ThrusterConfig(position=[px, py], direction=[dx, dy], force_max=force_max)
            )
        return cls(thrusters=thrusters)


# ---------------------------------------------------------------------------
# Spacecraft dynamics
# ---------------------------------------------------------------------------

class Spacecraft2D:
    """
    2-D spacecraft rigid-body simulator.

    State:  s = [x, y, θ, vx, vy, ω]  (float64)
    Action: a = [a_0, …, a_{N−1}]   ∈ [0, 1]  per thruster

    The simulation runs at ``config.dt`` seconds per step.
    """

    def __init__(self, config: Optional[SpacecraftConfig] = None) -> None:
        self.config = config or SpacecraftConfig.default_8_thruster()
        self.n_thrusters = len(self.config.thrusters)
        self._precompute_thruster_matrices()
        self._state = np.zeros(6, dtype=np.float64)

    # ------------------------------------------------------------------
    # Pre-computation
    # ------------------------------------------------------------------

    def _precompute_thruster_matrices(self) -> None:
        """
        Build static matrices for body-frame force and torque.

        force_body_matrix  [2 × N]:  col i = direction_i * force_max_i
        torque_vector      [N]:      element i = (p × d)_z * force_max_i
        """
        n = self.n_thrusters
        self._force_mat = np.zeros((2, n), dtype=np.float64)
        self._torque_vec = np.zeros(n, dtype=np.float64)

        for i, t in enumerate(self.config.thrusters):
            self._force_mat[:, i] = t.direction * t.force_max
            # Cross product z-component: px*dy − py*dx
            self._torque_vec[i] = (
                t.position[0] * t.direction[1] - t.position[1] * t.direction[0]
            ) * t.force_max

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Reset to a given state (or zeros).

        Args:
            state: [x, y, θ, vx, vy, ω], or None → zero state.

        Returns:
            Copy of the initial state.
        """
        if state is None:
            self._state = np.zeros(6, dtype=np.float64)
        else:
            self._state = np.asarray(state, dtype=np.float64).copy()
            self._state[2] = self._wrap_angle(self._state[2])
        return self._state.copy()

    def step(self, activations: np.ndarray) -> np.ndarray:
        """
        Advance simulation by one timestep.

        Args:
            activations: Thruster commands, shape [n_thrusters], clamped to [0,1].

        Returns:
            New state [x, y, θ, vx, vy, ω].
        """
        a = np.clip(activations, 0.0, 1.0)
        x, y, theta, vx, vy, omega = self._state
        dt = self.config.dt

        # Body-frame force vector (shape [2])
        F_body = self._force_mat @ a

        # Rotate to world frame
        c, s = math.cos(theta), math.sin(theta)
        ax = (c * F_body[0] - s * F_body[1]) / self.config.mass
        ay = (s * F_body[0] + c * F_body[1]) / self.config.mass

        # Angular acceleration
        alpha = (self._torque_vec @ a) / self.config.inertia

        # Euler integration (position uses mid-point correction)
        self._state[0] = x + vx * dt + 0.5 * ax * dt * dt
        self._state[1] = y + vy * dt + 0.5 * ay * dt * dt
        theta_new = theta + omega * dt + 0.5 * alpha * dt * dt
        self._state[2] = self._wrap_angle(theta_new)
        self._state[3] = vx + ax * dt
        self._state[4] = vy + ay * dt
        self._state[5] = omega + alpha * dt

        return self._state.copy()

    def step_force_torque(
        self,
        fx_world: float,
        fy_world: float,
        torque: float,
    ) -> np.ndarray:
        """
        Advance the simulation by applying world-frame forces and yaw torque
        directly, bypassing the thruster geometry.

        Args:
            fx_world: Force in world-frame X direction (N).
            fy_world: Force in world-frame Y direction (N).
            torque:   Yaw torque (N·m).  Positive = CCW.

        Returns:
            New state [x, y, θ, vx, vy, ω].
        """
        x, y, theta, vx, vy, omega = self._state
        dt = self.config.dt

        ax = fx_world / self.config.mass
        ay = fy_world / self.config.mass
        alpha = torque / self.config.inertia

        self._state[0] = x + vx * dt + 0.5 * ax * dt * dt
        self._state[1] = y + vy * dt + 0.5 * ay * dt * dt
        self._state[2] = self._wrap_angle(theta + omega * dt + 0.5 * alpha * dt * dt)
        self._state[3] = vx + ax * dt
        self._state[4] = vy + ay * dt
        self._state[5] = omega + alpha * dt

        return self._state.copy()

    def step_velocity_target(
        self,
        vx_target: float,
        vy_target: float,
        omega_target: float,
        kp: float = 5.0,
    ) -> np.ndarray:
        """
        Advance the simulation by driving toward target velocities via a
        proportional controller.

        The controller computes the required world-frame force / torque and
        clips them to the platform's physical limits before applying.

        Args:
            vx_target:    Desired world-frame X velocity (m/s).
            vy_target:    Desired world-frame Y velocity (m/s).
            omega_target: Desired yaw rate (rad/s).
            kp:           Proportional gain (default 5.0).

        Returns:
            New state [x, y, θ, vx, vy, ω].
        """
        _, _, _, vx, vy, omega = self._state

        fx = kp * self.config.mass * (vx_target - vx)
        fy = kp * self.config.mass * (vy_target - vy)
        torque = kp * self.config.inertia * (omega_target - omega)

        # Clip to physical limits
        f_max = float(np.sum(np.linalg.norm(self._force_mat, axis=0)))
        tau_max = float(np.sum(np.abs(self._torque_vec)))
        fx = float(np.clip(fx, -f_max, f_max))
        fy = float(np.clip(fy, -f_max, f_max))
        torque = float(np.clip(torque, -tau_max, tau_max))

        return self.step_force_torque(fx, fy, torque)

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def get_thruster_transforms(self) -> np.ndarray:
        """
        Return thruster layout as a float array for the observation.

        Returns:
            Shape [n_thrusters, 5]: each row = [px, py, dx, dy, force_max]
        """
        T = np.zeros((self.n_thrusters, 5), dtype=np.float32)
        for i, t in enumerate(self.config.thrusters):
            T[i, 0:2] = t.position
            T[i, 2:4] = t.direction
            T[i, 4] = t.force_max
        return T

    def get_thruster_masks(self) -> np.ndarray:
        """Binary mask: 1.0 for every valid thruster slot."""
        return np.ones(self.n_thrusters, dtype=np.float32)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> np.ndarray:
        return self._state.copy()

    @property
    def position(self) -> np.ndarray:
        return self._state[:2].copy()

    @property
    def heading(self) -> float:
        return float(self._state[2])

    @property
    def linear_velocity(self) -> np.ndarray:
        return self._state[3:5].copy()

    @property
    def angular_velocity(self) -> float:
        return float(self._state[5])

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi
