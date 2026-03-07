# Copyright (c) Space Robotics Lab, SnT, University of Luxembourg, SpaceR

"""
Tests for the 2-D spacecraft physics simulation.

Run with:  pytest server/tests/test_physics.py -v
"""

import math

import numpy as np
import pytest
import sys
import os

# Allow import from the parent package directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from server.spacecraft_physics import Spacecraft2D, SpacecraftConfig, ThrusterConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def spacecraft():
    return Spacecraft2D(SpacecraftConfig.default_8_thruster())


@pytest.fixture
def minimal_spacecraft():
    """Single +X thruster for deterministic tests."""
    cfg = SpacecraftConfig(
        mass=10.0,
        inertia=1.0,
        dt=0.1,
        thrusters=[
            ThrusterConfig(position=np.array([0.0, 0.0]), direction=np.array([1.0, 0.0]), force_max=10.0)
        ],
    )
    return Spacecraft2D(cfg)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInitialisation:
    def test_reset_zeros(self, spacecraft):
        s = spacecraft.reset()
        assert s.shape == (6,)
        np.testing.assert_allclose(s, 0.0, atol=1e-9)

    def test_reset_given_state(self, spacecraft):
        target = np.array([1.0, 2.0, 0.5, 0.1, -0.1, 0.05])
        s = spacecraft.reset(target)
        np.testing.assert_allclose(s, target, atol=1e-9)

    def test_heading_wraps_on_reset(self, spacecraft):
        # Heading outside (−π, π] should be wrapped
        s = spacecraft.reset(np.array([0, 0, 4.0, 0, 0, 0]))
        assert -math.pi < s[2] <= math.pi

    def test_thruster_count_8(self):
        cfg = SpacecraftConfig.default_8_thruster()
        sc = Spacecraft2D(cfg)
        assert sc.n_thrusters == 8

    def test_thruster_count_4(self):
        cfg = SpacecraftConfig.default_4_thruster()
        sc = Spacecraft2D(cfg)
        assert sc.n_thrusters == 4


# ---------------------------------------------------------------------------
# Dynamics — linear acceleration
# ---------------------------------------------------------------------------

class TestLinearDynamics:
    def test_zero_thrust_no_motion(self, spacecraft):
        spacecraft.reset()
        s = spacecraft.step(np.zeros(spacecraft.n_thrusters))
        np.testing.assert_allclose(s[:2], 0.0, atol=1e-9)   # no position change
        np.testing.assert_allclose(s[3:5], 0.0, atol=1e-9)  # no velocity change

    def test_single_thruster_acceleration(self, minimal_spacecraft):
        """Thruster force 10 N, mass 10 kg → acceleration 1 m/s²."""
        minimal_spacecraft.reset()
        dt = minimal_spacecraft.config.dt  # 0.1 s
        s = minimal_spacecraft.step(np.array([1.0]))
        # Expected vx after 1 step: a*dt = 1.0 * 0.1 = 0.1 m/s
        assert abs(s[3] - 1.0 * dt) < 1e-9

    def test_linear_velocity_accumulates(self, minimal_spacecraft):
        """Constant thrust → linearly increasing velocity."""
        minimal_spacecraft.reset()
        dt = minimal_spacecraft.config.dt
        a_expected = 1.0  # 10 N / 10 kg
        for step in range(1, 6):
            s = minimal_spacecraft.step(np.array([1.0]))
            assert abs(s[3] - a_expected * step * dt) < 1e-6

    def test_heading_rotates_force(self):
        """With heading = π/2, +X body force should produce +Y world force."""
        cfg = SpacecraftConfig(
            mass=1.0, inertia=1.0, dt=1.0,
            thrusters=[ThrusterConfig(
                position=np.array([0.0, 0.0]),
                direction=np.array([1.0, 0.0]),
                force_max=1.0,
            )],
        )
        sc = Spacecraft2D(cfg)
        sc.reset(np.array([0, 0, math.pi / 2, 0, 0, 0]))
        s = sc.step(np.array([1.0]))
        # After 1 s: ay ≈ 1.0 m/s² (world +Y), ax ≈ 0
        assert abs(s[4] - 1.0) < 1e-9  # vy
        assert abs(s[3]) < 1e-9        # vx


# ---------------------------------------------------------------------------
# Dynamics — angular acceleration
# ---------------------------------------------------------------------------

class TestAngularDynamics:
    def test_zero_torque(self, spacecraft):
        spacecraft.reset()
        s0 = spacecraft.state
        # Fire only translational thrusters (indices 0–3, assumed zero torque if
        # position is on the axis of thrust)
        spacecraft.step(np.zeros(spacecraft.n_thrusters))
        assert abs(spacecraft.angular_velocity) < 1e-9

    def test_pure_torque_thruster(self):
        """Thruster at [0, r], force in +X: torque = px*dy − py*dx = 0 − r*1 = −r."""
        r = 1.0
        cfg = SpacecraftConfig(
            mass=1.0, inertia=1.0, dt=1.0,
            thrusters=[ThrusterConfig(
                position=np.array([0.0, r]),
                direction=np.array([1.0, 0.0]),
                force_max=1.0,
            )],
        )
        sc = Spacecraft2D(cfg)
        sc.reset()
        s = sc.step(np.array([1.0]))
        # Expected angular velocity after 1 s: α = torque/I = (0*0 − r*1)/1 = −r = −1.0
        assert abs(s[5] - (-r)) < 1e-9

    def test_heading_wraps(self, spacecraft):
        """Heading must stay in (−π, π] after many steps."""
        spacecraft.reset(np.array([0, 0, math.pi - 0.01, 0, 0, 1.0]))
        for _ in range(200):
            spacecraft.step(np.zeros(spacecraft.n_thrusters))
        assert -math.pi < spacecraft.heading <= math.pi


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

class TestObservationHelpers:
    def test_thruster_transforms_shape(self, spacecraft):
        T = spacecraft.get_thruster_transforms()
        assert T.shape == (spacecraft.n_thrusters, 5)
        assert T.dtype == np.float32

    def test_thruster_masks_all_ones(self, spacecraft):
        M = spacecraft.get_thruster_masks()
        assert M.shape == (spacecraft.n_thrusters,)
        np.testing.assert_allclose(M, 1.0)

    def test_transforms_direction_normalised(self, spacecraft):
        T = spacecraft.get_thruster_transforms()
        # Columns 2:4 are direction (dx, dy): each should be unit vector
        norms = np.linalg.norm(T[:, 2:4], axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_state_property(self, spacecraft):
        spacecraft.reset(np.array([1, 2, 0.5, 0.1, -0.1, 0.05]))
        s = spacecraft.state
        assert len(s) == 6

    def test_position_property(self, spacecraft):
        spacecraft.reset(np.array([3.0, 4.0, 0, 0, 0, 0]))
        np.testing.assert_allclose(spacecraft.position, [3.0, 4.0])

    def test_state_is_copy(self, spacecraft):
        spacecraft.reset()
        s = spacecraft.state
        s[0] = 999.0
        assert spacecraft.state[0] != 999.0  # internal state unaffected
