# Copyright (c) Space Robotics Lab, SnT, University of Luxembourg, SpaceR

"""
Tests for RANS task implementations.

Run with:  pytest server/tests/test_tasks.py -v
"""

import math
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from server.tasks import (
    GoToPositionTask,
    GoToPoseTask,
    TrackLinearVelocityTask,
    TrackLinearAngularVelocityTask,
    TASK_REGISTRY,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def zero_state():
    return np.zeros(6, dtype=np.float64)

def state(x=0, y=0, theta=0, vx=0, vy=0, omega=0):
    return np.array([x, y, theta, vx, vy, omega], dtype=np.float64)


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

class TestTaskRegistry:
    def test_all_tasks_registered(self):
        expected = {
            "GoToPosition",
            "GoToPose",
            "TrackLinearVelocity",
            "TrackLinearAngularVelocity",
        }
        assert expected == set(TASK_REGISTRY.keys())

    def test_registry_instantiates(self):
        for name, cls in TASK_REGISTRY.items():
            task = cls()
            task.reset(zero_state())  # should not raise


# ---------------------------------------------------------------------------
# GoToPositionTask
# ---------------------------------------------------------------------------

class TestGoToPositionTask:
    @pytest.fixture
    def task(self):
        return GoToPositionTask()

    def test_reset_returns_target(self, task):
        info = task.reset(zero_state())
        assert "target_position" in info
        assert len(info["target_position"]) == 2

    def test_observation_length(self, task):
        task.reset(zero_state())
        obs = task.get_observation(zero_state())
        assert obs.shape == (6,)
        assert task.num_observations == 6

    def test_reward_at_goal(self, task):
        task.reset(zero_state())
        # Move target to origin
        task._target = np.array([0.0, 0.0])
        reward, done, info = task.compute_reward(zero_state())
        # At target: error = 0 → reward ≈ 1.0
        assert reward > 0.99
        assert done  # position_error < tolerance

    def test_reward_far_from_goal(self, task):
        task.reset(zero_state())
        task._target = np.array([100.0, 0.0])
        reward, done, info = task.compute_reward(zero_state())
        assert reward < 0.01
        assert not done

    def test_reward_decreases_with_distance(self, task):
        task.reset(zero_state())
        rewards = []
        for dist in [0.1, 0.5, 1.0, 2.0]:
            task._target = np.array([dist, 0.0])
            r, _, _ = task.compute_reward(zero_state())
            rewards.append(r)
        # Reward should be monotonically decreasing
        for i in range(len(rewards) - 1):
            assert rewards[i] > rewards[i + 1]

    def test_body_frame_obs_at_origin(self, task):
        """With heading=0, body frame == world frame."""
        task._target = np.array([1.0, 0.0])
        obs = task.get_observation(state(x=0, y=0, theta=0))
        # dx_body ≈ 1.0, dy_body ≈ 0.0
        assert abs(obs[0] - 1.0) < 1e-6
        assert abs(obs[1]) < 1e-6

    def test_body_frame_obs_rotated(self, task):
        """With heading=π/2, world +X becomes body −Y."""
        task._target = np.array([1.0, 0.0])
        obs = task.get_observation(state(x=0, y=0, theta=math.pi / 2))
        # dx_body = cos(π/2)*1 + sin(π/2)*0 = 0
        # dy_body = -sin(π/2)*1 + cos(π/2)*0 = -1
        assert abs(obs[0]) < 1e-6
        assert abs(obs[1] - (-1.0)) < 1e-6

    def test_info_contains_position_error(self, task):
        task.reset(zero_state())
        task._target = np.array([1.0, 0.0])
        _, _, info = task.compute_reward(zero_state())
        assert "position_error_m" in info
        assert abs(info["position_error_m"] - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# GoToPoseTask
# ---------------------------------------------------------------------------

class TestGoToPoseTask:
    @pytest.fixture
    def task(self):
        return GoToPoseTask()

    def test_reset_returns_target(self, task):
        info = task.reset(zero_state())
        assert "target_position" in info
        assert "target_heading_rad" in info

    def test_observation_length(self, task):
        task.reset(zero_state())
        obs = task.get_observation(zero_state())
        assert obs.shape == (7,)
        assert task.num_observations == 7

    def test_reward_at_pose(self, task):
        task.reset(zero_state())
        task._target_pos = np.array([0.0, 0.0])
        task._target_heading = 0.0
        reward, done, info = task.compute_reward(state(x=0, y=0, theta=0))
        assert reward > 0.95  # both position and heading at goal
        assert done

    def test_heading_error_wraps(self, task):
        """Heading error should be symmetric around 0."""
        task._target_pos = np.array([0.0, 0.0])
        task._target_heading = math.pi - 0.01
        # State with heading = −π + 0.01 → error should be ~0.02, not ~2π
        s = state(theta=-math.pi + 0.01)
        _, _, info = task.compute_reward(s)
        assert info["heading_error_rad"] < 0.1

    def test_weights_sum_to_one(self, task):
        assert abs(task.position_weight + task.heading_weight - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# TrackLinearVelocityTask
# ---------------------------------------------------------------------------

class TestTrackLinearVelocityTask:
    @pytest.fixture
    def task(self):
        return TrackLinearVelocityTask()

    def test_reset_returns_target(self, task):
        info = task.reset(zero_state())
        assert "target_linear_velocity" in info
        assert len(info["target_linear_velocity"]) == 2

    def test_observation_length(self, task):
        task.reset(zero_state())
        obs = task.get_observation(zero_state())
        assert obs.shape == (6,)
        assert task.num_observations == 6

    def test_reward_at_target(self, task):
        task._target_vel = np.array([0.5, 0.3])
        reward, done, _ = task.compute_reward(state(vx=0.5, vy=0.3))
        assert reward > 0.99
        assert done

    def test_velocity_error_in_obs(self, task):
        task._target_vel = np.array([1.0, 0.0])
        obs = task.get_observation(state(vx=0.5, theta=0))
        # dvx = 1.0 − 0.5 = 0.5, dvy = 0 − 0 = 0
        assert abs(obs[0] - 0.5) < 1e-6
        assert abs(obs[1]) < 1e-6


# ---------------------------------------------------------------------------
# TrackLinearAngularVelocityTask
# ---------------------------------------------------------------------------

class TestTrackLinearAngularVelocityTask:
    @pytest.fixture
    def task(self):
        return TrackLinearAngularVelocityTask()

    def test_reset_returns_both_targets(self, task):
        info = task.reset(zero_state())
        assert "target_linear_velocity" in info
        assert "target_angular_velocity" in info

    def test_observation_length(self, task):
        task.reset(zero_state())
        obs = task.get_observation(zero_state())
        assert obs.shape == (8,)
        assert task.num_observations == 8

    def test_reward_at_target(self, task):
        task._target_linear_vel = np.array([0.3, -0.2])
        task._target_angular_vel = 0.5
        reward, done, _ = task.compute_reward(state(vx=0.3, vy=-0.2, omega=0.5))
        assert reward > 0.95
        assert done

    def test_partial_reward_only_linear(self, task):
        """When only linear velocity matches, reward should be between 0 and 1."""
        task._target_linear_vel = np.array([0.5, 0.0])
        task._target_angular_vel = 1.0
        # Linear matches, angular doesn't
        reward, done, info = task.compute_reward(state(vx=0.5, vy=0.0, omega=0.0))
        assert 0.0 < reward < 1.0
        assert not done  # angular error too large

    def test_weights_sum_to_one(self, task):
        assert abs(task.linear_weight + task.angular_weight - 1.0) < 1e-9
