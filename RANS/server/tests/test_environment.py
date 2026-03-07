# Copyright (c) Space Robotics Lab, SnT, University of Luxembourg, SpaceR

"""
Integration tests for RANSEnvironment (without the FastAPI server).

Run with:  pytest server/tests/test_environment.py -v
"""

import math
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from server.rans_environment import RANSEnvironment
from server.tasks import TASK_REGISTRY
from rans_env.models import SpacecraftAction, SpacecraftObservation, SpacecraftState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    return RANSEnvironment(task="GoToPosition", max_episode_steps=50)

@pytest.fixture(params=list(TASK_REGISTRY.keys()))
def env_all_tasks(request):
    return RANSEnvironment(task=request.param, max_episode_steps=50)


# ---------------------------------------------------------------------------
# Basic lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_reset_returns_observation(self, env):
        obs = env.reset()
        assert isinstance(obs, SpacecraftObservation)

    def test_reset_sets_task(self, env):
        obs = env.reset()
        assert obs.task == "GoToPosition"

    def test_reset_reward_zero(self, env):
        obs = env.reset()
        assert obs.reward == 0.0
        assert obs.done is False

    def test_step_returns_observation(self, env):
        env.reset()
        n = env._spacecraft.n_thrusters
        result = env.step(SpacecraftAction(thrusters=[0.0] * n))
        assert isinstance(result, SpacecraftObservation)

    def test_step_increments_counter(self, env):
        env.reset()
        n = env._spacecraft.n_thrusters
        for i in range(5):
            env.step(SpacecraftAction(thrusters=[0.0] * n))
        assert env._step_count == 5

    def test_step_limit_triggers_done(self, env):
        env.reset()
        n = env._spacecraft.n_thrusters
        result = None
        for _ in range(50):  # max_episode_steps = 50
            result = env.step(SpacecraftAction(thrusters=[0.0] * n))
        assert result.done is True

    def test_multiple_resets(self, env):
        for _ in range(3):
            obs = env.reset()
            assert obs.done is False
            assert env._step_count == 0


# ---------------------------------------------------------------------------
# Observation structure
# ---------------------------------------------------------------------------

class TestObservationStructure:
    def test_state_obs_length(self, env):
        obs = env.reset()
        # GoToPosition has 6 state observations
        assert len(obs.state_obs) == 6

    def test_thruster_transforms_shape(self, env):
        obs = env.reset()
        # 8-thruster default → [8, 5]
        assert len(obs.thruster_transforms) == 8
        assert all(len(row) == 5 for row in obs.thruster_transforms)

    def test_thruster_masks_all_ones(self, env):
        obs = env.reset()
        assert all(m == 1.0 for m in obs.thruster_masks)

    def test_mass_positive(self, env):
        obs = env.reset()
        assert obs.mass > 0.0
        assert obs.inertia > 0.0

    def test_info_contains_step(self, env):
        env.reset()
        n = env._spacecraft.n_thrusters
        result = env.step(SpacecraftAction(thrusters=[0.0] * n))
        assert "step" in result.info

    def test_goal_reached_in_info(self, env):
        obs = env.reset()
        n = env._spacecraft.n_thrusters
        result = env.step(SpacecraftAction(thrusters=[0.0] * n))
        assert "goal_reached" in result.info


# ---------------------------------------------------------------------------
# State property
# ---------------------------------------------------------------------------

class TestStateProperty:
    def test_state_is_spacecraft_state(self, env):
        env.reset()
        assert isinstance(env.state, SpacecraftState)

    def test_state_task(self, env):
        env.reset()
        assert env.state.task == "GoToPosition"

    def test_state_tracks_steps(self, env):
        env.reset()
        n = env._spacecraft.n_thrusters
        for _ in range(3):
            env.step(SpacecraftAction(thrusters=[0.0] * n))
        assert env.state.step_count == 3

    def test_state_physical_values(self, env):
        env.reset()
        s = env.state
        # After reset, state values should be finite
        assert math.isfinite(s.x)
        assert math.isfinite(s.y)
        assert math.isfinite(s.heading_rad)
        assert -math.pi < s.heading_rad <= math.pi


# ---------------------------------------------------------------------------
# Action validation
# ---------------------------------------------------------------------------

class TestActionValidation:
    def test_wrong_action_type_raises(self, env):
        env.reset()
        with pytest.raises((ValueError, TypeError)):
            env.step(object())  # not a SpacecraftAction

    def test_partial_activation_clamped(self, env):
        env.reset()
        # Activations out of [0,1] should be clamped, not raise
        n = env._spacecraft.n_thrusters
        result = env.step(SpacecraftAction(thrusters=[2.0] * n))
        assert isinstance(result, SpacecraftObservation)

    def test_short_activation_padded(self, env):
        env.reset()
        # Fewer thrusters than expected → should be zero-padded
        result = env.step(SpacecraftAction(thrusters=[1.0]))
        assert isinstance(result, SpacecraftObservation)


# ---------------------------------------------------------------------------
# All tasks smoke test
# ---------------------------------------------------------------------------

class TestAllTasks:
    def test_reset_step_smoke(self, env_all_tasks):
        env = env_all_tasks
        obs = env.reset()
        assert isinstance(obs, SpacecraftObservation)
        n = env._spacecraft.n_thrusters
        result = env.step(SpacecraftAction(thrusters=[0.5] * n))
        assert isinstance(result, SpacecraftObservation)
        assert math.isfinite(result.reward)
        assert 0.0 <= result.reward <= 1.0

    def test_observation_length_correct(self, env_all_tasks):
        env = env_all_tasks
        obs = env.reset()
        expected_lengths = {
            "GoToPosition": 6,
            "GoToPose": 7,
            "TrackLinearVelocity": 6,
            "TrackLinearAngularVelocity": 8,
        }
        expected = expected_lengths[env._task_name]
        assert len(obs.state_obs) == expected, (
            f"{env._task_name}: expected {expected} obs, got {len(obs.state_obs)}"
        )


# ---------------------------------------------------------------------------
# Reward bounds
# ---------------------------------------------------------------------------

class TestRewardBounds:
    def test_reward_in_zero_one(self, env_all_tasks):
        env = env_all_tasks
        env.reset()
        n = env._spacecraft.n_thrusters
        for _ in range(10):
            result = env.step(SpacecraftAction(thrusters=np.random.random(n).tolist()))
            assert 0.0 <= result.reward <= 1.0 + 1e-9, (
                f"Reward {result.reward} out of [0,1]"
            )
