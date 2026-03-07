#!/usr/bin/env python3
# Copyright (c) Space Robotics Lab, SnT, University of Luxembourg, SpaceR
# RANS: arXiv:2310.07393 — OpenEnv training examples

"""
Gymnasium Wrapper for RANS
===========================
Wraps ``RANSEnvironment`` in a standard ``gymnasium.Env`` interface so any
Gymnasium-compatible RL library can be used for training:

    • Stable-Baselines3  (PPO, SAC, TD3, …)
    • CleanRL
    • RLlib
    • TorchRL

The wrapper runs the environment **locally** (in-process) — no HTTP server
needed.  For server-based training, replace ``RANSEnvironment()`` with the
``RANSEnv`` WebSocket client (see remote_train_sb3.py).

Usage
-----
    # Standalone check
    python examples/gymnasium_wrapper.py

    # Stable-Baselines3 PPO (requires: pip install stable-baselines3)
    from examples.gymnasium_wrapper import make_rans_env
    from stable_baselines3 import PPO

    env = make_rans_env(task="GoToPosition")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=200_000)
    model.save("rans_ppo_go_to_position")
"""

from __future__ import annotations

import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    print("gymnasium is required:  pip install gymnasium")
    sys.exit(1)

# Local import (no server needed)
sys.path.insert(0, __file__.replace("examples/gymnasium_wrapper.py", ""))
from server.rans_environment import RANSEnvironment
from server.spacecraft_physics import SpacecraftConfig
from rans_env.models import SpacecraftAction


class RANSGymnasiumEnv(gym.Env):
    """
    Gymnasium-compatible wrapper around ``RANSEnvironment``.

    Observation space:
        Flat Box containing [state_obs, thruster_transforms (flattened),
        thruster_masks, mass, inertia].

    Action space:
        Box([0, 1]^n_thrusters)  — continuous thruster activations.

    Parameters
    ----------
    task:
        RANS task name.
    spacecraft_config:
        Physical platform configuration.
    task_config:
        Dict of task hyper-parameters.
    max_episode_steps:
        Hard step limit per episode.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        task: str = "GoToPosition",
        spacecraft_config: Optional[SpacecraftConfig] = None,
        task_config: Optional[Dict[str, Any]] = None,
        max_episode_steps: int = 500,
    ) -> None:
        super().__init__()
        self._env = RANSEnvironment(
            task=task,
            spacecraft_config=spacecraft_config,
            task_config=task_config,
            max_episode_steps=max_episode_steps,
        )
        sc = self._env._spacecraft

        # --- action space ---
        n = sc.n_thrusters
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(n,), dtype=np.float32
        )

        # --- observation space ---
        # state_obs (task-dependent) + transforms [n×5] + masks [n] + mass + inertia
        obs0 = self._env.reset()
        flat_obs = self._flatten(obs0)
        dim = flat_obs.shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32
        )

        self._last_obs = flat_obs

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        obs = self._env.reset()
        self._last_obs = self._flatten(obs)
        return self._last_obs, {"task": obs.task}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        result = self._env.step(
            SpacecraftAction(thrusters=action.tolist())
        )
        flat_obs = self._flatten(result)
        reward = float(result.reward or 0.0)
        terminated = bool(result.done)
        truncated = False  # RANSEnvironment merges step-limit into done
        self._last_obs = flat_obs
        return flat_obs, reward, terminated, truncated, result.info or {}

    def render(self) -> None:
        pass  # headless — use result.info for diagnostics

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten(obs) -> np.ndarray:
        """Flatten the SpacecraftObservation into a 1-D float32 array."""
        parts = [
            np.array(obs.state_obs, dtype=np.float32),
            np.array(obs.thruster_transforms, dtype=np.float32).flatten(),
            np.array(obs.thruster_masks, dtype=np.float32),
            np.array([obs.mass, obs.inertia], dtype=np.float32),
        ]
        return np.concatenate(parts)


def make_rans_env(
    task: str = "GoToPosition",
    task_config: Optional[Dict[str, Any]] = None,
    max_episode_steps: int = 500,
) -> RANSGymnasiumEnv:
    """
    Factory that returns a ``gymnasium.Env``-compatible RANS environment.

    Example::

        from examples.gymnasium_wrapper import make_rans_env
        from stable_baselines3 import PPO

        env = make_rans_env(task="GoToPose")
        model = PPO("MlpPolicy", env, verbose=1, n_steps=2048)
        model.learn(total_timesteps=500_000)
    """
    return RANSGymnasiumEnv(task=task, task_config=task_config,
                            max_episode_steps=max_episode_steps)


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    print("RANS Gymnasium Wrapper — smoke test")
    print("=" * 50)

    for task in ["GoToPosition", "GoToPose",
                 "TrackLinearVelocity", "TrackLinearAngularVelocity"]:
        env = make_rans_env(task=task, max_episode_steps=100)
        obs, info = env.reset()
        print(f"\nTask: {task}")
        print(f"  obs shape:    {obs.shape}")
        print(f"  action shape: {env.action_space.shape}")

        total_reward = 0.0
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        print(f"  total_reward: {total_reward:.3f}")
        print(f"  goal_reached: {info.get('goal_reached', False)}")
        env.close()

    print("\nAll tasks OK.")


if __name__ == "__main__":
    _smoke_test()
