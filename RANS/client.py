# Copyright (c) Space Robotics Lab, SnT, University of Luxembourg, SpaceR
# RANS: arXiv:2310.07393 — OpenEnv-compatible implementation

"""
RANSEnv — OpenEnv client for the RANS spacecraft navigation environment.

Usage (synchronous)::

    from rans_env import RANSEnv, SpacecraftAction

    with RANSEnv(base_url="http://localhost:8000").sync() as env:
        result = env.reset()
        n = len(result.observation.thruster_masks)
        result = env.step(SpacecraftAction(thrusters=[1, 0, 0, 0, 0, 0, 0, 0]))
        print(result.reward, result.done)

Usage (async)::

    import asyncio
    from rans_env import RANSEnv, SpacecraftAction

    async def main():
        async with RANSEnv(base_url="http://localhost:8000") as env:
            result = await env.reset()
            result = await env.step(SpacecraftAction(thrusters=[0.0] * 8))
            print(result.reward, result.done)

    asyncio.run(main())

Docker::

    env = RANSEnv.from_docker_image("rans-env:latest", env={"RANS_TASK": "GoToPose"})

HuggingFace Spaces::

    env = RANSEnv.from_env("dpang/rans-env")
"""

from __future__ import annotations

from typing import Any, Dict

try:
    from openenv.core.env_client import EnvClient, StepResult
    _OPENENV_AVAILABLE = True
except ImportError:
    EnvClient = object  # type: ignore[assignment,misc]
    StepResult = None   # type: ignore[assignment,misc]
    _OPENENV_AVAILABLE = False

from rans_env.models import SpacecraftAction, SpacecraftObservation, SpacecraftState


class RANSEnv(EnvClient):
    """
    Client for the RANS spacecraft navigation OpenEnv environment.

    Implements the three ``EnvClient`` abstract methods that handle
    JSON serialisation of actions and deserialisation of observations.

    Parameters
    ----------
    base_url:
        HTTP/WebSocket URL of the running server,
        e.g. ``"http://localhost:8000"`` or ``"ws://localhost:8000"``.
    """

    # ------------------------------------------------------------------
    # EnvClient abstract method implementations
    # ------------------------------------------------------------------

    def _step_payload(self, action: SpacecraftAction) -> Dict[str, Any]:
        """Serialise SpacecraftAction → JSON dict for the WebSocket message."""
        return {"thrusters": action.thrusters}

    def _parse_result(self, payload: Dict[str, Any]) -> "StepResult[SpacecraftObservation]":
        """
        Deserialise the server response into a typed StepResult.

        The server sends::

            {
              "observation": { "state_obs": [...], "thruster_transforms": [...],
                               "thruster_masks": [...], "mass": 10.0, "inertia": 0.5,
                               "task": "GoToPosition", "reward": 0.42, "done": false,
                               "info": {...} },
              "reward": 0.42,
              "done": false
            }
        """
        obs_dict = payload.get("observation", payload)
        observation = SpacecraftObservation(
            state_obs=obs_dict.get("state_obs", []),
            thruster_transforms=obs_dict.get("thruster_transforms", []),
            thruster_masks=obs_dict.get("thruster_masks", []),
            mass=obs_dict.get("mass", 10.0),
            inertia=obs_dict.get("inertia", 0.5),
            task=obs_dict.get("task", "GoToPosition"),
            reward=float(obs_dict.get("reward") or 0.0),
            done=bool(obs_dict.get("done", False)),
            info=obs_dict.get("info", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward") or observation.reward,
            done=payload.get("done", observation.done),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> SpacecraftState:
        """Deserialise the /state response into a SpacecraftState."""
        return SpacecraftState(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
            task=payload.get("task", "GoToPosition"),
            x=payload.get("x", 0.0),
            y=payload.get("y", 0.0),
            heading_rad=payload.get("heading_rad", 0.0),
            vx=payload.get("vx", 0.0),
            vy=payload.get("vy", 0.0),
            angular_velocity_rads=payload.get("angular_velocity_rads", 0.0),
            total_reward=payload.get("total_reward", 0.0),
            goal_reached=payload.get("goal_reached", False),
        )
