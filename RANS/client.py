# Copyright (c) Space Robotics Lab, SnT, University of Luxembourg, SpaceR
# RANS: arXiv:2310.07393 — OpenEnv-compatible implementation

"""
RANSEnv — OpenEnv client for the RANS spacecraft navigation environment.

Usage (async)::

    import asyncio
    from rans_env import RANSEnv, SpacecraftAction

    async def main():
        async with RANSEnv(base_url="http://localhost:8000") as env:
            obs = await env.reset()
            print("Task:", obs.task)
            print("Observation:", obs.state_obs)

            # Zero-thrust step
            n = len(obs.thruster_masks)
            result = await env.step(SpacecraftAction(thrusters=[0.0] * n))
            print("Reward:", result.reward)
            print("Done:", result.done)

    asyncio.run(main())

Usage (synchronous)::

    from rans_env import RANSEnv, SpacecraftAction

    with RANSEnv(base_url="http://localhost:8000").sync() as env:
        obs = env.reset()
        result = env.step(SpacecraftAction(thrusters=[1, 0, 0, 0, 0, 0, 0, 0]))

Docker::

    env = RANSEnv.from_docker_image(
        "rans-env:latest",
        env={"RANS_TASK": "GoToPose"},
    )

HuggingFace Spaces::

    env = RANSEnv.from_env("openenv/rans-env")
"""

from __future__ import annotations

try:
    from openenv.core.env_client import EnvClient
except ImportError:
    EnvClient = object  # type: ignore[assignment,misc]

from rans_env.models import SpacecraftAction, SpacecraftObservation, SpacecraftState


class RANSEnv(EnvClient):
    """
    Client for the RANS spacecraft navigation OpenEnv environment.

    All functionality (``reset``, ``step``, ``state``, ``sync``,
    ``from_docker_image``, ``from_env``) is provided by the ``EnvClient``
    base class from openenv-core.

    The client is typed: it sends ``SpacecraftAction`` objects and receives
    ``SpacecraftObservation`` objects.

    Parameters
    ----------
    base_url:
        Base URL of the running RANS server, e.g. ``"http://localhost:8000"``.
    """

    action_type = SpacecraftAction
    observation_type = SpacecraftObservation
    state_type = SpacecraftState
