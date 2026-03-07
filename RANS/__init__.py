"""
rans_env — RANS spacecraft navigation environment for OpenEnv.

Paper: "RANS: Highly-Parallelised Simulator for Reinforcement Learning based
Autonomous Navigating Spacecrafts", El-Hariry, Richard, Olivares-Mendez (2023).
arXiv:2310.07393

Quick start::

    from rans_env import RANSEnv, SpacecraftAction

    with RANSEnv(base_url="http://localhost:8000").sync() as env:
        obs = env.reset()
        result = env.step(SpacecraftAction(thrusters=[0] * 8))
        print(result.reward)
"""

from .client import RANSEnv
from .models import SpacecraftAction, SpacecraftObservation, SpacecraftState

__all__ = [
    "RANSEnv",
    "SpacecraftAction",
    "SpacecraftObservation",
    "SpacecraftState",
]
