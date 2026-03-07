#!/usr/bin/env python3
# Copyright (c) Space Robotics Lab, SnT, University of Luxembourg, SpaceR
# RANS: arXiv:2310.07393 — OpenEnv training examples

"""
Training via the OpenEnv Client
================================
Uses the ``RANSEnv`` HTTP/WebSocket client to train against a remote server.
This is the canonical OpenEnv usage pattern — the environment runs in an
isolated Docker container, and the training loop runs separately.

Flow:
  ┌─────────────────────────────┐        ┌──────────────────────┐
  │  Training process           │  HTTP/ │  RANS Docker server  │
  │  (this script)              │  WS    │  (rans_env.server)   │
  │                             │◄──────►│                      │
  │  policy  →  RANSEnv.step() │        │  RANSEnvironment     │
  │          ←  observation     │        │  spacecraft physics  │
  │          ←  reward          │        │                      │
  └─────────────────────────────┘        └──────────────────────┘

Prerequisites
-------------
1. Start the server:
       RANS_TASK=GoToPosition uvicorn rans_env.server.app:app --port 8000
   or via Docker:
       docker run -e RANS_TASK=GoToPosition -p 8000:8000 rans-env

2. Run this script:
       python examples/openenv_client_train.py --url http://localhost:8000

Notes on parallelism
--------------------
OpenEnv supports concurrent sessions on one server.  For parallel environment
collection (common in PPO), start multiple server instances on different ports
and use ``AsyncVectorRANSEnv`` below, or use Docker Compose.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from typing import List

import numpy as np


# ---------------------------------------------------------------------------
# Async vectorised environment using OpenEnv client
# ---------------------------------------------------------------------------

class AsyncVectorRANSEnv:
    """
    Wraps N concurrent ``RANSEnv`` sessions as a vectorised environment.

    Each session connects to the same server (OpenEnv supports concurrent
    WebSocket sessions).  Actions and observations are batched across sessions.

    Parameters
    ----------
    base_url:
        Server URL, e.g. ``"http://localhost:8000"``.
    n_envs:
        Number of parallel sessions.
    """

    def __init__(self, base_url: str, n_envs: int = 4) -> None:
        self.base_url = base_url
        self.n_envs = n_envs
        self._clients = []

    async def __aenter__(self):
        from rans_env import RANSEnv
        self._clients = [RANSEnv(base_url=self.base_url) for _ in range(self.n_envs)]
        for c in self._clients:
            await c.__aenter__()
        return self

    async def __aexit__(self, *args):
        for c in self._clients:
            await c.__aexit__(*args)

    async def reset(self) -> List:
        results = await asyncio.gather(*[c.reset() for c in self._clients])
        return results

    async def step(self, actions) -> List:
        from rans_env import SpacecraftAction
        coros = [
            c.step(SpacecraftAction(thrusters=a.tolist()))
            for c, a in zip(self._clients, actions)
        ]
        return await asyncio.gather(*coros)


# ---------------------------------------------------------------------------
# Simple async training loop (random policy — replace with your own)
# ---------------------------------------------------------------------------

async def run_training(args: argparse.Namespace) -> None:
    """
    Minimal async training loop showing the OpenEnv client API.

    Replace the random action selection with your policy's forward pass.
    The structure here is compatible with any policy that maps
    ``obs → action``:

        import torch
        from examples.ppo_train import ActorCritic
        policy = ActorCritic(obs_dim, act_dim)
        policy.load_state_dict(torch.load("rans_ppo_GoToPosition.pt")["policy"])

        obs_tensor = torch.FloatTensor(obs_flat)
        action = policy.act_deterministic(obs_tensor).numpy()
    """
    try:
        from rans_env import RANSEnv, SpacecraftAction
    except ImportError:
        print("Install the RANS package:  pip install -e .")
        sys.exit(1)

    print(f"\nRANS OpenEnv Client Training")
    print(f"  server={args.url}  n_envs={args.n_envs}  episodes={args.episodes}")
    print("=" * 60)

    async with AsyncVectorRANSEnv(args.url, n_envs=args.n_envs) as vec_env:
        # Reset all environments
        results = await vec_env.reset()
        n_thrusters = len(results[0].thruster_masks)
        obs_dim = len(results[0].state_obs) + n_thrusters * 5 + n_thrusters + 2
        print(f"  n_thrusters={n_thrusters}  obs_dim={obs_dim}")
        print(f"  Task: {results[0].task}")

        ep_rewards = [0.0] * args.n_envs
        ep_counts = [0] * args.n_envs
        all_ep_rewards: List[float] = []
        t0 = time.perf_counter()
        total_steps = 0

        while len(all_ep_rewards) < args.episodes * args.n_envs:
            # ── Choose actions ─────────────────────────────────────────
            # Replace with: actions = policy(obs_batch)
            actions = [
                np.random.randint(0, 2, size=n_thrusters).astype(np.float32)
                for _ in range(args.n_envs)
            ]

            # ── Step all environments ───────────────────────────────────
            results = await vec_env.step(actions)
            total_steps += args.n_envs

            for i, result in enumerate(results):
                ep_rewards[i] += result.reward or 0.0
                ep_counts[i] += 1

                if result.done:
                    all_ep_rewards.append(ep_rewards[i])
                    n_done = len(all_ep_rewards)
                    mean100 = np.mean(all_ep_rewards[-100:])
                    print(f"  env={i}  ep={n_done:4d}  "
                          f"reward={ep_rewards[i]:6.3f}  "
                          f"steps={ep_counts[i]:4d}  "
                          f"mean100={mean100:.3f}  "
                          f"|  {result.info.get('goal_reached', False)}")
                    ep_rewards[i] = 0.0
                    ep_counts[i] = 0

                    # Reset this environment
                    reset_result = await vec_env._clients[i].reset()  # noqa: SLF001

        elapsed = time.perf_counter() - t0
        fps = total_steps / elapsed
        print(f"\nDone.  {total_steps} steps  |  {fps:.0f} steps/s")
        print(f"Mean episode reward: {np.mean(all_ep_rewards):.3f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RANS training via OpenEnv client"
    )
    parser.add_argument("--url", default="http://localhost:8000",
                        help="Running RANS server URL")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Number of concurrent environment sessions")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Total episodes to collect (per env)")
    args = parser.parse_args()
    asyncio.run(run_training(args))


if __name__ == "__main__":
    main()
