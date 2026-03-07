#!/usr/bin/env python3
# Copyright (c) Space Robotics Lab, SnT, University of Luxembourg, SpaceR
# RANS: arXiv:2310.07393 — OpenEnv training examples

"""
Random Agent — Sanity Check
============================
Verifies the RANS environment works end-to-end by running a random agent.
This is the first script to run after starting the server.

Requires a running RANS server:
    uvicorn rans_env.server.app:app --host 0.0.0.0 --port 8000

Run this script:
    python examples/random_agent.py
    python examples/random_agent.py --task GoToPose --episodes 5
"""

import argparse
import random
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="RANS random agent")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--task", default="GoToPosition",
                        choices=["GoToPosition", "GoToPose",
                                 "TrackLinearVelocity", "TrackLinearAngularVelocity"])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=200)
    args = parser.parse_args()

    try:
        from rans_env import RANSEnv, SpacecraftAction
    except ImportError:
        print("Install the RANS package first:  pip install -e .")
        sys.exit(1)

    print(f"\nRANS Random Agent  —  task={args.task}  server={args.url}")
    print("=" * 60)

    with RANSEnv(base_url=args.url).sync() as env:
        for ep in range(1, args.episodes + 1):
            result = env.reset()
            obs = result.observation
            n_thrusters = len(obs.thruster_masks)

            print(f"\nEpisode {ep}  |  thrusters={n_thrusters}  |  task={obs.task}")
            print(f"  Initial state_obs: {[f'{v:.3f}' for v in obs.state_obs]}")

            total_reward = 0.0
            t0 = time.perf_counter()

            for step in range(1, args.max_steps + 1):
                # Random binary thruster activations
                action = SpacecraftAction(
                    thrusters=[random.choice([0.0, 1.0]) for _ in range(n_thrusters)]
                )
                result = env.step(action)
                total_reward += result.reward or 0.0

                if result.done:
                    print(f"  Step {step:4d}  |  reward={result.reward:.4f}  "
                          f"|  DONE  ({result.info})")
                    break

                if step % 50 == 0:
                    print(f"  Step {step:4d}  |  reward={result.reward:.4f}  "
                          f"|  cumulative={total_reward:.3f}")

            elapsed = time.perf_counter() - t0
            fps = step / elapsed
            print(f"  Episode done  |  steps={step}  total_reward={total_reward:.3f}  "
                  f"|  {fps:.0f} steps/s")

    print("\nDone.")


if __name__ == "__main__":
    main()
