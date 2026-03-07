#!/usr/bin/env python3
# Copyright (c) Space Robotics Lab, SnT, University of Luxembourg, SpaceR
# RANS: arXiv:2310.07393 — OpenEnv training examples

"""
PPO Training for RANS
======================
Trains a spacecraft navigation policy using Proximal Policy Optimization (PPO),
the same algorithm used in the original RANS paper (via rl-games).

This implementation runs the environment locally (no HTTP server) and uses
pure PyTorch — no extra RL library required.

Architecture
------------
  Policy network:  MLP  obs → [64, 64] → action_mean, log_std
  Value network:   MLP  obs → [64, 64] → value
  Algorithm:       PPO with GAE advantage estimation

Usage
-----
    # GoToPosition (default)
    python examples/ppo_train.py

    # GoToPose, more steps
    python examples/ppo_train.py --task GoToPose --timesteps 500000

    # Continue from checkpoint
    python examples/ppo_train.py --checkpoint rans_ppo_GoToPosition.pt

    # Use trained policy
    python examples/ppo_train.py --eval --checkpoint rans_ppo_GoToPosition.pt

Requirements
------------
    pip install torch numpy
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# ---------------------------------------------------------------------------
# Local imports (no server needed)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from examples.gymnasium_wrapper import make_rans_env


# ---------------------------------------------------------------------------
# Neural network policy
# ---------------------------------------------------------------------------

def _mlp(in_dim: int, hidden: List[int], out_dim: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.Tanh()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """
    Shared-trunk actor-critic network.

    The actor outputs a Gaussian distribution over continuous thruster
    activations in [0, 1].  A Sigmoid is applied to the mean so it stays
    in a valid range; log_std is a learnable parameter.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: List[int] = None) -> None:
        super().__init__()
        if hidden is None:
            hidden = [64, 64]
        self.actor_mean = _mlp(obs_dim, hidden, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = _mlp(obs_dim, hidden, 1)

    def forward(self, obs: torch.Tensor):
        mean = torch.sigmoid(self.actor_mean(obs))   # ∈ (0, 1)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        value = self.critic(obs).squeeze(-1)
        return dist, value

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        dist, value = self(obs)
        action = dist.sample().clamp(0.0, 1.0)
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value

    @torch.no_grad()
    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        mean = torch.sigmoid(self.actor_mean(obs))
        return mean.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    def __init__(self, n_steps: int, obs_dim: int, act_dim: int, device: str) -> None:
        self.n = n_steps
        self.device = device
        self.obs = torch.zeros(n_steps, obs_dim, device=device)
        self.actions = torch.zeros(n_steps, act_dim, device=device)
        self.log_probs = torch.zeros(n_steps, device=device)
        self.rewards = torch.zeros(n_steps, device=device)
        self.values = torch.zeros(n_steps, device=device)
        self.dones = torch.zeros(n_steps, device=device)
        self.ptr = 0

    def add(self, obs, action, log_prob, reward, value, done) -> None:
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.rewards[i] = reward
        self.values[i] = value
        self.dones[i] = done
        self.ptr += 1

    def reset(self) -> None:
        self.ptr = 0

    def compute_returns_and_advantages(
        self, last_value: torch.Tensor, gamma: float = 0.99, lam: float = 0.95
    ) -> tuple:
        """GAE-λ advantage estimation."""
        advantages = torch.zeros_like(self.rewards)
        last_gae = 0.0
        for t in reversed(range(self.n)):
            next_val = last_value if t == self.n - 1 else self.values[t + 1]
            next_done = 0.0 if t == self.n - 1 else self.dones[t + 1]
            delta = (self.rewards[t]
                     + gamma * next_val * (1 - next_done)
                     - self.values[t])
            last_gae = delta + gamma * lam * (1 - self.dones[t]) * last_gae
            advantages[t] = last_gae
        returns = advantages + self.values
        return advantages, returns


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def ppo_update(
    policy: ActorCritic,
    optimizer: optim.Optimizer,
    buffer: RolloutBuffer,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    clip_eps: float = 0.2,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    n_epochs: int = 10,
    batch_size: int = 64,
) -> dict:
    """Single PPO update over the collected rollout."""
    n = buffer.n
    idx = torch.randperm(n, device=buffer.device)

    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    n_updates = 0

    for _ in range(n_epochs):
        for start in range(0, n, batch_size):
            mb = idx[start: start + batch_size]
            obs_b = buffer.obs[mb]
            act_b = buffer.actions[mb]
            old_lp_b = buffer.log_probs[mb]
            adv_b = advantages[mb]
            ret_b = returns[mb]

            # Normalise advantages
            adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

            dist, value = policy(obs_b)
            log_prob = dist.log_prob(act_b).sum(-1)
            entropy = dist.entropy().sum(-1).mean()

            ratio = (log_prob - old_lp_b).exp()
            surr1 = ratio * adv_b
            surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv_b
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (value - ret_b).pow(2).mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

            stats["policy_loss"] += policy_loss.item()
            stats["value_loss"] += value_loss.item()
            stats["entropy"] += entropy.item()
            n_updates += 1

    return {k: v / n_updates for k, v in stats.items()}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nRANS PPO Training")
    print(f"  task={args.task}  device={device}  steps={args.timesteps}")
    print("=" * 60)

    # Environment
    env = make_rans_env(task=args.task, max_episode_steps=args.episode_steps)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    print(f"  obs_dim={obs_dim}  act_dim={act_dim}")

    # Policy
    policy = ActorCritic(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        policy.load_state_dict(ckpt["policy"])
        optimizer.load_state_dict(ckpt["optimizer"])
        print(f"  Loaded checkpoint: {args.checkpoint}")

    buffer = RolloutBuffer(args.n_steps, obs_dim, act_dim, device)

    # Tracking
    ep_rewards: List[float] = []
    ep_lengths: List[int] = []
    ep_reward = 0.0
    ep_length = 0
    best_mean_reward = -float("inf")

    obs_np, _ = env.reset()
    obs = torch.from_numpy(obs_np).float().to(device)
    total_steps = 0
    update_num = 0
    t0 = time.perf_counter()

    while total_steps < args.timesteps:
        # --- Collect rollout ---
        buffer.reset()
        for _ in range(args.n_steps):
            action, log_prob, value = policy.act(obs)
            action_np = action.cpu().numpy()

            next_obs_np, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            buffer.add(obs, action, log_prob,
                       torch.tensor(reward, device=device),
                       value,
                       torch.tensor(float(done), device=device))

            ep_reward += reward
            ep_length += 1
            total_steps += 1

            if done:
                ep_rewards.append(ep_reward)
                ep_lengths.append(ep_length)
                ep_reward = 0.0
                ep_length = 0
                next_obs_np, _ = env.reset()

            obs = torch.from_numpy(next_obs_np).float().to(device)

        # Bootstrap value for last observation
        with torch.no_grad():
            _, last_value = policy(obs)

        advantages, returns = buffer.compute_returns_and_advantages(
            last_value, gamma=args.gamma, lam=args.lam
        )

        # --- PPO update ---
        stats = ppo_update(
            policy, optimizer, buffer, advantages, returns,
            clip_eps=args.clip_eps, entropy_coef=args.entropy_coef,
            n_epochs=args.n_epochs, batch_size=args.batch_size,
        )
        update_num += 1

        # --- Logging ---
        if update_num % args.log_interval == 0:
            mean_rew = np.mean(ep_rewards[-100:]) if ep_rewards else float("nan")
            mean_len = np.mean(ep_lengths[-100:]) if ep_lengths else float("nan")
            elapsed = time.perf_counter() - t0
            fps = total_steps / elapsed
            print(f"  Update {update_num:5d}  |  steps={total_steps:7d}  "
                  f"|  mean_reward={mean_rew:6.3f}  mean_len={mean_len:5.0f}  "
                  f"|  fps={fps:.0f}  "
                  f"|  pi_loss={stats['policy_loss']:.4f}  "
                  f"|  v_loss={stats['value_loss']:.4f}")

        # --- Checkpoint ---
        if ep_rewards:
            mean_rew = np.mean(ep_rewards[-100:])
            if mean_rew > best_mean_reward:
                best_mean_reward = mean_rew
                ckpt_path = f"rans_ppo_{args.task}.pt"
                torch.save({"policy": policy.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "total_steps": total_steps,
                            "best_mean_reward": best_mean_reward}, ckpt_path)

    env.close()
    print(f"\nTraining complete.  Best mean reward: {best_mean_reward:.3f}")
    print(f"Checkpoint saved to: rans_ppo_{args.task}.pt")


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    device = "cpu"
    env = make_rans_env(task=args.task, max_episode_steps=args.episode_steps)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    policy = ActorCritic(obs_dim, act_dim).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    print(f"\nEvaluating {args.checkpoint}  task={args.task}")
    print(f"  Best training reward: {ckpt.get('best_mean_reward', '?'):.3f}")
    print("=" * 60)

    for ep in range(args.eval_episodes):
        obs_np, _ = env.reset()
        total_reward = 0.0
        steps = 0
        while True:
            obs = torch.from_numpy(obs_np).float().to(device)
            action = policy.act_deterministic(obs).numpy()
            obs_np, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break
        print(f"  Episode {ep + 1:2d}  |  steps={steps:4d}  "
              f"|  reward={total_reward:.3f}  "
              f"|  goal={info.get('goal_reached', '?')}")

    env.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="RANS PPO training")
    parser.add_argument("--task", default="GoToPosition",
                        choices=["GoToPosition", "GoToPose",
                                 "TrackLinearVelocity", "TrackLinearAngularVelocity"])
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="Rollout length before each PPO update")
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Log every N PPO updates")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to a .pt checkpoint to load or save")
    parser.add_argument("--eval", action="store_true",
                        help="Run evaluation only (requires --checkpoint)")
    parser.add_argument("--eval-episodes", type=int, default=10)
    args = parser.parse_args()

    if args.eval:
        if not args.checkpoint:
            print("--eval requires --checkpoint PATH")
            sys.exit(1)
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
