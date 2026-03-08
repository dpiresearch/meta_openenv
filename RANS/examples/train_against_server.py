#!/usr/bin/env python3
# Copyright (c) Space Robotics Lab, SnT, University of Luxembourg, SpaceR
# RANS: arXiv:2310.07393 — OpenEnv training examples

"""
PPO Training Against a Running RANS Server
==========================================
Trains a spacecraft navigation policy via the OpenEnv HTTP/WebSocket client,
connecting to a RANS server running locally (uvicorn) or in Docker.

This is the canonical OpenEnv training pattern:

    ┌─────────────────────────────┐   HTTP/WS   ┌──────────────────────┐
    │  ppo_train (this script)    │ ──────────► │  uvicorn / Docker    │
    │  RemoteRANSGymnasiumEnv     │             │  RANSEnvironment     │
    │  ActorCritic + PPO          │ ◄────────── │  spacecraft physics  │
    └─────────────────────────────┘             └──────────────────────┘

Start the server first:
    uvicorn rans_env.server.app:app --host 0.0.0.0 --port 8000

Then run this script:
    python examples/train_against_server.py --task GoToPosition
    python examples/train_against_server.py --task GoToPose --url http://localhost:8000
    python examples/train_against_server.py --eval --checkpoint rans_ppo_remote_GoToPosition.pt

Requirements:
    pip install torch gymnasium openenv-core
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    print("gymnasium is required:  pip install gymnasium")
    sys.exit(1)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Normal
except ImportError:
    print("torch is required:  pip install torch")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ---------------------------------------------------------------------------
# Remote gymnasium wrapper (uses OpenEnv HTTP client)
# ---------------------------------------------------------------------------

class RemoteRANSGymnasiumEnv(gym.Env):
    """
    Gymnasium-compatible environment that connects to a running RANS server
    via the OpenEnv ``RANSEnv`` WebSocket/HTTP client.

    Identical observation and action spaces to ``RANSGymnasiumEnv``, but all
    physics runs inside the server process (or Docker container).
    """

    metadata = {"render_modes": []}

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        super().__init__()
        from rans_env import RANSEnv, SpacecraftAction

        self._SpacecraftAction = SpacecraftAction
        # EnvClient is already synchronous (WebSocket-based); just connect.
        self._client = RANSEnv(base_url=base_url)
        self._client.connect()

        # Probe the environment to determine spaces
        result = self._client.reset()
        obs = result.observation
        flat = self._flatten(obs)

        n = len(obs.thruster_masks)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(n,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(flat.shape[0],), dtype=np.float32
        )
        self._last_flat = flat
        self._task = obs.task

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        result = self._client.reset()
        self._last_flat = self._flatten(result.observation)
        return self._last_flat, {"task": result.observation.task}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        result = self._client.step(
            self._SpacecraftAction(thrusters=action.tolist())
        )
        flat = self._flatten(result.observation)
        reward = float(result.reward or 0.0)
        done = bool(result.done)
        self._last_flat = flat
        return flat, reward, done, False, result.observation.info or {}

    def close(self) -> None:
        self._client.disconnect()

    @staticmethod
    def _flatten(obs) -> np.ndarray:
        return np.concatenate([
            np.array(obs.state_obs, dtype=np.float32),
            np.array(obs.thruster_transforms, dtype=np.float32).flatten(),
            np.array(obs.thruster_masks, dtype=np.float32),
            np.array([obs.mass, obs.inertia], dtype=np.float32),
        ])


# ---------------------------------------------------------------------------
# Re-use ActorCritic and PPO from ppo_train.py
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
    def __init__(self, obs_dim: int, act_dim: int, hidden: List[int] = None):
        super().__init__()
        hidden = hidden or [64, 64]
        self.actor_mean = _mlp(obs_dim, hidden, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = _mlp(obs_dim, hidden, 1)

    def forward(self, obs):
        mean = torch.sigmoid(self.actor_mean(obs))
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std), self.critic(obs).squeeze(-1)

    @torch.no_grad()
    def act(self, obs):
        dist, value = self(obs)
        action = dist.sample().clamp(0.0, 1.0)
        return action, dist.log_prob(action).sum(-1), value

    @torch.no_grad()
    def act_deterministic(self, obs):
        return torch.sigmoid(self.actor_mean(obs)).clamp(0.0, 1.0)


class RolloutBuffer:
    def __init__(self, n: int, obs_dim: int, act_dim: int, device: str):
        self.n, self.device = n, device
        self.obs      = torch.zeros(n, obs_dim, device=device)
        self.actions  = torch.zeros(n, act_dim, device=device)
        self.log_probs = torch.zeros(n, device=device)
        self.rewards  = torch.zeros(n, device=device)
        self.values   = torch.zeros(n, device=device)
        self.dones    = torch.zeros(n, device=device)
        self.ptr = 0

    def add(self, obs, action, log_prob, reward, value, done):
        i = self.ptr
        self.obs[i], self.actions[i] = obs, action
        self.log_probs[i], self.rewards[i] = log_prob, reward
        self.values[i], self.dones[i] = value, done
        self.ptr += 1

    def reset(self): self.ptr = 0

    def compute_gae(self, last_value, gamma=0.99, lam=0.95):
        adv = torch.zeros_like(self.rewards)
        last_gae = 0.0
        for t in reversed(range(self.n)):
            nv = last_value if t == self.n - 1 else self.values[t + 1]
            nd = 0.0 if t == self.n - 1 else self.dones[t + 1]
            delta = self.rewards[t] + gamma * nv * (1 - nd) - self.values[t]
            last_gae = delta + gamma * lam * (1 - self.dones[t]) * last_gae
            adv[t] = last_gae
        return adv, adv + self.values


def ppo_update(policy, optimizer, buf, adv, returns,
               clip=0.2, ent=0.01, vf=0.5, epochs=10, bs=64):
    n = buf.n
    stats = {"pi": 0.0, "vf": 0.0, "ent": 0.0}
    n_updates = 0
    for _ in range(epochs):
        for s in range(0, n, bs):
            mb = torch.randperm(n, device=buf.device)[s:s+bs]
            a_b = (adv[mb] - adv[mb].mean()) / (adv[mb].std() + 1e-8)
            dist, val = policy(buf.obs[mb])
            lp = dist.log_prob(buf.actions[mb]).sum(-1)
            r = (lp - buf.log_probs[mb]).exp()
            pi_loss = -torch.min(r * a_b, r.clamp(1-clip, 1+clip) * a_b).mean()
            vf_loss = (val - returns[mb]).pow(2).mean()
            loss = pi_loss + vf * vf_loss - ent * dist.entropy().sum(-1).mean()
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            stats["pi"] += pi_loss.item()
            stats["vf"] += vf_loss.item()
            n_updates += 1
    return {key: val / max(n_updates, 1) for key, val in stats.items()}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nRANS PPO — Remote Training via OpenEnv Client")
    print(f"  server : {args.url}")
    print(f"  task   : connecting…  (task set by RANS_TASK on server)")
    print(f"  device : {device}")
    print(f"  steps  : {args.timesteps:,}")
    print("=" * 60)

    env = RemoteRANSGymnasiumEnv(base_url=args.url)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    print(f"  task      : {env._task}")
    print(f"  obs_dim   : {obs_dim}")
    print(f"  act_dim   : {act_dim}  (thrusters)")
    print()

    policy = ActorCritic(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    if args.checkpoint and os.path.exists(args.checkpoint):
        ck = torch.load(args.checkpoint, map_location=device)
        policy.load_state_dict(ck["policy"])
        optimizer.load_state_dict(ck["optimizer"])
        print(f"  Loaded checkpoint: {args.checkpoint}")

    buf = RolloutBuffer(args.n_steps, obs_dim, act_dim, device)

    ep_rewards: List[float] = []
    ep_lengths: List[int] = []
    ep_reward = ep_len = 0.0
    best_mean = -float("inf")

    obs_np, _ = env.reset()
    obs = torch.from_numpy(obs_np).float().to(device)
    total_steps = update_num = 0
    t0 = time.perf_counter()

    while total_steps < args.timesteps:
        buf.reset()
        for _ in range(args.n_steps):
            action, log_prob, value = policy.act(obs)
            next_obs_np, reward, terminated, truncated, info = env.step(
                action.cpu().numpy()
            )
            done = terminated or truncated
            buf.add(obs, action, log_prob,
                    torch.tensor(reward, device=device),
                    value,
                    torch.tensor(float(done), device=device))
            ep_reward += reward
            ep_len += 1
            total_steps += 1
            if done:
                ep_rewards.append(ep_reward)
                ep_lengths.append(ep_len)
                ep_reward = ep_len = 0.0
                next_obs_np, _ = env.reset()
            obs = torch.from_numpy(next_obs_np).float().to(device)

        with torch.no_grad():
            _, last_val = policy(obs)
        adv, returns = buf.compute_gae(last_val, args.gamma, args.lam)

        stats = ppo_update(policy, optimizer, buf, adv, returns,
                           clip=args.clip_eps, ent=args.entropy_coef,
                           epochs=args.n_epochs, bs=args.batch_size)
        update_num += 1

        if update_num % args.log_interval == 0:
            mean_rew = np.mean(ep_rewards[-100:]) if ep_rewards else float("nan")
            fps = total_steps / (time.perf_counter() - t0)
            print(f"  update {update_num:4d} | steps {total_steps:7,} | "
                  f"mean_rew {mean_rew:6.3f} | fps {fps:4.0f} | "
                  f"pi {stats['pi']:+.4f} vf {stats['vf']:.4f}")

        if ep_rewards:
            mean_rew = np.mean(ep_rewards[-100:])
            if mean_rew > best_mean:
                best_mean = mean_rew
                ck_path = args.checkpoint or f"rans_ppo_remote_{env._task}.pt"
                torch.save({"policy": policy.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "best_mean_reward": best_mean,
                            "task": env._task}, ck_path)

    env.close()
    print(f"\nTraining complete.  Best mean reward: {best_mean:.3f}")
    print(f"Checkpoint: {args.checkpoint or f'rans_ppo_remote_{env._task}.pt'}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    env = RemoteRANSGymnasiumEnv(base_url=args.url)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    ck = torch.load(args.checkpoint, map_location="cpu")
    policy = ActorCritic(obs_dim, act_dim)
    policy.load_state_dict(ck["policy"])
    policy.eval()

    print(f"\nEvaluating {args.checkpoint} against {args.url}")
    print(f"  task: {env._task} | best training reward: {ck.get('best_mean_reward', '?'):.3f}")
    print("=" * 60)

    for ep in range(args.eval_episodes):
        obs_np, _ = env.reset()
        total_r, steps = 0.0, 0
        while True:
            action = policy.act_deterministic(
                torch.from_numpy(obs_np).float()
            ).numpy()
            obs_np, r, term, trunc, info = env.step(action)
            total_r += r; steps += 1
            if term or trunc: break
        print(f"  ep {ep+1:2d} | steps {steps:4d} | reward {total_r:.3f} | "
              f"goal {info.get('goal_reached', '?')}")
    env.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="RANS PPO training via OpenEnv client")
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--n-steps", type=int, default=1024)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--log-interval", type=int, default=5)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--eval", action="store_true")
    p.add_argument("--eval-episodes", type=int, default=10)
    args = p.parse_args()

    if args.eval:
        if not args.checkpoint:
            print("--eval requires --checkpoint"); sys.exit(1)
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
