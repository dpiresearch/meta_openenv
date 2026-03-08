---
title: RANS Spacecraft Navigation Environment
emoji: 🛸
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - robotics
  - spacecraft
---

# RANS — OpenEnv Environment

**RANS: Reinforcement Learning based Autonomous Navigation for Spacecrafts**

OpenEnv-compatible implementation of the paper:

> El-Hariry, Richard, Olivares-Mendez (2023).
> *"RANS: Highly-Parallelised Simulator for Reinforcement Learning based Autonomous Navigating Spacecrafts."*
> [arXiv:2310.07393](https://arxiv.org/abs/2310.07393)

Original GPU implementation (Isaac Gym): [elharirymatteo/RANS](https://github.com/elharirymatteo/RANS)

**Live HuggingFace Space:** https://huggingface.co/spaces/dpang/rans-env

---

## Overview

This package wraps a pure-Python/NumPy 2-D spacecraft physics simulation (no Isaac Gym required) into an [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compatible environment.  The server can run inside a standard Docker container on CPU and exposes the standard OpenEnv HTTP/WebSocket API.

### Supported Tasks

| Task | Description | Obs size | Reward |
|------|-------------|----------|--------|
| `GoToPosition` | Reach target (x, y) | 6 | exp(−‖Δp‖²/2σ²) |
| `GoToPose` | Reach target (x, y, θ) | 7 | weighted position + heading |
| `TrackLinearVelocity` | Maintain (vx, vy) | 6 | exp(−‖Δv‖²/2σ²) |
| `TrackLinearAngularVelocity` | Maintain (vx, vy, ω) | 8 | weighted linear + angular |

### Spacecraft Model

- **Platform**: 2-D rigid body (MFP2D — Modular Floating Platform)
- **State**: `[x, y, θ, vx, vy, ω]`
- **Thrusters**: 8-thruster default layout (configurable)
- **Action**: continuous activation ∈ [0, 1] per thruster
- **Integration**: Euler, 50 Hz (dt = 0.02 s)

---

## Quick Start

### Run locally (no Docker)

```bash
pip install -e ".[dev]"
RANS_TASK=GoToPosition uvicorn rans_env.server.app:app --host 0.0.0.0 --port 8000
```

### Client usage (async)

```python
import asyncio
from rans_env import RANSEnv, SpacecraftAction

async def main():
    async with RANSEnv(base_url="http://localhost:8000") as env:
        obs = await env.reset()
        print(f"Task: {obs.task}")
        print(f"Initial obs: {obs.state_obs}")

        n = len(obs.thruster_masks)  # 8 thrusters
        result = await env.step(SpacecraftAction(thrusters=[0.0] * n))
        print(f"Reward: {result.reward:.4f},  Done: {result.done}")

asyncio.run(main())
```

### Client usage (synchronous)

```python
from rans_env import RANSEnv, SpacecraftAction

with RANSEnv(base_url="http://localhost:8000").sync() as env:
    obs = env.reset()
    for _ in range(500):
        n = len(obs.thruster_masks)
        result = env.step(SpacecraftAction(thrusters=[0.5] * n))
        obs = result.observation
        if result.done:
            obs = env.reset()
```

### Docker

```bash
# Build
docker build -f server/Dockerfile -t rans-env .

# Run GoToPose task
docker run -e RANS_TASK=GoToPose -p 8000:8000 rans-env
```

---

## Project Structure

```
RANS/
├── __init__.py                  # Public API: RANSEnv, SpacecraftAction, ...
├── client.py                    # RANSEnv OpenEnv client
├── models.py                    # SpacecraftAction / Observation / State
├── openenv.yaml                 # OpenEnv environment manifest
├── pyproject.toml               # Package configuration
└── server/
    ├── app.py                   # FastAPI entry-point (create_app)
    ├── rans_environment.py      # RANSEnvironment (Environment subclass)
    ├── spacecraft_physics.py    # 2-D rigid-body dynamics (NumPy)
    ├── tasks/
    │   ├── base.py              # BaseTask ABC
    │   ├── go_to_position.py    # GoToPositionTask
    │   ├── go_to_pose.py        # GoToPoseTask
    │   ├── track_linear_velocity.py
    │   └── track_linear_angular_velocity.py
    ├── tests/
    │   ├── test_physics.py      # Physics unit tests
    │   ├── test_tasks.py        # Task unit tests
    │   └── test_environment.py  # Integration tests
    └── Dockerfile
```

---

## Configuration

### Environment variables (Docker / server)

| Variable | Default | Description |
|----------|---------|-------------|
| `RANS_TASK` | `GoToPosition` | Task name |
| `RANS_MAX_STEPS` | `500` | Max steps per episode |

### Task hyper-parameters

Pass a dict to `RANSEnvironment(task_config={...})`:

```python
env = RANSEnvironment(
    task="GoToPosition",
    task_config={
        "tolerance": 0.05,       # success threshold (m)
        "reward_sigma": 0.5,     # Gaussian reward width
        "spawn_max_radius": 5.0, # max target distance (m)
    },
)
```

---

## Observation Format

`SpacecraftObservation` fields:

| Field | Shape | Description |
|-------|-------|-------------|
| `state_obs` | [6–8] | Task-specific error / velocity observations |
| `thruster_transforms` | [8 × 5] | `[px, py, dx, dy, F_max]` per thruster |
| `thruster_masks` | [8] | 1.0 = thruster present |
| `mass` | scalar | Platform mass (kg) |
| `inertia` | scalar | Moment of inertia (kg·m²) |
| `task` | str | Active task name |
| `reward` | scalar | Step reward ∈ [0, 1] |
| `done` | bool | Episode ended |
| `info` | dict | Diagnostics (error values, goal_reached, step) |

---

## Training an RL Agent

Three example scripts cover different training scenarios:

### 1. Sanity check — random agent (`examples/random_agent.py`)

First verify the server is reachable and the environment works:

```bash
# Start server (one terminal)
RANS_TASK=GoToPosition uvicorn rans_env.server.app:app --port 8000

# Run random agent (another terminal)
python examples/random_agent.py --task GoToPosition --episodes 5
```

### 2. PPO training — local, no server (`examples/ppo_train.py`)

Trains a MLP policy with PPO directly against `RANSEnvironment` (no HTTP
server required).  Uses pure PyTorch — no additional RL library needed.

```bash
pip install torch gymnasium

# Train GoToPosition (300 k steps)
python examples/ppo_train.py --task GoToPosition --timesteps 300000

# Train GoToPose
python examples/ppo_train.py --task GoToPose --timesteps 500000

# Evaluate a saved checkpoint
python examples/ppo_train.py --eval --checkpoint rans_ppo_GoToPosition.pt \
       --task GoToPosition --eval-episodes 20
```

Key hyper-parameters (all match the original RANS paper):

| Flag | Default | Description |
|------|---------|-------------|
| `--n-steps` | 2048 | Rollout length per update |
| `--n-epochs` | 10 | PPO epochs per rollout |
| `--gamma` | 0.99 | Discount factor |
| `--lam` | 0.95 | GAE-λ |
| `--clip-eps` | 0.2 | PPO clipping |
| `--lr` | 3e-4 | Adam learning rate |

### 3. Gymnasium wrapper — use with any RL library (`examples/gymnasium_wrapper.py`)

Wraps `RANSEnvironment` as a `gymnasium.Env` for compatibility with
Stable-Baselines3, CleanRL, RLlib, TorchRL, etc:

```python
from examples.gymnasium_wrapper import make_rans_env

env = make_rans_env(task="GoToPosition")
print(env.observation_space)   # Box(56,)
print(env.action_space)        # Box(8,)  — thruster activations in [0, 1]

# Stable-Baselines3
from stable_baselines3 import PPO, SAC

model = PPO("MlpPolicy", env, verbose=1, n_steps=2048)
model.learn(total_timesteps=500_000)
model.save("rans_sb3_ppo")

# Or SAC for off-policy training
model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500_000)
```

### 4. Remote training via OpenEnv client (`examples/openenv_client_train.py`)

Train against a running Docker server using `N` concurrent WebSocket
sessions (the canonical OpenEnv pattern):

```bash
# Start server
docker run -e RANS_TASK=GoToPosition -p 8000:8000 rans-env

# Train with 4 parallel environment sessions
python examples/openenv_client_train.py --url http://localhost:8000 \
       --n-envs 4 --episodes 50
```

### Observation & action spaces

| | |
|---|---|
| **Observation** | Flat vector: `[state_obs, thruster_transforms (flat), masks, mass, inertia]` |
| **Action** | `float32[8]` — thruster activations ∈ [0, 1] |
| **Reward** | Scalar ∈ [0, 1] — exponential decay from target error |
| **Done** | `True` when goal reached **or** step limit hit |

Observation sizes by task:

| Task | `state_obs` | total obs dim |
|------|------------|---------------|
| GoToPosition | 6 | 56 |
| GoToPose | 7 | 57 |
| TrackLinearVelocity | 6 | 56 |
| TrackLinearAngularVelocity | 8 | 58 |

---

## Tests

```bash
pip install -e ".[dev]"
pytest server/tests/ -v
```

---

## Citation

```bibtex
@misc{elhariry2023rans,
  title   = {RANS: Highly-Parallelised Simulator for Reinforcement Learning
             based Autonomous Navigating Spacecrafts},
  author  = {El-Hariry, Matteo and Richard, Antoine and Olivares-Mendez, Miguel},
  year    = {2023},
  eprint  = {2310.07393},
  archivePrefix = {arXiv},
}
```
