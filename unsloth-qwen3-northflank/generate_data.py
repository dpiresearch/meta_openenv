"""
generate_data.py
================
Collect expert spacecraft control trajectories from the RANS environment and
convert them into Qwen3-style conversational training data (SFT format).

Controller strategy
-------------------
For each task a proportional controller is derived from the pseudoinverse of
the thruster force/torque matrix.  A velocity-damping term prevents overshoot.
This gives physically sensible actions without requiring a pre-trained RL agent.

Output format (JSONL)
---------------------
Each line is one control decision rendered as a multi-turn conversation:
  {"messages": [
    {"role": "system",    "content": SYSTEM_PROMPT},
    {"role": "user",      "content": "<spacecraft state + task description>"},
    {"role": "assistant", "content": "<think>reasoning</think>\n<action>...</action>"}
  ]}

Usage (standalone)
------------------
    python generate_data.py                       # uses defaults
    RANS_DATA_EPISODES=200 python generate_data.py
    RANS_TASKS=GoToPosition,GoToPose python generate_data.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ── RANS imports (works whether installed as package or run from repo root) ──
_REPO_ROOT = Path(__file__).parent.parent / "RANS"
if _REPO_ROOT.exists():
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from server.rans_environment import RANSEnvironment
    from server.spacecraft_physics import Spacecraft2D, SpacecraftConfig
except ImportError:
    from rans_env.server.rans_environment import RANSEnvironment
    from rans_env.server.spacecraft_physics import Spacecraft2D, SpacecraftConfig

# ─── Configuration ────────────────────────────────────────────────────────────

TASKS         = os.environ.get("RANS_TASKS", "GoToPosition,GoToPose,TrackLinearVelocity,TrackLinearAngularVelocity").split(",")
EPISODES      = int(os.environ.get("RANS_DATA_EPISODES", "100"))       # per task
MAX_STEPS     = int(os.environ.get("RANS_MAX_STEPS",     "200"))
MIN_REWARD    = float(os.environ.get("RANS_MIN_REWARD",  "0.05"))      # filter low-quality steps
OUTPUT_FILE   = os.environ.get("RANS_DATA_OUTPUT",       "/data/rans_trajectories.jsonl")
SEED          = int(os.environ.get("RANS_SEED",          "42"))

np.random.seed(SEED)

# ─── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Spacecraft navigation AI. 8 thrusters, activations in [0,1].
T0/T1=+x, T2/T3=-x, T4=(-x+y CCW), T5=(+x-y CCW), T6=(+x+y CW), T7=(-x-y CW).
Reason inside <think>...</think>, then output <action>[t0,t1,t2,t3,t4,t5,t6,t7]</action>."""

# ─── Proportional controller ─────────────────────────────────────────────────

def _build_force_matrix(config: SpacecraftConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the body-frame force-torque matrix A such that:
        [Fx_body; Fy_body; τ] = A @ activations

    Returns (A [3×N], pinv_A [N×3]).
    """
    n = len(config.thrusters)
    A = np.zeros((3, n), dtype=np.float64)
    for i, t in enumerate(config.thrusters):
        A[0, i] = t.direction[0] * t.force_max    # body Fx
        A[1, i] = t.direction[1] * t.force_max    # body Fy
        A[2, i] = (
            t.position[0] * t.direction[1] - t.position[1] * t.direction[0]
        ) * t.force_max                             # torque
    return A, np.linalg.pinv(A)


def _proportional_thrusters(
    desired_body_force: np.ndarray,  # [Fx_body, Fy_body, τ]
    pinv_A: np.ndarray,
    scale: float = 1.5,
) -> List[float]:
    """Map desired body-frame force/torque to thruster activations in [0,1]."""
    raw = pinv_A @ desired_body_force * scale
    activations = np.clip(raw, 0.0, 1.0)
    return activations.tolist()


def _compute_action_goto_position(
    obs: np.ndarray,
    pinv_A: np.ndarray,
    mass: float,
    kp: float = 3.0,
    kd: float = 1.5,
) -> List[float]:
    """
    obs = [Δx_body, Δy_body, cos(θ), sin(θ), vx, vy]
    Proportional-derivative controller in body frame.
    """
    dx_b, dy_b = obs[0], obs[1]
    theta = math.atan2(obs[3], obs[2])
    vx_w, vy_w = obs[4], obs[5]
    # rotate world velocity to body frame
    c, s = math.cos(theta), math.sin(theta)
    vx_b = c * vx_w + s * vy_w
    vy_b = -s * vx_w + c * vy_w

    fx_b = kp * mass * dx_b - kd * mass * vx_b
    fy_b = kp * mass * dy_b - kd * mass * vy_b
    return _proportional_thrusters(np.array([fx_b, fy_b, 0.0]), pinv_A)


def _compute_action_goto_pose(
    obs: np.ndarray,
    pinv_A: np.ndarray,
    mass: float,
    inertia: float,
    kp: float = 3.0,
    kd: float = 1.5,
    kr: float = 2.0,
    kdr: float = 1.0,
) -> List[float]:
    """
    obs = [Δx_body, Δy_body, cos(Δθ), sin(Δθ), vx, vy, ω]
    """
    dx_b, dy_b = obs[0], obs[1]
    d_theta = math.atan2(obs[3], obs[2])
    # approximate body velocity
    theta_approx = 0.0  # obs doesn't include absolute heading directly
    # use world velocities as proxy (small heading error assumption)
    vx_b, vy_b = obs[4], obs[5]
    omega = obs[6]

    fx_b = kp * mass * dx_b - kd * mass * vx_b
    fy_b = kp * mass * dy_b - kd * mass * vy_b
    tau  = kr * inertia * d_theta - kdr * inertia * omega
    return _proportional_thrusters(np.array([fx_b, fy_b, tau]), pinv_A)


def _compute_action_track_vel(
    obs: np.ndarray,
    pinv_A: np.ndarray,
    mass: float,
    kp: float = 4.0,
) -> List[float]:
    """
    obs = [Δvx, Δvy, cos(θ), sin(θ), vx, vy]
    Δv = target_vel - current_vel (world frame)
    Rotate error to body frame.
    """
    dvx_w, dvy_w = obs[0], obs[1]
    theta = math.atan2(obs[3], obs[2])
    c, s = math.cos(theta), math.sin(theta)
    dvx_b = c * dvx_w + s * dvy_w
    dvy_b = -s * dvx_w + c * dvy_w

    fx_b = kp * mass * dvx_b
    fy_b = kp * mass * dvy_b
    return _proportional_thrusters(np.array([fx_b, fy_b, 0.0]), pinv_A)


def _compute_action_track_vel_angular(
    obs: np.ndarray,
    pinv_A: np.ndarray,
    mass: float,
    inertia: float,
    kp: float = 4.0,
    kr: float = 3.0,
) -> List[float]:
    """
    obs = [Δvx, Δvy, cos(θ), sin(θ), vx, vy, ω, Δω]  (8 values)
    """
    dvx_w, dvy_w = obs[0], obs[1]
    theta = math.atan2(obs[3], obs[2])
    c, s = math.cos(theta), math.sin(theta)
    dvx_b = c * dvx_w + s * dvy_w
    dvy_b = -s * dvx_w + c * dvy_w
    d_omega = obs[7] if len(obs) > 7 else 0.0

    fx_b = kp * mass * dvx_b
    fy_b = kp * mass * dvy_b
    tau  = kr * inertia * d_omega
    return _proportional_thrusters(np.array([fx_b, fy_b, tau]), pinv_A)


TASK_CONTROLLERS = {
    "GoToPosition":              _compute_action_goto_position,
    "GoToPose":                  _compute_action_goto_pose,
    "TrackLinearVelocity":       _compute_action_track_vel,
    "TrackLinearAngularVelocity": _compute_action_track_vel_angular,
}

# ─── Reasoning text generation ────────────────────────────────────────────────

def _direction_word(value: float, pos_word: str, neg_word: str, threshold: float = 0.05) -> str:
    if abs(value) < threshold:
        return "aligned"
    return pos_word if value > 0 else neg_word


def _generate_reasoning(task: str, obs: np.ndarray, action: List[float], info: Dict[str, Any]) -> str:
    """
    Generate physics-grounded chain-of-thought reasoning for a given observation.
    """
    lines: List[str] = []

    if task == "GoToPosition":
        dx_b, dy_b = obs[0], obs[1]
        cos_t, sin_t = obs[2], obs[3]
        vx_w, vy_w  = obs[4], obs[5]
        theta_deg    = math.degrees(math.atan2(sin_t, cos_t))
        dist         = math.sqrt(dx_b**2 + dy_b**2)
        speed        = math.sqrt(vx_w**2 + vy_w**2)

        fwd  = _direction_word(dx_b,  "forward",  "backward")
        side = _direction_word(dy_b,  "left",     "right")

        lines.append(f"Task: GoToPosition. I need to maneuver the spacecraft to the target position.")
        lines.append(f"In the body frame, the target is {dx_b:+.3f} m in x (body-forward) and {dy_b:+.3f} m in y (body-left).")
        lines.append(f"That is {dist:.3f} m away. My heading is {theta_deg:.1f}°. Current speed is {speed:.3f} m/s.")
        if dist > 0.1:
            lines.append(f"I need to accelerate {fwd} (body x) and {side} (body y).")
        else:
            lines.append("I am very close to the target. Apply braking to stop.")

        # Thruster reasoning
        t_str = ", ".join(f"T{i}={v:.2f}" for i, v in enumerate(action) if v > 0.05)
        lines.append(f"For body +x force I use T0/T1; for -x I use T2/T3; for +y I use T4/T6; for -y I use T5/T7.")
        lines.append(f"Velocity damping reduces overshoot. Resulting activation: {t_str if t_str else 'coast (all ≈ 0)'}.")

    elif task == "GoToPose":
        dx_b, dy_b  = obs[0], obs[1]
        d_theta      = math.atan2(obs[3], obs[2])
        omega        = obs[6]
        dist         = math.sqrt(dx_b**2 + dy_b**2)

        fwd  = _direction_word(dx_b,  "forward", "backward")
        side = _direction_word(dy_b,  "left",    "right")
        rot  = _direction_word(d_theta, "CCW",    "CW")

        lines.append("Task: GoToPose. I must reach the target position AND heading.")
        lines.append(f"Position error: ({dx_b:+.3f}, {dy_b:+.3f}) m in body frame — {dist:.3f} m from target.")
        lines.append(f"Heading error: {math.degrees(d_theta):.1f}° ({rot} rotation needed).")
        lines.append(f"Angular velocity: {omega:.3f} rad/s.")
        lines.append(f"I command translation {fwd}/{side} and {rot} torque simultaneously.")
        lines.append("CCW torque: T1, T2, T4, T5. CW torque: T0, T3, T6, T7. Translation as GoToPosition.")

    elif task == "TrackLinearVelocity":
        dvx, dvy    = obs[0], obs[1]
        cos_t, sin_t = obs[2], obs[3]
        theta_deg    = math.degrees(math.atan2(sin_t, cos_t))
        speed_err    = math.sqrt(dvx**2 + dvy**2)

        lines.append("Task: TrackLinearVelocity. I must match the target velocity vector.")
        lines.append(f"Velocity error in world frame: Δvx={dvx:+.3f} m/s, Δvy={dvy:+.3f} m/s (magnitude {speed_err:.3f} m/s).")
        lines.append(f"My heading is {theta_deg:.1f}°. I rotate the error to body frame to select thrusters.")
        lines.append("I apply proportional thrust in the direction of the velocity error.")

    elif task == "TrackLinearAngularVelocity":
        dvx, dvy    = obs[0], obs[1]
        speed_err    = math.sqrt(dvx**2 + dvy**2)
        d_omega      = obs[7] if len(obs) > 7 else 0.0

        lines.append("Task: TrackLinearAngularVelocity. I must match both target linear and angular velocity.")
        lines.append(f"Linear velocity error: Δvx={dvx:+.3f}, Δvy={dvy:+.3f} m/s (|err|={speed_err:.3f} m/s).")
        lines.append(f"Angular velocity error: Δω={d_omega:+.3f} rad/s.")
        lines.append("I combine linear thrust (body-frame rotation of world-frame error) with torque correction.")

    action_str = "[" + ", ".join(f"{v:.3f}" for v in action) + "]"
    lines.append(f"Computed thruster activations: {action_str}")
    return "\n".join(lines)


def _format_observation(task: str, obs: np.ndarray, info: Dict[str, Any], step: int) -> str:
    """Format the spacecraft state as a human-readable user message."""
    lines: List[str] = [f"Task: {task} | Step {step}"]

    if task == "GoToPosition":
        dx_b, dy_b  = obs[0], obs[1]
        cos_t, sin_t = obs[2], obs[3]
        vx, vy       = obs[4], obs[5]
        theta         = math.atan2(sin_t, cos_t)
        lines.append(f"Body-frame target offset: Δx={dx_b:+.4f} m, Δy={dy_b:+.4f} m")
        lines.append(f"Heading: {math.degrees(theta):.2f}° (cos={cos_t:.4f}, sin={sin_t:.4f})")
        lines.append(f"World-frame velocity: vx={vx:+.4f} m/s, vy={vy:+.4f} m/s")
        if "position_error_m" in info:
            lines.append(f"Position error to target: {info['position_error_m']:.4f} m")

    elif task == "GoToPose":
        dx_b, dy_b  = obs[0], obs[1]
        d_theta      = math.atan2(obs[3], obs[2])
        vx, vy, omega = obs[4], obs[5], obs[6]
        lines.append(f"Body-frame target offset: Δx={dx_b:+.4f} m, Δy={dy_b:+.4f} m")
        lines.append(f"Heading error: Δθ={math.degrees(d_theta):.2f}° (cos={obs[2]:.4f}, sin={obs[3]:.4f})")
        lines.append(f"World-frame velocity: vx={vx:+.4f} m/s, vy={vy:+.4f} m/s")
        lines.append(f"Angular velocity: ω={omega:+.4f} rad/s")
        if "position_error_m" in info:
            lines.append(f"Position error: {info['position_error_m']:.4f} m  |  Heading error: {info.get('heading_error_rad', 0):.4f} rad")

    elif task == "TrackLinearVelocity":
        dvx, dvy    = obs[0], obs[1]
        cos_t, sin_t = obs[2], obs[3]
        vx, vy       = obs[4], obs[5]
        theta         = math.atan2(sin_t, cos_t)
        lines.append(f"Velocity error (world frame): Δvx={dvx:+.4f} m/s, Δvy={dvy:+.4f} m/s")
        lines.append(f"Current velocity: vx={vx:+.4f} m/s, vy={vy:+.4f} m/s")
        lines.append(f"Heading: {math.degrees(theta):.2f}°")
        if "velocity_error_ms" in info:
            lines.append(f"Speed error magnitude: {info['velocity_error_ms']:.4f} m/s")

    elif task == "TrackLinearAngularVelocity":
        dvx, dvy    = obs[0], obs[1]
        cos_t, sin_t = obs[2], obs[3]
        vx, vy       = obs[4], obs[5]
        omega        = obs[6]
        d_omega      = obs[7] if len(obs) > 7 else 0.0
        theta         = math.atan2(sin_t, cos_t)
        lines.append(f"Linear velocity error: Δvx={dvx:+.4f} m/s, Δvy={dvy:+.4f} m/s")
        lines.append(f"Current velocity: vx={vx:+.4f} m/s, vy={vy:+.4f} m/s")
        lines.append(f"Angular velocity error: Δω={d_omega:+.4f} rad/s  |  Current ω={omega:+.4f} rad/s")
        lines.append(f"Heading: {math.degrees(theta):.2f}°")

    if info.get("goal_reached"):
        lines.append("Note: goal almost reached — apply fine braking.")
    return "\n".join(lines)


# ─── Trajectory collection ────────────────────────────────────────────────────

def collect_trajectories(
    task: str,
    n_episodes: int,
    max_steps: int,
    min_reward: float,
) -> List[Dict]:
    """Run the environment with the proportional controller, return SFT samples."""
    env = RANSEnvironment(
        task=task,
        max_episode_steps=max_steps,
    )
    config   = SpacecraftConfig.default_8_thruster()
    _, pinv_A = _build_force_matrix(config)
    controller = TASK_CONTROLLERS[task]

    samples: List[Dict] = []

    for ep in range(n_episodes):
        obs_obj = env.reset()
        obs     = np.array(obs_obj.state_obs, dtype=np.float32)
        mass    = obs_obj.mass
        inertia = obs_obj.inertia

        for step in range(max_steps):
            # Compute proportional control action
            if task == "GoToPosition":
                action = controller(obs, pinv_A, mass)
            elif task == "GoToPose":
                action = controller(obs, pinv_A, mass, inertia)
            elif task == "TrackLinearVelocity":
                action = controller(obs, pinv_A, mass)
            else:  # TrackLinearAngularVelocity
                action = controller(obs, pinv_A, mass, inertia)

            # Build SpacecraftAction
            from models import SpacecraftAction
            act_obj = SpacecraftAction(thrusters=action)
            obs_next = env.step(act_obj)
            reward   = obs_next.reward
            info     = obs_next.info

            # Only keep steps with meaningful reward signal
            if reward >= min_reward:
                reasoning = _generate_reasoning(task, obs, action, info)
                action_str = "[" + ", ".join(f"{v:.4f}" for v in action) + "]"
                assistant_content = (
                    f"<think>\n{reasoning}\n</think>\n"
                    f"<action>{action_str}</action>"
                )

                samples.append({
                    "messages": [
                        {"role": "system",    "content": SYSTEM_PROMPT},
                        {"role": "user",      "content": _format_observation(task, obs, info, step)},
                        {"role": "assistant", "content": assistant_content},
                    ]
                })

            obs = np.array(obs_next.state_obs, dtype=np.float32)
            if obs_next.done:
                break

        if (ep + 1) % 10 == 0:
            print(f"  [{task}] Episode {ep+1}/{n_episodes} — {len(samples)} samples so far")

    return samples


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with output_path.open("w") as fout:
        for task in TASKS:
            task = task.strip()
            print(f"\nCollecting trajectories for task: {task}")
            samples = collect_trajectories(task, EPISODES, MAX_STEPS, MIN_REWARD)
            for s in samples:
                fout.write(json.dumps(s) + "\n")
            total += len(samples)
            print(f"  → {len(samples)} samples written for {task}")

    print(f"\nDone. Total samples: {total} → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
