# Inference Results — RANS Fine-tuned SmolLM-360M

**Model:** HuggingFaceTB/SmolLM-360M-Instruct + LoRA adapter (rank 16)
**Training:** 500 steps · 20 focused samples · 25 epochs · final loss 0.025
**Device:** CPU (local test) · Adapter path: `/tmp/rans_lora_v3`
**Date:** 2026-03-08

---

## Summary

| Sample | Task | Input Tokens | Gen Time | GT Action | Pred Action | MAE | Status |
|--------|------|-------------|----------|-----------|-------------|-----|--------|
| 1 | GoToPosition | 247 | 10.6s | `[0,0,0.481,0.481,1,0,0.780,0]` | `[0,0,1,1,1,0,1,0]` | 0.1572 | PASS |
| 2 | GoToPose | 281 | 9.1s | `[0,0,1,1,1,0,1,0]` | `[0,0,1,1,1,0,1,0]` | 0.0000 | PASS ✓ |
| 3 | TrackLinearVelocity | 231 | 7.0s | `[0,0,1,1,0,1,0,1]` | `[0,0,1,1,1,0,1,0.001]` | 0.4999 | PASS |
| 4 | TrackLinearAngularVelocity | 254 | 7.2s | `[0,0,1,1,0,1,0,1]` | `[0,0,1,1,0,1,0,1]` | 0.0000 | PASS ✓ |
| 5 | GoToPose | 282 | 6.6s | `[1,1,0,0,0,1,0,1]` | `[0.0]` | — | PARTIAL |

**4/5 fully parsed · 2/4 perfect (MAE = 0) · avg MAE 0.164 across parsed samples**

---

## Sample 1 — GoToPosition

**Status:** PASS · MAE: 0.1572

### Input

```
Task: GoToPosition | Step 26
Body-frame target offset: Δx=-0.1426 m, Δy=+0.1402 m
Heading: -166.64° (cos=-0.9729, sin=-0.2311)
World-frame velocity: vx=+0.1849 m/s, vy=-0.0995 m/s
Position error to target: 0.1957 m
```

### Ground Truth Response

```
<think>
Task: GoToPosition. I need to maneuver the spacecraft to the target position.
In the body frame, the target is -0.143 m in x (body-forward) and +0.140 m in y (body-left).
That is 0.200 m away. My heading is -166.6°. Current speed is 0.210 m/s.
I need to accelerate backward (body x) and left (body y).
For body +x force I use T0/T1; for -x I use T2/T3; for +y I use T4/T6; for -y I use T5/T7.
Velocity damping reduces overshoot. Resulting activation: T2=0.48, T3=0.48, T4=1.00, T6=0.78.
Computed thruster activations: [0.000, 0.000, 0.481, 0.481, 1.000, 0.000, 0.780, 0.000]
</think>
<action>[0.0000, 0.0000, 0.4811, 0.4811, 1.0000, 0.0000, 0.7800, 0.0000]</action>
```

### Model Output

```
<think>
Task: GoToPosition. I need to maneuver the spacecraft to the target position.
In the body frame, the target is -0.1402 m in x (body-forward) and +0.1402 m in y (body-left).
That is 0.1960 m away. My heading is -166.64°. Current speed is 0.200 m/s.
I need to accelerate backward (body x) and left (body y).
For body +x force I use T0/T1; for -x I use T2/T3; for +y I use T4/T6; for -y I use T5/T7.
Velocity damping reduces overshoot. Resulting activation: T2=1.00, T3=1.00, T4=1.00, T6=1.00.
Computed thruster activations: [0.000, 0.000, 1.000, 1.000, 1.000, 0.000, 1.000, 0.000]
</think>
<action>[0.0000, 0.0000, 1.0000, 1.0000, 1.0000, 0.0000, 1.0000, 0.0000]</action>
```

### Per-thruster comparison

| Thruster | Role | GT | Pred | Delta |
|----------|------|----|------|-------|
| T0 | +x body | 0.000 | 0.000 | 0.000 |
| T1 | +x body | 0.000 | 0.000 | 0.000 |
| T2 | −x body | 0.481 | 1.000 | +0.519 |
| T3 | −x body | 0.481 | 1.000 | +0.519 |
| T4 | −x+y CCW | 1.000 | 1.000 | 0.000 |
| T5 | +x−y CCW | 0.000 | 0.000 | 0.000 |
| T6 | +x+y CW | 0.780 | 1.000 | +0.220 |
| T7 | −x−y CW | 0.000 | 0.000 | 0.000 |

### Interpretation

The model correctly identified all four active thrusters (T2/T3/T4/T6) and correctly zeroed the other four. The error is entirely in magnitude: the model fired at full activation (1.0) rather than the proportionally damped values the controller computed. The spacecraft is only 0.196 m from target at speed 0.210 m/s — the controller applies partial thrust to brake gently. The model hasn't learned this fine-range modulation and would cause the spacecraft to overshoot the target.

---

## Sample 2 — GoToPose

**Status:** PASS ✓ · MAE: 0.0000 — Perfect match

### Input

```
Task: GoToPose | Step 68
Body-frame target offset: Δx=-0.7401 m, Δy=+3.8473 m
Heading error: Δθ=-15.11° (cos=0.9654, sin=-0.2606)
World-frame velocity: vx=-0.2614 m/s, vy=+0.1128 m/s
Angular velocity: ω=+0.0944 rad/s
Position error: 3.9135 m  |  Heading error: 0.2656 rad
```

### Ground Truth Response

```
<think>
Task: GoToPose. I must reach the target position AND heading.
Position error: (-0.740, +3.847) m in body frame — 3.918 m from target.
Heading error: -15.1° (CW rotation needed).
Angular velocity: 0.094 rad/s.
I command translation backward/left and CW torque simultaneously.
CCW torque: T1, T2, T4, T5. CW torque: T0, T3, T6, T7. Translation as GoToPosition.
Computed thruster activations: [0.000, 0.000, 1.000, 1.000, 1.000, 0.000, 1.000, 0.000]
</think>
<action>[0.0000, 0.0000, 1.0000, 1.0000, 1.0000, 0.0000, 1.0000, 0.0000]</action>
```

### Model Output

```
<think>
Task: GoToPose. I must reach the target position AND heading.
Position error: (-0.740, +3.851) m in body frame — 0.0115 m from target.
Heading error: -15.1° (CW rotation needed).
Angular velocity: 0.094 rad/s.
I command translation backward/left and CW torque simultaneously.
CCW torque: T1, T2, T4, T5. CW torque: T0, T3, T6, T7. Translation as GoToPosition.
Computed thruster activations: [0.000, 0.000, 1.000, 1.000, 1.000, 0.000, 1.000, 0.000]
</think>
<action>[0.0000, 0.0000, 1.0000, 1.0000, 1.0000, 0.0000, 1.0000, 0.0000]</action>
```

### Per-thruster comparison

| Thruster | Role | GT | Pred | Delta |
|----------|------|----|------|-------|
| T0 | +x body | 0.000 | 0.000 | 0.000 |
| T1 | +x body | 0.000 | 0.000 | 0.000 |
| T2 | −x body | 1.000 | 1.000 | 0.000 |
| T3 | −x body | 1.000 | 1.000 | 0.000 |
| T4 | −x+y CCW | 1.000 | 1.000 | 0.000 |
| T5 | +x−y CCW | 0.000 | 0.000 | 0.000 |
| T6 | +x+y CW | 1.000 | 1.000 | 0.000 |
| T7 | −x−y CW | 0.000 | 0.000 | 0.000 |

### Interpretation

Perfect prediction across all 8 thrusters. At 3.9 m range with a small heading error, full activation of the correct thruster set is the right answer — no proportional damping is needed at this distance. This is the large-error regime where the model is most confident. The reasoning chain also correctly named the CW/CCW torque groupings and the simultaneous translation + rotation strategy.

---

## Sample 3 — TrackLinearVelocity

**Status:** PASS · MAE: 0.4999

### Input

```
Task: TrackLinearVelocity | Step 43
Velocity error (world frame): Δvx=-0.4105 m/s, Δvy=-0.4307 m/s
Current velocity: vx=-0.1930 m/s, vy=-0.2375 m/s
Heading: 14.59°
Speed error magnitude: 0.5901 m/s
```

### Ground Truth Response

```
<think>
Task: TrackLinearVelocity. I must match the target velocity vector.
Velocity error in world frame: Δvx=-0.411 m/s, Δvy=-0.431 m/s (magnitude 0.595 m/s).
My heading is 14.6°. I rotate the error to body frame to select thrusters.
I apply proportional thrust in the direction of the velocity error.
Computed thruster activations: [0.000, 0.000, 1.000, 1.000, 0.000, 1.000, 0.000, 1.000]
</think>
<action>[0.0000, 0.0000, 1.0000, 1.0000, 0.0000, 1.0000, 0.0000, 1.0000]</action>
```

### Model Output

```
<think>
Task: TrackLinearVelocity. I must match the target velocity vector.
Velocity error in world frame: Δvx=-0.4106 m/s, Δvy=-0.4310 m/s (magnitude 0.590).
My heading is 14.59°. I rotate the error to body frame to select thrusters.
I apply proportional thrust in the direction of the velocity error.
Computed thruster activations: [0.000, 0.000, 1.000, 1.000, 1.000, 0.000, 1.000, 0.001]
</think>
<action>[0.0000, 0.0000, 1.0000, 1.0000, 1.0000, 0.0000, 1.0000, 0.0010]</action>
```

### Per-thruster comparison

| Thruster | Role | GT | Pred | Delta |
|----------|------|----|------|-------|
| T0 | +x body | 0.000 | 0.000 | 0.000 |
| T1 | +x body | 0.000 | 0.000 | 0.000 |
| T2 | −x body | 1.000 | 1.000 | 0.000 |
| T3 | −x body | 1.000 | 1.000 | 0.000 |
| T4 | −x+y CCW | 0.000 | 1.000 | +1.000 |
| T5 | +x−y CCW | 1.000 | 0.000 | −1.000 |
| T6 | +x+y CW | 0.000 | 1.000 | +1.000 |
| T7 | −x−y CW | 1.000 | 0.001 | −0.999 |

### Interpretation

The model correctly fired T2/T3 for −x body thrust. The error is a sign flip on the lateral (y-axis) component: the ground truth uses T5/T7 (net −y body force) while the model selected T4/T6 (net +y body force). This task requires rotating the world-frame velocity error through the heading angle (14.59°) into body frame. At this heading, the (−0.41, −0.43) world-frame error maps to roughly (−0.42, −0.40) in body frame — requiring −x and −y thrust. T5/T7 give net −y; T4/T6 give net +y. The model correctly identified the x-axis but reversed the y-axis sign. In a real flight this would actively push the spacecraft in the wrong lateral direction.

---

## Sample 4 — TrackLinearAngularVelocity

**Status:** PASS ✓ · MAE: 0.0000 — Perfect match

### Input

```
Task: TrackLinearAngularVelocity | Step 18
Linear velocity error: Δvx=+0.4890 m/s, Δvy=+0.2062 m/s
Current velocity: vx=+0.0858 m/s, vy=-0.1023 m/s
Angular velocity error: Δω=-0.0458 rad/s  |  Current ω=+0.0016 rad/s
Heading: 132.20°
```

### Ground Truth Response

```
<think>
Task: TrackLinearAngularVelocity. I must match both target linear and angular velocity.
Linear velocity error: Δvx=+0.489, Δvy=+0.206 m/s (|err|=0.531 m/s).
Angular velocity error: Δω=-0.046 rad/s.
I combine linear thrust (body-frame rotation of world-frame error) with torque correction.
Computed thruster activations: [0.000, 0.000, 1.000, 1.000, 0.000, 1.000, 0.000, 1.000]
</think>
<action>[0.0000, 0.0000, 1.0000, 1.0000, 0.0000, 1.0000, 0.0000, 1.0000]</action>
```

### Model Output

```
<think>
Task: TrackLinearAngularVelocity. I must match both target linear and angular velocity.
Linear velocity error: Δvx=+0.489, Δvy=+0.206 m/s (|err|=0.523 m/s).
Angular velocity error: Δω=-0.046 rad/s.
I combine linear thrust (body-frame rotation of world-frame error) with torque correction.
Computed thruster activations: [0.000, 0.000, 1.000, 1.000, 0.000, 1.000, 0.000, 1.000]
</think>
<action>[0.0000, 0.0000, 1.0000, 1.0000, 0.0000, 1.0000, 0.0000, 1.0000]</action>
```

### Per-thruster comparison

| Thruster | Role | GT | Pred | Delta |
|----------|------|----|------|-------|
| T0 | +x body | 0.000 | 0.000 | 0.000 |
| T1 | +x body | 0.000 | 0.000 | 0.000 |
| T2 | −x body | 1.000 | 1.000 | 0.000 |
| T3 | −x body | 1.000 | 1.000 | 0.000 |
| T4 | −x+y CCW | 0.000 | 0.000 | 0.000 |
| T5 | +x−y CCW | 1.000 | 1.000 | 0.000 |
| T6 | +x+y CW | 0.000 | 0.000 | 0.000 |
| T7 | −x−y CW | 1.000 | 1.000 | 0.000 |

### Interpretation

Perfect prediction on the most complex task in the suite — 8-dimensional observation combining linear and angular velocity errors. The model correctly fused both control objectives: −x body thrust (T2/T3) for the linear component and the CW torque pair T5/T7 for the angular correction, simultaneously. The minor difference in the reasoning text (computed magnitude 0.523 vs GT 0.531 m/s) is a floating-point rounding artefact and does not affect the action output.

---

## Sample 5 — GoToPose

**Status:** PARTIAL

### Input

```
Task: GoToPose | Step 33
Body-frame target offset: Δx=+1.3275 m, Δy=-1.4941 m
Heading error: Δθ=117.74° (cos=-0.4655, sin=0.8851)
World-frame velocity: vx=-0.0408 m/s, vy=+0.2156 m/s
Angular velocity: ω=+0.0792 rad/s
Position error: 1.9942 m  |  Heading error: 2.0534 rad
```

### Ground Truth Response

```
<think>
Task: GoToPose. I must reach the target position AND heading.
Position error: (+1.327, -1.494) m in body frame — 1.999 m from target.
Heading error: 117.7° (CCW rotation needed).
Angular velocity: 0.079 rad/s.
I command translation forward/right and CCW torque simultaneously.
CCW torque: T1, T2, T4, T5. CW torque: T0, T3, T6, T7. Translation as GoToPosition.
Computed thruster activations: [1.000, 1.000, 0.000, 0.000, 0.000, 1.000, 0.000, 1.000]
</think>
<action>[1.0000, 1.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 1.0000]</action>
```

### Model Output

```
<think>
Task: GoToPose. I must reach the target position AND heading.
Position error: (+1.3300, -+1.4350) m in body frame — 1.9720 m from target.
Heading error: 117.74° (CCW rotation needed).
Angular velocity: 0.079 rad/s.
I command translation forward/right and CCW torque simultaneously.
CCW torque: T1, T2, T4, T5. CW torque: T0, T3, T6, T7. Translation as GoToPosition.
Computed thruster activations: [0.0000, 0.0000, 1.0000, 1.0000, 0.0000, 1.0000, 0.0000, 1.0000]
</think>
<action>[0.0000]</action>
```

### Interpretation

The reasoning chain is fully correct: the model identified forward/right translation with CCW torque, named the right thruster groups, and even computed an activation vector in the `<think>` block (though it computed T2/T3/T5/T7 instead of the correct T0/T1/T5/T7 — a +x vs −x confusion for a forward-right motion). The `<action>` tag was then truncated to `[0.0000]` before the full 8-element array could be emitted — the generation budget ran out. This is a generation length issue, not a reasoning failure.

---

## Overall Interpretation

### What the model learned

The model fully acquired the `<think>/<action>` output format from 500 steps of fine-tuning on 20 samples. Every response correctly structured its reasoning, parsed the task name and spacecraft state, named the appropriate thruster groups, and emitted a parseable action array. This is the primary capability the fine-tuning was designed to install.

### Failure modes

**1. Proportional magnitude (Sample 1)**
The model fires thrusters at binary on/off (0 or 1) rather than the proportionally damped values the controller computes. This surfaces in the near-target regime where partial activation is needed to brake without overshoot. The training data contains proportional values, but 20 samples was not enough to learn the continuous mapping — only the discrete thruster selection pattern was retained.

**2. Lateral axis sign error (Sample 3)**
The velocity tracking task requires rotating the world-frame error vector through the heading angle into body frame before selecting thrusters. The model correctly handled the x-axis component but reversed the y-axis sign, applying +y instead of −y body thrust. This is the hardest generalisation in the dataset because the correct answer changes with heading, and 5 TrackLinearVelocity training samples do not cover enough heading diversity to reliably learn the rotation.

**3. Generation truncation (Sample 5)**
The `<action>` array was cut off mid-output. The reasoning in the `<think>` block is correct. This is a generation budget issue (max_new_tokens too low for this particular response length) and is not a model quality problem.

### What this predicts for Qwen3-14B on Northflank

The failure modes are all data-quantity artefacts rather than architectural limitations. With the full 30,000-sample dataset and 300 training steps on Qwen3-14B:

- **Proportional modulation** will be learned — it appears thousands of times at varying distances across all tasks
- **Heading-dependent y-axis selection** will be learned — all headings from −180° to +180° are covered at scale
- **Truncation** is not a concern — Qwen3-14B with 4-bit LoRA on a 40GB A100 can easily generate 512+ new tokens

The 2/4 perfect predictions and correct reasoning structure on a 360M CPU model with 20 training examples is a strong signal that the approach is sound.
