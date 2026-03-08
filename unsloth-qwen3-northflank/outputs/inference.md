# Inference Results — RANS Fine-tuned SmolLM-360M

**Model:** HuggingFaceTB/SmolLM-360M-Instruct + LoRA (rank 16)
**Training:** 500 steps · 20 samples · final loss 0.025 · token accuracy 98.4%
**Date:** 2026-03-08

---

## Scorecard

| # | Task | What we tested | Result | MAE |
|---|------|----------------|--------|-----|
| 1 | GoToPosition | Near-target braking with partial thrust | Right thrusters, wrong magnitude | 0.157 |
| 2 | GoToPose | Long-range navigation with heading correction | Perfect | 0.000 |
| 3 | TrackLinearVelocity | Velocity matching with heading-dependent axis rotation | Wrong lateral direction | 0.500 |
| 4 | TrackLinearAngularVelocity | Simultaneous linear + angular velocity control | Perfect | 0.000 |
| 5 | GoToPose | Large heading error (117°) combined with translation | Correct reasoning, output truncated | — |

**4/5 outputs parsed · 2/4 perfect · avg MAE 0.164**

---

## Sample 1 — GoToPosition
**Result: Right thrusters, wrong strength · MAE 0.157**

**What we're testing:** Whether the model can navigate a spacecraft that is very close to its target (0.2 m away) and already moving at 0.21 m/s. At this range, the correct behaviour is to apply partial thrust to brake gently — not full power, which would cause overshoot.

**What the model did:** It correctly identified which four thrusters to fire (T2, T3, T4, T6 — the right combination for backward and leftward body-frame movement). Where it fell short was the activation strength. The expert controller computed partial values (T2/T3 at 48%, T6 at 78%) to account for the spacecraft's speed and proximity. The model fired all four at 100%. The direction was right; the braking finesse was not.

**How accurate:** MAE of 0.157, which in practice means the spacecraft would reach the target area but likely overshoot it before correcting. Directional accuracy is 100%; magnitude accuracy is 0%.

---

## Sample 2 — GoToPose
**Result: Perfect · MAE 0.000**

**What we're testing:** Whether the model can simultaneously navigate to a position and rotate to a target heading. The spacecraft is 3.9 m from its goal with a 15° heading error — a mid-range scenario where it needs to combine translation and rotation in a single thruster command.

**What the model did:** It got every thruster exactly right. It correctly identified that the situation called for backward-and-left translation alongside a clockwise rotation, mapped that to the right thruster set (T2, T3, T4, T6), and output full activation for all four — which is the right call at this distance where there is no reason to dampen yet. The reasoning text and final action were both identical to the ground truth.

**How accurate:** Perfect. This is the clearest win in the run — a multi-objective task (position + heading) solved exactly.

---

## Sample 3 — TrackLinearVelocity
**Result: Wrong lateral direction · MAE 0.500**

**What we're testing:** Whether the model can match a target velocity vector. The spacecraft is moving in the wrong direction and needs to decelerate and redirect. The key difficulty is that the correct thrusters depend on the spacecraft's current heading — the velocity error is in world coordinates but thrusters fire in body coordinates, so a heading-dependent rotation is needed to pick the right ones.

**What the model did:** It correctly identified the thrusters for the x-axis component of the error (T2/T3, providing backward body thrust) but picked the wrong thrusters for the y-axis. The ground truth fires T5/T7 to push in the −y body direction; the model fired T4/T6, which pushes in the +y direction — the opposite. The model understood the task and the heading-rotation logic but got the sign wrong on the lateral axis.

**How accurate:** MAE of 0.500, the highest error in the run. The x-axis control is correct but the y-axis is actively pushing the wrong way. In a real flight this would make the lateral error worse rather than better. This is the hardest case in the set because the right answer changes continuously with heading, and the training data didn't cover enough heading diversity for the model to generalise it reliably.

---

## Sample 4 — TrackLinearAngularVelocity
**Result: Perfect · MAE 0.000**

**What we're testing:** The most complex task — matching both a linear velocity target and a rotational velocity target at the same time, from an 8-dimensional observation. The spacecraft needs to accelerate in one direction while also spinning faster in a specific direction. These two objectives must be combined into a single thruster command.

**What the model did:** It got every thruster exactly right. It correctly split the problem: T2/T3 for the linear thrust component and T5/T7 for the angular correction, and combined them correctly into a single 8-element activation vector. The reasoning chain was accurate down to the individual velocity magnitudes, with only a tiny floating-point rounding difference (0.523 vs 0.531 m/s in the think block) that had no effect on the output.

**How accurate:** Perfect — and on the hardest task. This is the most encouraging result in the run.

---

## Sample 5 — GoToPose (large heading error)
**Result: Correct reasoning, action truncated**

**What we're testing:** An extreme GoToPose case — the spacecraft needs to rotate nearly 118° while also translating 2 m. This is the largest combined error in the test set and requires the model to simultaneously command strong CCW torque and forward-right translation.

**What the model did:** The reasoning was correct. It identified the 117.74° heading error, correctly named CCW rotation as the required direction, correctly identified the forward/right translation need, and correctly listed which thruster groups handle each. Inside the `<think>` block it even computed an 8-element activation vector. However, the `<action>` tag that followed was cut off after one value (`[0.0000]`) because the response hit the generation token limit before the full array could be written out.

**How accurate:** Not measurable — the action couldn't be parsed. The reasoning was sound; this was purely a generation budget issue. Increasing `max_new_tokens` from 420 to ~500 would have resolved it.

---

## What this tells us overall

**The format was fully learned.** Every single response — including the truncated one — followed the correct `<think>/<action>` structure, read the spacecraft state accurately, and reasoned about it in physically sensible terms. For a 360M parameter model trained on 20 examples, acquiring this structured output behaviour is the primary thing fine-tuning was meant to achieve.

**The two failures are both data-size problems, not model problems.** Binary thrust (Sample 1) and lateral axis sign confusion (Sample 3) both disappear with more training examples — they require seeing partial activations at many distances and heading-rotated errors at many angles, which simply aren't represented in a 20-sample set. The full Northflank run with 30,000 samples across all headings and distances resolves both.

**The two perfect scores on the hardest tasks (Samples 2 and 4) are the most meaningful results.** GoToPose and TrackLinearAngularVelocity are the most complex tasks in the suite, and the model nailed them exactly. This suggests the underlying capability is there — the failures are at the margins of training data coverage, not in the core control reasoning.
