# Deploying Unsloth Qwen3-14B × RANS Spacecraft Control to Northflank

## What this does

A two-stage GPU job that:

1. **Data generation** — runs the RANS 2D spacecraft simulator with a
   proportional controller across all four navigation tasks
   (`GoToPosition`, `GoToPose`, `TrackLinearVelocity`, `TrackLinearAngularVelocity`).
   Each environment step becomes one SFT training sample: a conversational
   turn where the LLM reasons about the spacecraft state and outputs the
   optimal 8-thruster command.

2. **Fine-tuning** — loads the generated JSONL, applies Unsloth + LoRA
   fine-tuning of Qwen3-14B (4-bit quantized) on those trajectories.
   The model learns to emit `<think>…reasoning…</think><action>[t0,…,t7]</action>`.

---

## Repository layout

```
meta_openenv/
├── RANS/                          ← spacecraft simulator source
│   ├── server/
│   │   ├── rans_environment.py
│   │   ├── spacecraft_physics.py
│   │   └── tasks/
│   ├── models.py
│   └── __init__.py
└── unsloth-qwen3-northflank/
    ├── Dockerfile                 ← builds the job image
    ├── generate_data.py           ← stage 1: RANS → JSONL
    ├── train.py                   ← stage 2: JSONL → LoRA weights
    ├── pipeline.py                ← orchestrates both stages
    └── DEPLOY.md                  ← this file
```

---

## GPU requirements

| GPU         | VRAM   | Notes                                             |
|-------------|--------|---------------------------------------------------|
| L4          | 24 GB  | Minimum — 4-bit only, small batch                 |
| A100 40 GB  | 40 GB  | **Recommended** — comfortable headroom for 14B 4-bit |
| A100 80 GB  | 80 GB  | Best — can increase batch size or use 8-bit        |

Northflank GPU pricing: L4 ≈ $80/mo | A100 40GB ≈ $274/mo | H100 ≈ $587/mo

---

## Step-by-step deployment

### 1. Push repo to GitHub

```bash
cd /path/to/meta_openenv
git add RANS/ unsloth-qwen3-northflank/
git commit -m "Add RANS × Qwen3 Northflank pipeline"
git push
```

### 2. Create a Northflank project

[northflank.com](https://northflank.com) → **New Project** → name it `rans-qwen3`.

### 3. Store secrets

**Project → Secrets → New Secret Group** named `hf-credentials`:
- `HF_TOKEN` = your HuggingFace token (needed to download Qwen3-14B)

### 4. Create two persistent volumes

| Name           | Mount path  | Size   | Purpose                          |
|----------------|-------------|--------|----------------------------------|
| `rans-data`    | `/data`     | 10 GB  | Generated JSONL trajectories     |
| `rans-output`  | `/output`   | 50 GB  | LoRA adapter weights             |

**Project → New Resource → Volume** for each.

### 5. Create a Build Service

1. **New Service → Build Service**
2. Connect your GitHub repo, branch `main`
3. Build settings:
   - Type: **Dockerfile**
   - Dockerfile path: `unsloth-qwen3-northflank/Dockerfile`
   - Context: `/` (repo root, so COPY RANS/ works)
4. Plan: `nf-compute-20` (2 vCPU, 4 GB — no GPU needed for builds)
5. Click **Build** and wait for the image.

### 6. Create the GPU Job

1. **New Service → Job**
2. **Deployment source**: your build service image
3. **Command**: `python /app/pipeline.py`

**Resources tab:**
- vCPU: 8 cores
- Memory: 32 GB
- Ephemeral storage: 20 GB (HF model cache during download)
- GPU: Enable → **A100 40GB × 1**

**Volumes tab:**
- `rans-data`   → `/data`
- `rans-output` → `/output`

**Environment variables:**

| Variable              | Value                                  | Notes                          |
|-----------------------|----------------------------------------|--------------------------------|
| `RANS_TASKS`          | `GoToPosition,GoToPose,TrackLinearVelocity,TrackLinearAngularVelocity` | Tasks to generate data for |
| `RANS_DATA_EPISODES`  | `100`                                  | Episodes per task              |
| `RANS_MAX_STEPS`      | `200`                                  | Max steps per episode          |
| `RANS_MIN_REWARD`     | `0.05`                                 | Min reward to include a step   |
| `RANS_DATA_OUTPUT`    | `/data/rans_trajectories.jsonl`        | Where to write data            |
| `MODEL_NAME`          | `unsloth/Qwen3-14B-unsloth-bnb-4bit`  | Base model                     |
| `MAX_SEQ_LENGTH`      | `2048`                                 |                                |
| `LOAD_IN_4BIT`        | `true`                                 |                                |
| `LORA_R`              | `32`                                   |                                |
| `BATCH_SIZE`          | `2`                                    |                                |
| `GRAD_ACCUM`          | `4`                                    |                                |
| `MAX_STEPS`           | `300`                                  | Training steps                 |
| `LEARNING_RATE`       | `2e-4`                                 |                                |
| `OUTPUT_DIR`          | `/output/qwen3_rans_lora`              |                                |

**Secret injection:** Link `hf-credentials` → injects `HF_TOKEN`.

### 7. Run the job

Click **Run Job**. Logs will show:
```
Collecting trajectories for task: GoToPosition
  [GoToPosition] Episode 10/100 — 892 samples so far
  ...
Done. Total samples: ~15000 → /data/rans_trajectories.jsonl

Loading model: unsloth/Qwen3-14B-unsloth-bnb-4bit
Applying LoRA adapters...
Starting fine-tuning on RANS spacecraft control data...
...
Saved.
```

---

## Re-running individual stages

| Goal                      | Set env var        | Value  |
|---------------------------|--------------------|--------|
| Skip data gen (data exists) | `SKIP_DATA_GEN` | `true` |
| Data gen only (no training) | `SKIP_TRAIN`    | `true` |

---

## What the training data looks like

Each JSONL line is one thruster decision formatted as a chat:

**User:**
```
Task: GoToPosition | Step 12
Body-frame target offset: Δx=+0.8431 m, Δy=-0.2107 m
Heading: 23.45° (cos=0.9171, sin=0.3980)
World-frame velocity: vx=+0.0412 m/s, vy=-0.0234 m/s
Position error to target: 0.8690 m
```

**Assistant:**
```
<think>
Task: GoToPosition. I need to maneuver the spacecraft to the target position.
In the body frame, the target is +0.843 m in x (body-forward) and -0.211 m in y (body-left).
That is 0.869 m away. My heading is 23.5°. Current speed is 0.047 m/s.
I need to accelerate forward (body x) and right (body y).
For body +x force I use T0/T1; for -x I use T2/T3; for +y I use T4/T6; for -y I use T5/T7.
Velocity damping reduces overshoot. Resulting activation: T0=0.72, T1=0.72.
</think>
<action>[0.7234, 0.7234, 0.0000, 0.0000, 0.1052, 0.0000, 0.1052, 0.0000]</action>
```

---

## Configuration reference (all env vars)

### Data generation (`generate_data.py`)

| Variable            | Default                               |
|---------------------|---------------------------------------|
| `RANS_TASKS`        | All four tasks (comma-separated)      |
| `RANS_DATA_EPISODES`| `100`                                 |
| `RANS_MAX_STEPS`    | `200`                                 |
| `RANS_MIN_REWARD`   | `0.05`                                |
| `RANS_DATA_OUTPUT`  | `/data/rans_trajectories.jsonl`       |
| `RANS_SEED`         | `42`                                  |

### Fine-tuning (`train.py`)

| Variable          | Default                                |
|-------------------|----------------------------------------|
| `MODEL_NAME`      | `unsloth/Qwen3-14B-unsloth-bnb-4bit`  |
| `MAX_SEQ_LENGTH`  | `2048`                                 |
| `LOAD_IN_4BIT`    | `true`                                 |
| `LORA_R`          | `32`                                   |
| `LORA_ALPHA`      | `32`                                   |
| `BATCH_SIZE`      | `2`                                    |
| `GRAD_ACCUM`      | `4`                                    |
| `MAX_STEPS`       | `300`                                  |
| `WARMUP_STEPS`    | `10`                                   |
| `LEARNING_RATE`   | `2e-4`                                 |
| `RANS_DATA_OUTPUT`| `/data/rans_trajectories.jsonl`        |
| `OUTPUT_DIR`      | `/output/qwen3_rans_lora`              |
| `HF_TOKEN`        | (from secret)                          |
| `HF_REPO`         | (optional — push adapters to Hub)      |

### Pipeline (`pipeline.py`)

| Variable        | Default  |
|-----------------|----------|
| `SKIP_DATA_GEN` | `false`  |
| `SKIP_TRAIN`    | `false`  |

---

## Northflank docs reference

- GPU workloads: https://northflank.com/docs/v1/application/gpu-workloads/gpus-on-northflank
- Configure GPU: https://northflank.com/docs/v1/application/gpu-workloads/configure-and-optimise-workloads-for-gpus
- Run a job: https://northflank.com/docs/v1/application/run/run-an-image-once-or-on-a-schedule
