# RANS × Qwen3 Fine-Tuning — Step-by-Step Runbook

This document explains exactly how to reproduce the two-stage pipeline that
generates spacecraft control training data from the RANS simulator (Stage 1)
and fine-tunes a language model on it (Stage 2).

Two paths are covered:

- **Local / CPU** — runs on any laptop without a GPU. Uses a small test model
  (SmolLM-360M) to verify the pipeline end-to-end in under two minutes.
- **Northflank / GPU** — production run using Qwen3-14B on an A100 GPU.

---

## Repository layout

```
meta_openenv/
├── RANS/                                  ← spacecraft simulator
│   ├── server/
│   │   ├── rans_environment.py
│   │   ├── spacecraft_physics.py
│   │   └── tasks/
│   ├── models.py
│   └── __init__.py
└── unsloth-qwen3-northflank/
    ├── generate_data.py                   ← Stage 1
    ├── train.py                           ← Stage 2
    ├── pipeline.py                        ← runs both stages
    ├── Dockerfile
    ├── outputs/
    │   ├── rans_trajectories.jsonl        ← Stage 1 output
    │   ├── trainer_state.json             ← Stage 2 raw metrics
    │   └── training_metrics.json          ← Stage 2 summary metrics
    └── RUNBOOK.md                         ← this file
```

---

## Prerequisites

### Software

| Requirement | Version used | Notes |
|-------------|-------------|-------|
| Python | 3.13.3 | 3.10+ should work |
| pip | any recent | comes with Python |
| git | any | to clone the repo |

### Python packages (local / CPU path)

Install with one command from the repo root:

```bash
pip3 install torch numpy transformers peft trl datasets \
             accelerate sentencepiece huggingface_hub
```

Exact versions used during development:

| Package | Version |
|---------|---------|
| torch | 2.9.0 |
| numpy | 2.2.6 |
| transformers | 5.3.0 |
| peft | 0.18.1 |
| trl | 0.29.0 |
| datasets | 4.6.1 |
| accelerate | 1.13.0 |
| sentencepiece | 0.2.1 |
| huggingface_hub | 1.6.0 |

### Python packages (GPU / Northflank path — additional)

```bash
pip3 install "unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git" \
             bitsandbytes xformers
```

Unsloth requires a CUDA-capable NVIDIA GPU.  On CPU it is not needed —
`train.py` detects its absence and falls back automatically.

### Repository

```bash
git clone https://github.com/YOUR_ORG/meta_openenv.git
cd meta_openenv
```

---

## Stage 1 — Data Generation

**Script:** `unsloth-qwen3-northflank/generate_data.py`

**What it does:**
Runs the RANS 2D spacecraft simulator for each of the four navigation tasks.
At each simulation step a proportional controller (derived from the pseudoinverse
of the thruster force-torque matrix) computes the optimal 8-thruster command.
Every step is formatted as a conversational training example with physics-based
chain-of-thought reasoning and saved to a JSONL file.

### Step 1 — Create output directories

```bash
mkdir -p /tmp/rans_data
mkdir -p /tmp/rans_output
```

You can use any paths you like. The paths are passed as environment variables
in the next step.

### Step 2 — Run the data generator

```bash
RANS_DATA_EPISODES=10 \
RANS_MAX_STEPS=100 \
RANS_DATA_OUTPUT=/tmp/rans_data/rans_trajectories.jsonl \
python3 unsloth-qwen3-northflank/generate_data.py
```

The script must be run from the **repo root** (`meta_openenv/`) so it can
locate the RANS simulator source at `RANS/server/`.

### Step 3 — Watch the progress

The script prints one line per 10 episodes per task:

```
Collecting trajectories for task: GoToPosition
  [GoToPosition] Episode 10/10 — 575 samples so far
  → 575 samples written for GoToPosition

Collecting trajectories for task: GoToPose
  ...
Done. Total samples: 3121 → /tmp/rans_data/rans_trajectories.jsonl
```

### Step 4 — Verify the output

Check that the file exists and contains valid JSON:

```bash
wc -l /tmp/rans_data/rans_trajectories.jsonl
```

Expected output: one integer equal to the total sample count.

Inspect a single example:

```bash
python3 -c "
import json
with open('/tmp/rans_data/rans_trajectories.jsonl') as f:
    sample = json.loads(f.readline())
print('USER:\n', sample['messages'][1]['content'])
print()
print('ASSISTANT:\n', sample['messages'][2]['content'])
"
```

A valid sample looks like this:

```
USER:
Task: GoToPosition | Step 9
Body-frame target offset: Δx=-2.1340 m, Δy=-1.2033 m
Heading: 82.81° (cos=0.1252, sin=0.9921)
World-frame velocity: vx=+0.0407 m/s, vy=-0.1076 m/s
Position error to target: 2.4475 m

ASSISTANT:
<think>
Task: GoToPosition. I need to maneuver the spacecraft to the target position.
In the body frame, the target is -2.134 m in x (body-forward) and -1.203 m in y (body-left).
That is 2.450 m away. My heading is 82.8°. Current speed is 0.115 m/s.
I need to accelerate backward (body x) and right (body y).
For body +x force I use T0/T1; for -x I use T2/T3; for +y I use T4/T6; for -y I use T5/T7.
Velocity damping reduces overshoot. Resulting activation: T2=1.00, T3=1.00, T5=1.00, T7=1.00.
Computed thruster activations: [0.000, 0.000, 1.000, 1.000, 0.000, 1.000, 0.000, 1.000]
</think>
<action>[0.0000, 0.0000, 1.0000, 1.0000, 0.0000, 1.0000, 0.0000, 1.0000]</action>
```

### Environment variable reference

All parameters are optional. Defaults are shown.

| Variable | Default | Description |
|----------|---------|-------------|
| `RANS_TASKS` | `GoToPosition,GoToPose,TrackLinearVelocity,TrackLinearAngularVelocity` | Comma-separated list of tasks to generate data for |
| `RANS_DATA_EPISODES` | `100` | Number of episodes to simulate per task |
| `RANS_MAX_STEPS` | `200` | Maximum steps per episode |
| `RANS_MIN_REWARD` | `0.05` | Minimum step reward to include a sample (filters low-quality steps) |
| `RANS_DATA_OUTPUT` | `/data/rans_trajectories.jsonl` | Where to write the JSONL file |
| `RANS_SEED` | `42` | Random seed for reproducibility |

### Expected output sizes

| Episodes per task | Tasks | Approx. samples | File size |
|-------------------|-------|-----------------|-----------|
| 10 | 4 | ~3,000 | ~7.5 MB |
| 100 | 4 | ~30,000 | ~75 MB |
| 500 | 4 | ~150,000 | ~375 MB |

---

## Stage 2 — Fine-Tuning

**Script:** `unsloth-qwen3-northflank/train.py`

**What it does:**
Loads the JSONL produced by Stage 1, tokenises it using the target model's
chat template, and fine-tunes the model with LoRA adapters.

The script has two automatic modes:

- **CPU mode** (no GPU / no unsloth): uses `HuggingFaceTB/SmolLM-360M-Instruct`
  with standard `transformers` + `peft`. Runs in ~10 seconds for 5 steps.
  Good for verifying the pipeline locally.
- **GPU mode** (CUDA + unsloth installed): uses `unsloth/Qwen3-14B-unsloth-bnb-4bit`
  with Unsloth's optimised kernels and 4-bit quantization.

### Step 1 — (GPU only) Set your HuggingFace token

Qwen3-14B is a gated model. You need a HuggingFace account and token.

```bash
export HF_TOKEN=hf_your_token_here
```

Skip this step for the CPU/SmolLM path — it is not a gated model.

### Step 2 — Run the trainer

```bash
RANS_DATA_OUTPUT=/tmp/rans_data/rans_trajectories.jsonl \
OUTPUT_DIR=/tmp/rans_output/qwen3_rans_lora \
python3 unsloth-qwen3-northflank/train.py
```

Again, run from the **repo root**.

The script will print which mode it detected:

```
Mode: CPU fallback (test)          ← no GPU / no unsloth
Model: HuggingFaceTB/SmolLM-360M-Instruct
4-bit quantization: False
Max training steps: 5
```

or on GPU:

```
Mode: GPU + Unsloth
Model: unsloth/Qwen3-14B-unsloth-bnb-4bit
4-bit quantization: True
Max training steps: 300
```

### Step 3 — Watch training progress

Training logs one line per step:

```
{'loss': '1.827', 'grad_norm': '0.067', 'learning_rate': '0.0002',
 'mean_token_accuracy': '0.6419', 'num_tokens': '512', 'epoch': '0.00032'}
{'loss': '1.820', ...}
...
Training complete.
Saving LoRA adapters to /tmp/rans_output/qwen3_rans_lora ...
Saved.
```

### Step 4 — Verify the saved adapter

```bash
ls -lh /tmp/rans_output/qwen3_rans_lora/
```

Expected files:

```
adapter_config.json       ← LoRA configuration
adapter_model.safetensors ← trained adapter weights
tokenizer.json
tokenizer_config.json
chat_template.jinja
checkpoint-N/             ← intermediate checkpoint
```

Inspect the training metrics:

```bash
python3 -c "
import json
with open('/tmp/rans_output/qwen3_rans_lora/checkpoint-5/trainer_state.json') as f:
    state = json.load(f)
for entry in state['log_history']:
    print(f\"step {entry['step']:>3}  loss={entry['loss']:.4f}  "
          f\"acc={entry['mean_token_accuracy']:.4f}  lr={entry['learning_rate']:.2e}\")
print(f\"\\nFinal loss: {state['train_loss']:.4f}\")
"
```

Expected output:

```
step   1  loss=1.8271  acc=0.6419  lr=2.00e-04
step   2  loss=1.8200  acc=0.6419  lr=1.81e-04
step   3  loss=1.8130  acc=0.6419  lr=1.31e-04
step   4  loss=1.8078  acc=0.6438  lr=6.91e-05
step   5  loss=1.8048  acc=0.6458  lr=1.91e-05

Final loss: 1.8145
```

### Environment variable reference

| Variable | CPU default | GPU default | Description |
|----------|------------|-------------|-------------|
| `MODEL_NAME` | `HuggingFaceTB/SmolLM-360M-Instruct` | `unsloth/Qwen3-14B-unsloth-bnb-4bit` | Model to fine-tune |
| `TEST_MODEL` | (same as above) | — | Override the CPU fallback model |
| `MAX_SEQ_LENGTH` | `512` | `2048` | Token context window |
| `LOAD_IN_4BIT` | `false` | `true` | 4-bit quantization (GPU only) |
| `LORA_R` | `16` | `32` | LoRA rank |
| `LORA_ALPHA` | `16` | `32` | LoRA alpha |
| `BATCH_SIZE` | `1` | `2` | Per-device batch size |
| `GRAD_ACCUM` | `1` | `4` | Gradient accumulation steps |
| `MAX_STEPS` | `5` | `300` | Total training steps |
| `WARMUP_STEPS` | `0` | `10` | LR warmup steps |
| `LEARNING_RATE` | `2e-4` | `2e-4` | Peak learning rate |
| `SEED` | `3407` | `3407` | Random seed |
| `RANS_DATA_OUTPUT` | `/data/rans_trajectories.jsonl` | same | Path to Stage 1 JSONL |
| `OUTPUT_DIR` | `/output/qwen3_rans_lora` | same | Where to save adapter weights |
| `HF_TOKEN` | — | required | HuggingFace token for gated models |
| `HF_REPO` | — | optional | If set, pushes adapter to the Hub after training |

---

## Running both stages together

`pipeline.py` runs Stage 1 then Stage 2 in sequence:

```bash
RANS_DATA_EPISODES=10 \
RANS_MAX_STEPS=100 \
RANS_DATA_OUTPUT=/tmp/rans_data/rans_trajectories.jsonl \
OUTPUT_DIR=/tmp/rans_output/qwen3_rans_lora \
python3 unsloth-qwen3-northflank/pipeline.py
```

To skip a stage (e.g. data already exists):

```bash
# Skip data generation, only train
SKIP_DATA_GEN=true \
RANS_DATA_OUTPUT=/tmp/rans_data/rans_trajectories.jsonl \
OUTPUT_DIR=/tmp/rans_output/qwen3_rans_lora \
python3 unsloth-qwen3-northflank/pipeline.py

# Generate data only, skip training
SKIP_TRAIN=true \
RANS_DATA_EPISODES=100 \
python3 unsloth-qwen3-northflank/pipeline.py
```

---

## Complete command reference — local CPU run

Copy and paste to reproduce the exact run used to produce the outputs in this repo:

```bash
# 1. Install dependencies
pip3 install torch numpy transformers peft trl datasets \
             accelerate sentencepiece huggingface_hub

# 2. Navigate to repo root
cd /path/to/meta_openenv

# 3. Create output directories
mkdir -p /tmp/rans_data /tmp/rans_output

# 4. Stage 1 — generate RANS training data
RANS_DATA_EPISODES=10 \
RANS_MAX_STEPS=100 \
RANS_DATA_OUTPUT=/tmp/rans_data/rans_trajectories.jsonl \
python3 unsloth-qwen3-northflank/generate_data.py

# 5. Stage 2 — fine-tune on RANS data (CPU, SmolLM-360M, 5 steps)
RANS_DATA_OUTPUT=/tmp/rans_data/rans_trajectories.jsonl \
OUTPUT_DIR=/tmp/rans_output/qwen3_rans_lora \
python3 unsloth-qwen3-northflank/train.py

# 6. Inspect outputs
wc -l /tmp/rans_data/rans_trajectories.jsonl
ls -lh /tmp/rans_output/qwen3_rans_lora/
```

Expected total wall-clock time on a modern laptop: **under 2 minutes**.

---

## Complete command reference — Northflank GPU run

See `DEPLOY.md` for the full Northflank setup guide. The environment variables
to set in the Northflank job panel for a full production run:

```bash
# Stage 1 settings
RANS_TASKS=GoToPosition,GoToPose,TrackLinearVelocity,TrackLinearAngularVelocity
RANS_DATA_EPISODES=100
RANS_MAX_STEPS=200
RANS_MIN_REWARD=0.05
RANS_DATA_OUTPUT=/data/rans_trajectories.jsonl

# Stage 2 settings
MODEL_NAME=unsloth/Qwen3-14B-unsloth-bnb-4bit
MAX_SEQ_LENGTH=2048
LOAD_IN_4BIT=true
LORA_R=32
LORA_ALPHA=32
BATCH_SIZE=2
GRAD_ACCUM=4
MAX_STEPS=300
WARMUP_STEPS=10
LEARNING_RATE=2e-4
OUTPUT_DIR=/output/qwen3_rans_lora

# Secret (inject from Northflank secret group, do not paste in plain text)
HF_TOKEN=<from secret group hf-credentials>
```

Entrypoint command: `python /app/pipeline.py`

GPU required: **NVIDIA A100 40 GB** (minimum: L4 24 GB for 4-bit only).

Expected runtime: ~30 min (Stage 1) + ~2–4 hours (Stage 2, 300 steps on A100).

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'server'`

You are not running from the repo root. Always run scripts as:

```bash
cd /path/to/meta_openenv
python3 unsloth-qwen3-northflank/generate_data.py
```

### `FileNotFoundError: Training data not found at /data/rans_trajectories.jsonl`

Stage 2 cannot find the Stage 1 output. Set `RANS_DATA_OUTPUT` to the actual
path where you saved the JSONL:

```bash
RANS_DATA_OUTPUT=/tmp/rans_data/rans_trajectories.jsonl python3 unsloth-qwen3-northflank/train.py
```

### `TypeError: SFTTrainer.__init__() got an unexpected keyword argument 'tokenizer'`

Your TRL version is ≥ 0.26 which renamed `tokenizer` to `processing_class`.
The current `train.py` already uses `processing_class`. If you see this error,
make sure you have the latest version of this file from the repo.

### `ModuleNotFoundError: No module named 'unsloth'`

Expected on CPU. `train.py` detects this and automatically switches to the
standard `transformers` + `peft` path with SmolLM-360M. No action needed.

### Stage 2 loss not decreasing

With only 5 steps the decrease is small but visible (1.827 → 1.805).
For meaningful learning use `MAX_STEPS=300` or more on a GPU.

### `OutOfMemoryError` on GPU

Reduce `BATCH_SIZE` to `1` and/or `LORA_R` to `16`.
Alternatively use a smaller GPU-friendly model:

```bash
MODEL_NAME=unsloth/Qwen3-8B-unsloth-bnb-4bit
```

---

## Output file reference

| File | Location | Description |
|------|----------|-------------|
| `rans_trajectories.jsonl` | `RANS_DATA_OUTPUT` | Stage 1: one JSON object per line, each a complete conversational training example |
| `adapter_config.json` | `OUTPUT_DIR/` | LoRA configuration (rank, alpha, target modules) |
| `adapter_model.safetensors` | `OUTPUT_DIR/` | Trained LoRA adapter weights |
| `tokenizer.json` | `OUTPUT_DIR/` | Tokenizer for the fine-tuned model |
| `trainer_state.json` | `OUTPUT_DIR/checkpoint-N/` | Full per-step training log (loss, accuracy, LR, grad norm) |
| `training_metrics.json` | `outputs/` | Human-readable summary of the run (in this repo) |
