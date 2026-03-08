"""
pipeline.py
===========
Orchestrates the two-stage RANS → Qwen3 fine-tuning pipeline in a single
Northflank job run:

  Stage 1 — Data generation
    Runs the RANS environment with a proportional controller across all four
    tasks to produce labelled (observation → thruster) trajectories.
    Output: /data/rans_trajectories.jsonl

  Stage 2 — Fine-tuning
    Loads the JSONL, applies Unsloth + LoRA fine-tuning of Qwen3-14B.
    Output: /output/qwen3_rans_lora/  (LoRA adapter weights + tokenizer)

Both stages are governed by environment variables; see generate_data.py and
train.py for the full list.  Key overrides:

  SKIP_DATA_GEN=true   — skip stage 1 (use pre-existing /data/*.jsonl)
  SKIP_TRAIN=true      — skip stage 2 (data generation only)
"""

import os
import subprocess
import sys
from pathlib import Path

SKIP_DATA_GEN = os.environ.get("SKIP_DATA_GEN", "false").lower() == "true"
SKIP_TRAIN    = os.environ.get("SKIP_TRAIN",    "false").lower() == "true"

HERE = Path(__file__).parent


def run(script: str) -> None:
    print(f"\n{'='*60}")
    print(f"  Running: {script}")
    print(f"{'='*60}\n")
    result = subprocess.run(
        [sys.executable, str(HERE / script)],
        check=True,
    )
    print(f"\n  {script} finished (exit code {result.returncode}).")


def main() -> None:
    if not SKIP_DATA_GEN:
        run("generate_data.py")
    else:
        print("SKIP_DATA_GEN=true — skipping data generation.")

    if not SKIP_TRAIN:
        run("train.py")
    else:
        print("SKIP_TRAIN=true — skipping fine-tuning.")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
