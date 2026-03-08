"""
train.py
========
Fine-tune a language model on RANS spacecraft control trajectories.

Production mode (Northflank GPU):
  Uses Unsloth + LoRA on Qwen3-14B (4-bit).  Requires CUDA + unsloth installed.

Local / CPU test mode (automatic fallback):
  When unsloth is not installed or no GPU is detected, falls back to standard
  transformers + PEFT with a small model (HuggingFaceTB/SmolLM-360M-Instruct).
  Set TEST_MODEL env var to override the fallback model.

All parameters are controlled via environment variables.
"""

import json
import os
from pathlib import Path

import torch

# ─── Configuration ────────────────────────────────────────────────────────────

HAS_GPU     = torch.cuda.is_available()
try:
    import unsloth  # noqa: F401
    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False

GPU_MODE    = HAS_GPU and HAS_UNSLOTH

MODEL_NAME  = os.environ.get(
    "MODEL_NAME",
    "unsloth/Qwen3-14B-unsloth-bnb-4bit" if GPU_MODE
    else os.environ.get("TEST_MODEL", "HuggingFaceTB/SmolLM-360M-Instruct"),
)
MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH",   "512" if not GPU_MODE else "2048"))
LOAD_IN_4BIT   = os.environ.get("LOAD_IN_4BIT", "true").lower() == "true" and GPU_MODE

LORA_R         = int(os.environ.get("LORA_R",         "16" if not GPU_MODE else "32"))
LORA_ALPHA     = int(os.environ.get("LORA_ALPHA",      "16" if not GPU_MODE else "32"))

BATCH_SIZE     = int(os.environ.get("BATCH_SIZE",      "1" if not GPU_MODE else "2"))
GRAD_ACCUM     = int(os.environ.get("GRAD_ACCUM",      "1" if not GPU_MODE else "4"))
MAX_STEPS      = int(os.environ.get("MAX_STEPS",       "5" if not GPU_MODE else "300"))
WARMUP_STEPS   = int(os.environ.get("WARMUP_STEPS",    "0" if not GPU_MODE else "10"))
LEARNING_RATE  = float(os.environ.get("LEARNING_RATE", "2e-4"))
SEED           = int(os.environ.get("SEED",            "3407"))

DATA_FILE      = os.environ.get("RANS_DATA_OUTPUT",    "/data/rans_trajectories.jsonl")
OUTPUT_DIR     = os.environ.get("OUTPUT_DIR",          "/output/qwen3_rans_lora")
HF_TOKEN       = os.environ.get("HF_TOKEN",            None)
HF_REPO        = os.environ.get("HF_REPO",             None)

print(f"Mode: {'GPU + Unsloth' if GPU_MODE else 'CPU fallback (test)'}")
print(f"Model: {MODEL_NAME}")
print(f"4-bit quantization: {LOAD_IN_4BIT}")
print(f"Max training steps: {MAX_STEPS}")

# ─── HuggingFace login ────────────────────────────────────────────────────────

if HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN)
    print("Logged in to HuggingFace.")

# ─── Load model ───────────────────────────────────────────────────────────────

if GPU_MODE:
    from unsloth import FastLanguageModel

    print(f"\nLoading model with Unsloth: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name      = MODEL_NAME,
        max_seq_length  = MAX_SEQ_LENGTH,
        load_in_4bit    = LOAD_IN_4BIT,
        load_in_8bit    = False,
        full_finetuning = False,
        device_map      = "balanced",
        token           = HF_TOKEN,
    )

    print("Applying LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r                          = LORA_R,
        target_modules             = ["q_proj", "k_proj", "v_proj", "o_proj",
                                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha                 = LORA_ALPHA,
        lora_dropout               = 0,
        bias                       = "none",
        use_gradient_checkpointing = "unsloth",
        random_state               = SEED,
        use_rslora                 = False,
        loftq_config               = None,
    )

else:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    print(f"\nLoading model with transformers (CPU): {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype = torch.float32,
        token       = HF_TOKEN,
    )

    print("Applying LoRA adapters (PEFT)...")
    lora_config = LoraConfig(
        task_type    = TaskType.CAUSAL_LM,
        r            = LORA_R,
        lora_alpha   = LORA_ALPHA,
        lora_dropout = 0.05,
        bias         = "none",
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

# ─── Dataset ──────────────────────────────────────────────────────────────────

data_path = Path(DATA_FILE)
if not data_path.exists():
    raise FileNotFoundError(
        f"Training data not found at {DATA_FILE}. "
        "Run generate_data.py first, or set RANS_DATA_OUTPUT to the correct path."
    )

print(f"\nLoading RANS trajectory data from {DATA_FILE} ...")
records = []
with data_path.open() as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

print(f"  {len(records)} training examples loaded.")

# Render messages → text using the tokenizer's chat template
texts = []
for rec in records:
    try:
        text = tokenizer.apply_chat_template(
            rec["messages"],
            tokenize              = False,
            add_generation_prompt = False,
        )
    except Exception:
        # Fallback: simple concatenation if no chat template
        parts = [f"{m['role'].upper()}: {m['content']}" for m in rec["messages"]]
        text  = "\n\n".join(parts)
    texts.append(text)

from datasets import Dataset as HFDataset
dataset = HFDataset.from_dict({"text": texts}).shuffle(seed=SEED)
print(f"  Dataset ready: {len(dataset)} examples.")

# ─── Training ─────────────────────────────────────────────────────────────────

from trl import SFTTrainer, SFTConfig

print("\nStarting fine-tuning on RANS spacecraft control data...")
trainer = SFTTrainer(
    model              = model,
    processing_class   = tokenizer,
    train_dataset      = dataset,
    eval_dataset       = None,
    args = SFTConfig(
        dataset_text_field          = "text",
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,
        warmup_steps                = WARMUP_STEPS,
        max_steps                   = MAX_STEPS,
        learning_rate               = LEARNING_RATE,
        logging_steps               = 1,
        optim                       = "adamw_8bit" if GPU_MODE else "adamw_torch",
        weight_decay                = 0.001,
        lr_scheduler_type           = "cosine",
        seed                        = SEED,
        output_dir                  = OUTPUT_DIR,
        report_to                   = "none",
        fp16                        = False,
        bf16                        = GPU_MODE and torch.cuda.is_bf16_supported(),
        max_length                  = MAX_SEQ_LENGTH,
        use_cpu                     = not HAS_GPU,
    ),
)

trainer_stats = trainer.train()
print(f"\nTraining complete: {trainer_stats}")

# ─── Save ─────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Saving LoRA adapters to {OUTPUT_DIR} ...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Saved.")

if HF_REPO and HF_TOKEN:
    print(f"Pushing to HuggingFace Hub: {HF_REPO}")
    model.push_to_hub(HF_REPO, token=HF_TOKEN)
    tokenizer.push_to_hub(HF_REPO, token=HF_TOKEN)
    print("Pushed.")
