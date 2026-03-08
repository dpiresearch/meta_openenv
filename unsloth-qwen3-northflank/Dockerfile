# ─── Unsloth Qwen3-14B × RANS Spacecraft Control ─────────────────────────────
# GPU: NVIDIA L4 24GB minimum (4-bit) | A100 40/80GB recommended
# CUDA 12.1+

FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        "unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git" \
        transformers==4.46.3 \
        "trl==0.15.2" \
        datasets \
        accelerate \
        bitsandbytes \
        peft \
        huggingface_hub \
        sentencepiece \
        protobuf \
        xformers \
        numpy \
        scipy

# ── RANS environment source ───────────────────────────────────────────────────
# Copy the RANS server code so generate_data.py can import it directly.
# (No openenv-core needed — the fallback pydantic imports handle it.)
COPY RANS/server/         /app/RANS/server/
COPY RANS/models.py       /app/RANS/models.py
COPY RANS/__init__.py     /app/RANS/__init__.py

# Add RANS to the Python path
ENV PYTHONPATH="/app/RANS:${PYTHONPATH}"

# ── Pipeline scripts ──────────────────────────────────────────────────────────
COPY unsloth-qwen3-northflank/generate_data.py /app/generate_data.py
COPY unsloth-qwen3-northflank/train.py         /app/train.py
COPY unsloth-qwen3-northflank/pipeline.py      /app/pipeline.py

# Persistent mount points (attach Northflank volumes here)
RUN mkdir -p /data /output

# Default: run the full pipeline (data gen → fine-tune)
CMD ["python", "/app/pipeline.py"]
