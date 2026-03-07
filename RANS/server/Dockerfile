# RANS spacecraft navigation environment — OpenEnv Docker image
# arXiv:2310.07393

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy the environment package
COPY . /app/

# Install Python dependencies
RUN pip install --no-cache-dir -e ".[dev]"

# Default task (override via docker run -e RANS_TASK=GoToPose)
ENV RANS_TASK=GoToPosition
ENV RANS_MAX_STEPS=500

# Expose the OpenEnv server port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run the server
CMD ["uvicorn", "rans_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
