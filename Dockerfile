# ── Stage 1: Builder ──────────────────────────────────────────────────────────
# Install Python dependencies in a separate layer to keep final image lean.
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
# linux/amd64 is required by the judging VM (per Participant Guide).
# If building on Apple Silicon: docker buildx build --platform linux/amd64 ...
FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application code and the local model GGUF
COPY src/       ./src/
COPY config/    ./config/
COPY models/    ./models/
COPY run.py     ./run.py

# ── Input / output directories ────────────────────────────────────────────────
# The harness mounts /input (read) and /output (write) at runtime.
# Create them so the container can write without permission errors.
RUN mkdir -p /input /output

# ── Environment defaults ──────────────────────────────────────────────────────
# These are overridden by the harness at evaluation time.
# FIREWORKS_API_KEY, FIREWORKS_BASE_URL, ALLOWED_MODELS are NOT set here —
# they are injected by the harness. Setting them here would be a rules violation.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CONFIG_PATH=config/models.yaml \
    INPUT_PATH=/input/tasks.json \
    OUTPUT_PATH=/output/results.json

# ── Startup ───────────────────────────────────────────────────────────────────
# Container must start and be ready within 60 seconds (Participant Guide).
# run.py reads /input/tasks.json, processes all tasks, writes /output/results.json,
# and exits with code 0 on success.
CMD ["python", "run.py"]
