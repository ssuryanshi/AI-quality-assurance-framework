# ============================================================
# Dockerfile - AI Model QA & Hallucination Detection Framework
# ============================================================
# Multi-stage build for a lightweight evaluation container.
#
# Build:  docker build -t ai-qa-framework .
# Run:    docker run --env-file .env ai-qa-framework
# Test:   docker run ai-qa-framework pytest tests/ -v

FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies (for matplotlib)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# ── Install Python dependencies ──
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ──
COPY . .

# Create output directories
RUN mkdir -p reports/csv reports/charts baselines

# ── Default entrypoint: run evaluation ──
ENTRYPOINT ["python", "scripts/run_evaluation.py"]
CMD ["--config", "config.yaml"]
