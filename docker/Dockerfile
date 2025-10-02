# WMTP Fine-Tuning Framework Docker Image
# Base image aligned with pyproject (torch 2.4.x) and CUDA 12.1
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv by copying from official uv image (same method as GitHub Actions)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy only dependency manifests first (maximize layer caching)
COPY pyproject.toml uv.lock ./

# Install Python dependencies using uv (frozen lock for reproducibility)
RUN uv sync --frozen

# Copy project sources
COPY src/ ./src/
COPY configs/ ./configs/
COPY tests/ ./tests/
COPY scripts/ ./scripts/

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Create essential directories for S3 downloads
RUN mkdir -p /app/models /app/datasets

# Make scripts executable
RUN chmod +x /app/scripts/*.py /app/scripts/*.sh 2>/dev/null || true

# Default command - can be overridden in vessl.yaml or docker run
CMD ["uv", "run", "python", "-m", "src.cli.__main__", "--help"]
