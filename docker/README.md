# Docker & VESSL Deployment

This directory contains all files necessary for Docker containerization and VESSL GPU cluster deployment.

## Directory Structure

```
docker/
├── Dockerfile       # Docker image build specification
├── vessl.yaml      # VESSL platform deployment configuration
├── .dockerignore   # Build context ignores
└── README.md       # This file
```

## Files

### Dockerfile
- Base image: PyTorch 2.4.0 with CUDA 12.1 (aligned with pyproject)
- Installs uv for package management
- Copies project sources and installs dependencies with `uv sync --frozen`
- Default entrypoint: `python -m src.cli.__main__ --help`

### vessl.yaml
- Parameterizes algorithm via `WMTP_ALGO` env (`baseline-mtp | critic-wmtp | rho1-wmtp`)
- Selects corresponding `configs/config.*.yaml` and `configs/recipe.*.yaml`
- Includes GPU preset, secrets, and training command

## Usage

### Build & Push
```bash
# From project root
make build IMAGE_TAG=latest
make push REGISTRY=ghcr.io/wooshikwon IMAGE_TAG=latest
```

### Local Testing with Docker
```bash
# Interactive shell with GPU (mount configs/models/datasets as needed)
make run-bash

# Or run training directly (example: rho1)
docker run --rm --gpus all \
  -e HF_TOKEN=$HF_TOKEN \
  -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  -e AWS_DEFAULT_REGION=eu-north-1 \
  -v $(pwd)/configs:/app/configs \
  ghcr.io/wooshikwon/wmtp:latest \
  uv run python -m src.cli.train \
    --config configs/config.rho1_wmtp.yaml \
    --recipe configs/recipe.rho1_wmtp.yaml
```

### Deploying to VESSL
```bash
# Submit job (defaults to rho1; override WMTP_ALGO in file or via VESSL UI)
make vessl-run
```

## Environment Variables

Configure via VESSL secrets or export locally:
- `HF_TOKEN`: HuggingFace API token
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`
- `MLFLOW_TRACKING_URI`, `MLFLOW_REGISTRY_URI` (S3 URIs recommended in production)

## Workflow

1. Build image with dependencies
2. Push to container registry
3. Submit VESSL run with `docker/vessl.yaml`
4. Train (and optionally evaluate) using selected algorithm
5. Monitor GPU/memory utilization via VESSL
6. Models/checkpoints/MLflow artifacts stored to S3 per config
