# Docker & VESSL Deployment

This directory contains all files necessary for Docker containerization and VESSL GPU cluster deployment.

## Directory Structure

```
docker/
├── Dockerfile       # Docker image build specification
├── vessl.yaml      # VESSL platform deployment configuration
└── README.md       # This file
```

## Files

### Dockerfile
- Base image: PyTorch 2.4.0 with CUDA 12.1
- Installs uv for package management
- Copies project code and dependencies
- Default entrypoint for training/evaluation

### vessl.yaml
- VESSL cluster resource allocation (GPUs, memory, CPUs)
- Environment variables and secrets configuration
- Training/evaluation command execution
- Monitoring and restart policies

## Usage

### Building Docker Image
```bash
# From project root
docker build -f docker/Dockerfile -t wmtp:latest .
```

### Deploying to VESSL
```bash
# Using VESSL CLI
vessl run create --file docker/vessl.yaml

# Or push image to registry first
docker tag wmtp:latest <registry>/wmtp:latest
docker push <registry>/wmtp:latest
```

### Local Testing with Docker
```bash
# Run with GPU support
docker run --gpus all \
  -e HF_TOKEN=$HF_TOKEN \
  -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  -e AWS_DEFAULT_REGION=eu-north-1 \
  -v $(pwd):/app \
  wmtp:latest \
  uv run python -m src.cli.train \
    --config configs/config.local.yaml \
    --recipe configs/recipe.rho1.yaml
```

## Environment Variables

Required environment variables (set in VESSL secrets or .env):
- `HF_TOKEN`: HuggingFace API token
- `AWS_ACCESS_KEY_ID`: AWS access key
- `AWS_SECRET_ACCESS_KEY`: AWS secret key
- `AWS_DEFAULT_REGION`: AWS region (eu-north-1)
- `S3_BUCKET_NAME`: S3 bucket name (wmtp)

## VESSL Configuration

The `vessl.yaml` file is configured for:
- **Default**: 4x A100 GPUs (`v1-a100-4-pod`)
- **Testing**: Single A100 (`v1-a100-1-pod`)
- **Budget**: 4x V100 GPUs (`v1-v100-4-pod`)

Modify the `preset` field in `vessl.yaml` to change resource allocation.

## Workflow

1. **Build**: Docker image with all dependencies
2. **Push**: Upload image to container registry
3. **Deploy**: Create VESSL run with configuration
4. **Execute**: Automatic training/evaluation as defined in vessl.yaml
5. **Monitor**: GPU/memory usage tracked by VESSL
6. **Results**: Stored in S3 (models, checkpoints, MLflow artifacts)
