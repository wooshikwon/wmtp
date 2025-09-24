# WMTP Deployment Guide

본 문서는 WMTP Fine-Tuning Framework의 Docker 컨테이너화 및 VESSL 배포 가이드입니다.

## Prerequisites

### Local Development
- Docker 24.0+ with NVIDIA Container Toolkit
- NVIDIA GPU with CUDA 12.1+ support
- 16GB+ RAM
- 100GB+ disk space

### VESSL Deployment
- VESSL CLI installed (`pip install vessl`)
- VESSL account with GPU quota
- AWS S3 credentials for MLflow tracking

## Quick Start

### 1. Build Docker Image

```bash
# Using Makefile
make build

# Using build script
./scripts/build_push.sh build

# Manual build
docker build -t wmtp:latest .
```

### 2. Test Locally

```bash
# Check GPU availability
make check-gpu

# Run interactive shell
make run-bash

# Run training test
make run-train
```

### 3. Push to Registry

```bash
# Set registry and push
make push REGISTRY=ghcr.io/username

# Or using script
./scripts/build_push.sh -r ghcr.io/username push
```

## Docker Configuration

### Base Image
- **Image**: `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime`
- **Python**: 3.11+
- **PyTorch**: 2.4.0
- **CUDA**: 12.1
- **cuDNN**: 9

### Package Management
- **Tool**: `uv` (Astral's fast Python package manager)
- **Lock file**: `uv.lock` for reproducible builds
- **Command**: `uv sync --frozen`

## VESSL Deployment

### 1. Configure Secrets

VESSL UI에서 다음 시크릿 설정:
- `AWS_ACCESS_KEY_ID`: S3 액세스 키
- `AWS_SECRET_ACCESS_KEY`: S3 시크릿 키
- `MLFLOW_TRACKING_TOKEN`: MLflow 인증 토큰 (선택)

### 2. Update vessl.yaml

```yaml
resources:
  cluster: your-cluster-name  # 클러스터 이름 변경
  preset: v1-a100-4-pod        # GPU 프리셋 선택

image: your-registry/wmtp:latest  # 레지스트리 URL 업데이트
```

### 3. Submit Job

```bash
# Submit to VESSL
vessl run -f configs/vessl.yaml

# Monitor logs
vessl logs <run-id>

# Check status
vessl status <run-id>
```

## Environment Variables

### Required
- `MLFLOW_TRACKING_URI`: MLflow 서버 URI
- `AWS_ACCESS_KEY_ID`: S3 액세스 (S3 사용시)
- `AWS_SECRET_ACCESS_KEY`: S3 시크릿 (S3 사용시)

### Optional
- `CUDA_VISIBLE_DEVICES`: GPU 선택
- `PYTORCH_CUDA_ALLOC_CONF`: CUDA 메모리 설정
- `TRANSFORMERS_VERBOSITY`: 로깅 레벨

## Volume Mounts

### Local Development

```bash
docker run -v $(PWD)/models:/app/models \
           -v $(PWD)/dataset:/app/dataset \
           -v $(PWD)/outputs:/app/outputs \
           wmtp:latest
```

### VESSL Volumes

```yaml
volumes:
  - name: wmtp-data
    mount_path: /data
    size: 500Gi
  - name: wmtp-models
    mount_path: /models
    size: 200Gi
```

## Resource Specifications

### GPU Presets

| Preset | GPUs | Memory | Use Case |
|--------|------|--------|----------|
| v1-a100-1-pod | 1x A100 40GB | 120GB | Testing, small experiments |
| v1-a100-4-pod | 4x A100 40GB | 480GB | Full training runs |
| v1-v100-4-pod | 4x V100 32GB | 360GB | Cost-effective training |

### Recommended Settings

```yaml
# For Rho-1 experiments
resources:
  preset: v1-a100-4-pod
env:
  PYTORCH_CUDA_ALLOC_CONF: "max_split_size_mb:512"

# For Critic experiments
resources:
  preset: v1-a100-4-pod
env:
  PYTORCH_CUDA_ALLOC_CONF: "max_split_size_mb:256"
```

## Monitoring

### Local Monitoring

```bash
# GPU utilization
nvidia-smi -l 1

# Docker stats
docker stats <container-id>

# Container logs
docker logs -f <container-id>
```

### VESSL Monitoring

- GPU 사용률: 80% 임계값 (5분 지속)
- 메모리 사용률: 90% 임계값
- 자동 재시작: 실패시 최대 3회

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

```bash
# Reduce batch size in recipe.yaml
batch_size: 8  # From 16

# Or adjust memory config
PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
```

#### 2. Docker Build Fails

```bash
# Clear cache and rebuild
docker system prune -af
make build-nocache
```

#### 3. VESSL Job Stuck

```bash
# Check detailed logs
vessl logs <run-id> --tail 100

# Cancel job
vessl cancel <run-id>
```

#### 4. S3 Connection Error

```bash
# Verify credentials
aws s3 ls s3://wmtp-artifacts/ --region ap-northeast-2

# Check environment variables
docker run --rm wmtp:latest env | grep AWS
```

## Production Checklist

### Before Deployment

- [ ] Update image tag from `latest` to version tag
- [ ] Set production MLflow tracking URI
- [ ] Configure S3 credentials in VESSL secrets
- [ ] Test with small dataset first
- [ ] Verify GPU allocation and limits

### After Deployment

- [ ] Monitor first 30 minutes of training
- [ ] Check MLflow metrics logging
- [ ] Verify checkpoint saving
- [ ] Review GPU/memory utilization

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Build and Push Docker

on:
  push:
    tags:
      - 'v*'

jobs:
  docker:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2

      - name: Login to GitHub Container Registry
        uses: docker/login-action@v2
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push
        uses: docker/build-push-action@v4
        with:
          push: true
          tags: |
            ghcr.io/${{ github.repository }}:${{ github.ref_name }}
            ghcr.io/${{ github.repository }}:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

## Advanced Configuration

### Multi-Stage Build (Future)

```dockerfile
# Build stage
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime AS builder
# ... build steps ...

# Runtime stage
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
# ... runtime setup ...
```

### Custom Entry Points

```bash
# Training with custom config
docker run wmtp:latest uv run python -m src.cli.train \
  --config /custom/config.yaml \
  --recipe /custom/recipe.yaml

# Evaluation only
docker run wmtp:latest uv run python -m src.cli.eval \
  --checkpoint /models/best.pt
```

## Support

문제 발생시 다음을 확인하세요:
1. Docker 및 NVIDIA 드라이버 버전
2. Container logs 전체
3. MLflow UI의 에러 메시지
4. VESSL 대시보드의 리소스 사용량

추가 지원이 필요한 경우 GitHub Issues를 통해 문의하세요.
