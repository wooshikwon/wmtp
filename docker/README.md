# WMTP Docker 배포 가이드

이 문서는 WMTP(Weighted Multi-Token Prediction) 프로젝트를 Docker 컨테이너로 빌드하고 배포하기 위한 완전한 가이드입니다.

## 📋 사전 요구사항

- Docker Desktop 설치 (로컬 빌드용)
- GitHub Container Registry (ghcr.io) 접근 권한
- AWS S3 접근 권한 (모델/MLflow용, 선택적)
- HuggingFace 토큰 (모델 다운로드용, 선택적)

## 🏗️ Docker 이미지 빌드

### 로컬 빌드 및 테스트

```bash
# 프로젝트 루트에서 실행
cd /path/to/wmtp

# Docker 이미지 빌드
docker build -t wmtp:local -f docker/Dockerfile .

# 로컬에서 테스트 실행 (baseline-mtp 예시)
docker run --rm \
  -v $(pwd)/tests:/app/tests \
  -v $(pwd)/configs:/app/configs \
  -e WMTP_ALGO=baseline-mtp \
  -e ENV_MODE=test \
  wmtp:local \
  bash -c "uv run python -m src.cli.train --config tests/configs/config.local_test.yaml --recipe configs/recipe.mtp_baseline.yaml --dry-run"
```

### GitHub Container Registry 푸시

```bash
# 이미지 태깅
docker tag wmtp:local ghcr.io/wooshikwon/wmtp:latest
docker tag wmtp:local ghcr.io/wooshikwon/wmtp:v1.0.0

# GitHub 로그인
echo $GITHUB_TOKEN | docker login ghcr.io -u wooshikwon --password-stdin

# 푸시
docker push ghcr.io/wooshikwon/wmtp:latest
docker push ghcr.io/wooshikwon/wmtp:v1.0.0
```

## 🚀 Docker 실행 가이드

### 1. 로컬에서 컨테이너 실행

```bash
# Baseline MTP 테스트
docker run --rm \
  -v $(pwd)/tests:/app/tests \
  -v $(pwd)/configs:/app/configs \
  ghcr.io/wooshikwon/wmtp:latest \
  uv run python -m src.cli.train \
    --config tests/configs/config.local_test.yaml \
    --recipe tests/configs/recipe.mtp_baseline.yaml \
    --verbose

# Critic WMTP 테스트
docker run --rm \
  -v $(pwd)/tests:/app/tests \
  -v $(pwd)/configs:/app/configs \
  ghcr.io/wooshikwon/wmtp:latest \
  uv run python -m src.cli.train \
    --config tests/configs/config.local_test.yaml \
    --recipe tests/configs/recipe.critic_wmtp.yaml \
    --verbose
```

### 2. GPU 환경에서 실행

```bash
# GPU 사용 (nvidia-docker 필요)
docker run --rm --gpus all \
  -v $(pwd)/tests:/app/tests \
  -v $(pwd)/configs:/app/configs \
  -v $(pwd)/checkpoints:/app/checkpoints \
  ghcr.io/wooshikwon/wmtp:latest \
  uv run python -m src.cli.train \
    --config configs/config.gpu.yaml \
    --recipe configs/recipe.critic_wmtp.yaml \
    --run-name production_run \
    --verbose
```

## 📊 실험 모니터링

### MLflow UI 확인
```bash
# S3에 저장된 MLflow 데이터 확인
aws s3 ls s3://wmtp/mlflow/

# 로컬 MLflow UI 실행 (S3 백엔드)
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret
mlflow ui --backend-store-uri s3://wmtp/mlflow
```

### Docker 컨테이너 로그 확인
```bash
# 실행 중인 컨테이너 확인
docker ps

# 컨테이너 로그 확인
docker logs <container-id>

# 실시간 로그 스트리밍
docker logs <container-id> -f
```

## 🔧 문제 해결

### GPU 메모리 부족
```bash
# 설정 파일에서 배치 크기 조정
# configs/config.your_env.yaml
data:
  train:
    batch_size: 1
train:
  gradient_accumulation: 32
```

### 모델 다운로드 실패
```bash
# HuggingFace 토큰 확인
echo $HF_TOKEN

# 토큰 권한 확인 (read 권한 필요)
curl -H "Authorization: Bearer $HF_TOKEN" https://huggingface.co/api/whoami
```

### S3 접근 오류
```bash
# AWS 자격증명 확인
aws s3 ls s3://wmtp/ --region eu-north-1

# IAM 권한 확인 (s3:PutObject, s3:GetObject 필요)
```

## 🏗️ 디렉토리 구조

```
docker/
├── .dockerignore       # Docker 빌드 시 제외 파일
├── Dockerfile          # 컨테이너 빌드 정의
├── vessl.yaml         # VESSL 실행 설정
└── README.md          # 이 문서

configs/
├── config.vessl.yaml  # 프로덕션 환경 설정
├── recipe.mtp_baseline.yaml
├── recipe.critic_wmtp.yaml
├── recipe.rho1_wmtp_weighted.yaml
└── recipe.rho1_wmtp_tokenskip.yaml

tests/
├── configs/
│   └── config.local_test.yaml  # 테스트 환경 설정
├── test_models/
│   └── distilgpt2-mtp/         # 테스트용 작은 MTP 모델
└── test_dataset/               # 테스트용 작은 데이터셋
```

## 🎯 알고리즘별 특징

| 알고리즘 | 설명 | 특징 | 메모리 사용 |
|---------|------|------|------------|
| baseline-mtp | 균등 가중치 기준선 | 모든 토큰 w=1 | 낮음 |
| critic-wmtp | Value Function 기반 | 2단계 학습 필요 | 높음 |
| rho1-weighted | 참조 모델 CE 차이 | 연속적 가중치 | 중간 |
| rho1-tokenskip | 참조 모델 CE 차이 | 하위 30% 제외 | 낮음 |

## 📝 추가 참고사항

- **로컬 테스트**: `tests/configs/config.local_test.yaml` 사용 (작은 모델)
- **프로덕션 배포**: `configs/config.gpu.yaml` 사용 (전체 크기 모델)
- **GPU 리소스**: 테스트는 CPU/MPS 가능, 프로덕션은 GPU 권장
- **실행 시간**: 테스트 ~10-30분, 프로덕션 ~4-8시간

## 🆘 지원

문제가 발생하면 다음을 확인하세요:
1. [시스템 아키텍처 문서](../docs/WMTP_시스템_아키텍처.md)
2. [테스트 가이드](../tests/README.md)
3. [메인 README](../README.md)

프로젝트 이슈: https://github.com/wooshikwon/wmtp/issues
