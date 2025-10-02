# WMTP (Weighted Multi-Token Prediction)

> "Not All Tokens Are What You Need" - 토큰별 중요도 기반 효율적 언어 모델 학습 프레임워크

WMTP는 모든 토큰을 동등하게 취급하는 기존 MTP(Multi-Token Prediction) 방식을 개선하여, 토큰별 중요도에 따라 차등적으로 학습하는 혁신적인 접근법입니다.

## 🚀 빠른 시작 (Quick Start)

### 로컬 개발 (MacBook/Linux)
```bash
# 1. 저장소 클론 및 의존성 설치
git clone https://github.com/wooshikwon/wmtp.git && cd wmtp
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# 2. 로컬 테스트 실행 (4가지 알고리즘 중 선택)
uv run python -m src.cli.train \
  --config tests/configs/config.local_test.yaml \
  --recipe configs/recipe.critic_wmtp.yaml \
  --dry-run --verbose
```

### 프로덕션 배포 (GPU Cluster)
```bash
# 1. Docker 이미지 Pull
docker pull ghcr.io/wooshikwon/wmtp:latest

# 2. S3에서 리소스 다운로드 및 훈련 실행
docker run --gpus all \
  -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  ghcr.io/wooshikwon/wmtp:latest \
  bash -c "python scripts/download_resources.py && \
           uv run python -m src.cli.train \
           --config configs/config.production.yaml \
           --recipe configs/recipe.critic_wmtp.yaml"
```

> 💡 **상세 배포 가이드**: VESSL, RunPod, Kubernetes 등 다양한 GPU cluster에서 실행하는 방법은 [docs/SIMPLE_PRODUCTION_DEPLOYMENT.md](docs/SIMPLE_PRODUCTION_DEPLOYMENT.md)를 참고하세요.

## 🌐 지원 플랫폼

WMTP는 Docker 기반으로 다양한 GPU cluster에서 동일하게 실행됩니다:

| 플랫폼 | 지원 여부 | 설정 난이도 |
|--------|----------|-----------|
| **VESSL** | ✅ | 쉬움 |
| **RunPod** | ✅ | 쉬움 |
| **Kubernetes** | ✅ | 중간 |
| **AWS SageMaker** | ✅ | 중간 |
| **Bare Metal** | ✅ | 쉬움 |

**핵심 요구사항**:
- Docker + NVIDIA Container Toolkit
- 4x GPU (A100 권장)
- S3 접근 가능한 AWS credentials

**프로바이더별 실행 예제**: [docs/SIMPLE_PRODUCTION_DEPLOYMENT.md](docs/SIMPLE_PRODUCTION_DEPLOYMENT.md) 참조

## 📋 지원 알고리즘

| 알고리즘 | 설명 | 특징 |
|---------|------|------|
| **baseline-mtp** | 균등 가중치 기준선 | 모든 토큰 w=1 |
| **critic-wmtp** | Value Function 기반 | 2단계 학습, 강화학습 가치함수 |
| **rho1-weighted** | 참조 모델 CE 차이 | 연속적 가중치 |
| **rho1-tokenskip** | 참조 모델 CE 차이 | 하위 30% 토큰 제외 |

## 🗂️ 프로젝트 구조

```
wmtp/
├── src/                    # 메인 소스코드
│   ├── cli/               # CLI 인터페이스
│   ├── pipelines/         # 통합 훈련/평가 파이프라인
│   ├── components/        # 플러그인 아키텍처
│   ├── factory/           # 컴포넌트 팩토리
│   └── settings/          # 설정 스키마
├── configs/               # 프로덕션 설정 파일
├── tests/                 # 테스트 환경
│   ├── configs/          # 테스트용 설정
│   ├── test_models/      # 작은 테스트 모델
│   └── test_dataset/     # 축소 데이터셋
├── docker/               # 컨테이너 배포
└── .github/              # CI/CD 워크플로우
```

## 🔧 개발 워크플로우

### 로컬 개발
1. **환경 설정**: `uv sync` → 의존성 자동 설치
2. **코드 수정**: src/ 디렉토리에서 개발
3. **로컬 테스트**: MacBook MPS 또는 단일 GPU 환경에서 빠른 검증
   ```bash
   uv run python -m src.cli.train \
     --config tests/configs/config.local_test.yaml \
     --recipe configs/recipe.critic_wmtp.yaml \
     --dry-run
   ```

### CI/CD 자동화
1. **코드 푸시**: `git push origin main`
2. **자동 빌드**: GitHub Actions가 Docker 이미지 빌드
3. **자동 배포**: ghcr.io/wooshikwon/wmtp:latest 자동 푸시
4. **테스트 매트릭스**: 4개 알고리즘 자동 테스트

### 프로덕션 배포
1. **이미지 Pull**: `docker pull ghcr.io/wooshikwon/wmtp:latest`
2. **리소스 준비**: S3에 모델/데이터셋 업로드 (최초 1회)
3. **GPU Cluster 실행**: 어떤 플랫폼이든 동일한 Docker 명령어
4. **모니터링**: MLflow (S3 기반) + CloudWatch/Prometheus

> 📖 **상세 CI/CD 가이드**: [.github/CI-CD-GUIDE.md](.github/CI-CD-GUIDE.md)

## 📚 상세 가이드

- **🧪 테스트 실행**: [tests/README.md](tests/README.md)
  - MacBook MPS 환경 설정
  - 4가지 알고리즘 테스트 명령어
  - 테스트 모델/데이터셋 구조

- **🐳 프로덕션 배포**: [docs/SIMPLE_PRODUCTION_DEPLOYMENT.md](docs/SIMPLE_PRODUCTION_DEPLOYMENT.md)
  - Docker 기반 범용 GPU cluster 배포
  - VESSL, RunPod, Kubernetes, Bare Metal 예제
  - S3 리소스 다운로드 가이드
  - config.production.yaml 사용법

- **⚙️ CI/CD 설정**: [.github/CI-CD-GUIDE.md](.github/CI-CD-GUIDE.md)
  - GitHub Actions 워크플로우
  - 자동 빌드/테스트/배포 파이프라인
  - GitHub Secrets 설정

- **🏗️ 시스템 아키텍처**: [docs/WMTP_시스템_아키텍처.md](docs/WMTP_시스템_아키텍처.md)
  - 5계층 모듈러 아키텍처
  - 알고리즘별 구현 전략
  - 확장성 메커니즘

- **📖 연구 배경**: [docs/WMTP_학술_연구제안서.md](docs/WMTP_학술_연구제안서.md)
  - 이론적 배경 및 수식
  - 4가지 가중화 방법론
  - 실험 설계 및 평가

## 🏃‍♂️ 실행 예시

### 로컬 테스트 (MacBook MPS)
```bash
# Critic WMTP 알고리즘 빠른 테스트
uv run python -m src.cli.train \
  --config tests/configs/config.local_test.yaml \
  --recipe configs/recipe.critic_wmtp.yaml \
  --run-name test_critic_mps \
  --tags test,critic,mps \
  --verbose
```

### 프로덕션 GPU Cluster 실행

#### 기본 실행 (모든 플랫폼 공통)
```bash
docker run --gpus all \
  -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  ghcr.io/wooshikwon/wmtp:latest \
  bash -c "python scripts/download_resources.py && \
           uv run python -m src.cli.train \
           --config configs/config.production.yaml \
           --recipe configs/recipe.critic_wmtp.yaml \
           --run-name prod_critic_run \
           --tags production,critic"
```

#### 4개 알고리즘 순차 실행
```bash
for RECIPE in mtp_baseline critic_wmtp rho1_wmtp_weighted rho1_wmtp_tokenskip; do
  docker run --gpus all \
    -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
    ghcr.io/wooshikwon/wmtp:latest \
    bash -c "python scripts/download_resources.py && \
             uv run python -m src.cli.train \
             --config configs/config.production.yaml \
             --recipe configs/recipe.$RECIPE.yaml \
             --run-name prod_$RECIPE \
             --tags production,$RECIPE"
done
```

> 💡 **플랫폼별 상세 예제**: VESSL, RunPod, Kubernetes 등에서 실행하는 구체적인 방법은 [docs/SIMPLE_PRODUCTION_DEPLOYMENT.md](docs/SIMPLE_PRODUCTION_DEPLOYMENT.md) 참조
