# WMTP Fine-Tuning Framework

**Meta Multi-Token Prediction (MTP) 기반 파인튜닝 프레임워크**

본 프로젝트는 Meta의 Multi-Token Prediction 모델을 두 가지 실험 파이프라인(critic-weighted, Rho-1-weighted)으로 파인튜닝하고 평가하기 위한 프레임워크입니다.

## 핵심 특징

- **두 가지 파인튜닝 방식**: Critic-Weighted MTP, Rho-1 Reference-Weighted MTP
- **분산 학습**: FSDP 기반 A100 멀티-GPU 지원
- **MLflow 통합**: S3 백엔드 기반 실험 추적 및 모델 레지스트리
- **재현성 보장**: uv 패키지 관리자와 Docker 컨테이너화
- **자동화된 평가**: Meta MTP 논문 프로토콜 준수 (MBPP, CodeContests)

## 빠른 시작

### 필수 요구사항

- Python 3.11+
- CUDA 12.1+ (GPU 학습 시)
- uv 패키지 관리자
- Docker (VESSL 배포 시)

### 설치

```bash
# uv 설치 (아직 없다면)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 의존성 설치
uv sync --frozen
```

### 학습 실행

```bash
# Critic-Weighted MTP 학습
uv run python -m src.cli.train \
  --config configs/config.yaml \
  --recipe configs/recipe.critic.yaml

# Rho-1 Weighted MTP 학습
uv run python -m src.cli.train \
  --config configs/config.yaml \
  --recipe configs/recipe.rho1.yaml
```

### 평가 실행

```bash
uv run python -m src.cli.eval \
  --config configs/config.yaml \
  --recipe configs/recipe.yaml \
  --checkpoint runs/<run_id>/best.pt
```

## 프로젝트 구조

```
wmtp/
├── src/
│   ├── cli/                # 엔트리포인트 (train/eval)
│   ├── pipelines/          # 파이프라인 오케스트레이션
│   ├── components/         # 레지스트리 패턴 구성요소
│   │   ├── loader/        # 데이터/모델 로더
│   │   ├── scorer/        # 토큰 중요도 스코어러
│   │   ├── trainer/       # 학습 로직
│   │   ├── optimizer/     # 옵티마이저
│   │   └── evaluator/     # 평가 하네스
│   ├── factory/           # 컴포넌트 팩토리
│   ├── settings/          # Pydantic 스키마
│   └── utils/             # 공통 유틸리티
├── configs/               # 설정 파일
├── docker/                # Docker/VESSL 설정
├── tests/                 # 테스트
└── docs/                  # 문서
```

## 주요 모델

- **Base Model**: facebook/multi-token-prediction (7B, n_heads=4)
- **Reward Model**: sfair/Llama-3-8B-RM-Reward-Model
- **Reference Model**: Sheared LLaMA 1.3B

## 설정 파일

### config.yaml (환경 설정)
- 스토리지 모드 (local/s3)
- MLflow 트래킹 서버
- 분산 학습 설정
- 하드웨어 설정

### recipe.yaml (학습 설정)
- 알고리즘 선택 (critic-wmtp/rho1-wmtp)
- 하이퍼파라미터
- 데이터 설정
- 평가 프로토콜

## 개발

### 테스트 실행

```bash
# 전체 테스트
uv run pytest

# 빠른 테스트만
uv run pytest -m "not slow"
```

### 코드 스타일 검사

```bash
# 린팅
uv run ruff check .

# 포맷팅
uv run ruff format .
```

## Docker & VESSL 배포

```bash
# Docker 이미지 빌드
docker build -t wmtp:latest -f docker/Dockerfile .

# VESSL 실행
vessl run -f docker/vessl.yaml
```

## 문서

- [BLUEPRINT.md](docs/BLUEPRINT.md) - 시스템 아키텍처 및 설계 청사진
- [DEV_PRINCIPLES.md](docs/DEV_PRINCIPLE.md) - 개발 원칙 및 코딩 가이드
- [DEV_PLANS.md](docs/DEV_PLANS.md) - 단계별 개발 계획
- [CONTRIBUTING.md](CONTRIBUTING.md) - 기여 가이드

## 라이선스

MIT License - 자세한 내용은 [LICENSE](LICENSE) 파일 참조

## 문의

프로젝트 관련 문의사항은 이슈 트래커를 이용해 주세요.
