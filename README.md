# WMTP (Weighted Multi-Token Prediction)

> "Not All Tokens Are What You Need" - 토큰별 중요도 기반 효율적 언어 모델 학습 프레임워크

WMTP는 모든 토큰을 동등하게 취급하는 기존 MTP(Multi-Token Prediction) 방식을 개선하여, 토큰별 중요도에 따라 차등적으로 학습하는 혁신적인 접근법입니다.

## 🚀 빠른 시작 (Quick Start)

### 1. 환경 설정
```bash
# 1. 저장소 클론
git clone https://github.com/wooshikwon/wmtp.git
cd wmtp

# 2. uv로 의존성 설치
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# 3. 활성화 확인
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

### 2. 로컬 테스트 실행
```bash
# 4가지 알고리즘 중 하나 선택하여 테스트
uv run python -m src.cli.train \
  --config tests/configs/config.local_test.yaml \
  --recipe configs/recipe.critic_wmtp.yaml \
  --run-name test_critic \
  --tags test,critic \
  --dry-run
```

### 3. Docker & CI 배포
```bash
# GitHub Actions로 자동 빌드/배포
git push origin main  # → 자동으로 ghcr.io/wooshikwon/wmtp:latest 생성

# 또는 수동 빌드
docker build -t wmtp:local -f docker/Dockerfile .
```

### 4. VESSL GPU 클러스터 실행
```bash
# VESSL 시크릿 설정 후
vessl run -f docker/vessl.yaml --env WMTP_ALGO=critic-wmtp
```

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
3. **테스트**: MacBook MPS 환경에서 빠른 검증
4. **커밋**: git push → 자동 CI 실행

### 프로덕션 배포
1. **자동 빌드**: GitHub Actions → Docker 이미지 생성
2. **테스트**: 4개 알고리즘 매트릭스 테스트
3. **배포**: VESSL GPU 클러스터에서 실행
4. **모니터링**: MLflow + S3 기반 실험 추적

## 📚 상세 가이드

- **🧪 테스트 실행**: [tests/README.md](tests/README.md)
  - MacBook MPS 환경 설정
  - 4가지 알고리즘 테스트 명령어
  - 테스트 모델/데이터셋 구조

- **🐳 Docker 배포**: [docker/README.md](docker/README.md)
  - 컨테이너 빌드 및 레지스트리 푸시
  - VESSL GPU 클러스터 실행
  - 4가지 알고리즘별 실행 명령어

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
# Critic WMTP 알고리즘 테스트
uv run python -m src.cli.train \
  --config tests/configs/config.local_test.yaml \
  --recipe configs/recipe.critic_wmtp.yaml \
  --run-name test_critic_mps \
  --tags test,critic,mps \
  --verbose
```

### VESSL 클러스터 실행
```bash
# 4개 알고리즘 순차 실행
for ALGO in baseline-mtp critic-wmtp rho1-weighted rho1-tokenskip; do
  vessl run -f docker/vessl.yaml --env WMTP_ALGO=$ALGO --env ENV_MODE=test
done
```

### CI/CD 자동화
```bash
# 브랜치 푸시만으로 전체 파이프라인 실행
git push origin feature/new-algorithm
# → 린트 검사 → 테스트 → Docker 빌드 → 선택적 배포
```
