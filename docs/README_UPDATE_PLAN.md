# README.md 업데이트 계획서
## Docker & Cloud GPU Cluster 범용 가이드 전환

---

## 📋 목적 및 배경

### 현재 문제점
현재 README.md는 다음과 같은 한계를 가지고 있습니다:

1. **VESSL 중심 설명**: Quick Start와 실행 예시가 VESSL에 치우쳐 있음 (line 43-47, 132-137)
2. **프로덕션 설정 누락**: `config.production.yaml`이 존재하나 README에서 언급되지 않음
3. **리소스 다운로드 미설명**: `scripts/download_resources.py` 스크립트 사용법 부재
4. **범용성 부족**: RunPod, Kubernetes, Bare Metal 등 다양한 GPU cluster 가이드 없음
5. **Docker 워크플로우 불명확**: Docker 기반 프로덕션 배포 흐름이 명확하지 않음

### 개선 목표
1. ✅ **범용 Cloud GPU Cluster 가이드**: 어떤 GPU 환경에서도 동일하게 실행 가능하도록
2. ✅ **Docker 중심 프로덕션 워크플로우**: config.production.yaml + Docker + S3 기반 흐름 명확화
3. ✅ **문서 간 역할 분리**: README.md는 개요/빠른시작, SIMPLE_PRODUCTION_DEPLOYMENT.md는 상세 배포
4. ✅ **로컬/프로덕션 구분**: 개발 환경과 배포 환경을 명확히 구분

---

## 🔍 현재 상태 분석

### 기존 README.md 구조 (145 lines)
```
1-5:    타이틀 및 부제
7-21:   환경 설정 (uv 기반 로컬)
24-32:  로컬 테스트 실행
34-47:  Docker & CI 배포 (VESSL 중심) ⚠️
49-57:  지원 알고리즘 테이블
59-75:  프로젝트 구조
77-90:  개발 워크플로우
92-117: 상세 가이드 링크들
119-145: 실행 예시 (로컬 + VESSEL) ⚠️
```

### 관련 파일 현황
- ✅ `configs/config.production.yaml`: Docker 컨테이너 기반 프로덕션 설정
- ✅ `docker/Dockerfile`: PyTorch 2.4.0 + CUDA 12.1 + uv
- ✅ `scripts/download_resources.py`: S3 리소스 다운로드 스크립트
- ✅ `docs/SIMPLE_PRODUCTION_DEPLOYMENT.md`: 상세 배포 가이드 (4 Phase)
- ✅ `.github/workflows/docker-build-push.yml`: CI/CD 자동화

### 문서 간 역할 정의
| 문서 | 역할 | 대상 독자 |
|------|------|----------|
| **README.md** | 프로젝트 개요, 빠른 시작, 구조 소개 | 모든 사용자 (첫 방문자) |
| **SIMPLE_PRODUCTION_DEPLOYMENT.md** | 프로덕션 배포 상세 가이드 | 운영 담당자 |
| **WMTP_시스템_아키텍처.md** | 시스템 설계 및 구현 전략 | 개발자 |
| **WMTP_학술_연구제안서.md** | 연구 배경 및 이론 | 연구자 |

---

## 📝 Phase별 개선 계획

### Phase 1: Quick Start 섹션 개선 (20분)
**목표**: 로컬 개발과 프로덕션 배포를 명확히 구분

#### 수정 위치: Line 7-47
**변경 전**: 단일 Quick Start (로컬 중심, VESSL 언급)
**변경 후**: 두 가지 Quick Start 경로 제공

```markdown
## 🚀 빠른 시작 (Quick Start)

### 로컬 개발 (MacBook/Linux)
```bash
# 1. 저장소 클론 및 의존성 설치
git clone https://github.com/wooshikwon/wmtp.git && cd wmtp
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# 2. 로컬 테스트 실행
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

> 💡 **상세 배포 가이드**: VESSL, RunPod, Kubernetes 등 다양한 GPU cluster에서 실행하는 방법은 [SIMPLE_PRODUCTION_DEPLOYMENT.md](docs/SIMPLE_PRODUCTION_DEPLOYMENT.md)를 참고하세요.
```

**핵심 변경사항**:
- ❌ VESSL 전용 명령어 제거 (line 43-47)
- ✅ Docker 기반 범용 실행 커맨드 추가
- ✅ config.production.yaml 사용
- ✅ download_resources.py 스크립트 소개
- ✅ SIMPLE_PRODUCTION_DEPLOYMENT.md 링크 추가

---

### Phase 2: 프로덕션 배포 섹션 추가 (15분)
**목표**: 다양한 GPU cluster 환경 지원 명시

#### 추가 위치: Line 48 이후 (알고리즘 테이블 앞)
**새로운 섹션 추가**:

```markdown
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

**프로바이더별 실행 예제**: [SIMPLE_PRODUCTION_DEPLOYMENT.md](docs/SIMPLE_PRODUCTION_DEPLOYMENT.md) 참조
```

**핵심 변경사항**:
- ✅ 범용 플랫폼 지원 명시
- ✅ 요구사항 간단 명시
- ✅ 상세 가이드 링크

---

### Phase 3: 개발 워크플로우 업데이트 (10분)
**목표**: Docker 기반 CI/CD 플로우 명확화

#### 수정 위치: Line 77-90
**변경 전**: 로컬 개발 중심 워크플로우
**변경 후**: 로컬 → CI/CD → 프로덕션 전체 흐름

```markdown
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
```

**핵심 변경사항**:
- ✅ 3단계 워크플로우 (로컬 → CI/CD → 프로덕션)
- ✅ Docker 기반 배포 플로우
- ✅ 실제 명령어 예시 포함

---

### Phase 4: 실행 예시 업데이트 (15분)
**목표**: VESSL 전용 예제를 범용 예제로 교체

#### 수정 위치: Line 119-145
**변경 전**: 로컬 테스트 + VESSL 클러스터 예제
**변경 후**: 로컬 테스트 + 범용 Docker 예제

```markdown
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

> 💡 **플랫폼별 상세 예제**: VESSL, RunPod, Kubernetes 등에서 실행하는 구체적인 방법은 [SIMPLE_PRODUCTION_DEPLOYMENT.md](docs/SIMPLE_PRODUCTION_DEPLOYMENT.md) 참조
```

**핵심 변경사항**:
- ❌ VESSL vessl.yaml 기반 예제 제거
- ✅ 범용 Docker 명령어 추가
- ✅ 4개 알고리즘 자동화 스크립트
- ✅ 상세 가이드 링크

---

### Phase 5: 상세 문서 링크 추가 (5분)
**목표**: SIMPLE_PRODUCTION_DEPLOYMENT.md 참조 추가

#### 수정 위치: Line 92-117
**변경 전**: 기존 4개 문서 링크
**변경 후**: 프로덕션 배포 가이드 추가 (5개 문서)

```markdown
## 📚 상세 가이드

- **🧪 테스트 실행**: [tests/README.md](tests/README.md)
  - MacBook MPS 환경 설정
  - 4가지 알고리즘 테스트 명령어
  - 테스트 모델/데이터셋 구조

- **🐳 프로덕션 배포**: [docs/SIMPLE_PRODUCTION_DEPLOYMENT.md](docs/SIMPLE_PRODUCTION_DEPLOYMENT.md) ⭐ NEW
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
```

**핵심 변경사항**:
- ✅ SIMPLE_PRODUCTION_DEPLOYMENT.md 추가 (⭐ NEW 표시)
- ✅ Docker 중심 설명 강조
- ✅ 문서 우선순위 재정렬 (테스트 → 배포 → CI/CD → 아키텍처 → 연구)

---

## ✅ 검증 계획

### 1단계: 로컬 문서 검증
```bash
# README.md 마크다운 문법 확인
markdownlint README.md

# 링크 유효성 확인
markdown-link-check README.md
```

### 2단계: Docker 빌드 테스트
```bash
# Dockerfile 빌드 성공 확인
docker build -t wmtp:test -f docker/Dockerfile .

# Quick Start 명령어 실제 실행
docker run --rm \
  -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  wmtp:test \
  python scripts/download_resources.py
```

### 3단계: 문서 일관성 확인
- [ ] README.md와 SIMPLE_PRODUCTION_DEPLOYMENT.md 내용 중복 없는지 확인
- [ ] config.production.yaml 경로 일치 확인
- [ ] 4개 recipe 파일명 일치 확인
- [ ] 모든 링크 작동 확인

---

## 📊 예상 소요 시간

| Phase | 작업 내용 | 예상 시간 |
|-------|----------|----------|
| Phase 1 | Quick Start 섹션 개선 | 20분 |
| Phase 2 | 프로덕션 배포 섹션 추가 | 15분 |
| Phase 3 | 개발 워크플로우 업데이트 | 10분 |
| Phase 4 | 실행 예시 업데이트 | 15분 |
| Phase 5 | 상세 문서 링크 추가 | 5분 |
| **검증** | 문서/Docker 테스트 | 10분 |
| **총계** | | **약 75분** |

---

## 🎯 핵심 개선사항 요약

### 제거되는 내용
- ❌ VESSL 전용 명령어 (`vessl run -f docker/vessl.yaml`)
- ❌ VESSL 중심 설명
- ❌ 단일 플랫폼 가이드

### 추가되는 내용
- ✅ 범용 Docker 실행 커맨드
- ✅ config.production.yaml 사용 예제
- ✅ download_resources.py 스크립트 소개
- ✅ 다양한 GPU cluster 플랫폼 지원 명시
- ✅ SIMPLE_PRODUCTION_DEPLOYMENT.md 링크
- ✅ 로컬/프로덕션 명확한 구분

### 유지되는 내용
- ✅ 연구 철학 ("Not All Tokens Are What You Need")
- ✅ 4가지 알고리즘 소개
- ✅ 프로젝트 구조
- ✅ uv 기반 로컬 개발 가이드
- ✅ 기존 상세 문서 링크들

---

## 🔄 개발 원칙 준수 확인

### [원칙 1] 앞/뒤 흐름 확인
- ✅ 현재 README.md 구조 완전 분석
- ✅ 관련 파일들 (config, Dockerfile, scripts) 검토 완료
- ✅ SIMPLE_PRODUCTION_DEPLOYMENT.md와의 관계 파악

### [원칙 2] 기존 구조 존중, 중복 제거
- ✅ README.md 기본 구조 유지 (타이틀 → Quick Start → 알고리즘 → 구조 → 가이드)
- ✅ SIMPLE_PRODUCTION_DEPLOYMENT.md와 중복 방지 (README는 개요, SIMPLE은 상세)
- ✅ 링크 참조로 문서 간 역할 분리

### [원칙 3] 삭제 승인
- ✅ VESSL 전용 내용 제거는 "범용성 확보"를 위한 필수적 조치
- ✅ 기존 구조를 파괴하지 않고 섹션 내 내용만 교체

### [원칙 4] 하위 호환성 무시, 깨끗한 코드
- ✅ 문서 업데이트이므로 하위 호환성 고려 불필요
- ✅ VESSL 전용 내용을 과감히 제거하고 범용 가이드로 전환

### [원칙 5] 구현 결과 검토 및 객관적 기술
- ✅ Phase별 변경 내용 명확히 문서화
- ✅ 검증 계획 수립으로 구현 후 검토 프로세스 확보

### [원칙 6] 패키지 의존성 도구 활용
- ✅ uv 기반 로컬 개발 워크플로우 유지
- ✅ Docker 기반 프로덕션 배포 (환경 격리)

---

## 📌 다음 단계

이 계획서를 검토 후 승인하시면:
1. Phase 1부터 순차적으로 README.md 수정 진행
2. 각 Phase 완료 후 diff 확인 및 중간 검토
3. 전체 완료 후 최종 검증 (문서/Docker 테스트)
4. 커밋 및 푸시

**예상 완료 시간**: 약 75분 (검증 포함)
