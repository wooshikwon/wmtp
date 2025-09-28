# WMTP Docker 배포 가이드

이 문서는 WMTP(Weighted Multi-Token Prediction) 프로젝트를 Docker 컨테이너로 빌드하고 VESSL GPU 클러스터에서 실행하기 위한 완전한 가이드입니다.

## 📋 사전 요구사항

- Docker Desktop 설치 (로컬 빌드용)
- VESSL CLI 설치 및 로그인
- GitHub Container Registry (ghcr.io) 접근 권한
- AWS S3 접근 권한 (MLflow 추적용)
- HuggingFace 토큰 (모델 다운로드용)

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

## 🚀 VESSL 실행 가이드

### 1. VESSL Secrets 설정

VESSL 웹 콘솔에서 다음 시크릿을 설정하세요:

```yaml
AWS_ACCESS_KEY_ID: "your-aws-key"
AWS_SECRET_ACCESS_KEY: "your-aws-secret"
HF_TOKEN: "your-huggingface-token"
```

### 2. VESSL 클러스터 설정

```bash
# VESSL CLI 로그인
vessl login

# 사용 가능한 클러스터 확인
vessl cluster list

# 기본 클러스터 설정
vessl configure
```

### 3. 4가지 알고리즘 테스트 실행

#### Baseline MTP (균등 가중치 기준선)
```bash
vessl run \
  --image ghcr.io/wooshikwon/wmtp:latest \
  --cluster default \
  --resource v1-a100-1-pod \
  --env WMTP_ALGO=baseline-mtp \
  --env ENV_MODE=test \
  --env AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  --env AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  --env HF_TOKEN=$HF_TOKEN \
  --command "cd /app && uv run python -m src.cli.train --config tests/configs/config.local_test.yaml --recipe configs/recipe.mtp_baseline.yaml --run-name vessl_baseline_test --tags vessl,baseline,test --verbose"
```

#### Critic WMTP (Value Function 기반)
```bash
vessl run \
  --image ghcr.io/wooshikwon/wmtp:latest \
  --cluster default \
  --resource v1-a100-1-pod \
  --env WMTP_ALGO=critic-wmtp \
  --env ENV_MODE=test \
  --env AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  --env AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  --env HF_TOKEN=$HF_TOKEN \
  --command "cd /app && uv run python -m src.cli.train --config tests/configs/config.local_test.yaml --recipe configs/recipe.critic_wmtp.yaml --run-name vessl_critic_test --tags vessl,critic,test --verbose"
```

#### Rho1 WMTP Weighted (연속 가중치)
```bash
vessl run \
  --image ghcr.io/wooshikwon/wmtp:latest \
  --cluster default \
  --resource v1-a100-1-pod \
  --env WMTP_ALGO=rho1-weighted \
  --env ENV_MODE=test \
  --env AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  --env AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  --env HF_TOKEN=$HF_TOKEN \
  --command "cd /app && uv run python -m src.cli.train --config tests/configs/config.local_test.yaml --recipe configs/recipe.rho1_wmtp_weighted.yaml --run-name vessl_rho1_weighted_test --tags vessl,rho1,weighted,test --verbose"
```

#### Rho1 WMTP Token Skip (이진 선택)
```bash
vessl run \
  --image ghcr.io/wooshikwon/wmtp:latest \
  --cluster default \
  --resource v1-a100-1-pod \
  --env WMTP_ALGO=rho1-tokenskip \
  --env ENV_MODE=test \
  --env AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  --env AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  --env HF_TOKEN=$HF_TOKEN \
  --command "cd /app && uv run python -m src.cli.train --config tests/configs/config.local_test.yaml --recipe configs/recipe.rho1_wmtp_tokenskip.yaml --run-name vessl_rho1_tokenskip_test --tags vessl,rho1,tokenskip,test --verbose"
```

### 4. YAML 파일을 통한 실행 (권장)

`vessl.yaml`을 사용하여 더 간편하게 실행할 수 있습니다:

```bash
# vessl.yaml 수정 (알고리즘 선택)
# env.WMTP_ALGO: baseline-mtp | critic-wmtp | rho1-weighted | rho1-tokenskip
# env.ENV_MODE: test | production

# VESSL 실행
vessl run -f docker/vessl.yaml

# 특정 알고리즘으로 오버라이드
vessl run -f docker/vessl.yaml --env WMTP_ALGO=critic-wmtp

# 프로덕션 모드로 실행 (더 큰 모델 사용)
vessl run -f docker/vessl.yaml \
  --env ENV_MODE=production \
  --resource v1-a100-4-pod
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

### VESSL 실행 로그 확인
```bash
# 실행 중인 작업 목록
vessl run list

# 특정 실행의 로그 확인
vessl run logs <run-id>

# 실시간 로그 스트리밍
vessl run logs <run-id> -f
```

## 🔧 문제 해결

### GPU 메모리 부족
```yaml
# vessl.yaml에서 배치 크기 조정
env:
  BATCH_SIZE: 1
  GRADIENT_ACCUMULATION: 32
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

- **테스트 환경**: `ENV_MODE=test`는 작은 DistilGPT2-MTP 모델 사용
- **프로덕션 환경**: `ENV_MODE=production`은 전체 크기 모델 사용
- **GPU 리소스**: 테스트는 A100 1개, 프로덕션은 A100 4개 권장
- **실행 시간**: 테스트 ~30분, 프로덕션 ~4-8시간

## 🆘 지원

문제가 발생하면 다음을 확인하세요:
1. [시스템 아키텍처 문서](../docs/WMTP_시스템_아키텍처.md)
2. [테스트 가이드](../tests/README.md)
3. [메인 README](../README.md)

VESSL 관련 문의: support@vessl.ai
프로젝트 이슈: https://github.com/wooshikwon/wmtp/issues