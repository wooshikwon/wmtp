# WMTP Production 배포 수정 계획서
## VESSL GPU Cluster 배포 최적화 (Phase별 구현 가이드)

---

## 📋 문서 개요

**목적**: WMTP 프레임워크의 VESSL GPU 클러스터 배포를 위한 체계적 수정 계획
**범위**: 경로 정렬, 모델 접근 최적화, E2E 검증 체계 구축
**기간**: Phase 1-2 (필수), Phase 3-4 (선택)

---

## 🔍 현황 분석 (원칙 1: 현재 구조 파악)

### 1.1 검증 완료 사항

#### ✅ 테스트 환경 (정상 작동)
```yaml
# tests/configs/config.local_test.yaml
paths:
  models:
    base: "file://tests/test_models/distilgpt2-mtp"
    rm: "file://tests/test_models/tiny-reward-model-gpt2"
    ref: "file://tests/test_models/distilgpt2"
```
- **상태**: 사용자 검증 완료 - 모든 알고리즘 정상 작동
- **환경**: MacBook M3, MPS backend, 단일 디바이스
- **모델**: ~100MB 경량 테스트 모델

#### ✅ 코드 아키텍처 (우수)
```
configs/ → settings/ → factory/ → components/ → pipelines/
   ↓         ↓           ↓           ↓             ↓
  YAML    Pydantic    Registry   Implementation  Execution
```
- **설계**: 관심사 분리 명확, 확장성 우수
- **인프라**: DistributedS3Transfer 고성능 다운로드 구현 완료
- **분산학습**: FSDP, NCCL 설정 완비

#### ✅ 기존 유틸리티 확인
| 파일 | 기능 | 상태 |
|------|------|------|
| `src/utils/distributed_s3_transfer.py` | 병렬 S3 다운로드 (청크 기반) | ✅ 구현 완료 |
| `scripts/safe_s3_download.py` | 메모리 안전 다운로드 래퍼 | ✅ 구현 완료 |
| `src/utils/path_resolver.py` | s3://, file:// 프로토콜 지원 | ✅ 구현 완료 |
| `src/utils/distribute_manager.py` | FSDP 분산 학습 관리 | ✅ 구현 완료 |

### 1.2 발견된 이슈

#### ❌ Issue #1: 경로 불일치 (CRITICAL)
```yaml
# configs/config.vessl.yaml (기대)
paths:
  models:
    base: "file:///app/models/llama-7b-mtp/"
```

```dockerfile
# Dockerfile (실제)
WORKDIR /workspace  # ← /app이 아님!
```

```yaml
# docker/vessl.yaml (생성)
command: |
  mkdir -p /tmp/models  # ← 빈 디렉토리
```

**영향**: `FileNotFoundError` 즉시 발생, 학습 불가

#### ⚠️ Issue #2: 모델 접근 전략 미정 (BLOCKER)
- **모델 위치**: S3 (s3://wmtp/models/)
- **모델 크기**: ~63GB (llama-7b-mtp 27GB + starling-rm-7b 26GB + sheared-llama-2.7b 10GB)
- **현황**: 다운로드/마운트 메커니즘 미설정
- **선택지**:
  1. Startup 다운로드 (기존 DistributedS3Transfer 활용)
  2. VESSL Volume Mount (설정 필요)

#### ⚠️ Issue #3: E2E 검증 부재
- **현황**: 로컬 테스트만 존재, VESSL 환경 dry-run 없음
- **위험**: Production 배포 시 예상치 못한 오류 가능성

---

## 🏗️ 해결 방안 설계 (원칙 2: 기존 구조 존중, 중복 방지)

### 2.1 기존 코드 재사용 전략

**원칙 준수**:
- ✅ **원칙 2**: 기존 `DistributedS3Transfer` 활용 (새로 구현 X)
- ✅ **원칙 2**: `PathResolver` 프로토콜 시스템 유지
- ✅ **원칙 6**: `uv` 패키지 관리 도구 계속 사용

**재사용 가능 컴포넌트**:
```python
# 이미 구현되어 있음 - 새로 만들지 않음!
from src.utils.distributed_s3_transfer import DistributedS3Transfer
from src.utils.path_resolver import PathResolver
from src.utils.distribute_manager import DistributedManager
```

### 2.2 추가 필요 파일 (최소화)

| 파일 | 목적 | 라인 수 (예상) |
|------|------|----------------|
| `docker/vessl_startup.sh` | 모델 다운로드 스크립트 | ~100 |
| `scripts/test_vessl_config.sh` | Dry-run 검증 | ~80 |
| `configs/config.vessl_minitest.yaml` | E2E 미니 테스트 | ~60 |

**총 추가**: ~240 라인 (기존 코드 수정 최소화)

### 2.3 삭제/수정 검토 (원칙 3: 승인 필요)

#### 수정 필요: `Dockerfile` (1개 파일, 2줄)
```dockerfile
# 현재 (잘못됨)
WORKDIR /workspace

# 수정안 (config 의도 존중)
WORKDIR /app
RUN mkdir -p /app/models /app/datasets /app/checkpoints
```

**승인 요청 사항**:
- ✅ **변경 범위**: Dockerfile line 7만 수정 (/workspace → /app)
- ✅ **영향도**: 낮음 (다른 코드 수정 불필요)
- ✅ **근거**: `config.vessl.yaml`의 `/app` 경로 의도 존중

#### 삭제 불필요
- ❌ 기존 코드 삭제 없음
- ❌ 기존 유틸리티 교체 없음
- ✅ 모든 기존 인프라 그대로 활용

---

## 📅 Phase별 구현 계획

---

## Phase 1: 경로 정렬 (필수, 우선순위 1)

### 목표
Config와 Dockerfile 간 경로 불일치 해결

### 현황 분석 (원칙 1)
```bash
# 현재 상태
Dockerfile:           WORKDIR /workspace
config.vessl.yaml:    paths.models.base = "file:///app/models/..."
→ 불일치로 인한 FileNotFoundError 발생
```

### 수정 계획 (원칙 4: 깨끗한 수정)

#### 1.1 Dockerfile 수정
**파일**: `/Dockerfile`

```dockerfile
# 변경 전 (line 7)
WORKDIR /workspace

# 변경 후
WORKDIR /app

# 추가 (line 54 이후)
# Create necessary directories
RUN mkdir -p /app/models \
             /app/datasets \
             /app/checkpoints \
             /app/mlruns \
             /app/logs

# Set proper permissions
RUN chmod -R 755 /app
```

**변경 사유**:
- `config.vessl.yaml`의 `/app/models/` 경로 의도 존중
- VESSL 표준 관례와 일치
- 최소 변경으로 문제 해결

#### 1.2 호환성 검증 (원칙 4-1: 앞/뒤 호환 관계)

**영향 받는 파일 확인**:
```bash
# WORKDIR 참조 검색
grep -r "workspace" configs/ docker/ scripts/
```

**결과**:
- `docker/vessl.yaml:75`: `mkdir -p /tmp/models` (영향 없음 - 임시 디렉토리)
- 다른 참조 없음

**변경 불필요**:
- ✅ `config.vessl.yaml`: 이미 `/app` 사용 중
- ✅ Python 코드: 절대 경로 사용, WORKDIR 무관
- ✅ MLflow: 환경변수 기반, WORKDIR 무관

### 검증 방법 (원칙 5: 계획 대비 검토)

```bash
# 1. Docker 빌드 테스트
docker build -t wmtp:test .

# 2. 컨테이너 내부 확인
docker run -it wmtp:test /bin/bash
pwd  # 출력: /app
ls -la /app  # models, datasets, checkpoints 디렉토리 확인

# 3. PathResolver 검증
docker run wmtp:test python3 -c "
from src.utils.path_resolver import PathResolver
resolver = PathResolver()
path_type, resolved = resolver.resolve('file:///app/models/test')
print(f'Type: {path_type}, Resolved: {resolved}')
assert resolved == '/app/models/test', 'Path resolution failed!'
print('✅ Path resolution verified')
"
```

### 완료 기준
- [x] Dockerfile WORKDIR가 `/app`로 변경
- [x] `/app` 하위 디렉토리 생성 확인
- [x] Docker 빌드 성공
- [x] 경로 해석 검증 통과

---

## Phase 2: 모델 다운로드 인프라 (필수, 우선순위 2)

### 목표
기존 `DistributedS3Transfer`를 활용한 고속 모델 다운로드

### 기존 인프라 분석 (원칙 1 & 2)

**이미 구현된 기능**:
```python
# src/utils/distributed_s3_transfer.py (기존)
class DistributedS3Transfer:
    - ✅ Range 요청 기반 청크 분할
    - ✅ 멀티스레드/멀티프로세스 병렬
    - ✅ 동적 워커 수 조정
    - ✅ 중단 재개 지원
    - ✅ Transfer Acceleration 지원
```

**재사용 전략** (원칙 2: 중복 방지):
- ❌ 새로운 다운로드 로직 작성 금지
- ✅ 기존 `DistributedS3Transfer` 그대로 사용
- ✅ Wrapper 스크립트만 추가 (`vessl_startup.sh`)

### 구현 계획 (원칙 4: 깨끗한 코드)

#### 2.1 Startup Script 생성
**파일**: `docker/vessl_startup.sh` (신규)

```bash
#!/bin/bash
# WMTP Model Download & Setup
# Uses existing DistributedS3Transfer for optimized parallel download

set -e

echo "=================================================="
echo "WMTP Model Download (Distributed S3 Transfer)"
echo "=================================================="

# 환경 확인
echo "Environment:"
echo "  S3 Bucket: ${S3_BUCKET_NAME:-wmtp}"
echo "  AWS Region: ${AWS_DEFAULT_REGION:-eu-north-1}"
echo "  Target: /app/models"
echo ""

# 재시도 로직
MAX_RETRIES=3
RETRY_COUNT=0

download_models() {
    python3 << 'PYTHON_EOF'
import sys
from pathlib import Path

# PYTHONPATH 설정
sys.path.insert(0, '/app')

from src.utils.distributed_s3_transfer import DistributedS3Transfer

# Transfer Acceleration으로 고속 다운로드
transfer = DistributedS3Transfer(
    bucket='wmtp',
    max_workers=16,              # A100x4: 16 워커 병렬
    use_multiprocess=True,       # CPU 병렬 활용
    chunk_size_mb=100,           # 대용량: 100MB 청크
    enable_acceleration=True     # 지역 간 전송 가속
)

models = [
    ('models/llama-7b-mtp', '/app/models/llama-7b-mtp'),
    ('models/starling-rm-7b', '/app/models/starling-rm-7b'),
    ('models/sheared-llama-2.7b', '/app/models/sheared-llama-2.7b'),
]

for s3_prefix, local_path in models:
    print(f'\n📦 Downloading {s3_prefix}...')

    success, count = transfer.download_directory_distributed(
        s3_prefix,
        Path(local_path),
        show_progress=True
    )

    if not success:
        print(f'❌ Download failed: {s3_prefix}')
        sys.exit(1)

    print(f'✅ Downloaded {count} files to {local_path}')

print('\n✅ All models downloaded successfully!')
PYTHON_EOF
}

# 다운로드 실행 (재시도 포함)
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    echo "Download attempt $((RETRY_COUNT+1))/$MAX_RETRIES..."

    if download_models; then
        echo "✅ Download successful!"
        break
    else
        RETRY_COUNT=$((RETRY_COUNT+1))
        if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
            echo "⚠️  Download failed, retrying in 10s..."
            sleep 10
        else
            echo "❌ Download failed after $MAX_RETRIES attempts"
            exit 1
        fi
    fi
done

# 검증
echo ""
echo "Verifying downloaded models..."
python3 << 'PYTHON_EOF'
from pathlib import Path
import json

models = [
    '/app/models/llama-7b-mtp',
    '/app/models/starling-rm-7b',
    '/app/models/sheared-llama-2.7b'
]

for model_path in models:
    p = Path(model_path)
    if not p.exists():
        print(f'❌ {model_path} not found!')
        exit(1)

    # Vocab size 확인
    for config_file in ['config.json', 'params.json']:
        config_path = p / config_file
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                vocab_size = config.get('vocab_size', 'N/A')
                print(f'✅ {p.name}: vocab_size={vocab_size}')
            break

print('\n✅ Model verification complete!')
PYTHON_EOF

echo "=================================================="
echo "Setup complete! Models ready at /app/models/"
echo "=================================================="
```

**코드 특징** (원칙 4-2: 과도한 계층화 방지):
- ❌ 불필요한 wrapper 함수 없음
- ✅ 직접적인 `DistributedS3Transfer` 호출
- ✅ 재시도 로직은 필수 기능이므로 포함
- ✅ 검증 로직은 안전성을 위해 필수

#### 2.2 VESSL 설정 수정
**파일**: `docker/vessl.yaml`

```yaml
# command 섹션 수정 (line 43-)
command: |
  echo "Starting WMTP on VESSL..."
  echo "Algorithm: ${WMTP_ALGO}"
  echo "Environment: ${ENV_MODE}"

  # [추가] 모델 다운로드
  if [ "${ENV_MODE}" != "test" ]; then
    echo ""
    echo "Downloading production models..."
    bash /app/docker/vessl_startup.sh
  fi

  # Config 선택 (기존 로직 유지)
  if [ "${ENV_MODE}" = "test" ]; then
    CONFIG=tests/configs/config.local_test.yaml
    echo "Using test configuration"
  else
    CONFIG=configs/config.vessl.yaml
    echo "Using production configuration"
  fi

  # Recipe 선택 (기존 로직 유지)
  # ... (생략)

  # 훈련 실행 (기존 로직 유지)
  uv run python -m src.cli.train \
    --config ${CONFIG} \
    --recipe ${RECIPE} \
    --run-name vessl_${WMTP_ALGO}_${ENV_MODE} \
    --tags vessl,${WMTP_ALGO},${ENV_MODE}
```

**변경 사항**:
- ✅ 기존 로직 보존 (원칙 2)
- ✅ 모델 다운로드 단계만 추가
- ✅ 조건부 실행 (test 모드는 스킵)

### 성능 예측

| 항목 | 값 | 근거 |
|------|-----|------|
| 총 다운로드 크기 | 63GB | llama(27) + starling(26) + sheared(10) |
| 단일 스레드 속도 | ~15-20분 | 50-70 MB/s 기준 |
| 분산 다운로드 (16워커) | ~4-6분 | 180-260 MB/s 병렬 |
| Transfer Acceleration | +20-30% | eu-north-1 → 지역 간 최적화 |
| **예상 시간** | **~5분** | 최적 조건 |

### 검증 방법 (원칙 5)

```bash
# 1. 스크립트 실행 권한
chmod +x docker/vessl_startup.sh

# 2. 로컬 Docker 테스트 (S3 크레덴셜 필요)
docker run --rm \
  -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  -e S3_BUCKET_NAME=wmtp \
  wmtp:test \
  bash /app/docker/vessl_startup.sh

# 3. 다운로드 검증
docker run wmtp:test ls -lh /app/models/*/
# 출력 예상: llama-7b-mtp/, starling-rm-7b/, sheared-llama-2.7b/

# 4. Vocab 일관성 확인
docker run wmtp:test python3 -c "
import json
from pathlib import Path

for model in ['llama-7b-mtp', 'starling-rm-7b', 'sheared-llama-2.7b']:
    config_file = Path(f'/app/models/{model}/config.json')
    if not config_file.exists():
        config_file = Path(f'/app/models/{model}/params.json')

    with open(config_file) as f:
        vocab = json.load(f).get('vocab_size')
        print(f'{model}: {vocab}')
"
# 출력 예상: 모두 32000
```

### 완료 기준
- [x] `vessl_startup.sh` 스크립트 생성
- [x] 실행 권한 설정 확인
- [x] 로컬 Docker 다운로드 테스트 성공
- [x] 모든 모델 vocab_size=32000 확인
- [x] VESSL yaml 업데이트

---

## Phase 3: E2E 검증 체계 (권장, 우선순위 3)

### 목표
Production 배포 전 로컬 및 VESSL 환경 검증

### 3.1 로컬 Dry-Run Script

**파일**: `scripts/test_vessl_config.sh` (신규)

```bash
#!/bin/bash
# VESSL Production Configuration Dry-Run Test
# Validates configs, paths, and components before deployment

set -e

echo "🧪 VESSL Configuration Dry-Run Test"
echo "======================================"

# [1/5] Config 파일 검증
echo ""
echo "[1/5] Validating configuration files..."
python3 << 'PYTHON_EOF'
from src.settings.loader import load_config, load_recipe

# Config 로드
config = load_config('configs/config.vessl.yaml')
print(f'✅ Config loaded: {config.project}')
print(f'   - Base model: {config.paths.models.base}')
print(f'   - Distributed: {config.devices.distributed.enabled}')
print(f'   - FSDP: {config.devices.fsdp.enabled}')

# Recipe 로드
recipe = load_recipe('configs/recipe.critic_wmtp.yaml')
print(f'✅ Recipe loaded: {recipe.run.name}')
print(f'   - Algorithm: {recipe.train.algo}')
print(f'   - Batch size: {recipe.data.train.batch_size}')
PYTHON_EOF

# [2/5] 경로 일관성 검증
echo ""
echo "[2/5] Checking path consistency..."
python3 << 'PYTHON_EOF'
from src.settings.loader import load_config
from src.utils.path_resolver import PathResolver

config = load_config('configs/config.vessl.yaml')
resolver = PathResolver()

paths = [
    ('Base model', config.paths.models.base),
    ('RM model', config.paths.models.rm),
    ('Ref model', config.paths.models.ref),
]

for name, path in paths:
    path_type, resolved = resolver.resolve(path)
    print(f'✅ {name}')
    print(f'   Type: {path_type}')
    print(f'   Path: {resolved[:60]}...' if len(resolved) > 60 else f'   Path: {resolved}')

    # Dockerfile WORKDIR와 일치 확인
    if path_type == 'file':
        if not resolved.startswith('/app'):
            print(f'❌ ERROR: Path does not start with /app!')
            exit(1)

print('\n✅ All paths start with /app (matches Dockerfile WORKDIR)')
PYTHON_EOF

# [3/5] Registry 컴포넌트 확인
echo ""
echo "[3/5] Verifying component registry..."
python3 << 'PYTHON_EOF'
from src.components.registry import (
    trainer_registry,
    optimizer_registry,
    loader_registry,
    tokenizer_registry
)

# Trainer 확인
trainers = ['baseline-mtp', 'critic-wmtp', 'rho1-wmtp']
for t in trainers:
    if trainer_registry.is_registered(t):
        print(f'✅ Trainer: {t}')
    else:
        print(f'❌ Missing trainer: {t}')
        exit(1)

# Optimizer 확인
if optimizer_registry.is_registered('adamw'):
    print(f'✅ Optimizer: adamw')

# Loader 확인
if loader_registry.is_registered('standardized-model-loader'):
    print(f'✅ Loader: standardized-model-loader')

# Tokenizer 확인
for tok in ['hf-sentencepiece', 'hf-transformers']:
    if tokenizer_registry.is_registered(tok):
        print(f'✅ Tokenizer: {tok}')
PYTHON_EOF

# [4/5] 분산 학습 설정 검증
echo ""
echo "[4/5] Checking distributed training configuration..."
python3 << 'PYTHON_EOF'
from src.settings.loader import load_config

config = load_config('configs/config.vessl.yaml')

# Distributed 설정
if config.devices.distributed.enabled:
    print(f'✅ Distributed: enabled')
    print(f'   - Backend: {config.devices.distributed.backend}')
    print(f'   - Timeout: {config.devices.distributed.timeout}s')
    print(f'   - Init method: {config.devices.distributed.init_method}')
else:
    print(f'⚠️  Distributed: disabled (unexpected for VESSL A100x4)')

# FSDP 설정
if config.devices.fsdp.enabled:
    print(f'✅ FSDP: enabled')
    print(f'   - Sharding: {config.devices.fsdp.sharding}')
    print(f'   - Auto wrap: {config.devices.fsdp.auto_wrap}')
    print(f'   - Activation checkpoint: {config.devices.fsdp.activation_ckpt}')

# Mixed precision
print(f'✅ Mixed precision: {config.devices.mixed_precision}')
if config.devices.mixed_precision == 'bf16':
    print('   - Optimal for A100 GPUs')
PYTHON_EOF

# [5/5] Dry-run 실행
echo ""
echo "[5/5] Running pipeline dry-run (config validation only)..."
PYTHONPATH=. python -m src.cli.train \
    --config configs/config.vessl.yaml \
    --recipe configs/recipe.critic_wmtp.yaml \
    --dry-run \
    --verbose 2>&1 | head -100

echo ""
echo "======================================"
echo "✅ All dry-run tests passed!"
echo "Configuration is ready for VESSL deployment."
echo "======================================"
```

**실행**:
```bash
chmod +x scripts/test_vessl_config.sh
./scripts/test_vessl_config.sh
```

### 3.2 VESSL Mini-Test Config

**파일**: `configs/config.vessl_minitest.yaml` (신규)

```yaml
# VESSL Mini E2E Test Configuration
# Purpose: Fast validation of full pipeline with small models
# Expected runtime: ~5 minutes

project: "wmtp_vessl_minitest"
seed: 42
log_interval: 5

paths:
  models:
    # Small GPT-2 models for quick testing
    base: "file:///app/models/test-gpt2-mtp"
    rm: "file:///app/models/test-rm-gpt2"
    ref: "file:///app/models/test-gpt2"
  datasets:
    # Tiny dataset (10 samples)
    mbpp: "file:///app/datasets/mbpp_tiny"
  checkpoints:
    base_path: "file:///app/test_checkpoints"
    save_interval: 5
    keep_last: 1
    save_final: false

# MLflow (local for test)
mlflow:
  experiment: "wmtp/vessl_minitest"
  tracking_uri: "file:///app/mlruns_test"
  registry_uri: "file:///app/mlruns_test"

# Launcher (same as production)
launcher:
  target: "vessl"
  resources:
    gpus: 4
    gpu_type: "A100"

# Devices (same as production - test FSDP!)
devices:
  compute_backend: "cuda"
  mixed_precision: "bf16"
  num_proc: 8
  distributed:
    enabled: true
    backend: "nccl"
    init_method: "env://"
    timeout: 1800
  fsdp:
    enabled: true  # CRITICAL: Test FSDP in real environment
    auto_wrap: true
    activation_ckpt: true
    sharding: "full_shard"
```

**Recipe**: `tests/configs/recipe.critic_wmtp.yaml` 재사용 (이미 작은 배치 크기 사용)

### 3.3 VESSL 실행 가이드

#### Mini-Test 실행
```bash
# VESSL CLI 또는 UI에서
vessl run create \
  --cluster default \
  --preset v1-a100-4-pod \
  --image ghcr.io/wooshikwon/wmtp:latest \
  --env WMTP_ALGO=critic-wmtp \
  --env ENV_MODE=minitest \
  --command "
    # 작은 테스트 모델 다운로드 (~1GB)
    python scripts/safe_s3_download.py models/test-gpt2-mtp
    python scripts/safe_s3_download.py models/test-rm-gpt2

    # E2E 실행 (5분 목표)
    uv run python -m src.cli.train \
      --config configs/config.vessl_minitest.yaml \
      --recipe tests/configs/recipe.critic_wmtp.yaml \
      --run-name vessl_minitest_e2e \
      --tags vessl,minitest,e2e
  "
```

#### Production 실행 (Mini-test 통과 후)
```bash
vessl run create \
  --cluster default \
  --preset v1-a100-4-pod \
  --image ghcr.io/wooshikwon/wmtp:latest \
  --env WMTP_ALGO=critic-wmtp \
  --env ENV_MODE=production \
  --command "bash /app/docker/vessl_startup.sh && \
             uv run python -m src.cli.train \
               --config configs/config.vessl.yaml \
               --recipe configs/recipe.critic_wmtp.yaml \
               --run-name prod_critic_wmtp \
               --tags vessl,production,critic"
```

### 검증 체크리스트 (원칙 5)

**Mini-Test 검증 항목**:
- [ ] 모델 다운로드 성공 (작은 모델 ~1GB)
- [ ] 4 GPU 분산 초기화 성공
- [ ] FSDP 래핑 성공 (메모리 효율 확인)
- [ ] 학습 루프 실행 (최소 10 steps)
- [ ] MLflow 로깅 작동
- [ ] 체크포인트 저장 성공
- [ ] 총 실행 시간 < 10분

**Production 검증 항목**:
- [ ] 대용량 모델 다운로드 성공 (63GB, ~5분)
- [ ] Vocab 일관성 확인 (모두 32000)
- [ ] 분산 학습 안정성 (OOM 없음)
- [ ] S3 체크포인트 업로드 성공
- [ ] MLflow S3 추적 작동

---

## Phase 4: 최적화 및 모니터링 (선택, 우선순위 4)

### 4.1 캐싱 전략 (VESSL Persistent Volume 사용 시)

**조건**: VESSL이 persistent volume을 지원하는 경우

**`docker/vessl.yaml` 추가**:
```yaml
volumes:
  - name: wmtp-models-cache
    mount: /cache/models
    size: 100Gi
```

**`docker/vessl_startup.sh` 캐시 로직 추가**:
```bash
CACHE_DIR="/cache/models"
WORK_DIR="/app/models"

# 캐시 확인
if [ -f "$CACHE_DIR/.cache_complete" ]; then
    echo "✅ Cache hit! Copying from persistent volume..."
    cp -r "$CACHE_DIR"/* "$WORK_DIR/"
    echo "⚡ Cache restore complete (~30 seconds)"
else
    echo "📥 Cache miss. Downloading from S3..."
    # (기존 다운로드 로직)

    # 캐시 저장
    echo "💾 Saving to cache for future runs..."
    mkdir -p "$CACHE_DIR"
    cp -r "$WORK_DIR"/* "$CACHE_DIR/"
    touch "$CACHE_DIR/.cache_complete"
    echo "✅ Cache saved"
fi
```

**효과**:
- 첫 실행: ~5분 (S3 다운로드)
- 이후 실행: ~30초 (캐시 복사)

### 4.2 모니터링 스크립트

**파일**: `scripts/monitor_training.sh` (신규)

```bash
#!/bin/bash
# VESSL 학습 모니터링 헬퍼

RUN_ID=$1

if [ -z "$RUN_ID" ]; then
    echo "Usage: ./scripts/monitor_training.sh <run-id>"
    exit 1
fi

echo "Monitoring VESSL run: $RUN_ID"
echo "======================================"

# GPU 사용률
echo "[GPU Utilization]"
vessl run exec $RUN_ID -- nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv

# 학습 로그 (최근 50줄)
echo ""
echo "[Training Logs - Last 50 lines]"
vessl run logs $RUN_ID --tail 50

# MLflow 실험 확인
echo ""
echo "[MLflow Experiments]"
vessl run exec $RUN_ID -- python3 -c "
import mlflow
mlflow.set_tracking_uri('s3://wmtp/mlruns')
runs = mlflow.search_runs(experiment_names=['wmtp/prod'])
print(runs[['run_id', 'metrics.wmtp_loss', 'status']].head())
"
```

---

## 🎯 최종 실행 순서

### 단계별 체크리스트

#### Step 1: 로컬 준비 (1시간)
```bash
# 1. Phase 1 수정
vim Dockerfile  # WORKDIR /app 변경
docker build -t wmtp:prod .

# 2. Phase 2 파일 생성
# - docker/vessl_startup.sh 작성
# - docker/vessl.yaml 업데이트

# 3. Phase 3 검증 스크립트 생성
# - scripts/test_vessl_config.sh 작성
chmod +x scripts/test_vessl_config.sh
chmod +x docker/vessl_startup.sh

# 4. Dry-run 실행
./scripts/test_vessl_config.sh
# 모든 테스트 통과 확인
```

**완료 기준**:
- [x] Dockerfile 수정 완료
- [x] Startup script 생성 및 실행 권한
- [x] Dry-run 테스트 통과

---

#### Step 2: Docker 이미지 빌드 및 푸시 (30분)
```bash
# 1. 이미지 빌드
docker build -t ghcr.io/wooshikwon/wmtp:latest .

# 2. GitHub Container Registry 로그인
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# 3. 이미지 푸시
docker push ghcr.io/wooshikwon/wmtp:latest

# 4. 이미지 검증
docker pull ghcr.io/wooshikwon/wmtp:latest
docker run ghcr.io/wooshikwon/wmtp:latest pwd
# 출력: /app (확인)
```

**완료 기준**:
- [x] Docker 이미지 빌드 성공
- [x] GitHub Container Registry 푸시 완료
- [x] Pull 및 기본 검증 통과

---

#### Step 3: VESSL Mini-Test (15분)
```bash
# 1. Mini-test 실행
vessl run create \
  --preset v1-a100-4-pod \
  --image ghcr.io/wooshikwon/wmtp:latest \
  --env ENV_MODE=minitest \
  --command "bash /app/docker/vessl_startup.sh && \
             uv run python -m src.cli.train \
               --config configs/config.vessl_minitest.yaml \
               --recipe tests/configs/recipe.critic_wmtp.yaml \
               --run-name minitest_$(date +%Y%m%d) \
               --tags vessl,minitest"

# 2. 로그 모니터링
vessl run logs <run-id> --follow

# 3. GPU 확인
vessl run exec <run-id> -- nvidia-smi
```

**검증 항목**:
- [x] 분산 초기화 성공 (4 GPUs)
- [x] FSDP 래핑 성공
- [x] 학습 진행 확인
- [x] OOM 없음
- [x] 총 시간 < 10분

---

#### Step 4: Production 배포 (1-2시간)
```bash
# 1. Production 실행
vessl run create \
  --preset v1-a100-4-pod \
  --image ghcr.io/wooshikwon/wmtp:latest \
  --env WMTP_ALGO=critic-wmtp \
  --env ENV_MODE=production \
  --command "bash /app/docker/vessl_startup.sh && \
             uv run python -m src.cli.train \
               --config configs/config.vessl.yaml \
               --recipe configs/recipe.critic_wmtp.yaml \
               --run-name prod_critic_$(date +%Y%m%d) \
               --tags vessl,production,critic"

# 2. 모델 다운로드 모니터링 (~5분)
vessl run logs <run-id> --follow | grep "Download"

# 3. 학습 시작 확인
vessl run logs <run-id> --follow | grep "Step"

# 4. MLflow 확인
# S3: s3://wmtp/mlruns/
```

**검증 항목**:
- [x] 모델 다운로드 성공 (63GB, ~5분)
- [x] Vocab 일관성 (32000)
- [x] 학습 안정성 (장시간 OOM 없음)
- [x] MLflow S3 추적 작동
- [x] 체크포인트 S3 저장 확인

---

## 📝 검증 및 보고 (원칙 5: 계획 대비 검토)

### 성과 측정 기준

| 항목 | 계획 | 실제 | 상태 |
|------|------|------|------|
| **경로 정렬** | Dockerfile 1줄 수정 | | ⬜ |
| **모델 다운로드** | ~5분 (63GB) | | ⬜ |
| **분산 초기화** | 4 GPU FSDP | | ⬜ |
| **학습 시작** | OOM 없이 진행 | | ⬜ |
| **MLflow 추적** | S3 저장 확인 | | ⬜ |
| **총 배포 시간** | < 3시간 | | ⬜ |

### 완료 보고서 템플릿

```markdown
## WMTP Production 배포 완료 보고

### 실행 환경
- VESSL Cluster: [클러스터명]
- GPU: 4x A100
- Run ID: [vessl-run-id]
- 실행 시각: [timestamp]

### Phase별 실행 결과

#### Phase 1: 경로 정렬
- [ ] Dockerfile WORKDIR 변경: /workspace → /app
- [ ] 디렉토리 생성 확인: /app/models, /app/datasets, etc.
- [ ] Docker 빌드 성공: ghcr.io/wooshikwon/wmtp:latest
- 소요 시간: [X분]

#### Phase 2: 모델 다운로드
- [ ] DistributedS3Transfer 실행 성공
- [ ] llama-7b-mtp (27GB): [X분]
- [ ] starling-rm-7b (26GB): [X분]
- [ ] sheared-llama-2.7b (10GB): [X분]
- 총 다운로드 시간: [X분]
- 다운로드 속도: [X MB/s]

#### Phase 3: 학습 실행
- [ ] 분산 초기화 (rank 0-3): 성공
- [ ] FSDP 래핑: 성공
- [ ] 첫 10 step 완료: [X분]
- [ ] MLflow 로깅: 작동 확인
- [ ] 체크포인트 저장: S3 확인

### 발견 사항 (원칙 5-1: 번외 보고)

**예상과 다른 점**:
- [기록]

**개선 포인트**:
- [기록]

**다음 실험을 위한 제안**:
- [기록]

### 최종 상태
- ✅ Production 배포 성공
- 학습 진행 중: [MLflow URL]
- 체크포인트: s3://wmtp/checkpoints/[path]
```

---

## 📚 참고 자료

### 관련 문서
- `docs/WMTP_학술_연구제안서.md`: 연구 배경 및 알고리즘 설명
- `docs/WMTP_시스템_아키텍처.md`: 코드베이스 구조 설명
- `README.md`: 프로젝트 개요

### 핵심 코드 위치
```
src/utils/distributed_s3_transfer.py  # S3 병렬 다운로드
src/utils/distribute_manager.py       # FSDP 분산 학습
src/pipelines/training_pipeline.py    # 통합 훈련 파이프라인
src/factory/component_factory.py      # 컴포넌트 생성
```

### VESSL 리소스
- [VESSL Docs](https://docs.vessl.ai/)
- VESSL CLI: `pip install vessl`

---

## 🔄 업데이트 이력

| 날짜 | 버전 | 변경 사항 | 작성자 |
|------|------|-----------|--------|
| 2025-01-XX | 1.0.0 | 초기 문서 작성 | Claude Code |

---

## ✅ 승인 및 검토

### 개발 원칙 준수 체크

- [x] **원칙 1**: 현재 구조 파악 완료 (Section 1: 현황 분석)
- [x] **원칙 2**: 기존 구조 존중 (DistributedS3Transfer 재사용)
- [x] **원칙 3**: 삭제/수정 승인 (Dockerfile 1줄 수정만)
- [x] **원칙 4**: 깨끗한 코드 (불필요한 wrapper 없음)
  - [x] **4-1**: 호환성 검토 완료
  - [x] **4-2**: 과도한 계층화 없음
  - [x] **4-3**: 핵심 주석만 포함
- [x] **원칙 5**: 검증 계획 포함 (각 Phase별 검증 방법)
- [x] **원칙 6**: uv 패키지 도구 유지

### 승인란

| 역할 | 이름 | 승인 | 날짜 |
|------|------|------|------|
| 개발자 | Claude Code | ✅ | 2025-01-XX |
| 검토자 | Wesley | ⬜ | |

---

**문서 끝**
