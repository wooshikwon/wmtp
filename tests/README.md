# WMTP 테스트 환경 가이드

이 문서는 MacBook MPS(Metal Performance Shaders) 환경에서 WMTP 알고리즘을 테스트하기 위한 가이드입니다.

## 환경 설정

### 전제 조건
- Python 3.10+ with `uv` 패키지 관리자
- MacBook M1/M2/M3 (MPS 지원)
- 8GB+ RAM

### uv 환경 설정
```bash
# uv 설치 (아직 없다면)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 프로젝트 의존성 설치
uv sync

# 가상환경 활성화
source .venv/bin/activate
```

## 테스트 환경 구조

### 1. 테스트 모델 (`tests/test_models/`)

#### DistilGPT2-MTP
**경로**: `tests/test_models/distilgpt2-mtp/`

작은 규모의 DistilGPT2 모델(6층, 768차원)에 MTP(Multi-Token Prediction) 래퍼를 씌운 테스트용 모델입니다.

**주요 특징**:
- **Base Model**: DistilGPT2 (82M 파라미터)
- **MTP Heads**: 4개 (t+1, t+2, t+3, t+4 예측)
- **Architecture**: `DistilGPT2MTP` (커스텀 클래스)
- **MPS 최적화**: MacBook에서 효율적 실행을 위한 최적화 포함

**HuggingFace 호환 디렉토리 구조**:
```
distilgpt2-mtp/
├── config.json           # 모델 설정 (mtp_config 포함)
├── modeling.py           # MTP 래퍼 구현
├── pytorch_model.bin     # 가중치 파일
├── tokenizer.json        # 토크나이저
├── tokenizer_config.json
├── special_tokens_map.json
└── vocab.json
```

**MTP 래퍼 구현 방식**:
```python
class DistilGPT2MTP(nn.Module):
    def __init__(self, config=None):
        # 기본 DistilGPT2 모델 로드
        self.base_model = AutoModelForCausalLM.from_pretrained("distilgpt2")

        # 추가 MTP 헤드 생성 (3개 추가 = 총 4개 예측)
        self.extra_heads = nn.ModuleList([
            nn.Linear(hidden_size, vocab_size)
            for _ in range(n_future_tokens - 1)
        ])
```

#### 보조 모델들
- **tiny-reward-model-gpt2**: Critic-WMTP의 Value Head 학습용 보상 모델
- **distilgpt2**: Rho1-WMTP의 참조 모델로 사용

### 2. 테스트 데이터셋 (`tests/test_dataset/`)

#### MBPP (Most Basic Programming Problems)
**경로**: `tests/test_dataset/mbpp/`

간단한 파이썬 프로그래밍 문제 데이터셋의 축소 버전입니다.

**파일 구조**:
```
mbpp/
├── train_mini.jsonl      # 학습용 미니 샘플 (10개)
├── validation.jsonl      # 검증용 샘플
└── test_mini.jsonl       # 테스트용 미니 샘플
```

**데이터 형식** (JSONL):
```json
{
  "task_id": "mbpp_1",
  "text": "Write a function to find the minimum of three numbers.",
  "code": "def min_of_three(a, b, c):\n    return min(a, b, c)",
  "test_list": ["assert min_of_three(1, 2, 3) == 1"]
}
```

#### 기타 데이터셋
- **humaneval**: 함수 구현 벤치마크 (축소 버전)
- **contest**: 경쟁 프로그래밍 문제 (축소 버전)

### 3. 테스트 설정 파일 (`tests/configs/`)

#### config.local_test.yaml
MacBook MPS 환경에 최적화된 하드웨어 설정:
- **compute_backend**: `mps` (Metal Performance Shaders)
- **mixed_precision**: `fp32` (MPS는 bf16 미지원)
- **메모리 최적화**: 체크포인트 최소화, 배치 크기 조정

## 테스트 명령어

### 기본 사용법
```bash
# uv 환경에서 실행
uv run python -m src.cli.train \
    --config tests/configs/config.local_test.yaml \
    --recipe tests/configs/recipe.{알고리즘}.yaml \
    --run-name test_{알고리즘} \
    --tags test,{태그} \
    --verbose
```

### 1. Baseline MTP (균등 가중치)
```bash
uv run python -m src.cli.train \
    --config tests/configs/config.local_test.yaml \
    --recipe tests/configs/recipe.mtp_baseline.yaml \
    --run-name test_baseline \
    --tags test,baseline,mps \
    --verbose
```

### 2. Critic-WMTP (Value Function 기반)
```bash
uv run python -m src.cli.train \
    --config tests/configs/config.local_test.yaml \
    --recipe tests/configs/recipe.critic_wmtp.yaml \
    --run-name test_critic \
    --tags test,critic,mps \
    --verbose
```

**특징**: 2단계 학습 (Stage1: Value Head 사전학습 → Stage2: 메인 학습)

### 3. Rho1-WMTP Weighted (연속 가중치)
```bash
uv run python -m src.cli.train \
    --config tests/configs/config.local_test.yaml \
    --recipe tests/configs/recipe.rho1_wmtp_weighted.yaml \
    --run-name test_rho1_weighted \
    --tags test,rho1,weighted,mps \
    --verbose
```

### 4. Rho1-WMTP Token Skip (이진 선택)
```bash
uv run python -m src.cli.train \
    --config tests/configs/config.local_test.yaml \
    --recipe tests/configs/recipe.rho1_wmtp_tokenskip.yaml \
    --run-name test_rho1_skip \
    --tags test,rho1,tokenskip,mps \
    --verbose
```

**특징**: 하위 30% 토큰을 완전히 제외하는 aggressive한 선택

### 추가 옵션

#### Dry-run (설정 검증만)
```bash
uv run python -m src.cli.train \
    --config tests/configs/config.local_test.yaml \
    --recipe tests/configs/recipe.mtp_baseline.yaml \
    --dry-run \
    --verbose
```

#### 체크포인트에서 재개
```bash
uv run python -m src.cli.train \
    --config tests/configs/config.local_test.yaml \
    --recipe tests/configs/recipe.critic_wmtp.yaml \
    --resume test_checkpoints/test_critic/checkpoint_100.pt \
    --verbose
```

## MPS 최적화 특징

MacBook에서의 효율적 실행을 위한 최적화:

1. **메모리 관리**:
   - 작은 배치 크기 (1-2)
   - Gradient accumulation으로 effective batch size 확보
   - 체크포인트 저장 최소화

2. **데이터 타입**:
   - FP32 사용 (MPS는 BF16 미지원)
   - 필요시 FP16으로 대체 가능

3. **병렬화**:
   - FSDP 비활성화 (단일 GPU)
   - DataLoader workers 조정 (4-8)

## 테스트 결과 확인

### MLflow UI
```bash
# MLflow 서버 시작
mlflow ui --backend-store-uri file:///tmp/mlflow_m3

# 브라우저에서 http://localhost:5000 접속
```

### 로그 파일
```bash
# 실행 로그
tail -f test_checkpoints/test_{알고리즘}/train.log

# 텐서보드 (선택적)
tensorboard --logdir test_checkpoints/
```

## 트러블슈팅

### 일반적인 문제

1. **CUDA/MPS 오류**:
```bash
# MPS 사용 확인
python -c "import torch; print(torch.backends.mps.is_available())"
```

2. **메모리 부족**:
- 배치 크기 감소: `data.train.batch_size: 1`
- Gradient accumulation 증가: `train.gradient_accumulation: 32`

3. **토크나이저 오류**:
```bash
# 토크나이저 캐시 재생성
rm -rf ~/.cache/huggingface/
```

## 성능 벤치마크 (M3 Max 기준)

| 알고리즘 | 스텝당 시간 | 메모리 사용 | 수렴 스텝 |
|---------|------------|------------|----------|
| Baseline MTP | ~0.8s | ~4GB | 100-200 |
| Critic-WMTP | ~1.2s | ~5GB | 150-250 |
| Rho1-Weighted | ~1.0s | ~4.5GB | 100-200 |
| Rho1-TokenSkip | ~0.9s | ~4GB | 80-150 |

## 종합 테스트 스크립트 (선택적)

여러 알고리즘을 순차적으로 테스트하려면 `test_m3_pipeline.py`를 사용할 수 있습니다:

```bash
# 모든 알고리즘 순차 테스트
uv run python tests/script/test_m3_pipeline.py \
    --config tests/configs/config.local_test.yaml \
    --all \
    --verbose

# 특정 알고리즘만 테스트
uv run python tests/script/test_m3_pipeline.py \
    --config tests/configs/config.local_test.yaml \
    --recipe tests/configs/recipe.critic_wmtp.yaml \
    --verbose
```

**참고**: CLI 직접 실행이 더 유연하고 디버깅이 용이하므로, 개별 알고리즘 테스트 시에는 위의 직접 실행 명령어를 권장합니다.

## 추가 리소스

- [WMTP 연구 제안서](../docs/WMTP_학술_연구제안서.md)
- [시스템 아키텍처](../docs/WMTP_시스템_아키텍처.md)
- [메인 README](../README.md)
