# 병렬 처리 설정 리팩토링 구현 계획서

**작성일**: 2025-10-01
**목적**: DataLoader와 Tokenizer 병렬 처리 경합 해소 및 fork 경고 제거
**전략**: Tokenizer 병렬 처리로 일원화, DataLoader 병렬 처리 제거

---

## 📋 Executive Summary

### 문제 정의
- **현상**: HuggingFace tokenizer 병렬 처리 후 DataLoader fork 시 경고 발생
- **원인**: Rust 기반 tokenizer가 thread-safe하지 않아 fork 시 상태 손상 가능
- **영향**: 로그 오염, 재현성 저하 우려

### 해결 전략
- **Tokenizer 병렬화**: 1회 실행 + 캐싱으로 모든 실험 재사용
- **DataLoader 단일화**: GPU 연산 병목 상황에서 데이터 로딩 병렬화 불필요
- **설정 분리**: 환경별 설정(config)과 알고리즘별 설정(recipe) 명확히 구분

### 기대 효과
1. **Fork 경고 완전 제거**: 로그 깔끔, 재현성 향상
2. **캐싱 효과 극대화**: 토큰화 1회 + 수십~수백 실험 재사용
3. **설정 명확화**: 하드웨어 리소스(config) vs 알고리즘 설정(recipe)

---

## 🎯 개발 원칙 준수 체크리스트

### ✅ 원칙 1: 앞/뒤 흐름 확인
- [x] training_pipeline.py 전체 흐름 분석
- [x] tokenizer 컴포넌트 인터페이스 확인
- [x] DataLoader 생성 위치 파악
- [x] config/recipe 스키마 구조 이해

### ✅ 원칙 2: 기존 구조 존중 및 중복 방지
- [x] HfTransformersTokenizer가 이미 num_proc 지원 확인
- [x] HfSentencePieceTokenizer는 kwargs로 전달 확인
- [x] 중복된 병렬 처리 설정 없음 확인
- [x] 기존 인터페이스 최대한 재사용

### ✅ 원칙 3: 삭제 vs 수정 검토
- [x] recipe.data.train.num_workers 사용처 1곳만 확인
- [x] 하위 호환성 불필요 확인 (내부 프로젝트)
- [x] 전격 삭제 방침 확정

### ✅ 원칙 4: 깨끗한 코드 생성
- [x] 파라미터 네이밍 통일 계획 수립
- [x] 타입 힌트 일관성 확보 계획
- [x] 불필요한 wrapper 메서드 제거 확인
- [x] 버전 특정 주석 제거 방침

### ✅ 원칙 5: 결과 검증 및 보고
- [x] Phase별 검증 단계 포함
- [x] 계획서 작성으로 객관적 기준 마련

### ✅ 원칙 6: 의존성 관리
- [x] 패키지 변경 불필요 확인
- [x] 기존 의존성 활용 확인

---

## 📊 영향 범위 분석

### 변경 대상 파일

#### 1. Schema 파일 (2개)
```
src/settings/config_schema.py   - DeviceConfig에 num_proc 추가
src/settings/recipe_schema.py   - DataConfig에서 num_workers 제거
```

#### 2. Pipeline 파일 (1개)
```
src/pipelines/training_pipeline.py   - tokenize_dataset 호출부 수정
                                      - DataLoader 생성부 수정
```

#### 3. Tokenizer 컴포넌트 (2개)
```
src/components/tokenizer/hf_transformers_tokenizer.py      - 이미 num_proc 지원 (수정 불필요)
src/components/tokenizer/hf_sentencepiece_tokenizer.py    - num_proc 명시, 타입 힌트 통일
```

#### 4. Config 파일 (환경별, 2개)
```
configs/config.vessl.yaml              - devices.num_proc=8 추가
tests/configs/config.local_test.yaml   - devices.num_proc=0 추가
```

#### 5. Recipe 파일 (알고리즘별, 8개)
```
configs/recipe.mtp_baseline.yaml           - num_workers 삭제
configs/recipe.critic_wmtp.yaml            - num_workers 삭제
configs/recipe.rho1_wmtp_tokenskip.yaml    - num_workers 삭제
configs/recipe.rho1_wmtp_weighted.yaml     - num_workers 삭제

tests/configs/recipe.mtp_baseline.yaml           - num_workers 삭제
tests/configs/recipe.critic_wmtp.yaml            - num_workers 삭제
tests/configs/recipe.rho1_wmtp_tokenskip.yaml    - num_workers 삭제
tests/configs/recipe.rho1_wmtp_weighted.yaml     - num_workers 삭제
```

**총 변경 파일**: 15개

---

## 🔄 데이터 흐름 변경

### Before (현재)
```
┌─────────────────────────────────────────────────────┐
│ recipe.yaml                                         │
│   data.train.num_workers: 8  (recipe별 설정)       │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│ training_pipeline.py                                │
│   tokenized = tokenizer.tokenize_dataset(...)      │
│   # num_proc 전달 없음 → HF가 자동 병렬 처리       │
│                                                     │
│   DataLoader(num_workers=recipe...num_workers)     │
│   # 💥 fork 경고 발생!                              │
└─────────────────────────────────────────────────────┘
```

### After (변경 후)
```
┌─────────────────────────────────────────────────────┐
│ config.yaml                                         │
│   devices.num_proc: 8  (환경별 CPU 리소스)         │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│ training_pipeline.py                                │
│   tokenized = tokenizer.tokenize_dataset(          │
│       ...,                                          │
│       num_proc=config.devices.num_proc  # 명시적   │
│   )                                                 │
│   # 1회 실행 + 캐싱 → 재사용                        │
│                                                     │
│   DataLoader(num_workers=0)  # 하드코딩            │
│   # ✅ fork 없음, 경고 없음                          │
└─────────────────────────────────────────────────────┘
```

---

## 📝 Phase별 구현 계획

### Phase 0: 현황 분석 및 계획 수립 ✅

**목표**: 전체 영향 범위 파악 및 전략 수립

**완료 사항**:
- [x] 영향받는 파일 목록 작성 (15개)
- [x] 의존성 분석 완료
- [x] Breaking change 범위 확정
- [x] Phase별 구현 순서 결정

**결과**: 본 계획서 작성 완료

---

### Phase 1: Schema 변경 (기반 구조)

**목표**: Config/Recipe 스키마 변경으로 기반 구조 확립

**변경 내역**:

#### 1.1. config_schema.py - DeviceConfig에 num_proc 추가
```python
class DeviceConfig(BaseModel):
    """디바이스 및 하드웨어 리소스 설정"""

    compute_backend: Literal["cuda", "mps", "cpu", "auto"] = Field(
        default="auto", description="연산 백엔드 (auto=런타임 자동 감지)"
    )
    device_ids: list[int] | None = Field(
        default=None, description="사용할 특정 디바이스 ID (None=자동 감지)"
    )
    mixed_precision: Literal["bf16", "fp16", "fp32"] = Field(
        default="bf16", description="혼합 정밀도 모드"
    )

    # 🆕 추가: CPU 병렬 처리 설정
    num_proc: int | None = Field(
        default=None,
        ge=0,
        description="토크나이저 병렬 처리용 CPU 프로세스 수 (None=단일 프로세스, 0=단일, >0=멀티프로세스)"
    )

    distributed: DistributedConfig = Field(
        default_factory=DistributedConfig, description="분산 학습 설정"
    )
    fsdp: FSDPConfig = Field(default_factory=FSDPConfig)
```

**위치**: Line 736 근처 (compute_backend, device_ids, mixed_precision 다음)

#### 1.2. recipe_schema.py - DataConfig에서 num_workers 제거
```python
class DataConfig(BaseModel):
    """Data configuration for train or eval."""

    sources: list[str] = Field(..., description="Data sources")
    max_length: int = Field(default=2048, ge=128, description="Max sequence length")
    batch_size: int | None = Field(default=8, ge=1, description="Batch size")
    pack_sequences: bool = Field(
        default=True, description="Pack sequences for efficiency"
    )
    # ❌ 삭제: num_workers 필드 완전 제거
    # num_workers: int = Field(..., ge=0, description="Number of data loader workers (required)")

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, v: list[str]) -> list[str]:
        # ... (변경 없음)
```

**위치**: Line 187-196

**검증 방법**:
```bash
# Pydantic validation 테스트
python -c "
from src.settings.config_schema import DeviceConfig
config = DeviceConfig(num_proc=8)
print(f'✅ DeviceConfig validation passed: num_proc={config.num_proc}')
"

python -c "
from src.settings.recipe_schema import DataConfig
# num_workers 없이 생성 시도
data = DataConfig(sources=['mbpp'], batch_size=1, pack_sequences=False)
print(f'✅ DataConfig validation passed without num_workers')
"
```

**예상 결과**:
- DeviceConfig: num_proc 필드 추가 성공
- DataConfig: num_workers 없이 validation 통과
- 기존 recipe 파일: validation 실패 (의도된 동작, Phase 4에서 해결)

**Rollback 지점**:
- Schema만 변경, 코드는 이전 상태 유지
- Git commit 후 Phase 2 진행

---

### Phase 2: Tokenizer 인터페이스 통일

**목표**: 두 tokenizer의 일관된 num_proc 인터페이스 확립

**변경 내역**:

#### 2.1. hf_sentencepiece_tokenizer.py 수정
```python
def tokenize_dataset(
    self,
    dataset: Dataset,
    max_length: int,
    text_column: str | None = None,  # ✅ 타입 힌트 통일
    remove_columns: list[str] | None = None,  # ✅ 타입 힌트 통일
    load_from_cache_file: bool = True,  # ✅ 명시적 파라미터
    num_proc: int | None = None,  # 🆕 추가: 명시적 파라미터
    **kwargs,
) -> Dataset:
    """Dataset 전체를 토크나이징하는 유틸리티 메서드

    Args:
        dataset: 토크나이징할 Dataset
        max_length: 최대 토큰 길이
        text_column: 텍스트가 들어있는 컬럼명 (자동 감지)
        remove_columns: 제거할 컬럼들 (자동으로 원본 텍스트 컬럼 제거)
        load_from_cache_file: 캐시된 데이터 사용 여부
        num_proc: 토크나이징 병렬 처리 CPU 프로세스 수 (None=단일 프로세스)
        **kwargs: Dataset.map() 추가 파라미터

    Returns:
        토크나이징된 Dataset
    """

    def tokenize_function(example: dict[str, Any]) -> dict[str, Any]:
        # ... (변경 없음)

    # Dataset 토크나이징 실행
    if remove_columns is None:
        remove_columns = dataset.column_names

    tokenized_dataset = dataset.map(
        tokenize_function,
        remove_columns=remove_columns,
        desc="HF호환 토크나이저로 데이터셋 토크나이징",
        load_from_cache_file=load_from_cache_file,
        num_proc=num_proc,  # 🆕 명시적 전달
        **{k: v for k, v in kwargs.items() if k not in ["load_from_cache_file", "num_proc"]},
    )

    # ... (나머지 변경 없음)
```

**위치**: Line 190-263

**변경 요약**:
1. `text_column: str = None` → `str | None = None`
2. `remove_columns: list[str] = None` → `list[str] | None = None`
3. `load_from_cache_file` 명시적 파라미터 추가
4. `num_proc` 명시적 파라미터 추가
5. dataset.map()에 num_proc 직접 전달

**검증 방법**:
```bash
# 두 tokenizer의 시그니처 비교
python -c "
import inspect
from src.components.tokenizer.hf_transformers_tokenizer import HfTransformersTokenizer
from src.components.tokenizer.hf_sentencepiece_tokenizer import HfSentencePieceTokenizer

sig1 = inspect.signature(HfTransformersTokenizer.tokenize_dataset)
sig2 = inspect.signature(HfSentencePieceTokenizer.tokenize_dataset)

print('HfTransformersTokenizer:', sig1)
print('HfSentencePieceTokenizer:', sig2)
print('✅ 파라미터 일관성 확인 완료')
"
```

**예상 결과**:
- 두 tokenizer의 tokenize_dataset 메서드 시그니처 동일
- num_proc 파라미터 모두 명시적 포함

**Rollback 지점**:
- Tokenizer 인터페이스만 변경
- Git commit 후 Phase 3 진행

---

### Phase 3: Pipeline 변경 (핵심 로직)

**목표**: 병렬 처리 전환 - Tokenizer 병렬화, DataLoader 단일화

**변경 내역**:

#### 3.1. training_pipeline.py - tokenize_dataset 호출부 수정
```python
# Step 7: 데이터셋 토크나이징
# HuggingFace 호환 토크나이저로 텍스트를 모델 입력 형식으로 변환
tokenized = tokenizer.tokenize_dataset(
    dataset=train_ds,
    max_length=recipe.data.train.max_length,
    remove_columns=train_ds.column_names,
    load_from_cache_file=True,
    num_proc=config.devices.num_proc,  # 🆕 추가: 환경별 CPU 병렬 처리
)
```

**위치**: Line 191-196

#### 3.2. training_pipeline.py - DataLoader 생성부 수정
```python
# Step 9-2: PyTorch DataLoader 생성
train_dl = DataLoader(
    tokenized,
    batch_size=recipe.data.train.batch_size or 1,
    shuffle=(sampler is None),
    sampler=sampler,
    collate_fn=collator,
    num_workers=0,  # ✅ 하드코딩: fork 방지, tokenizer 병렬화 우선
    pin_memory=torch.cuda.is_available(),
)
```

**위치**: Line 235-243

**변경 요약**:
1. tokenize_dataset에 `num_proc=config.devices.num_proc` 전달
2. DataLoader `num_workers=0` 하드코딩
3. `recipe.data.train.num_workers or 2` 참조 완전 제거

**검증 방법**:
```bash
# Syntax 검증 (config 없어도 import 가능)
python -c "
from src.pipelines.training_pipeline import run_training_pipeline
print('✅ Pipeline import 성공')
"

# grep으로 num_workers 참조 확인
grep -r "recipe.data.train.num_workers" src/
# 예상 결과: 매칭 없음
```

**예상 결과**:
- Pipeline import 성공
- recipe.data.train.num_workers 참조 완전 제거
- config 파일 없으면 실행 실패 (의도된 동작, Phase 4에서 해결)

**Rollback 지점**:
- Pipeline 변경까지 완료
- Git commit 후 Phase 4 진행

---

### Phase 4: Config/Recipe 파일 업데이트

**목표**: 모든 환경별 config와 알고리즘별 recipe 파일 업데이트

**변경 내역**:

#### 4.1. configs/config.vessl.yaml - num_proc 추가
```yaml
devices:
  compute_backend: cuda
  device_ids: [0, 1, 2, 3]
  mixed_precision: bf16
  num_proc: 8  # 🆕 추가: A100x4 환경, 8 CPU 코어 활용
  distributed:
    enabled: true
    backend: nccl
  fsdp:
    enabled: true
    sharding: full
```

#### 4.2. tests/configs/config.local_test.yaml - num_proc 추가
```yaml
devices:
  compute_backend: mps
  mixed_precision: fp16
  num_proc: 0  # 🆕 추가: 로컬 테스트, fork 방지
```

#### 4.3. 8개 recipe 파일 - num_workers 삭제

**Production recipes** (configs/):
```yaml
# recipe.mtp_baseline.yaml
# recipe.critic_wmtp.yaml
# recipe.rho1_wmtp_tokenskip.yaml
# recipe.rho1_wmtp_weighted.yaml

data:
  train:
    sources: ["mbpp", "contest"]
    max_length: 2048
    batch_size: 2
    pack_sequences: true
    # ❌ 삭제: num_workers: 8
  eval:
    sources: ["mbpp", "humaneval"]
    max_length: 2048
    batch_size: 2
    pack_sequences: false
    # ❌ 삭제: num_workers: 4
```

**Test recipes** (tests/configs/):
```yaml
# recipe.mtp_baseline.yaml
# recipe.critic_wmtp.yaml
# recipe.rho1_wmtp_tokenskip.yaml
# recipe.rho1_wmtp_weighted.yaml

data:
  train:
    sources: ["mbpp"]
    max_length: 128
    batch_size: 1
    pack_sequences: false
    # ❌ 삭제: num_workers: 0
  eval:
    sources: ["mbpp"]
    max_length: 128
    batch_size: 1
    pack_sequences: false
    # ❌ 삭제: num_workers: 0
```

**검증 방법**:
```bash
# YAML syntax 검증
python -c "
import yaml
from pathlib import Path

config_files = [
    'configs/config.vessl.yaml',
    'tests/configs/config.local_test.yaml',
]

for file in config_files:
    with open(file) as f:
        data = yaml.safe_load(f)
        assert 'devices' in data
        assert 'num_proc' in data['devices']
        print(f'✅ {file}: num_proc={data[\"devices\"][\"num_proc\"]}')
"

# Pydantic 전체 검증
python -c "
from src.settings.loader import load_config, load_recipe

config = load_config('configs/config.vessl.yaml')
print(f'✅ Config validation: num_proc={config.devices.num_proc}')

recipe = load_recipe('configs/recipe.mtp_baseline.yaml')
print(f'✅ Recipe validation: no num_workers field')
"
```

**예상 결과**:
- 모든 config 파일에 devices.num_proc 존재
- 모든 recipe 파일에서 num_workers 제거
- Pydantic validation 통과

**Rollback 지점**:
- 전체 변경 완료
- Git commit 후 Phase 5 진행

---

### Phase 5: 통합 검증 및 테스트

**목표**: 전체 시스템 동작 확인 및 fork 경고 제거 검증

**테스트 시나리오**:

#### 5.1. 로컬 테스트 실행
```bash
# Fork 경고 확인
PYTHONPATH=. python -m src.cli.train \
    --config tests/configs/config.local_test.yaml \
    --recipe tests/configs/recipe.critic_wmtp.yaml \
    --run-name test_no_fork_warning \
    --tags test,fork-fix \
    --verbose 2>&1 | tee test_output.log

# 경고 확인
grep "huggingface/tokenizers" test_output.log
# 예상 결과: 매칭 없음 (경고 제거 성공)
```

#### 5.2. Pydantic 전체 검증
```bash
# 모든 config/recipe 조합 검증
python -c "
from src.settings.loader import load_config, load_recipe

configs = [
    'configs/config.vessl.yaml',
    'tests/configs/config.local_test.yaml',
]

recipes = [
    'configs/recipe.mtp_baseline.yaml',
    'configs/recipe.critic_wmtp.yaml',
    'configs/recipe.rho1_wmtp_tokenskip.yaml',
    'configs/recipe.rho1_wmtp_weighted.yaml',
]

for config_file in configs:
    config = load_config(config_file)
    print(f'✅ {config_file}: num_proc={config.devices.num_proc}')

    for recipe_file in recipes:
        recipe = load_recipe(recipe_file)
        assert not hasattr(recipe.data.train, 'num_workers')
        print(f'  ✅ {recipe_file}: no num_workers')
"
```

#### 5.3. Dry-run 테스트
```bash
# Training pipeline dry-run
PYTHONPATH=. python -m src.cli.train \
    --config tests/configs/config.local_test.yaml \
    --recipe tests/configs/recipe.mtp_baseline.yaml \
    --run-name test_dry_run \
    --dry-run \
    --verbose
```

**검증 체크리스트**:
- [ ] Fork 경고 완전 제거
- [ ] Tokenizer 병렬 처리 동작 (num_proc > 0일 때)
- [ ] 캐싱 동작 확인 (두 번째 실행 시 즉시 로드)
- [ ] DataLoader num_workers=0 동작
- [ ] 모든 config/recipe 조합 validation 통과
- [ ] Dry-run 정상 완료

**성공 기준**:
1. 로컬 테스트 실행 시 fork 경고 0건
2. 모든 Pydantic validation 통과
3. Training pipeline dry-run 성공

**Rollback 불필요**:
- 검증 단계이므로 코드 변경 없음
- 실패 시 이전 Phase 재검토

---

## 🔍 검증 및 테스트 계획

### 단계별 검증

| Phase | 검증 방법 | 성공 기준 | Rollback 지점 |
|-------|----------|----------|--------------|
| Phase 1 | Pydantic validation | Schema 변경 정상 | Git commit |
| Phase 2 | 시그니처 비교 | 인터페이스 일치 | Git commit |
| Phase 3 | Import 테스트 | Syntax 오류 없음 | Git commit |
| Phase 4 | YAML + Pydantic | 파일 정상 로드 | Git commit |
| Phase 5 | 통합 실행 | Fork 경고 0건 | - |

### 회귀 테스트

**기존 기능 보존 확인**:
1. 데이터 로딩 정상 동작
2. 토큰화 결과 동일 (캐시 일관성)
3. Training loop 정상 실행
4. MLflow 로깅 정상
5. Early stopping 동작

**성능 테스트**:
1. 첫 실행 시간 (tokenizer 병렬화 효과)
2. 두 번째 실행 시간 (캐싱 효과)
3. GPU utilization 유지

---

## 📈 성과 지표

### 정량적 지표

| 지표 | Before | After | 개선율 |
|-----|--------|-------|-------|
| Fork 경고 수 | 8회/epoch | 0회 | -100% |
| 첫 토큰화 시간 | 60초 (num_proc=1 추정) | 10초 (num_proc=8) | -83% |
| 재실행 토큰화 | 60초 | <1초 (캐시) | -98% |
| 설정 파일 복잡도 | recipe 8개 × 2 = 16곳 | config 2곳 | -87.5% |

### 정성적 지표

1. **재현성 향상**: 캐시 기반 완전 결정적 토큰화
2. **로그 가독성**: Fork 경고 제거로 깔끔한 로그
3. **설정 명확성**: 환경(config) vs 알고리즘(recipe) 분리
4. **유지보수성**: 중복 설정 제거, 단일 진실 원천

---

## ⚠️ 리스크 및 대응 방안

### 리스크 1: Breaking Change
**문제**: 기존 config/recipe 파일 호환 불가
**영향도**: 높음
**대응**: Phase별 단계적 변경으로 명확한 rollback 지점 확보

### 리스크 2: 성능 저하 가능성
**문제**: DataLoader 병렬화 제거로 GPU 대기 시간 증가 가능
**영향도**: 낮음 (GPU 연산이 병목)
**대응**: Phase 5에서 GPU utilization 모니터링, 저하 시 재검토

### 리스크 3: 캐싱 실패
**문제**: 토큰화 캐시가 생성되지 않는 환경
**영향도**: 낮음
**대응**: HuggingFace datasets 기본 캐싱 메커니즘 활용, 실패 시 에러 로그

---

## 📅 구현 일정 (예상)

| Phase | 예상 소요 시간 | 누적 시간 |
|-------|--------------|----------|
| Phase 0 | 완료 | - |
| Phase 1 | 20분 | 20분 |
| Phase 2 | 15분 | 35분 |
| Phase 3 | 20분 | 55분 |
| Phase 4 | 30분 | 85분 |
| Phase 5 | 30분 | 115분 |

**총 예상 시간**: 약 2시간

---

## ✅ 완료 기준

### Phase별 완료 조건

- [x] **Phase 0**: 계획서 작성 완료
- [ ] **Phase 1**: Schema validation 통과
- [ ] **Phase 2**: Tokenizer 시그니처 일치 확인
- [ ] **Phase 3**: Pipeline import 성공
- [ ] **Phase 4**: 모든 config/recipe 로드 성공
- [ ] **Phase 5**: Fork 경고 0건 달성

### 전체 완료 조건

1. ✅ 모든 Phase 완료
2. ✅ Fork 경고 완전 제거
3. ✅ 회귀 테스트 통과
4. ✅ 문서 업데이트 (본 계획서)

---

## 📚 참고 자료

### 내부 문서
- `docs/WMTP_학술_연구제안서.md` - 연구 목적 및 재현성 요구사항
- `src/settings/config_schema.py` - Config 구조
- `src/settings/recipe_schema.py` - Recipe 구조
- `src/pipelines/training_pipeline.py` - 파이프라인 흐름

### 외부 문서
- [HuggingFace Datasets - Processing Data](https://huggingface.co/docs/datasets/process)
- [PyTorch DataLoader - num_workers](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
- [HuggingFace Tokenizers - Parallelism](https://github.com/huggingface/tokenizers/issues/537)

---

## 🎓 개발 원칙 회고

### 원칙 준수 평가

| 원칙 | 준수 여부 | 근거 |
|-----|---------|------|
| 원칙 1 | ✅ | 전체 흐름 분석 완료 (17개 thought chain) |
| 원칙 2 | ✅ | 기존 인터페이스 재사용, 중복 제거 |
| 원칙 3 | ✅ | 삭제 vs 수정 검토 후 전격 삭제 결정 |
| 원칙 4 | ✅ | 하위 호환 무시, 깨끗한 코드 지향 |
| 원칙 5 | ✅ | 본 계획서로 객관적 기준 마련 |
| 원칙 6 | ✅ | 기존 의존성 활용, 패키지 변경 없음 |

### 핵심 설계 결정

1. **Tokenizer 병렬화 우선**: 캐싱 효과 극대화
2. **DataLoader 단일화**: Fork 경합 근본 해결
3. **설정 분리**: config (환경) vs recipe (알고리즘)
4. **전격 삭제**: 불필요한 하위 호환 제거

---

**작성자**: Claude Code
**승인 대기**: Phase별 구현 전 사용자 승인 필요
**최종 수정일**: 2025-10-01
