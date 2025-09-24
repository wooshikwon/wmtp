# Loader 내부 메서드 최소화 계획

## 1. 현재 상태 분석

### 1.1 UnifiedModelLoader 내부 메서드
```
model_loader.py (7개 private 메서드):
├── _setup_huggingface()
├── _setup_mtp_native()
├── _setup_mtp_s3()
├── _load_all_hf_models()
├── _load_hf_model()
├── _load_hf_from_path()
└── _load_hf_tokenizer()
```

### 1.2 UnifiedDatasetLoader 내부 메서드
```
dataset_loader.py (12개 private 메서드):
├── _setup_mbpp()
├── _load_mbpp()
├── _load_mbpp_raw()
├── _preprocess_mbpp()
├── _setup_codecontests()
├── _load_codecontests()
├── _load_codecontests_raw()
├── _preprocess_codecontests()
├── _setup_custom()
├── _load_custom()
├── _load_from_json()
└── _load_from_disk()
```

### 1.3 기존 utils 파일
```
src/utils/
├── hf.py       # HuggingFace 관련 유틸리티
├── s3.py       # S3 관련 유틸리티
├── mlflow.py   # MLflow 관련
├── dist.py     # 분산 학습 관련
└── eval.py     # 평가 관련
```

## 2. 리팩토링 설계

### 2.1 새로운 utils 파일 구조
```
src/utils/
├── hf.py          # [확장] HuggingFace 모델 로딩
├── mtp.py         # [신규] Facebook MTP 모델 로딩
├── dataset.py     # [신규] 데이터셋 전처리 및 로딩
├── s3.py          # [유지] S3 관련
└── io.py          # [신규] 파일 I/O 유틸리티
```

### 2.2 메서드 이동 매핑

#### A. model_loader.py → utils 이동

| 현재 메서드 | 이동 위치 | 새 함수명 |
|------------|-----------|-----------|
| `_setup_huggingface()` | - | 삭제 (빈 메서드) |
| `_load_all_hf_models()` | utils/hf.py | `load_all_models()` |
| `_load_hf_model()` | utils/hf.py | `load_model_with_tokenizer()` |
| `_load_hf_from_path()` | utils/hf.py | `load_from_local_path()` |
| `_load_hf_tokenizer()` | utils/hf.py | `load_tokenizer()` |
| `_setup_mtp_native()` | utils/mtp.py | `setup_native_mtp_model()` |
| `_setup_mtp_s3()` | utils/mtp.py | `setup_s3_mtp_model()` |

#### B. dataset_loader.py → utils 이동

| 현재 메서드 | 이동 위치 | 새 함수명 |
|------------|-----------|-----------|
| `_setup_mbpp()` | - | 삭제 (빈 메서드) |
| `_load_mbpp()` | utils/dataset.py | `load_mbpp_dataset()` |
| `_load_mbpp_raw()` | utils/dataset.py | `load_mbpp_raw()` |
| `_preprocess_mbpp()` | utils/dataset.py | `preprocess_mbpp()` |
| `_setup_codecontests()` | - | 삭제 (빈 메서드) |
| `_load_codecontests()` | utils/dataset.py | `load_codecontests_dataset()` |
| `_load_codecontests_raw()` | utils/dataset.py | `load_codecontests_raw()` |
| `_preprocess_codecontests()` | utils/dataset.py | `preprocess_codecontests()` |
| `_setup_custom()` | - | 삭제 (빈 메서드) |
| `_load_custom()` | utils/dataset.py | `load_custom_dataset()` |
| `_load_from_json()` | utils/io.py | `load_dataset_from_json()` |
| `_load_from_disk()` | utils/io.py | `load_dataset_from_disk()` |

## 3. 리팩토링 후 Loader 구조

### 3.1 단순화된 UnifiedModelLoader
```python
class UnifiedModelLoader(ModelLoader):
    def __init__(self, config):
        # 초기화만

    def setup(self, ctx):
        # loader_type에 따라 utils 함수 호출
        if self.loader_type == "mtp-native":
            self.model = mtp.setup_native_mtp_model(self.model_paths, self.device)
            ctx["mtp_model"] = self.model
        # huggingface는 setup 불필요

    def run(self, ctx):
        # loader_type에 따라 utils 함수 호출
        if self.loader_type == "huggingface":
            return hf.load_all_models(self.model_configs, self.cache_dir, self.s3_manager)
        elif self.loader_type == "mtp-native":
            return mtp.get_native_model_dict(self.model)
        # ...
```

### 3.2 단순화된 UnifiedDatasetLoader
```python
class UnifiedDatasetLoader(DatasetLoader):
    def __init__(self, config):
        # 초기화만

    def run(self, ctx):
        # dataset_name에 따라 utils 함수 호출
        if self.dataset_name == "mbpp":
            dataset = dataset_utils.load_mbpp_dataset(
                self.local_path, ctx.get("split"),
                ctx.get("max_length"), ctx.get("add_solution"),
                self.cache_dir, self.s3_manager
            )
        elif self.dataset_name == "codecontests":
            dataset = dataset_utils.load_codecontests_dataset(
                self.local_path, ctx.get("split"),
                ctx.get("language"), ctx.get("difficulty"),
                self.cache_dir, self.s3_manager
            )
        # ...
        return {"dataset": dataset, ...}
```

## 4. 구현 단계

### Phase 1: Utils 파일 생성
1. `src/utils/mtp.py` 생성
   - Facebook MTP 모델 관련 함수들
2. `src/utils/dataset.py` 생성
   - 데이터셋 로딩 및 전처리 함수들
3. `src/utils/io.py` 생성
   - 파일 I/O 관련 함수들

### Phase 2: 기존 utils 확장
1. `src/utils/hf.py` 확장
   - 모델 로딩 관련 함수 추가

### Phase 3: 메서드 이동
1. model_loader.py의 private 메서드들을 utils로 이동
2. dataset_loader.py의 private 메서드들을 utils로 이동

### Phase 4: Loader 클래스 단순화
1. UnifiedModelLoader를 라우터로 변경
2. UnifiedDatasetLoader를 라우터로 변경

### Phase 5: 테스트 및 검증
1. 기존 테스트가 동작하는지 확인
2. 파이프라인 호환성 검증

## 5. 장점

### 5.1 코드 구조 개선
- **Loader 단순화**: 각 loader가 100줄 이내로 축소
- **재사용성**: utils 함수들을 다른 곳에서도 사용 가능
- **테스트 용이성**: 각 util 함수를 독립적으로 테스트 가능

### 5.2 유지보수성
- **명확한 책임 분리**: Loader는 라우팅, Utils는 실제 로직
- **함수 단위 수정**: 특정 기능 수정 시 해당 util 함수만 수정
- **확장 용이**: 새 loader type/dataset 추가 시 util 함수 추가만 필요

### 5.3 가독성
- **Flat structure**: 깊은 중첩 없이 단순한 조건문
- **명시적 함수명**: 각 함수가 하는 일이 명확
- **작은 함수**: 각 함수가 한 가지 일만 수행

## 6. 예상 결과

### Before
```
UnifiedModelLoader: ~400줄 (7개 private 메서드)
UnifiedDatasetLoader: ~500줄 (12개 private 메서드)
```

### After
```
UnifiedModelLoader: ~100줄 (라우터 역할만)
UnifiedDatasetLoader: ~100줄 (라우터 역할만)

utils/mtp.py: ~150줄 (MTP 관련 함수)
utils/dataset.py: ~400줄 (데이터셋 관련 함수)
utils/hf.py: +100줄 (기존 + 모델 로딩 함수)
utils/io.py: ~50줄 (파일 I/O 함수)
```

## 7. 위험 요소 및 대응

### 7.1 순환 참조
- **위험**: utils가 loader를 참조하면 순환 참조 발생
- **대응**: utils는 독립적으로 작동, loader만 utils 참조

### 7.2 컨텍스트 전달
- **위험**: ctx 같은 복잡한 객체 전달 시 결합도 증가
- **대응**: 필요한 파라미터만 명시적으로 전달

### 7.3 S3Manager 의존성
- **위험**: utils 함수가 S3Manager에 의존
- **대응**: Optional 파라미터로 처리, None일 때 로컬만 사용

## 8. 결론

이 리팩토링을 통해:
- ✅ **Loader 코드 80% 감소**: 900줄 → 200줄
- ✅ **재사용 가능한 utils**: 독립적인 함수들
- ✅ **테스트 용이성**: 단위 테스트 작성 간편
- ✅ **확장성**: 새 기능 추가 시 utils에만 함수 추가

Loader는 단순한 라우터가 되고, 실제 로직은 테스트 가능한 utils 함수들로 분리됩니다.
