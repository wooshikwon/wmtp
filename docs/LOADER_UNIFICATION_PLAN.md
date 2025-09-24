# Loader 모듈 통합 계획

## 1. 현재 구조 분석

### 1.1 기존 Loader 구조
```
src/components/loader/
├── base_loader.py              # 추상 베이스 클래스
├── hf_local_s3_loader.py       # HuggingFace 모델 로더
├── mtp_native_loader.py        # Facebook MTP 네이티브 로더
├── facebook_mtp_s3_loader.py   # Facebook MTP S3 로더
├── dataset_mbpp_loader.py      # MBPP 데이터셋 로더
└── dataset_contest_loader.py   # CodeContests 데이터셋 로더
```

### 1.2 문제점
- **과도한 파일 분산**: 각 loader가 개별 파일로 존재
- **Registry 복잡성**: 각 loader가 개별적으로 registry에 등록
- **Factory 복잡성**: 많은 loader key 매핑 관리
- **확장성 제한**: 새 loader 추가 시 여러 곳 수정 필요

## 2. 통합 설계

### 2.1 목표 구조
```
src/components/loader/
├── base_loader.py        # 기존 유지 (추상 베이스)
├── model_loader.py        # 통합 모델 로더 (새로 생성)
└── dataset_loader.py      # 통합 데이터셋 로더 (새로 생성)
```

### 2.2 UnifiedModelLoader 설계
```python
@loader_registry.register("model-loader")
class UnifiedModelLoader(BaseLoader):
    """모든 모델 로딩 전략을 통합하는 단일 로더"""

    def __init__(self, config: dict):
        super().__init__(config)
        self.loader_type = config.get("loader_type")  # hf, mtp-native, mtp-s3
        self.model_configs = config.get("model_configs", {})

    def setup(self, ctx: dict):
        """loader_type에 따라 적절한 초기화 수행"""
        if self.loader_type == "huggingface":
            self._setup_huggingface(ctx)
        elif self.loader_type == "mtp-native":
            self._setup_mtp_native(ctx)
        elif self.loader_type == "mtp-s3":
            self._setup_mtp_s3(ctx)

    def run(self, ctx: dict) -> dict:
        """통합된 모델 로딩 인터페이스"""
        if self.loader_type == "huggingface":
            return self._load_huggingface_models(ctx)
        elif self.loader_type == "mtp-native":
            return self._load_mtp_native(ctx)
        elif self.loader_type == "mtp-s3":
            return self._load_mtp_s3(ctx)

    # Private methods for each strategy
    def _setup_huggingface(self, ctx): ...
    def _load_huggingface_models(self, ctx): ...
    def _setup_mtp_native(self, ctx): ...
    def _load_mtp_native(self, ctx): ...
    def _setup_mtp_s3(self, ctx): ...
    def _load_mtp_s3(self, ctx): ...
```

### 2.3 UnifiedDatasetLoader 설계
```python
@loader_registry.register("dataset-loader")
class UnifiedDatasetLoader(BaseLoader):
    """모든 데이터셋 로딩 전략을 통합하는 단일 로더"""

    def __init__(self, config: dict):
        super().__init__(config)
        self.dataset_name = config.get("dataset_name")  # mbpp, codecontests, custom
        self.dataset_config = config.get("dataset_config", {})

    def setup(self, ctx: dict):
        """데이터셋별 초기화"""
        if self.dataset_name == "mbpp":
            self._setup_mbpp(ctx)
        elif self.dataset_name == "codecontests":
            self._setup_codecontests(ctx)
        elif self.dataset_name == "custom":
            self._setup_custom(ctx)

    def run(self, ctx: dict) -> dict:
        """통합된 데이터셋 로딩 인터페이스"""
        if self.dataset_name == "mbpp":
            return self._load_mbpp(ctx)
        elif self.dataset_name == "codecontests":
            return self._load_codecontests(ctx)
        elif self.dataset_name == "custom":
            return self._load_custom(ctx)

    # Private methods for each dataset
    def _setup_mbpp(self, ctx): ...
    def _load_mbpp(self, ctx): ...
    def _setup_codecontests(self, ctx): ...
    def _load_codecontests(self, ctx): ...
    def _setup_custom(self, ctx): ...
    def _load_custom(self, ctx): ...
```

## 3. Factory 수정

### 3.1 ComponentFactory 변경
```python
class ComponentFactory:

    @classmethod
    def create_model_loader(cls, config: Config, recipe: Recipe) -> Loader:
        """통합 모델 로더 생성"""

        # Recipe에서 모델 타입 결정
        if recipe.model.base_id == "facebook/multi-token-prediction":
            if "7b_1t_4" in str(config.paths.models.base_local):
                loader_type = "mtp-native"
            else:
                loader_type = "mtp-s3"
        else:
            loader_type = "huggingface"

        # 통합 설정 생성
        loader_config = {
            "loader_type": loader_type,
            "model_configs": {
                "base_path": str(config.paths.models.base_local),
                "rm_path": str(config.paths.models.rm_local),
                "ref_path": str(config.paths.models.ref_local),
                "base_id": recipe.model.base_id,
                "rm_id": recipe.model.rm_id,
                "ref_id": recipe.model.ref_id,
            },
            "cache_dir": str(config.paths.cache),
            "storage": config.storage.model_dump(),
        }

        # 단일 loader 생성
        return loader_registry.create("model-loader", loader_config)

    @classmethod
    def create_data_loader(cls, source: str, config: Config) -> Loader:
        """통합 데이터셋 로더 생성"""

        # 데이터셋 이름 매핑
        dataset_map = {
            "mbpp": "mbpp",
            "contest": "codecontests",
            "codecontests": "codecontests",
        }

        dataset_name = dataset_map.get(source, "custom")

        # 통합 설정 생성
        loader_config = {
            "dataset_name": dataset_name,
            "dataset_config": {
                "local_path": self._get_dataset_path(source, config),
                "split": "train",  # ctx에서 override 가능
                "max_samples": None,
            },
            "cache_dir": str(config.paths.cache),
            "storage": config.storage.model_dump(),
        }

        # 단일 loader 생성
        return loader_registry.create("dataset-loader", loader_config)
```

## 4. YAML 설정 구조

### 4.1 config.yaml (환경 설정)
```yaml
# 경로 설정은 그대로 유지
paths:
  models:
    base_local: "models/7b_1t_4"
    rm_local: "models/Llama_3_8B_RM"
    ref_local: "models/codellama_7b_python"
  datasets:
    mbpp_local: "dataset/mbpp"
    contest_local: "dataset/contest"
  cache: ".cache"

# Storage 설정 그대로 유지
storage:
  mode: "local"  # 또는 "s3"
```

### 4.2 recipe.yaml (알고리즘 설정)
```yaml
# 모델 설정 - Factory가 이를 보고 loader_type 결정
model:
  base_id: "facebook/multi-token-prediction"  # 또는 HF 모델 ID
  rm_id: "meta-llama/Llama-3-8B-RM"
  ref_id: "codellama/CodeLlama-7b-Python-hf"

# 데이터 설정 - Factory가 이를 보고 dataset_name 결정
data:
  train:
    sources: ["mbpp"]  # Factory가 "mbpp" -> dataset_name으로 전달
    max_length: 512
    batch_size: 2
```

## 5. 구현 단계

### Phase 1: 통합 Loader 생성
1. `model_loader.py` 생성
   - 기존 hf_local_s3_loader.py 로직 통합
   - 기존 mtp_native_loader.py 로직 통합
   - 기존 facebook_mtp_s3_loader.py 로직 통합

2. `dataset_loader.py` 생성
   - 기존 dataset_mbpp_loader.py 로직 통합
   - 기존 dataset_contest_loader.py 로직 통합
   - 확장 가능한 구조 설계

### Phase 2: Factory 수정
1. `ComponentFactory.create_model_loader()` 수정
   - Recipe 기반 loader_type 결정 로직
   - 통합 config 생성

2. `ComponentFactory.create_data_loader()` 수정
   - Source 기반 dataset_name 결정 로직
   - 통합 config 생성

### Phase 3: Registry 정리
1. `components/__init__.py` 수정
   - 기존 개별 loader import 제거
   - UnifiedModelLoader, UnifiedDatasetLoader만 import

2. Registry 등록 확인
   - "model-loader" 단일 키
   - "dataset-loader" 단일 키

### Phase 4: 테스트 및 검증
1. 기존 파이프라인 동작 확인
2. 모든 알고리즘 (mtp-baseline, critic-wmtp, rho1-wmtp) 테스트
3. 데이터셋 로딩 테스트

### Phase 5: 정리
1. 기존 개별 loader 파일들을 `_deprecated/` 폴더로 이동
2. 문서 업데이트
3. 최종 검증

## 6. 장점

### 6.1 단순성
- **Registry**: 2개 loader만 관리 (model-loader, dataset-loader)
- **Factory**: 단순한 분기 로직
- **Pipeline**: 변경 불필요, 동일한 인터페이스

### 6.2 확장성
- 새 모델 타입: `UnifiedModelLoader`에 메서드 추가
- 새 데이터셋: `UnifiedDatasetLoader`에 메서드 추가
- Factory 수정 최소화

### 6.3 유지보수성
- 모든 모델 로딩 로직이 한 파일에
- 모든 데이터셋 로딩 로직이 한 파일에
- 명확한 책임 분리

### 6.4 일관성
- 통일된 인터페이스
- 동일한 설정 구조
- 예측 가능한 동작

## 7. 마이그레이션 안전성

### 7.1 단계적 마이그레이션
1. 새 통합 loader 생성 (기존 코드 유지)
2. Factory에서 통합 loader 사용하도록 전환
3. 테스트 완료 후 기존 loader 제거

### 7.2 롤백 가능성
- 기존 loader 파일 보존 (_deprecated/)
- Factory에서 쉽게 전환 가능
- Git history 보존

## 8. 예상 결과

### Before (7개 파일, 복잡한 registry)
```
loader/
├── base_loader.py
├── hf_local_s3_loader.py
├── mtp_native_loader.py
├── facebook_mtp_s3_loader.py
├── dataset_mbpp_loader.py
├── dataset_contest_loader.py
└── __init__.py

Registry keys: 5개
Factory mappings: 복잡한 매핑 테이블
```

### After (3개 파일, 단순한 registry)
```
loader/
├── base_loader.py      # 유지
├── model_loader.py     # 통합
├── dataset_loader.py   # 통합
└── __init__.py

Registry keys: 2개 (model-loader, dataset-loader)
Factory logic: 단순한 조건문
```

## 9. 결론

이 통합 계획은:
- ✅ **코드 복잡도 감소**: 7개 → 3개 파일
- ✅ **Registry 단순화**: 5개 → 2개 키
- ✅ **Factory 로직 단순화**: 복잡한 매핑 → 단순 조건문
- ✅ **확장성 향상**: 새 loader 추가가 용이
- ✅ **파이프라인 영향 없음**: 인터페이스 동일

Recipe와 Config YAML의 구조는 그대로 유지되며, Factory가 모든 분기를 처리하므로 사용자 경험은 변하지 않습니다.

# [TODO] s3, local load 방식은 util로
