# WMTP Metadata 기반 로더 리팩토링 계획

## 🎯 목표
- metadata.json 기반 명확한 모델 로딩 전략 구현
- StandardizedModelLoader 구조 단순화 (12개+ → 4개 메서드)
- ComponentFactory 하드코딩 분기 제거
- 가시성 높고 유지보수 쉬운 구조 달성

## 📋 현재 문제점

### 1. 파일 기반 감지의 한계
- `modeling.py` 존재 여부로 MTP 모델 추측
- 예측 불가능하고 에러 prone한 방식

### 2. 복잡한 메서드 구조
- `_load_xxx_from_yyy` 패턴으로 12개+ 내부 메서드
- 중복된 S3/로컬 분기 로직
- 가독성과 유지보수성 저하

### 3. 하드코딩된 Factory 분기
- 알고리즘별 경로 매핑이 하드코딩됨
- Recipe와 모델 호환성 검증 부재

## 🚀 해결 전략

### Phase 1: 확장된 Metadata Schema

```json
{
  "wmtp_type": "base_model|reference_model|reward_model",
  "training_algorithm": "mtp|baseline|critic|rho1",
  "base_architecture": "gpt2|llama|mistral",
  "storage_version": "2.0",
  "loading_strategy": {
    "loader_type": "custom_mtp|huggingface",
    "model_class_name": "GPTMTPForCausalLM",
    "custom_module_file": "modeling.py",
    "transformers_class": "AutoModelForCausalLM|AutoModel|null",
    "state_dict_mapping": {
      "remove_prefix": "base_model.",
      "add_prefix": null,
      "key_transforms": {}
    },
    "required_files": ["config.json", "model.safetensors", "modeling.py"]
  },
  "algorithm_compatibility": ["baseline-mtp", "critic-wmtp", "rho1-wmtp"]
}
```

### Phase 2: 단순화된 Loader 구조

```python
class StandardizedModelLoader:
    def load_model(self, model_path: str) -> Any:
        """메인 로딩 인터페이스"""
        metadata = self._load_metadata(model_path)
        strategy = metadata["loading_strategy"]
        return self._load_with_strategy(model_path, strategy)

    def _load_with_strategy(self, path: str, strategy: dict) -> Any:
        """전략 패턴으로 통합 로딩"""
        loader_type = strategy["loader_type"]

        if loader_type == "custom_mtp":
            return self._load_custom_model(path, strategy)
        elif loader_type == "huggingface":
            return self._load_huggingface_model(path, strategy)

    def _load_custom_model(self, path: str, strategy: dict) -> Any:
        """커스텀 모델 로딩 (MTP 등)"""
        # 동적 모듈 로드 + state_dict 매핑

    def _load_huggingface_model(self, path: str, strategy: dict) -> Any:
        """HuggingFace 모델 로딩"""
        # transformers_class 기반 정확한 로딩
```

### Phase 3: 개선된 Factory 로직

```python
def create_model_loader(config: Config, recipe: Recipe = None) -> Loader:
    """알고리즘 호환성 기반 모델 자동 선택"""
    if not recipe:
        return loader_registry.create("standardized-model-loader", config)

    # 모든 가능한 모델 경로에서 metadata 확인
    candidate_paths = [
        config.paths.models.base,
        config.paths.models.ref,
        config.paths.models.rm
    ]

    compatible_model = self._find_compatible_model(
        recipe.train.algo,
        candidate_paths
    )

    loader_config = {**config, "model_path": compatible_model}
    return loader_registry.create("standardized-model-loader", loader_config)

def _find_compatible_model(self, algorithm: str, paths: list) -> str:
    """metadata의 algorithm_compatibility로 매칭"""
    for path in paths:
        metadata = load_metadata(path)
        if algorithm in metadata.get("algorithm_compatibility", []):
            return path
    raise ValueError(f"No compatible model found for {algorithm}")
```

## 📊 개선 효과 비교

| 항목 | Before (현재) | After (개선) |
|------|---------------|--------------|
| 내부 메서드 수 | 12개+ (`_load_xxx_from_yyy`) | 4개 핵심 메서드 |
| 모델 감지 방식 | 파일 존재 여부 추측 | metadata 명시적 전략 |
| Factory 분기 | 하드코딩된 알고리즘 매핑 | 호환성 자동 검증 |
| 확장성 | 새 모델 타입마다 메서드 추가 | metadata schema 확장만 |
| 테스트 용이성 | 복잡한 분기 로직 | 단순한 전략 패턴 |

## 🎯 구현 단계

### Phase 1: Metadata Schema 확장 ✅ 진행중
- [ ] 기존 모델들의 metadata.json 업데이트
- [ ] loading_strategy 필드 추가
- [ ] algorithm_compatibility 정의

### Phase 2: Loader 리팩토링
- [ ] 새로운 단순 구조로 재작성
- [ ] 기존 복잡한 메서드들 제거
- [ ] OptimizedS3Transfer 통합 유지

### Phase 3: Factory 개선
- [ ] 호환성 기반 모델 선택 로직 구현
- [ ] 하드코딩된 분기 제거

### Phase 4: 정리
- [ ] 레거시 메서드 완전 제거
- [ ] 테스트 코드 업데이트
- [ ] 문서화 완료

## 🚦 성공 기준
- [ ] 모든 기존 기능 동작 보장
- [ ] 코드 라인 수 30% 이상 감소
- [ ] 새로운 모델 타입 추가시 metadata만 수정
- [ ] Factory 알고리즘 매핑 자동화 완료
