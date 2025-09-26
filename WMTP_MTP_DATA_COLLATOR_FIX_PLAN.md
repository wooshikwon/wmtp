# WMTP MTP Data Collator 통합 수정 계획서
## Factory 패턴 기반 완전 아키텍처 통합

---

## 📋 **현재 구조 심층 분석**

### **🏗️ ComponentFactory 아키텍처 현황**

**Factory 메서드 패턴**:
```python
# 현재 ComponentFactory 메서드들
ComponentFactory.create_trainer(recipe, config)      # ✅ 존재
ComponentFactory.create_model_loader(config, recipe, type)  # ✅ 존재
ComponentFactory.create_data_loader(recipe, config)  # ✅ 존재
ComponentFactory.create_tokenizer(recipe, config)    # ✅ 존재
ComponentFactory.create_optimizer(recipe, config)    # ✅ 존재
ComponentFactory.create_pretrainer(recipe)           # ✅ 존재

# 누락된 메서드
ComponentFactory.create_collator(???, ???)           # ❌ 부재
```

**Registry 패턴 현황**:
```python
# 기존 Registry들
loader_registry = _CompatibilityAdapter("loader")     # ✅
trainer_registry = _CompatibilityAdapter("trainer")   # ✅
tokenizer_registry = _CompatibilityAdapter("tokenizer") # ✅
optimizer_registry = _CompatibilityAdapter("optimizer") # ✅
evaluator_registry = _CompatibilityAdapter("evaluator") # ✅
pretrainer_registry = _CompatibilityAdapter("pretrainer") # ✅

# 필요한 추가 Registry
collator_registry = _CompatibilityAdapter("collator") # ❌ 부재
```

### **🔄 Training Pipeline 데이터 플로우**

**현재 플로우**:
```
Step 4: ComponentFactory.create_model_loader() → base 모델
Step 5: ComponentFactory.create_tokenizer() → 토크나이저
Step 6: ComponentFactory.create_data_loader() → 데이터셋
Step 7: tokenizer.tokenize_dataset() → 토큰화
Step 8: 분산 샘플러 설정
Step 9: DataLoader(collate_fn=default_data_collator) ← 🚨 문제 지점
```

**Factory 패턴 위반**:
- 모든 컴포넌트는 Factory에서 생성되는데 **collator만 하드코딩**
- `pack_sequences` 설정이 Recipe에 있으나 **완전히 무시됨**

---

## 🚨 **확인된 문제 종합**

### **Critical Issues**

1. **KeyError: 'hidden_states'** (critic_head_pretrainer.py:221)
   - 원인: unsafe한 딕셔너리 접근
   - 영향: Critic-WMTP Stage 1 완전 블록

2. **배치 차원 불일치** (training_pipeline.py:224)
   - 원인: `default_data_collator` 패딩 부재
   - 영향: 모든 알고리즘 DataLoader 생성 실패

3. **Factory 패턴 불일치** (training_pipeline.py:224)
   - 원인: collator만 Factory 밖에서 하드코딩
   - 영향: 아키텍처 일관성 깨짐, pack_sequences 무시

### **Major Issues**

4. **MTP 라벨 생성 부재**
   - 문제: MTP는 `[B, S, H=4]` 라벨 필요하나 `[B, S]`만 생성
   - 영향: 모든 WMTP 알고리즘의 부정확한 손실 계산

---

## 🎯 **설계 결정사항**

### **Collator 분류 기준: 모델 타입**
- **MTPDataCollator**: MTP 모델용 (metadata: training_algorithm == "mtp")
- **LMDataCollator**: 일반 언어모델용 (metadata: training_algorithm != "mtp")

### **Factory 통합 필수성**
- 현재 모든 컴포넌트가 Factory 패턴 사용
- collator만 예외적으로 하드코딩되어 아키텍처 일관성 위반
- **Registry 패턴으로 완전 통합 필요**

---

## 🛠️ **완전 통합 구현 계획**

### **Phase 1: Collator 클래스 구현 (1시간)**

#### **1.1 Collator 기본 클래스 생성**
**파일**: `src/components/data/collators.py`

```python
"""WMTP Data Collator 구현체들"""

from transformers import DataCollatorForLanguageModeling
import torch
from typing import Dict, List, Any, Optional

from src.components.base import Component


class LMDataCollator(Component, DataCollatorForLanguageModeling):
    """일반 언어모델용 Data Collator

    표준 언어모델링을 위한 기본 패딩 및 라벨 생성
    """

    def __init__(self, config: Dict[str, Any]):
        Component.__init__(self, config)

        # tokenizer는 config에서 전달받음
        tokenizer = config.get("tokenizer")
        if not tokenizer:
            raise ValueError("tokenizer is required in config")

        DataCollatorForLanguageModeling.__init__(
            self,
            tokenizer=tokenizer,
            mlm=False,  # 인과적 언어 모델링
            pad_to_multiple_of=config.get("pad_to_multiple_of", 8)
        )

    def setup(self, inputs: Dict[str, Any]) -> None:
        """Component 인터페이스 구현"""
        pass

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Component 인터페이스 구현 - 실제로는 __call__ 사용"""
        return {"collator": self}


class MTPDataCollator(LMDataCollator):
    """MTP(Multi-Token Prediction)용 Data Collator

    LMDataCollator를 상속하여:
    1. 기본 패딩/마스킹 기능 유지
    2. MTP용 multi-horizon 라벨 생성 추가
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.horizon = config.get("horizon", 4)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 1. 기본 패딩 처리 (상속된 DataCollatorForLanguageModeling)
        batch = super().__call__(features)

        # 2. MTP 라벨 생성: [B, S] → [B, S, H]
        if "labels" in batch:
            batch["labels"] = self._create_mtp_labels(batch["labels"])

        return batch

    def _create_mtp_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """[B, S] 라벨을 [B, S, H] MTP 라벨로 변환

        각 위치 t에서 t+1, t+2, t+3, t+4 토큰을 라벨로 생성
        """
        B, S = labels.shape
        device = labels.device
        dtype = labels.dtype

        # MTP 라벨 텐서 초기화 (-100은 무시할 라벨)
        mtp_labels = torch.full((B, S, self.horizon), -100, dtype=dtype, device=device)

        # 각 horizon에 대해 라벨 생성
        for h in range(self.horizon):
            shift = h + 1  # t+1, t+2, t+3, t+4
            if shift < S:
                # 각 위치에서 shift만큼 앞의 토큰을 라벨로 사용
                mtp_labels[:, :S-shift, h] = labels[:, shift:]

        return mtp_labels
```

#### **1.2 Registry 등록**
**파일**: `src/components/data/__init__.py`

```python
"""Data Collator 컴포넌트들"""

from src.components.registry import registry
from .collators import LMDataCollator, MTPDataCollator

# Registry 등록
registry.register("lm-data-collator", category="collator")(LMDataCollator)
registry.register("mtp-data-collator", category="collator")(MTPDataCollator)

__all__ = ["LMDataCollator", "MTPDataCollator"]
```

#### **1.3 Registry Adapter 추가**
**파일**: `src/components/registry.py` (수정)

```python
# 기존 Registry들에 추가
collator_registry = _CompatibilityAdapter("collator")
```

### **Phase 2: ComponentFactory 통합 (30분)**

#### **2.1 Factory 메서드 추가**
**파일**: `src/factory/component_factory.py` (수정)

```python
from src.components.registry import collator_registry

class ComponentFactory:
    # ... 기존 메서드들 ...

    @staticmethod
    def create_collator(
        recipe: Recipe,
        config: Config,
        tokenizer: Any,
        model_metadata: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Data Collator 생성 - 모델 타입 기반 자동 선택

        모델 메타데이터의 training_algorithm에 따라 적절한 collator 선택:
        - training_algorithm == "mtp": MTPDataCollator
        - 기타: LMDataCollator

        Args:
            recipe: 훈련 레시피 (pack_sequences 등 설정)
            config: 환경 설정
            tokenizer: 토크나이저 인스턴스
            model_metadata: 모델 메타데이터 (알고리즘 판단용)

        Returns:
            Collator 인스턴스
        """
        # 1. 모델 타입 결정
        training_algorithm = "base"  # 기본값
        if model_metadata:
            training_algorithm = model_metadata.get("training_algorithm", "base")

        # 2. 알고리즘별 collator 선택
        if training_algorithm == "mtp":
            collator_key = "mtp-data-collator"
            horizon = model_metadata.get("horizon", 4)
        else:
            collator_key = "lm-data-collator"
            horizon = 1

        # 3. Collator 설정 구성
        collator_config = {
            "tokenizer": tokenizer,
            "horizon": horizon,
            "pad_to_multiple_of": 8,  # GPU 효율성
        }

        # 4. pack_sequences 설정 반영
        if hasattr(recipe.data.train, 'pack_sequences') and recipe.data.train.pack_sequences:
            # pack_sequences가 True일 때만 고급 패딩 적용
            collator_config["pad_to_multiple_of"] = 8
        else:
            collator_config["pad_to_multiple_of"] = None

        # 5. Registry에서 생성
        return collator_registry.create(collator_key, collator_config)
```

### **Phase 3: Training Pipeline 통합 (30분)**

#### **3.1 Pipeline 수정**
**파일**: `src/pipelines/training_pipeline.py` (수정)

**현재 (Step 9)**:
```python
# Step 9: PyTorch DataLoader 생성
train_dl = DataLoader(
    tokenized,
    batch_size=recipe.data.train.batch_size or 1,
    shuffle=(sampler is None),
    sampler=sampler,
    collate_fn=default_data_collator,  # ← 하드코딩
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)
```

**수정 후**:
```python
# Step 9-1: Data Collator 생성 (Factory 패턴)
# 모델 메타데이터 기반으로 적절한 collator 자동 선택
model_metadata = getattr(base, '_model_metadata', {})
collator = ComponentFactory.create_collator(
    recipe=recipe,
    config=config,
    tokenizer=tokenizer.tokenizer,  # 실제 HF tokenizer 객체
    model_metadata=model_metadata
)

console.print(f"[dim]🔍 Data Collator 생성 완료: {type(collator).__name__}[/dim]")

# Step 9-2: PyTorch DataLoader 생성 (Factory로 생성된 collator 사용)
train_dl = DataLoader(
    tokenized,
    batch_size=recipe.data.train.batch_size or 1,
    shuffle=(sampler is None),
    sampler=sampler,
    collate_fn=collator,  # ← Factory에서 생성된 collator
    num_workers=recipe.data.train.num_workers or 2,
    pin_memory=torch.cuda.is_available(),
)
```

#### **3.2 Import 추가**
**파일**: `src/pipelines/training_pipeline.py` (수정)

```python
# 기존 import에 추가
from src.components.data import LMDataCollator, MTPDataCollator  # 타입 힌트용

# default_data_collator import 제거 (더 이상 사용하지 않음)
# from transformers import default_data_collator  # ← 제거
```

### **Phase 4: Hidden States 안전 추출 (30분)**

#### **4.1 공통 유틸리티 생성**
**파일**: `src/utils/model_utils.py` (새 파일)

```python
"""모델 관련 공통 유틸리티"""

import torch
from typing import Any, Union


def extract_hidden_states(outputs: Any) -> torch.Tensor:
    """모델 출력에서 hidden_states를 안전하게 추출

    다양한 모델 출력 형태를 지원:
    - dict: outputs["hidden_states"]
    - object: outputs.hidden_states
    - object: outputs.last_hidden_state

    Args:
        outputs: 모델 출력 (dict, BaseModelOutput, 등)

    Returns:
        torch.Tensor: [B, S, D] 형태의 hidden states

    Raises:
        ValueError: hidden_states 추출 실패 시
    """
    hidden_states = None

    try:
        # Case 1: dict 형태에서 hidden_states 키 접근
        if isinstance(outputs, dict) and "hidden_states" in outputs:
            hs = outputs["hidden_states"]
            # list/tuple인 경우 마지막 레이어 선택
            hidden_states = hs[-1] if isinstance(hs, (list, tuple)) else hs

        # Case 2: object 형태에서 hidden_states 속성 접근
        elif hasattr(outputs, "hidden_states"):
            hs = outputs.hidden_states
            hidden_states = hs[-1] if isinstance(hs, (list, tuple)) else hs

        # Case 3: object 형태에서 last_hidden_state 속성 접근
        elif hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state

    except Exception as e:
        # 예상치 못한 오류는 로그만 남기고 계속 진행
        pass

    # 검증: hidden_states가 올바른 형태인지 확인
    if hidden_states is None:
        raise ValueError(
            f"Failed to extract hidden_states from model outputs. "
            f"Output type: {type(outputs)}, "
            f"Available keys/attributes: {_get_available_keys(outputs)}"
        )

    if not isinstance(hidden_states, torch.Tensor):
        raise ValueError(
            f"hidden_states must be torch.Tensor, got {type(hidden_states)}"
        )

    if hidden_states.ndim != 3:
        raise ValueError(
            f"Expected hidden_states shape [B, S, D], got {hidden_states.shape}"
        )

    return hidden_states


def _get_available_keys(outputs: Any) -> str:
    """디버깅을 위한 사용 가능한 키/속성 목록 반환"""
    if isinstance(outputs, dict):
        return f"dict keys: {list(outputs.keys())}"
    else:
        attrs = [attr for attr in dir(outputs) if not attr.startswith('_')]
        return f"object attrs: {attrs[:10]}..."  # 처음 10개만
```

#### **4.2 Critic Head Pretrainer 수정**
**파일**: `src/components/trainer/critic_head_pretrainer.py` (수정)

**현재 (218-222라인)**:
```python
hidden_states = (
    outputs.hidden_states[-1]
    if hasattr(outputs, "hidden_states")
    else outputs["hidden_states"][-1]  # ← KeyError 발생
)
```

**수정 후**:
```python
from src.utils.model_utils import extract_hidden_states

hidden_states = extract_hidden_states(outputs)
```

#### **4.3 기존 코드 정리**
**파일**: `src/components/trainer/critic_wmtp_trainer.py` (수정)

**현재 (447-452라인)**의 중복 로직을 공통 유틸리티로 교체:
```python
from src.utils.model_utils import extract_hidden_states

# 기존 447-452 라인 대체
try:
    hidden_states = extract_hidden_states(outputs)
except ValueError as e:
    raise ValueError(
        f"CriticWmtpTrainer requires valid hidden_states [B,S,D] from model outputs. "
        f"Error: {e}"
    )
```

---

## 🧪 **검증 계획**

### **단계별 테스트**

#### **Phase 1 테스트: Collator 단독**
```python
# Unit Test 예시
from src.components.data import MTPDataCollator, LMDataCollator

# MTP Collator 테스트
config = {"tokenizer": tokenizer, "horizon": 4}
mtp_collator = MTPDataCollator(config)

# 테스트 데이터
features = [
    {"input_ids": [1, 2, 3, 4, 5]},
    {"input_ids": [6, 7, 8]}
]

batch = mtp_collator(features)
assert batch["labels"].shape == (2, 5, 4)  # [B, S, H]
```

#### **Phase 2 테스트: Factory 통합**
```python
# Factory 테스트
collator = ComponentFactory.create_collator(
    recipe=recipe,
    config=config,
    tokenizer=tokenizer,
    model_metadata={"training_algorithm": "mtp", "horizon": 4}
)
assert isinstance(collator, MTPDataCollator)
```

#### **Phase 3 테스트: Pipeline End-to-End**
```bash
# 1. Dry-run 테스트
python -m src.cli.train --config tests/configs/config.local_test.yaml \
                        --recipe tests/configs/recipe.critic_wmtp.yaml \
                        --dry-run

# 2. 실제 훈련 테스트 (각 알고리즘별)
# Baseline (가장 단순)
python -m src.cli.train --config tests/configs/config.local_test.yaml \
                        --recipe tests/configs/recipe.mtp_baseline.yaml

# Critic (수정 대상)
python -m src.cli.train --config tests/configs/config.local_test.yaml \
                        --recipe tests/configs/recipe.critic_wmtp.yaml

# Rho1 (참조 모델 포함)
python -m src.cli.train --config tests/configs/config.local_test.yaml \
                        --recipe tests/configs/recipe.rho1_wmtp_weighted.yaml
```

---

## 📊 **예상 효과**

### **즉시 해결되는 문제**
- ✅ **KeyError: 'hidden_states'** → 안전한 추출 유틸리티로 해결
- ✅ **배치 차원 불일치** → DataCollatorForLanguageModeling 상속으로 해결
- ✅ **Factory 패턴 위반** → ComponentFactory.create_collator() 추가로 해결
- ✅ **pack_sequences 무시** → Factory에서 설정 반영으로 해결
- ✅ **MTP 라벨 부재** → MTPDataCollator._create_mtp_labels()로 해결

### **아키텍처 개선**
- **일관성**: 모든 컴포넌트가 Factory 패턴 사용
- **확장성**: 새로운 collator 타입을 Registry에 쉽게 추가
- **재사용성**: 공통 유틸리티로 hidden_states 추출 로직 통합
- **설정 주도**: Recipe의 pack_sequences 설정 정상 동작

### **성능 최적화**
- **메모리 효율성**: pad_to_multiple_of=8로 GPU 최적화
- **배치 안정성**: 모든 시퀀스가 동일 길이로 패딩
- **처리 속도**: DataCollatorForLanguageModeling의 최적화된 구현 활용

---

## ⚠️ **리스크 분석 및 완화**

### **예상 리스크**

1. **메모리 사용량 증가** (MTP 라벨 4배 증가)
   - **완화**: 테스트 환경에서 batch_size=1로 시작, 점진적 확대
   - **모니터링**: GPU 메모리 사용량 추적

2. **기존 알고리즘 호환성**
   - **완화**: LMDataCollator로 기존 동작 완전 보존
   - **검증**: baseline/rho1 알고리즘 먼저 테스트

3. **Factory 메서드 시그니처 변경**
   - **완화**: 기존 Factory 메서드는 변경하지 않고 추가만
   - **하위호환**: 기존 코드 영향 없음

### **롤백 계획**
```python
# 긴급 롤백: training_pipeline.py 한 줄만 수정
# 수정된 코드
collate_fn=collator,

# 롤백 코드
collate_fn=default_data_collator,
```

---

## ⏱️ **구현 타임라인**

| Phase | 작업 | 예상 시간 | 주요 산출물 |
|-------|------|----------|------------|
| 1 | Collator 클래스 + Registry | 1시간 | `collators.py`, `__init__.py` |
| 2 | ComponentFactory 통합 | 30분 | `component_factory.py` 수정 |
| 3 | Training Pipeline 수정 | 30분 | `training_pipeline.py` 수정 |
| 4 | Hidden States 유틸리티 | 30분 | `model_utils.py`, pretrainer 수정 |
| 5 | 통합 테스트 | 1.5시간 | 전체 파이프라인 검증 |

**총 예상 시간**: **3.5시간**

---

## 🎯 **성공 기준**

### **기능적 성공**
1. **모든 에러 제거**: KeyError, 배치 차원 불일치 완전 해결
2. **Factory 패턴 완성**: 모든 컴포넌트가 ComponentFactory 통과
3. **MTP 라벨 정상 생성**: `[B, S, H=4]` 형태 검증
4. **설정 반영**: pack_sequences 설정 정상 동작

### **성능 기준**
1. **메모리 효율성**: 기존 대비 적정 수준 (4배 이내) 유지
2. **처리 속도**: 기존 대비 90% 이상 성능 유지
3. **안정성**: 10회 연속 실행 모두 성공

### **아키텍처 품질**
1. **일관성**: 모든 컴포넌트가 동일한 Factory + Registry 패턴
2. **확장성**: 새로운 collator 타입 쉽게 추가 가능
3. **재사용성**: 공통 로직의 유틸리티 함수화

---

## 📋 **최종 파일 변경 요약**

### **새로 생성할 파일**
```
src/components/data/
├── __init__.py              # Registry 등록
└── collators.py             # LMDataCollator, MTPDataCollator

src/utils/
└── model_utils.py           # extract_hidden_states 유틸리티
```

### **수정할 파일**
```
src/components/registry.py                    # collator_registry 추가
src/factory/component_factory.py              # create_collator 메서드 추가
src/pipelines/training_pipeline.py            # Factory 기반 collator 사용
src/components/trainer/critic_head_pretrainer.py  # 안전한 hidden_states 추출
src/components/trainer/critic_wmtp_trainer.py     # 중복 로직 제거
```

---

## 🚀 **결론**

이 계획은 **완전한 Factory 패턴 통합**을 통해 WMTP 시스템의 아키텍처 일관성을 확립하면서 모든 Critical/Major 이슈를 동시 해결합니다:

### **핵심 혁신점**
1. **ComponentFactory.create_collator()** - 모델 메타데이터 기반 자동 선택
2. **MTPDataCollator/LMDataCollator** - 알고리즘별 최적화된 라벨 생성
3. **Registry 패턴 완성** - 모든 컴포넌트의 통일된 관리
4. **안전한 hidden_states 추출** - 공통 유틸리티로 재사용성 확보

### **아키텍처 완성도**
- ✅ **Factory 패턴**: 모든 컴포넌트가 ComponentFactory 경유
- ✅ **Registry 패턴**: collator_registry 추가로 통합 관리
- ✅ **설정 주도**: Recipe의 pack_sequences 등 설정 완전 반영
- ✅ **컴포넌트 독립성**: 각 collator가 독립적으로 테스트/확장 가능

이제 WMTP 시스템이 진정한 "Research-Grade Production System"으로 완성됩니다.