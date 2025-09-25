# ComponentFactory Recipe/Config 통합 개선 계획

## 🎯 목표
ComponentFactory의 모든 create_* 메서드가 recipe와 config만을 인자로 받아서 모든 컴포넌트를 생성하도록 통합

## 📊 현재 문제점 분석

### 불일치하는 인자 패턴
- ✅ `create_scorer(recipe)` - 이미 recipe만 사용
- ✅ `create_evaluator(recipe, config)` - 이미 recipe/config만 사용
- ✅ `create_pretrainer(recipe)` - 이미 recipe만 사용
- ❌ `create_trainer(recipe, config, scorer)` - scorer 별도 의존성
- ❌ `create_optimizer(recipe, model_params)` - model_params 별도 필요
- ❌ `create_data_loader(source, config)` - source 별도 인자
- ❌ `create_tokenizer(config, recipe, tokenizer_type)` - tokenizer_type 별도

### Pipeline에서의 복잡한 호출 패턴
```python
# 현재: 복잡한 별도 인자들과 의존성 관리
scorer = ComponentFactory.create_scorer(recipe)
trainer = ComponentFactory.create_trainer(recipe, config, scorer)
train_source = recipe.data.train.sources[0]
train_loader = ComponentFactory.create_data_loader(train_source, config)
tokenizer = ComponentFactory.create_tokenizer(config, recipe, "hf")
optimizer = ComponentFactory.create_optimizer(recipe, base.parameters())
```

## 🚀 종합 해결 방안

### Phase 1: Recipe 스키마 확장

#### 1.1 Model 스키마에 tokenizer_type 필드 추가
```python
# src/settings/recipe_schema.py
class Model(BaseModel):
    """모델 설정: WMTP에서 사용하는 모든 모델 정의"""

    # ... 기존 필드들
    base_id: str = Field(..., description="기본 MTP 모델 식별자")
    rm_id: str | None = Field(default=None, description="보상 모델 식별자")
    ref_id: str = Field(..., description="참조 모델 식별자")

    # 🆕 새 필드: 토크나이저 타입
    tokenizer_type: Literal["hf", "raw"] = Field(
        default="hf",
        description="Tokenizer interface type: hf=HuggingFace compatible, raw=SentencePiece direct"
    )

    # ... 기존 필드들 계속
```

#### 1.2 기존 data.sources 필드 활용
Recipe에는 이미 `recipe.data.train.sources` 필드가 존재하므로 추가 수정 불필요:
```python
class DataConfig(BaseModel):
    sources: list[str] = Field(..., description="Data sources")  # 이미 존재
```

### Phase 2: ComponentFactory 메서드 리팩토링

#### 2.1 create_tokenizer 통합 (완전 달성)
```python
@classmethod
def create_tokenizer(cls, recipe: Recipe, config: Config):
    """토크나이저 생성 - recipe/config만 사용하는 통합 패턴"""

    # 1. tokenizer_type을 recipe에서 가져옴 (더 이상 별도 인자 불필요)
    tokenizer_type = recipe.model.tokenizer_type

    # 2. registry_key 결정
    if tokenizer_type in ["hf", "huggingface", "hf-sentencepiece"]:
        registry_key = "hf"
    elif tokenizer_type in ["raw", "sentencepiece", "default"]:
        registry_key = "default"
    else:
        raise ValueError(f"지원되지 않는 tokenizer_type: {tokenizer_type}")

    # 3. config 직접 사용
    tokenizer_config = config.model_dump()

    # 4. registry 생성 및 반환
    return tokenizer_registry.create(registry_key, tokenizer_config)
```

#### 2.2 create_data_loader 통합 (완전 달성)
```python
@classmethod
def create_data_loader(cls, recipe: Recipe, config: Config) -> Loader:
    """데이터 로더 생성 - recipe/config만 사용하는 통합 패턴"""

    # 1. source를 recipe에서 자동 추출 (더 이상 별도 인자 불필요)
    source = recipe.data.train.sources[0]  # 첫 번째 훈련 소스 사용

    # 2. 소스별 데이터셋 경로 결정 (기존 로직 유지)
    dataset_path = None
    if source == "mbpp":
        dataset_path = str(config.paths.datasets.mbpp)
    elif source in ["contest", "codecontests"]:
        dataset_path = str(config.paths.datasets.contest)
    else:
        dataset_path = source

    # 3. 통합 데이터 로더 설정 (기존 로직 유지)
    loader_config = {
        "storage": config.storage.model_dump(),
        "paths": config.paths.model_dump(),
        "split": "train",
        "dataset_type": source,
    }

    # 4. UnifiedDataLoader 생성
    return loader_registry.create("unified-data-loader", loader_config)
```

#### 2.3 create_trainer 의존성 자동 관리 (완전 달성)
```python
@classmethod
def create_trainer(cls, recipe: Recipe, config: Config) -> Trainer:
    """트레이너 생성 - recipe/config만 사용, scorer 의존성 자동 관리"""

    # 1. scorer를 내부에서 자동 생성 (더 이상 별도 인자 불필요)
    if recipe.train.algo == "mtp-baseline":
        scorer = None  # Baseline: 균등 가중치
    else:
        scorer = cls.create_scorer(recipe)  # 자동으로 적합한 scorer 생성

    # 2. trainer 설정 구성 (기존 로직 유지)
    trainer_config = {
        "n_heads": recipe.model.mtp.n_heads,
        "horizon": recipe.model.mtp.horizon,
        "loss_config": {
            "weight_norm": recipe.loss.weight_norm,
            "lambda": recipe.loss.lambda_weight,
            "temperature": recipe.loss.temperature,
            "epsilon": recipe.loss.epsilon,
            "max_weight": recipe.loss.max_weight,
        },
        "full_finetune": recipe.train.full_finetune,
        "lora_config": recipe.train.lora.model_dump() if recipe.train.lora.enabled else None,
        "mixed_precision": config.devices.mixed_precision,
        "fsdp_config": config.devices.fsdp.model_dump() if config.devices.fsdp.enabled else None,
        "scorer": scorer,  # 자동 생성된 scorer 포함
    }

    # 3. registry 생성 및 반환
    return trainer_registry.create(recipe.train.algo, trainer_config)
```

#### 2.4 create_optimizer 부분 개선 (기술적 제약)
```python
@classmethod
def create_optimizer(cls, recipe: Recipe, model_params) -> Optimizer:
    """최적화기 생성 - model_params는 기술적 제약으로 유지"""

    # model_params는 실제 모델 인스턴스의 .parameters() 필요
    # recipe/config만으로는 해결 불가능한 기술적 제약

    # 기존 로직 유지하되 config 정보는 recipe에서 추출 가능
    optimizer_config = {
        "params": model_params,  # 여전히 별도 인자 필요
        "lr": recipe.optim.lr,
        "weight_decay": recipe.optim.weight_decay,
        "betas": recipe.optim.betas,
        "grad_clip": recipe.optim.grad_clip,
        "scheduler": recipe.optim.scheduler,
        "warmup_ratio": recipe.optim.warmup_ratio,
    }

    return optimizer_registry.create(recipe.optim.optimizer, optimizer_config)
```

### Phase 3: Pipeline 대폭 단순화

#### 3.1 training_pipeline.py 개선
```python
# Before: 복잡한 별도 인자들과 의존성 관리
scorer = ComponentFactory.create_scorer(recipe)
trainer = ComponentFactory.create_trainer(recipe, config, scorer)
train_source = recipe.data.train.sources[0]
train_loader = ComponentFactory.create_data_loader(train_source, config)
tokenizer_component = ComponentFactory.create_tokenizer(config, recipe, "hf")
optimizer = ComponentFactory.create_optimizer(recipe, base.parameters())

# After: recipe/config 중심의 깔끔한 호출
trainer = ComponentFactory.create_trainer(recipe, config)  # scorer 자동 생성
train_loader = ComponentFactory.create_data_loader(recipe, config)  # source 자동 추출
tokenizer_component = ComponentFactory.create_tokenizer(recipe, config)  # type은 recipe에서
optimizer = ComponentFactory.create_optimizer(recipe, base.parameters())  # 유일한 예외
```

#### 3.2 evaluation_pipeline.py 개선
```python
# Before
for source in sources:
    data_loader = ComponentFactory.create_data_loader(source, self.config)

# After
for source in sources:
    # recipe.data를 동적으로 수정하여 현재 source 반영
    current_recipe = recipe.model_copy(deep=True)
    current_recipe.data.train.sources = [source]
    data_loader = ComponentFactory.create_data_loader(current_recipe, self.config)
```

### Phase 4: YAML 설정 파일 업데이트

#### 4.1 기존 recipe YAML 파일들에 tokenizer_type 필드 추가
```yaml
# configs/recipe.mtp_baseline.yaml
model:
  base_id: "facebook/multi-token-prediction"
  ref_id: "codellama/CodeLlama-7b-Python-hf"
  tokenizer_type: "hf"  # 🆕 새 필드 추가
  tokenizer_pad_side: "right"
  mtp:
    n_heads: 4
    horizon: 4

# configs/recipe.critic.yaml
model:
  base_id: "facebook/multi-token-prediction"
  rm_id: "models/Llama_3_8B_RM"
  ref_id: "codellama/CodeLlama-7b-Python-hf"
  tokenizer_type: "hf"  # 🆕 새 필드 추가
  tokenizer_pad_side: "right"

# configs/recipe.rho1.yaml
model:
  base_id: "facebook/multi-token-prediction"
  ref_id: "codellama/CodeLlama-7b-Python-hf"
  tokenizer_type: "hf"  # 🆕 새 필드 추가
  tokenizer_pad_side: "right"
```

### Phase 5: 통합 테스트 및 검증

#### 5.1 기본 import 테스트
```python
from src.factory.component_factory import ComponentFactory
from src.settings.loader import load_config, load_recipe

config = load_config("configs/config.yaml")
recipe = load_recipe("configs/recipe.mtp_baseline.yaml")
```

#### 5.2 새 인터페이스 동작 테스트
```python
# 새로운 통합 인터페이스 테스트
trainer = ComponentFactory.create_trainer(recipe, config)
data_loader = ComponentFactory.create_data_loader(recipe, config)
tokenizer = ComponentFactory.create_tokenizer(recipe, config)
```

#### 5.3 기존 pipeline과의 호환성 검증
- training_pipeline.py 동작 확인
- evaluation_pipeline.py 동작 확인
- 동일한 결과 생성되는지 검증

## 📈 예상 효과

### 🎉 개선 지표
- **인터페이스 일관성**: 75% 메서드가 recipe/config만 사용 (3/4 메서드 완전 통합)
- **코드 단순화**: Pipeline 호출 코드 50% 이상 단순화
- **의존성 관리**: ComponentFactory 내부에서 자동 처리
- **설정 중심**: 선언적 YAML 설정 패턴 완성

### 🔧 기술적 개선
1. **일관성**: 모든 메서드가 유사한 인터페이스 패턴 사용
2. **캡슐화**: 컴포넌트 간 의존성을 Factory 내부에서 관리
3. **선언적**: YAML 설정만으로 모든 컴포넌트 구성 가능
4. **유지보수성**: Pipeline 코드가 설정 중심으로 단순화

### ⚠️ 제약사항
- `create_optimizer`의 model_params는 실제 모델 인스턴스 필요로 기술적 제약 유지
- 하지만 나머지 75% 메서드는 완전히 recipe/config로 통합 가능

## 🚀 구현 순서

1. **Phase 1**: Recipe Model 스키마에 tokenizer_type 필드 추가
2. **Phase 2.1**: create_tokenizer 리팩토링 (recipe/config만 사용)
3. **Phase 2.2**: create_data_loader 리팩토링 (recipe/config만 사용)
4. **Phase 2.3**: create_trainer 리팩토링 (scorer 자동 생성)
5. **Phase 3**: Pipeline 호출 방식 단순화
6. **Phase 4**: YAML 설정 파일들에 tokenizer_type 추가
7. **Phase 5**: 통합 테스트 및 검증

## 💡 결론

이 계획을 통해 ComponentFactory가 recipe/config 중심의 일관된 인터페이스를 가지게 되며, Pipeline 코드의 복잡성이 대폭 감소합니다. 사용자의 요구사항인 "recipe와 config만 인자로 받아서 모든것을 생성"하는 목표를 75% 달성할 수 있는 현실적이고 구체적인 방안입니다.

특히 model_params의 기술적 제약을 인정하면서도, 나머지 대부분의 메서드를 통합하여 전체적인 일관성과 사용성을 크게 향상시킬 수 있습니다.