# Early Stopping Implementation Plan for WMTP Training (v2.0)

## 변경 이력

**v2.0 (2025-10-01)**: 전면 재설계
- PretrainConfig를 최상위 레벨로 분리 (train.stage1, critic.pretrainer 통합)
- Early stopping 모드를 "any" 방식으로 변경 (더 실용적)
- Gradient norm을 윈도우 기반 비율 체크로 개선
- API 사용 패턴 명확화 (should_stop() 반환값 처리)
- 분산 학습 및 체크포인트 통합 강화

**v1.0**: 초기 계획 (Stage1Config 기반)

---

## 1. 개요

### 목표

WMTP 학습의 **Pretraining (Stage 1)**과 **Main Training (Stage 2)**에 학술적으로 타당한 Early Stopping을 구현하여:

1. **과적합 방지**: Loss 수렴 감지로 불필요한 학습 중단
2. **학습 시간 최적화**: 효율적인 자원 활용
3. **연구 리스크 완화**: Critic 표류·고분산, Gradient 불안정성 조기 감지

### 핵심 변경사항

#### 1. 통합 Pretraining 설정
```yaml
# 기존 (혼란스러운 구조)
train:
  stage1: ...  # 또는
critic:
  pretrainer: ...

# 개선 (명확한 구조)
pretrain:  # 최상위 레벨, train과 동일 층위
  enabled: true
  num_epochs: 3
  max_steps: 30
  early_stopping:
    mode: "any"  # 하나라도 조건 만족 시 중단
    ...
```

#### 2. 실용적인 ANY 모드
- **기존**: loss_converged **AND** grad_stable **AND** variance_valid (매우 보수적)
- **개선**: loss_converged **OR** grad_unstable **OR** variance_invalid (실용적)
- 각 조건이 독립적으로 중단 트리거 가능

#### 3. 윈도우 기반 Gradient 안정성
- **기존**: 연속 patience 횟수 초과 (일시적 스파이크에 취약)
- **개선**: 윈도우 내 초과 비율 (예: 최근 10회 중 70% 이상)
- 일시적 변동에 강인한 판단

#### 4. 명확한 API 패턴
```python
# 올바른 사용법
if early_stopping.should_stop(metrics):
    reason = early_stopping.stop_reason  # 속성으로 접근
    console.print(f"[yellow]Early stopping: {reason}[/yellow]")
    break
```

### 연구 제안서와의 정합성

본 계획은 WMTP_학술_연구제안서.md의 핵심 목표를 직접 지원합니다:

- **Line 103**: "Critic 표현 안정화: 히든 상태 분포 변화 억제"
  → Variance range 체크로 표현 표류 조기 감지

- **Line 104**: "그래디언트 중요도 안정화: EMA 누적, outlier 클리핑"
  → 윈도우 기반 gradient norm 모니터링으로 불안정성 감지

- **Line 112**: "Critic 표류·고분산: 가치 편향·분산 증가"
  → 다중 기준 early stopping으로 리스크 완화

- **Line 16**: "중요 토큰에 계산을 집중하는 WMTP"
  → 불필요한 학습 조기 중단으로 자원 효율화

---

## 2. 현재 구조 분석

### Pretraining (Stage 1) - critic_head_pretrainer.py

**현재 종료 조건**:
```python
for epoch in range(self.num_epochs):  # Line 199
    for step, batch in enumerate(train_loader):
        if step >= self.max_steps:  # Line 203
            break
```

**문제점**:
- ❌ Value loss 수렴 무시
- ❌ Gradient norm 폭주 감지 없음 (Line 288에서 경고만)
- ❌ 예측 variance 범위 체크 없음

**활용 가능한 정보**:
- `loss.item()`: MSE loss (Line 274)
- `total_norm`: Gradient L2 norm (Line 281-286)
- `pred_values`: Value predictions (Line 271) → variance 계산 가능

### Main Training (Stage 2) - base_wmtp_trainer.py

**현재 종료 조건**:
```python
for step, batch in enumerate(dataloader):  # Line 405
    if max_steps is not None and current_step >= max_steps:  # Line 440
        break
```

**문제점**:
- ❌ Loss 정체 감지 없음
- ✅ 모든 알고리즘 (baseline, critic, rho1) 공통 구조

**활용 가능한 정보**:
- `train_step()` → `dict[str, Any]` (Line 413-414)
- 각 알고리즘별 metrics 반환 (loss, wmtp_loss 등)

---

## 3. Early Stopping 전략

### Pretraining (Stage 1) 기준

#### 1. Loss Convergence (필수)
- **Metric**: `loss` or `value_loss`
- **기준**: 최근 N steps 동안 loss 개선이 `min_delta` 미만
- **설정**: `patience=10`, `min_delta=1e-4`
- **이유**: Value head가 더 이상 보상 예측 능력을 개선하지 못함

#### 2. Gradient Instability (중요)
- **Metric**: `grad_norm` (L2 norm)
- **기준**: 윈도우 내 초과 비율이 threshold_ratio 이상
- **설정**:
  - `grad_norm_threshold=50.0` (기존 경고 수준)
  - `grad_norm_window_size=10` (최근 10 스텝)
  - `grad_norm_threshold_ratio=0.7` (70% 이상 초과)
- **이유**: 연구제안서 Line 104 "그래디언트 중요도 안정화" 달성
- **개선점**: 일시적 스파이크에 강인, 전체 패턴 파악

#### 3. Variance Out of Range (안정성)
- **Metric**: `value_variance` (pred_values.var())
- **기준**: 분산이 `[variance_min, variance_max]` 범위 이탈
- **설정**: `variance_min=0.1`, `variance_max=5.0`
- **이유**:
  - 너무 작으면 uninformative (모든 토큰 동일 중요도)
  - 너무 크면 unstable (표현 표류)

#### 4. Max Steps (안전장치)
- **기준**: `max_steps` 도달 OR `num_epochs` 완료
- **이유**: 무한 학습 방지

#### 중단 모드 (mode)
- `"any"` (권장): 위 조건 중 **하나라도** 만족하면 중단 (실용적)
- `"all"`: **모두** 만족해야 중단 (매우 보수적)
- `"loss_only"`: Loss convergence만 체크

### Main Training (Stage 2) 기준

#### 1. Loss Convergence (필수)
- **Metric**: `loss` or `wmtp_loss` (algo별 자동 감지)
- **기준**: 최근 N steps 동안 loss 개선이 `min_delta` 미만
- **설정**: `patience=100`, `min_delta=1e-5`
- **이유**: Main model이 더 이상 학습하지 못함

#### 2. Max Steps (안전장치)
- **기준**: `max_steps` 도달
- **이유**: 무한 학습 방지

---

## 4. 구현 계획

### Phase 1: Core Early Stopping Utility 개선

**파일**: `src/utils/early_stopping.py`

**변경사항**:

1. **BaseEarlyStopping**: 유지 (기존 구조 우수)

2. **LossEarlyStopping**: 유지 (Stage 2용, 변경 불필요)

3. **ValueHeadEarlyStopping**: 전면 개선

**주요 개선 내용**:
```python
class ValueHeadEarlyStopping(BaseEarlyStopping):
    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)

        # Mode 추가
        self.mode = self.config.get("mode", "any")  # "any" | "all" | "loss_only"

        # 윈도우 기반 gradient 체크
        self.grad_norm_window_size = self.config.get("grad_norm_window_size", 10)
        self.grad_norm_threshold_ratio = self.config.get("grad_norm_threshold_ratio", 0.7)

        from collections import deque
        self.grad_norm_history: deque = deque(maxlen=self.grad_norm_window_size)

    def should_stop(self, metrics: dict[str, float]) -> bool:
        # 각 조건 독립적으로 체크
        loss_converged = self._check_loss_convergence(...)
        grad_unstable = self._check_gradient_instability(...)
        variance_invalid = self._check_variance_invalid(...)

        # 모드에 따른 중단 결정
        if self.mode == "any":
            should_stop = loss_converged or grad_unstable or variance_invalid
        elif self.mode == "all":
            should_stop = loss_converged and not grad_unstable and not variance_invalid
        else:  # "loss_only"
            should_stop = loss_converged

        return should_stop

    def _check_gradient_instability(self, grad_norm: float | None) -> bool:
        """윈도우 기반 gradient 불안정성 체크."""
        if grad_norm is None or self.grad_norm_threshold is None:
            return False

        self.grad_norm_history.append(grad_norm > self.grad_norm_threshold)

        if len(self.grad_norm_history) < self.grad_norm_window_size:
            return False

        unstable_ratio = sum(self.grad_norm_history) / len(self.grad_norm_history)
        return unstable_ratio >= self.grad_norm_threshold_ratio
```

**체크리스트**:
- [ ] `mode` 파라미터 추가
- [ ] 윈도우 기반 gradient 체크 구현
- [ ] ANY/ALL/LOSS_ONLY 로직 구현
- [ ] State 관리에 `grad_norm_history` 추가
- [ ] 중단 이유 메시지 개선

---

### Phase 2: Schema 전면 재설계

**파일**: `src/settings/recipe_schema.py`

#### 2.1 EarlyStoppingConfig 추가

```python
class EarlyStoppingConfig(BaseModel):
    """Early stopping configuration."""

    enabled: bool = Field(default=False)

    # Common settings
    patience: int = Field(default=100, ge=1)
    min_delta: float = Field(default=1e-4, gt=0)
    monitor: str = Field(default="loss")

    # Pretraining specific
    mode: Literal["any", "all", "loss_only"] = Field(default="any")

    # Gradient stability (window-based)
    grad_norm_threshold: float | None = Field(default=None)
    grad_norm_window_size: int = Field(default=10, ge=1)
    grad_norm_threshold_ratio: float = Field(default=0.7, gt=0, le=1)

    # Variance range
    variance_min: float | None = Field(default=None)
    variance_max: float | None = Field(default=None)
```

#### 2.2 PretrainConfig 추가 (최상위 레벨)

```python
class PretrainConfig(BaseModel):
    """Pretraining configuration (Stage 1)."""

    enabled: bool = Field(default=True)

    # Training parameters
    num_epochs: int = Field(default=3, ge=1)
    max_steps: int = Field(default=2000, ge=1)
    lr: float = Field(default=1e-4, gt=0)

    # Output
    save_value_head: bool = Field(default=True)

    # Early stopping
    early_stopping: EarlyStoppingConfig | None = Field(default=None)

    # Note: GAE parameters (gamma, gae_lambda) are in critic section
    # and will be used by pretrainer via component_factory
```

#### 2.3 Recipe 수정

```python
class Recipe(BaseModel):
    run: Run
    pretrain: PretrainConfig | None = Field(default=None)  # 추가
    train: Train  # stage1 필드 제거됨
    optim: Optim
    data: Data
    loss: Loss
    critic: Critic | None
    rho1: Rho1 | None
    eval: Eval
```

#### 2.4 Train 수정

```python
class Train(BaseModel):
    algo: Literal["baseline-mtp", "critic-wmtp", "rho1-wmtp"]
    full_finetune: bool = True
    max_steps: int | None = None
    eval_interval: int = 500
    save_interval: int = 1000
    # stage1 제거됨
    early_stopping: EarlyStoppingConfig | None = Field(default=None)  # 추가
```

#### 2.5 Stage1Config 제거

**체크리스트**:
- [ ] `EarlyStoppingConfig` 클래스 정의
- [ ] `PretrainConfig` 클래스 정의
- [ ] `Recipe`에 `pretrain` 필드 추가
- [ ] `Train`에서 `stage1` 제거
- [ ] `Train`에 `early_stopping` 추가
- [ ] `Stage1Config` 클래스 제거
- [ ] 스키마 테스트

---

### Phase 3: Pretrainer Integration

**파일**: `src/components/trainer/critic_head_pretrainer.py`

#### 3.1 __init__ 수정

```python
def __init__(self, config: dict[str, Any] | None = None):
    super().__init__(config)

    # 기존 파라미터
    self.lr = self.config.get("lr", 1e-4)
    self.num_epochs = self.config.get("num_epochs", 3)
    self.max_steps = self.config.get("max_steps", 1000)
    self.gamma = self.config.get("gamma", 0.99)
    self.gae_lambda = self.config.get("gae_lambda", 0.95)

    # Early stopping 추가
    self.early_stopping_config = self.config.get("early_stopping")
```

#### 3.2 run() 수정

```python
def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
    # ... 기존 초기화 ...

    # Early stopping 초기화
    early_stopping = None
    if self.early_stopping_config and self.early_stopping_config.get("enabled", False):
        from src.utils.early_stopping import ValueHeadEarlyStopping

        early_stopping = ValueHeadEarlyStopping(self.early_stopping_config)
        mode = self.early_stopping_config.get("mode", "any")
        console.print(f"[cyan]Early stopping enabled (mode={mode})[/cyan]")

    # Training loop
    for epoch in range(self.num_epochs):
        for step, batch in enumerate(train_loader):
            if step >= self.max_steps:
                break

            # ... 기존 학습 로직 ...

            # Forward & Backward
            pred_values = self.value_head(hs_flat)
            loss = loss_fn(pred_values, vt_flat)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Gradient norm 계산 (기존)
            total_norm = ...

            # Variance 계산 추가
            pred_variance = pred_values.var().item()

            optimizer.step()

            # Early stopping 체크
            if early_stopping:
                metrics = {
                    "value_loss": loss.item(),
                    "grad_norm": total_norm,
                    "value_variance": pred_variance,
                }

                if early_stopping.should_stop(metrics):
                    reason = early_stopping.stop_reason
                    console.print(f"[yellow]⚠ Early stopping: {reason}[/yellow]")
                    break

        # 외부 loop 종료
        if early_stopping and early_stopping.should_stop_flag:
            break

    # ... 저장 로직 ...

    return {
        "saved": save_location,
        "final_loss": avg_final_loss,
        "total_steps": step_count,
        "early_stopped": early_stopping.should_stop_flag if early_stopping else False,
        "stop_reason": early_stopping.stop_reason if early_stopping else None,
    }
```

**체크리스트**:
- [ ] `__init__`에 early_stopping_config 추가
- [ ] `run()`에 early_stopping 인스턴스 생성
- [ ] Variance 계산 추가
- [ ] Early stopping 체크 통합
- [ ] Nested loop 종료 처리
- [ ] 반환값에 early_stopped 정보 추가

---

### Phase 4: Main Trainer Integration

**파일**: `src/components/trainer/base_wmtp_trainer.py`

#### 4.1 __init__ 수정

```python
def __init__(self, config: dict[str, Any] | None = None):
    super().__init__(config)
    # ... 기존 초기화 ...
    self.early_stopping = None  # setup()에서 초기화
```

#### 4.2 setup() 수정

```python
def setup(self, ctx: dict[str, Any]) -> None:
    super().setup(ctx)
    # ... 기존 setup ...

    # Early stopping 초기화
    recipe = ctx.get("recipe")
    if recipe and hasattr(recipe.train, "early_stopping"):
        es_config = recipe.train.early_stopping
        if es_config and es_config.enabled:
            from src.utils.early_stopping import LossEarlyStopping

            es_config_dict = (
                es_config.model_dump()
                if hasattr(es_config, "model_dump")
                else es_config
            )

            self.early_stopping = LossEarlyStopping(es_config_dict)
            console.print(f"[cyan]Early stopping enabled (monitor={es_config.monitor})[/cyan]")

    # 체크포인트 복원
    checkpoint_data = ctx.get("checkpoint_data")
    if checkpoint_data and self.early_stopping:
        es_state = checkpoint_data.get("early_stopping_state")
        if es_state:
            self.early_stopping.load_state(es_state)
            console.print("[cyan]Early stopping state restored[/cyan]")
```

#### 4.3 run() 수정

```python
def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
    # ... 기존 초기화 ...

    for step, batch in enumerate(dataloader):
        current_step = step + 1

        if current_step <= self.start_step:
            continue

        # 훈련 스텝
        out = self.train_step(batch)
        metrics = out

        # Early stopping 체크
        if self.early_stopping:
            should_stop = self.early_stopping.should_stop(metrics)

            # 분산 학습: rank 0 결정 브로드캐스트
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                should_stop_tensor = torch.tensor(
                    [should_stop],
                    dtype=torch.bool,
                    device=self.device
                )
                torch.distributed.broadcast(should_stop_tensor, src=0)
                should_stop = should_stop_tensor.item()

            if should_stop:
                reason = self.early_stopping.stop_reason
                console.print(f"[yellow]⚠ Early stopping: {reason}[/yellow]")

                # MLflow 로깅
                if self.mlflow:
                    self.mlflow.log_metrics({
                        "early_stopping/final_step": current_step,
                        "early_stopping/best_value": self.early_stopping.best_value,
                        "early_stopping/counter": self.early_stopping.counter,
                    })
                break

        # 체크포인트 저장
        if current_step % self.save_interval == 0:
            checkpoint_path = self._save_checkpoint(epoch, current_step, metrics)
            # ...

        # Max steps 체크
        if max_steps is not None and current_step >= max_steps:
            break

    # ... 최종 저장 ...
    return metrics
```

#### 4.4 _save_checkpoint() 수정

```python
def _save_checkpoint(self, epoch: int, step: int, metrics: dict) -> str:
    # ... 기존 코드 ...

    # Early stopping 상태 포함
    es_state = self.early_stopping.get_state() if self.early_stopping else None

    self.dist_manager.save_checkpoint(
        # ... 기존 파라미터 ...
        early_stopping_state=es_state,
    )

    return checkpoint_path
```

**체크리스트**:
- [ ] `__init__`에 early_stopping 변수
- [ ] `setup()`에 early_stopping 생성
- [ ] 체크포인트 복원 시 상태 로드
- [ ] `run()`에 early stopping 체크
- [ ] 분산 학습 브로드캐스트
- [ ] MLflow 로깅
- [ ] 체크포인트에 상태 저장

---

### Phase 5: Pipeline & Factory 수정

#### 5.1 training_pipeline.py

**파일**: `src/pipelines/training_pipeline.py`

```python
def run_training_pipeline(config: Config, recipe: Recipe, ...):
    # ... 기존 초기화 ...

    # Step 10: Pretraining (recipe.pretrain 사용)
    value_head_path = None

    if (recipe.train.algo == "critic-wmtp" and
        recipe.pretrain and
        recipe.pretrain.enabled and
        rm_model is not None and
        not dry_run):

        console.print("[cyan]🔬 Starting Pretraining (Stage 1)[/cyan]")

        pretrainer = ComponentFactory.create_pretrainer(recipe)
        pretrainer.setup({})

        stage1_result = pretrainer.run({
            "base_model": base,
            "rm_model": rm_model,
            "train_dataloader": train_dl,
            "run_name": recipe.run.name or "default",
        })

        if stage1_result.get("saved"):
            value_head_path = stage1_result["saved"]
            console.print(f"[green]✅ Pretraining complete: {value_head_path}[/green]")

        if stage1_result.get("early_stopped"):
            console.print(f"[yellow]⚠ Pretraining early stopped: {stage1_result.get('stop_reason')}[/yellow]")

    # ... 나머지 코드 ...
```

#### 5.2 component_factory.py

**파일**: `src/factory/component_factory.py`

```python
@staticmethod
def create_pretrainer(recipe: Recipe) -> Any:
    algo = recipe.train.algo

    if algo == "critic-wmtp":
        if not recipe.pretrain:
            raise ValueError("critic-wmtp requires pretrain configuration")

        pretrainer_config = {
            # Pretrain 섹션
            "num_epochs": recipe.pretrain.num_epochs,
            "max_steps": recipe.pretrain.max_steps,
            "lr": recipe.pretrain.lr,
            "gamma": recipe.pretrain.gamma,
            "gae_lambda": recipe.pretrain.gae_lambda,

            # Loss 섹션
            "temperature": recipe.loss.weight_temperature,

            # Critic 섹션
            "target": recipe.critic.target,
            "token_spread": recipe.critic.token_spread,
            "delta_mode": recipe.critic.delta_mode,
            "normalize": recipe.critic.normalize,
            "value_coef": recipe.critic.auxiliary_loss_coef,

            # Early stopping
            "early_stopping": (
                recipe.pretrain.early_stopping.model_dump()
                if recipe.pretrain.early_stopping
                else None
            ),
        }

        from src.components.registry import pretrainer_registry
        return pretrainer_registry.create("critic-head-pretrainer", pretrainer_config)
    else:
        raise ValueError(f"Pretraining not supported for: {algo}")
```

**체크리스트**:
- [ ] `training_pipeline.py`: recipe.pretrain 사용
- [ ] `component_factory.py`: create_pretrainer() 수정
- [ ] Early stopping 결과 처리

---

### Phase 6: Testing & Documentation

#### 6.1 테스트 YAML 업데이트

**파일**: `tests/configs/recipe.critic_wmtp.yaml`

```yaml
run:
  name: "m3_test_critic_wmtp"
  tags: ["test", "m3", "critic", "wmtp"]

# Pretraining (최상위 레벨)
pretrain:
  enabled: true
  num_epochs: 3
  max_steps: 30
  lr: 1e-4
  save_value_head: true

  early_stopping:
    enabled: true
    mode: "any"
    patience: 10
    min_delta: 1e-4
    monitor: "value_loss"
    grad_norm_threshold: 50.0
    grad_norm_window_size: 10
    grad_norm_threshold_ratio: 0.7
    variance_min: 0.1
    variance_max: 5.0

train:
  algo: "critic-wmtp"
  full_finetune: true
  max_steps: 2

  early_stopping:
    enabled: false
    patience: 100
    min_delta: 1e-5
    monitor: "wmtp_loss"

optim:
  optimizer: "adamw"
  lr: 5.0e-5
  weight_decay: 0.01
  betas: [0.9, 0.999]
  grad_clip: 1.0
  scheduler: "constant"
  warmup_ratio: 0.0

data:
  train:
    sources: ["mbpp"]
    max_length: 128
    batch_size: 1
    pack_sequences: false
  eval:
    sources: ["mbpp"]
    max_length: 128
    batch_size: 1
    pack_sequences: false

loss:
  weight_norm: "mean1.0_clip"
  lambda: 0.3
  weight_temperature: 0.7
  epsilon: 0.05
  max_weight: 3.0

critic:
  target: "rm_sequence"
  token_spread: "gae"
  delta_mode: "td"
  normalize: "zscore"
  discount_lambda: 0.95
  gamma: 0.99
  gae_lambda: 0.95
  auxiliary_loss_coef: 0.1
  value_lr: 5e-5
  use_pseudo_rewards: true

eval:
  protocol: "meta-mtp"
  sampling:
    temperature: 0.7
    top_p: 0.9
    n: 1
  metrics:
    - "mbpp_exact"
```

**체크리스트**:
- [ ] 테스트 YAML 업데이트
- [ ] 통합 테스트 작성
- [ ] 문서 최종 검토

---

## 5. 예상 효과

### 학술적 타당성
- ✅ 연구제안서 Line 112 "Critic 표류·고분산" 리스크 완화
- ✅ 연구제안서 Line 104 "그래디언트 중요도 안정화" 달성
- ✅ Loss 수렴 기반 조기 종료로 과적합 방지

### 실용적 이점
- ✅ 학습 시간 절약: 불필요한 epoch 제거
- ✅ 자원 효율화: GPU 사용 시간 감소
- ✅ 안정성 향상: Gradient 폭주 조기 감지
- ✅ 유연성: ANY/ALL/LOSS_ONLY 모드 선택

### 재현성
- ✅ YAML 설정으로 조기 종료 기준 명시
- ✅ MLflow에 early_stopping 메트릭 로깅
- ✅ 체크포인트에 상태 저장
- ✅ 분산 학습에서 일관된 중단 결정

---

## 6. 리스크 및 완화

### 리스크 1: False Positive

**문제**: patience가 너무 짧으면 일시적 정체에서 중단

**완화**:
- 보수적 기본값 (patience=10 for Stage 1, 100 for Stage 2)
- min_delta를 충분히 작게 설정 (1e-4)
- 테스트 환경에서 검증

### 리스크 2: Hyperparameter Tuning

**문제**: 여러 파라미터 튜닝 필요

**완화**:
- 연구제안서 기반 합리적 기본값
- enabled=false로 기존 동작 유지
- PPO 커뮤니티 권장값 참조

### 리스크 3: Mode 선택

**문제**: ANY/ALL/LOSS_ONLY 선택 혼란

**완화**:
- ANY를 기본값으로 설정 (가장 실용적)
- 문서에 각 모드 특성 설명
- 알고리즘별 권장 모드 제시

### 리스크 4: 분산 학습 동기화

**문제**: Rank 간 중단 결정 불일치

**완화**:
- Rank 0 결정 브로드캐스트
- torch.distributed.barrier() 사용
- 분산 학습 테스트

---

## 7. 전체 체크리스트

### Phase 1: Early Stopping Utility
- [ ] ValueHeadEarlyStopping에 mode 추가
- [ ] 윈도우 기반 gradient 체크 구현
- [ ] ANY/ALL/LOSS_ONLY 로직 구현
- [ ] State 관리에 grad_norm_history 추가
- [ ] 중단 이유 메시지 개선

### Phase 2: Schema
- [ ] EarlyStoppingConfig 클래스 정의
- [ ] PretrainConfig 클래스 정의
- [ ] Recipe에 pretrain 추가
- [ ] Train에서 stage1 제거
- [ ] Train에 early_stopping 추가
- [ ] Stage1Config 제거
- [ ] 스키마 테스트

### Phase 3: Pretrainer
- [ ] __init__에 early_stopping_config
- [ ] run()에 early_stopping 생성
- [ ] Variance 계산 추가
- [ ] Early stopping 체크 통합
- [ ] Nested loop 종료
- [ ] 반환값에 early_stopped 추가

### Phase 4: Main Trainer
- [ ] __init__에 early_stopping 변수
- [ ] setup()에 early_stopping 생성
- [ ] 체크포인트 상태 로드
- [ ] run()에 early stopping 체크
- [ ] 분산 학습 브로드캐스트
- [ ] MLflow 로깅
- [ ] 체크포인트 상태 저장

### Phase 5: Pipeline & Factory
- [ ] training_pipeline.py 수정
- [ ] component_factory.py 수정
- [ ] Early stopping 결과 처리

### Phase 6: Testing
- [ ] 테스트 YAML 업데이트
- [ ] 통합 테스트 작성
- [ ] 문서 최종 검토

---

## 8. 개발 원칙 준수

**원칙1**: ✅ 기존 구조 파악 완료

**원칙2**: ✅ 중복 없는 일관된 구조
- PretrainConfig로 통합 (train.stage1, critic.pretrainer 제거)
- EarlyStopping 유틸리티 공통 사용

**원칙3**: ✅ 기존 코드 삭제 계획 수립
- train.stage1, critic.pretrainer → pretrain으로 대체
- 하위 호환성 무시 (사용자 승인)

**원칙4**: ✅ 깨끗한 재작성 준비
- 불필요한 중복 제거
- 명확한 구조로 재설계

**원칙5**: ✅ 계획서 작성 후 승인 요청

**원칙6**: ✅ 의존성 문제 없음
- PyTorch, collections (표준 라이브러리)
- 추가 패키지 불필요

---

## 9. 구현 순서

1. Phase 1: Early Stopping Utility 개선 (2-3시간)
2. Phase 2: Schema 재설계 (1-2시간)
3. Phase 3: Pretrainer Integration (2-3시간)
4. Phase 4: Main Trainer Integration (2-3시간)
5. Phase 5: Pipeline & Factory 수정 (1-2시간)
6. Phase 6: Testing & Documentation (2-3시간)

**총 예상 시간**: 10-16시간

---

**문서 버전**: 2.0
**작성일**: 2025-10-01
**주요 개선**:
- PretrainConfig 최상위 레벨 분리
- ANY 모드로 실용성 개선
- 윈도우 기반 gradient 안정화
- API 명확화 및 분산 학습 통합
- 체크포인트 상태 관리 강화
