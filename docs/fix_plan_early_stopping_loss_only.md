# Early Stopping loss_only 모드 수정 계획

## 1. 문제 정의

### 1.1 발생한 에러
```
TypeError: '<=' not supported between instances of 'NoneType' and 'float'
Location: src/utils/early_stopping.py:390 in _check_variance_invalid
```

### 1.2 사용자 의도
- **Stage 1 (Value Head Pretraining)**에서 `mode="loss_only"` 사용
- Variance/Gradient 체크를 recipe에서 제거
- **이유**:
  - Variance: 초기화 시 매우 낮아 즉시 수렴으로 오판
  - Gradient: Value Head는 작은 네트워크라 폭주 기준이 부적절

### 1.3 현재 Recipe 설정
```yaml
# tests/configs/recipe.critic_wmtp.yaml
pretrain:
  early_stopping:
    enabled: true
    mode: "loss_only"
    patience: 10
    min_delta: 1e-4
    monitor: "value_loss"
    # variance_min/max, grad_norm_* 설정 없음!
```

**Production도 동일:**
```yaml
# configs/recipe.critic_wmtp.yaml
pretrain:
  early_stopping:
    mode: "loss_only"  # Most reliable: loss convergence only
```

---

## 2. 근본 원인

### 2.1 설계 결함 발견

현재 `ValueHeadEarlyStopping.should_stop()` 메서드는 **mode와 무관하게 모든 체크 함수를 호출**합니다:

```python
# src/utils/early_stopping.py:266-269
def should_stop(self, metrics: dict[str, float]) -> bool:
    # ... 기본 체크 ...

    # 🔥 mode="loss_only"여도 항상 실행!
    loss_converged = self._check_loss_convergence(value_loss)
    grad_unstable = self._check_gradient_instability(grad_norm)      # ← 호출됨
    variance_invalid = self._check_variance_invalid(value_variance)  # ← None 에러!

    # ... mode별 분기 ...
    if self.mode == "loss_only":
        if loss_converged:  # ← 이것만 사용하는데 위에서 이미 다 호출함!
            ...
```

### 2.2 왜 에러가 발생했나

`_check_variance_invalid()` 내부:
```python
def _check_variance_invalid(self, variance: float | None) -> bool:
    if variance is None:
        return False

    # 🔥 variance_min/max가 None이면 TypeError!
    return not (self.variance_min <= variance <= self.variance_max)
```

**문제 흐름:**
1. Recipe에 `variance_min`/`variance_max` 설정 없음
2. `__init__()`에서 기본값 할당 시도하지만 어떤 이유로 None이 됨
3. `mode="loss_only"`인데도 `_check_variance_invalid()` 호출됨
4. `self.variance_min`이 None인 채로 비교 연산(`<=`) 시도 → TypeError

---

## 3. 해결 철학

### 3.1 설계 원칙

**"Mode별로 필요한 체크만 수행한다"**

- `loss_only`: Loss convergence만 체크
- `any`/`all`: Loss + Gradient + Variance 모두 체크

### 3.2 2단계 방어 전략

**Primary Defense (주 방어선)**: Mode 기반 조기 분기
- `loss_only` 모드면 gradient/variance 함수를 **아예 호출하지 않음**
- 불필요한 연산 제거 + None 에러 근본 차단

**Secondary Defense (보조 방어선)**: None 체크 방어 코드
- `any`/`all` 모드에서 설정이 누락된 경우를 대비
- `_check_variance_invalid()`에 variance_min/max None 체크 추가
- `_check_gradient_instability()`와 동일한 패턴 적용

---

## 4. Phase별 구현 계획

### Phase 1: should_stop() 메서드 리팩토링

#### 목표
`mode="loss_only"`일 때 loss만 체크하고 즉시 반환하도록 최적화

#### 변경 파일
`src/utils/early_stopping.py`

#### 변경 대상 메서드
`ValueHeadEarlyStopping.should_stop()` (line 242-316)

#### 현재 구조
```python
def should_stop(self, metrics: dict[str, float]) -> bool:
    if not self.enabled:
        return False

    # 필수 메트릭 확인
    value_loss = metrics.get(self.monitor)
    grad_norm = metrics.get("grad_norm")
    value_variance = metrics.get("value_variance")

    if value_loss is None:
        return False

    # 🔥 mode와 무관하게 모든 체크 함수 호출
    loss_converged = self._check_loss_convergence(value_loss)
    grad_unstable = self._check_gradient_instability(grad_norm)
    variance_invalid = self._check_variance_invalid(value_variance)

    # Mode별 중단 결정
    should_stop = False
    reasons = []

    if self.mode == "any":
        # 하나라도 만족하면 중단
        if loss_converged:
            reasons.append(...)
        if grad_unstable:
            reasons.append(...)
        if variance_invalid:
            reasons.append(...)
        should_stop = len(reasons) > 0

    elif self.mode == "all":
        # 모두 만족해야 중단
        if loss_converged and not grad_unstable and not variance_invalid:
            reasons.append(...)
            should_stop = True

    else:  # "loss_only"
        # Loss convergence만 체크
        if loss_converged:
            reasons.append(...)
            should_stop = True

    # 중단 결정
    if should_stop:
        self.should_stop_flag = True
        self.stop_reason = f"Stage 1 early stop ({self.mode} mode): {'; '.join(reasons)}"
        return True

    return False
```

#### 변경 후 구조
```python
def should_stop(self, metrics: dict[str, float]) -> bool:
    if not self.enabled:
        return False

    # 필수 메트릭 확인
    value_loss = metrics.get(self.monitor)
    if value_loss is None:
        return False

    # 🎯 loss_only 모드는 여기서 조기 처리
    if self.mode == "loss_only":
        if self._check_loss_convergence(value_loss):
            self.should_stop_flag = True
            self.stop_reason = (
                f"Stage 1 early stop (loss_only mode): "
                f"loss converged ({value_loss:.6f}, patience={self.patience})"
            )
            return True
        return False

    # any/all 모드만 여기 도달 → 모든 메트릭 가져오기
    grad_norm = metrics.get("grad_norm")
    value_variance = metrics.get("value_variance")

    # 모든 체크 함수 호출
    loss_converged = self._check_loss_convergence(value_loss)
    grad_unstable = self._check_gradient_instability(grad_norm)
    variance_invalid = self._check_variance_invalid(value_variance)

    # Mode별 중단 결정
    should_stop = False
    reasons = []

    if self.mode == "any":
        # 하나라도 만족하면 중단
        if loss_converged:
            reasons.append(
                f"loss converged ({value_loss:.6f}, patience={self.patience})"
            )
        if grad_unstable:
            reasons.append(
                f"gradient unstable (threshold={self.grad_norm_threshold}, "
                f"ratio={self.grad_norm_threshold_ratio})"
            )
        if variance_invalid:
            reasons.append(
                f"variance out of range ({value_variance:.4f}, "
                f"range=[{self.variance_min}, {self.variance_max}])"
            )
        should_stop = len(reasons) > 0

    elif self.mode == "all":
        # 모두 만족해야 중단
        if loss_converged and not grad_unstable and not variance_invalid:
            reasons.append(
                f"all conditions met: loss={value_loss:.6f}, grad stable, variance valid"
            )
            should_stop = True

    # 중단 결정
    if should_stop:
        self.should_stop_flag = True
        self.stop_reason = (
            f"Stage 1 early stop ({self.mode} mode): {'; '.join(reasons)}"
        )
        return True

    return False
```

#### 핵심 변경 포인트

1. **Line 257-260 제거**: grad_norm, value_variance를 먼저 가져오는 부분 삭제
2. **Line 262-264 이후에 loss_only 분기 추가**:
   ```python
   if self.mode == "loss_only":
       # loss만 체크하고 즉시 반환
       ...
       return True/False
   ```
3. **any/all 모드에서만 메트릭 가져오기**:
   ```python
   grad_norm = metrics.get("grad_norm")
   value_variance = metrics.get("value_variance")
   ```
4. **기존 loss_only 블록 (line 300-306) 제거**: 위에서 처리되므로 불필요

#### 보존 사항
- ✅ 메서드 시그니처 (`def should_stop(self, metrics: dict[str, float]) -> bool`)
- ✅ 반환값 타입 (`bool`)
- ✅ `any` 모드 로직 (line 275-290)
- ✅ `all` 모드 로직 (line 292-298)
- ✅ `should_stop_flag`, `stop_reason` 설정 방식
- ✅ 에러 메시지 포맷

#### 개선 효과
- ✅ `loss_only` 모드일 때 불필요한 함수 호출 제거 (성능 향상)
- ✅ None 에러 근본적으로 차단
- ✅ 코드 의도와 실제 동작 일치
- ✅ 각 모드의 역할이 명확해짐

#### 위험도
**낮음** - `any`/`all` 모드 로직은 그대로 유지, `loss_only`만 최적화

---

### Phase 2: _check_variance_invalid() 방어 코드 추가

#### 목표
`variance_min`/`variance_max`가 None일 때 안전하게 처리 (Secondary Defense)

#### 변경 파일
`src/utils/early_stopping.py`

#### 변경 대상 메서드
`ValueHeadEarlyStopping._check_variance_invalid()` (line 376-390)

#### 현재 코드
```python
def _check_variance_invalid(self, variance: float | None) -> bool:
    """Variance 범위 이탈 체크.

    Args:
        variance: Value 예측 분산

    Returns:
        분산이 범위를 벗어났는지 여부 (True면 조기 종료 사유)
    """
    if variance is None:
        # Variance가 없으면 체크하지 않음 (유효함)
        return False

    # 범위를 벗어났으면 True (invalid)
    return not (self.variance_min <= variance <= self.variance_max)
```

#### 변경 후 코드
```python
def _check_variance_invalid(self, variance: float | None) -> bool:
    """Variance 범위 이탈 체크.

    Args:
        variance: Value 예측 분산

    Returns:
        분산이 범위를 벗어났는지 여부 (True면 조기 종료 사유)
    """
    # 설정이 없으면 체크하지 않음
    if self.variance_min is None or self.variance_max is None:
        return False

    # Variance가 없으면 체크하지 않음
    if variance is None:
        return False

    # 범위를 벗어났으면 True (invalid)
    return not (self.variance_min <= variance <= self.variance_max)
```

#### 핵심 변경 포인트

**Line 385와 386 사이에 3줄 추가:**
```python
# 설정이 없으면 체크하지 않음
if self.variance_min is None or self.variance_max is None:
    return False
```

#### 참고 패턴
`_check_gradient_instability()` (line 347-360)에서 이미 동일한 패턴 사용:
```python
def _check_gradient_instability(self, grad_norm: float | None) -> bool:
    # 🎯 이미 방어 코드가 있음!
    if grad_norm is None or self.grad_norm_threshold is None:
        return False
    ...
```

#### 개선 효과
- ✅ `any`/`all` 모드에서 variance 설정 누락 시에도 안전
- ✅ `_check_gradient_instability()`와 일관된 패턴
- ✅ Fail-safe 메커니즘 완성

#### 위험도
**매우 낮음** - 방어 코드만 추가, 기존 로직 무변경

---

### Phase 3: 테스트 및 검증

#### 3.1 단위 테스트

**테스트 파일:** `tests/test_early_stopping.py`

**테스트 시나리오:**

**Scenario 1: loss_only 모드 (정상 동작)**
```python
config = {
    "enabled": True,
    "mode": "loss_only",
    "patience": 10,
    "min_delta": 1e-4,
    "monitor": "value_loss",
    # variance_min/max 없음!
}
early_stop = ValueHeadEarlyStopping(config)

# Variance/gradient는 전달해도 무시됨
metrics = {
    "value_loss": 0.5,
    "grad_norm": 100.0,      # 무시
    "value_variance": 0.01   # 무시
}
result = early_stop.should_stop(metrics)
# 예상: False (아직 patience 안 차서)
```

**Scenario 2: loss_only 모드 (수렴)**
```python
# patience=10, min_delta=1e-4
for i in range(15):
    metrics = {"value_loss": 0.5}  # 개선 없음
    if early_stop.should_stop(metrics):
        print(f"Stopped at step {i}")
        break
# 예상: step 10에서 중단
```

**Scenario 3: any 모드 (설정 누락, 방어 코드 테스트)**
```python
config = {
    "enabled": True,
    "mode": "any",
    "patience": 10,
    # variance_min/max 없음!
}
early_stop = ValueHeadEarlyStopping(config)

metrics = {
    "value_loss": 0.5,
    "value_variance": 0.01  # variance_min/max None이지만 에러 없어야 함
}
result = early_stop.should_stop(metrics)
# 예상: 에러 없이 정상 동작
```

**Scenario 4: any 모드 (정상 동작)**
```python
config = {
    "enabled": True,
    "mode": "any",
    "patience": 10,
    "variance_min": 0.1,
    "variance_max": 5.0,
}
early_stop = ValueHeadEarlyStopping(config)

# Variance out of range
metrics = {
    "value_loss": 0.5,
    "value_variance": 0.05  # < 0.1
}
result = early_stop.should_stop(metrics)
# 예상: True (variance invalid)
```

**실행 명령:**
```bash
PYTHONPATH=. python -m pytest tests/test_early_stopping.py::TestValueHeadEarlyStopping -v
```

#### 3.2 통합 테스트

**원래 실패했던 명령어 재실행:**
```bash
PYTHONPATH=. python -m src.cli.train \
    --config tests/configs/config.local_test.yaml \
    --recipe tests/configs/recipe.critic_wmtp.yaml \
    --run-name test_critic_fixed \
    --tags test,fix,early-stopping \
    --verbose
```

**예상 결과:**
```
🔬 Starting Critic-WMTP Stage 1: Value Head Pretraining
Starting Stage 1: Value Head Pretraining
  - Hidden size: 768
  - Learning rate: 0.0001
  - Max steps: 30
  - Early stopping enabled (mode=loss_only)

Epoch 1/3
Training ━━━━━━━━━━━━━━━━━━━━ 100%
  Step 100: Loss = 0.xxxx

✅ Stage 1 Training Complete
  - Final avg loss: 0.xxxx
  - Total steps: 30
  - Value Head saved to: ./checkpoints/critic/test_critic_fixed/value_head_stage1.pt
```

#### 3.3 검증 체크리스트

- [ ] loss_only 모드에서 variance/gradient 함수가 호출되지 않는가?
- [ ] loss_only 모드에서 patience 기반 조기 종료가 정상 동작하는가?
- [ ] any 모드에서 모든 체크가 정상 동작하는가?
- [ ] all 모드에서 모든 체크가 정상 동작하는가?
- [ ] variance_min/max가 None일 때 에러가 발생하지 않는가?
- [ ] 기존 테스트 케이스들이 모두 통과하는가?
- [ ] Production recipe로도 정상 동작하는가?

---

## 5. 개발 원칙 준수 체크리스트

### [원칙 1] 앞/뒤 흐름 분석
- ✅ `should_stop()` → `_check_*` 메서드 호출 흐름 분석 완료
- ✅ `critic_head_pretrainer.py`에서 early_stopping 사용 패턴 확인
- ✅ Recipe 설정 → Config → EarlyStopping 초기화 흐름 파악

### [원칙 2] 기존 구조 존중 및 일관성
- ✅ 3가지 모드 (`any`, `all`, `loss_only`) 모두 유지
- ✅ `_check_gradient_instability()`의 None 체크 패턴을 `_check_variance_invalid()`에 적용
- ✅ 메서드 시그니처, 반환값, 에러 메시지 포맷 보존
- ✅ 기존 any/all 모드 로직 무변경

### [원칙 3] 삭제/재작성 판단
- ✅ 완전한 재작성 불필요, 로직 최적화만 수행
- ✅ loss_only 블록 중복 제거 (조기 분기로 이동)
- ✅ 핵심 로직은 유지, 실행 순서만 조정

### [원칙 4] 코드 품질
#### [원칙 4-1] 호환성 및 네이밍
- ✅ `should_stop(metrics)` 시그니처 유지 (호출부 변경 불필요)
- ✅ `stop_reason`, `should_stop_flag` 변수명 유지
- ✅ 메트릭 키 (`value_loss`, `grad_norm`, `value_variance`) 일관성 유지

#### [원칙 4-2] 메서드 계층
- ✅ 새로운 wrapper 메서드 추가 없음
- ✅ 기존 `_check_*` 메서드 활용
- ✅ 과도한 계층화 없음

#### [원칙 4-3] 주석 작성
- ✅ "Phase", "Version", "v2.0" 같은 임시 주석 없음
- ✅ 코드 동작에 대한 핵심 설명만 포함
- ✅ Docstring 유지 및 필요 시 업데이트

### [원칙 5] 검토 및 보고
- ✅ Phase별 구현 완료 후 사용자 승인 대기
- ✅ 계획서와 비교하여 객관적 보고
- ✅ 성과 과장 없이 실제 변경 사항만 기술

### [원칙 6] 의존성 관리
- ✅ 코드 수정만으로 해결 (의존성 변경 불필요)
- ✅ Python 표준 기능만 사용
- ✅ 외부 라이브러리 추가 없음

---

## 6. 예상 결과

### 수정 전
```
🔬 Starting Critic-WMTP Stage 1: Value Head Pretraining
...
Training ━━━━━━━━━━━━━━━━━━━━   0% -:--:--

❌ 예상치 못한 오류: '<=' not supported between instances of 'NoneType' and 'float'
```

### 수정 후 (loss_only 모드)
```
🔬 Starting Critic-WMTP Stage 1: Value Head Pretraining
Starting Stage 1: Value Head Pretraining
  - Hidden size: 768
  - Learning rate: 0.0001
  - Max steps: 30
  - Early stopping enabled (mode=loss_only)

Epoch 1/3
Training ━━━━━━━━━━━━━━━━━━━━ 100%
  Step 10: Loss = 0.5123
  ...

⚠ Early stopping: Stage 1 early stop (loss_only mode): loss converged (0.5123, patience=10)

✅ Stage 1 Training Complete
  - Final avg loss: 0.5123
  - Total steps: 25
  - Value Head saved to: ./checkpoints/critic/.../value_head_stage1.pt
  - Early stopped: Stage 1 early stop (loss_only mode): loss converged (0.5123, patience=10)
```

### 수정 후 (any 모드, 설정 누락)
```
# variance_min/max 없어도 에러 없음
✅ Stage 1 Training Complete
  - Final avg loss: 0.xxxx
  - Total steps: 30
```

---

## 7. 실행 계획

### Step 1: Phase 1 구현
1. `should_stop()` 메서드 리팩토링
2. 로컬 단위 테스트 실행
3. **사용자 승인 대기** → 결과 보고

### Step 2: Phase 2 구현
1. `_check_variance_invalid()` 방어 코드 추가
2. 로컬 단위 테스트 재실행
3. **사용자 승인 대기** → 결과 보고

### Step 3: Phase 3 검증
1. 전체 단위 테스트 실행
2. 통합 테스트 (원래 실패 명령어)
3. Production recipe 테스트
4. **최종 결과 보고**

---

## 8. 요약

### 핵심 변경
- `should_stop()`: loss_only 모드 조기 분기 추가
- `_check_variance_invalid()`: variance_min/max None 체크 추가

### 해결되는 문제
- ✅ loss_only 모드에서 TypeError 해결
- ✅ 불필요한 체크 함수 호출 제거
- ✅ 코드 의도와 실제 동작 일치
- ✅ Recipe에서 설정 누락해도 안전

### 사용자 판단 지지
- ✅ loss_only가 Stage 1에 가장 적합
- ✅ Variance/Gradient 체크가 오히려 방해됨
- ✅ Loss convergence가 가장 신뢰할 수 있는 지표

### 개발 원칙 100% 준수
- ✅ 기존 구조 존중 및 분석
- ✅ 일관된 패턴 적용
- ✅ 최소 수정 원칙
- ✅ Phase별 승인 기반 진행
