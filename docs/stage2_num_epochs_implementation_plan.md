# Stage 2 num_epochs 지원 구현 계획서

## 목적
Stage 2 (본 학습)에서 `num_epochs` 파라미터를 지원하여 LLM SFT 표준 학습 패턴을 따르도록 개선

## 문제 정의

### 현재 구조의 문제점
1. **Stage 1 (Value Head Pretraining)**: `num_epochs=3` 지원 ✅
2. **Stage 2 (Main Training)**: epoch 개념 없음, 단일 패스만 수행 ❌

### 영향
- **테스트 환경** (10 samples, max_steps=10)
  - 현재: 10 steps 학습 후 종료 (데이터 1회만 사용)
  - 개선 후: num_epochs=3 설정 시 30 steps (데이터 3회 반복)

- **프로덕션 환경** (10K+ samples, max_steps=10000)
  - 현재: 첫 10K steps만 학습 (전체 데이터의 일부만 사용)
  - 개선 후: num_epochs=3 설정 시 30K steps (전체 데이터 3회 반복)

### LLM SFT 표준 패턴
- Llama2/3: num_epochs=3
- Alpaca: num_epochs=3
- HuggingFace 기본값: num_epochs=3
- **결론**: num_epochs는 선택이 아니라 SFT의 핵심 파라미터

## 개발 원칙

모든 수정은 다음 원칙을 **반드시** 준수:

- **[원칙 1]** 추가하려는 기능의 앞/뒤 흐름을 직접 열어 확인 후, 분석하여 현재 구조를 먼저 파악
- **[원칙 2]** 기존 구조를 존중하여 필수 로직이 누락되지 않도록 하며, 중복 제거
- **[원칙 3]** 기존 코드 삭제가 필요한 경우 최선의 방안을 검토하여 승인받기
- **[원칙 4]** 삭제 결정 시 하위 호환성을 고려하지 말고, 중복 없이 깨끗하게 구현
  - **[원칙 4-1]** 앞/뒤 클래스와 메서드 호환 관계 검토, 통일성 있는 네이밍
  - **[원칙 4-2]** 과도하게 단순한 wrapper 메서드 지양
  - **[원칙 4-3]** 주석은 코드 동작의 핵심 설명만 (버전/phase 주석 금지)
- **[원칙 5]** 개발 완료 후 계획서와 비교하여 객관적으로 검토
- **[원칙 6]** 의존성 문제는 코드가 아니라 의존성 도구(uv)로 해결

## Phase별 구현 계획

### Phase 1: Recipe Schema 확장

**목표**: `Train` 클래스에 `num_epochs` 파라미터 추가

**수정 파일**:
- `src/settings/recipe_schema.py`

**구현 내용**:
```python
class Train(BaseModel):
    """Training configuration."""

    algo: Literal["baseline-mtp", "critic-wmtp", "rho1-wmtp"] = Field(...)
    full_finetune: bool = Field(default=True)

    # 추가: num_epochs (SFT 표준)
    num_epochs: int = Field(
        default=3,
        ge=1,
        description="Number of training epochs (SFT standard: 3)"
    )

    max_steps: int | None = Field(
        default=None,
        ge=1,
        description="Maximum training steps across all epochs (None for unlimited)"
    )

    eval_interval: int = Field(default=500, ge=1)
    save_interval: int = Field(default=1000, ge=1)
    early_stopping: EarlyStoppingConfig | None = Field(default=None)
```

**원칙 준수**:
- ✅ [원칙 1] PretrainConfig의 num_epochs 구조 참조 (lines 127-128)
- ✅ [원칙 2] Stage 1과 동일한 패턴 유지
- ✅ [원칙 4-3] 주석은 "SFT standard: 3"만 명시

**검증 방법**:
```bash
# Schema 검증
PYTHONPATH=. python -c "
from src.settings.recipe_schema import Train
t = Train(algo='baseline-mtp', num_epochs=3, max_steps=1000)
assert t.num_epochs == 3
print('✅ Phase 1 검증 완료')
"
```

**예상 영향**:
- 기존 recipe 파일: 기본값 3으로 동작 (하위 호환성 유지)
- 새 recipe 파일: 명시적 설정 권장

---

### Phase 2: Training Pipeline 수정

**목표**: `num_epochs`를 trainer에 전달하고, optimizer setup 로직 수정

**수정 파일**:
- `src/pipelines/training_pipeline.py`

**구현 내용**:

**2-1. Optimizer setup 수정 (line 171)**:
```python
# 현재
optimizer.setup({"num_training_steps": recipe.train.max_steps or 0})

# 수정 후
# num_training_steps 계산: min(max_steps, dataset_size * num_epochs)
dataset_size = len(tokenized)
num_epochs = recipe.train.num_epochs
max_steps = recipe.train.max_steps

if max_steps is None:
    # max_steps가 None이면 전체 epoch 기준
    num_training_steps = dataset_size * num_epochs
else:
    # max_steps와 전체 epoch 중 작은 값
    num_training_steps = min(max_steps, dataset_size * num_epochs)

optimizer.setup({"num_training_steps": num_training_steps})
```

**2-2. Trainer run 호출 수정 (line 329-331)**:
```python
# 현재
metrics = trainer.run(
    {"train_dataloader": train_dl, "max_steps": recipe.train.max_steps}
)

# 수정 후
metrics = trainer.run({
    "train_dataloader": train_dl,
    "num_epochs": recipe.train.num_epochs,
    "max_steps": recipe.train.max_steps
})
```

**원칙 준수**:
- ✅ [원칙 1] optimizer.setup 호출 구조 확인 (line 171)
- ✅ [원칙 2] 기존 max_steps 로직 존중하며 확장
- ✅ [원칙 4-1] ctx 딕셔너리에 일관된 키 추가

**검증 방법**:
```bash
# Dry-run으로 파이프라인 검증
PYTHONPATH=. python -m src.cli.train \
  --config tests/configs/config.local_test.yaml \
  --recipe tests/configs/recipe.mtp_baseline.yaml \
  --dry-run \
  --verbose
```

**예상 영향**:
- optimizer의 learning rate scheduler가 정확한 total steps 기반으로 동작
- warmup, cosine decay 등이 올바르게 계산됨

---

### Phase 3: Base WMTP Trainer 수정

**목표**: Epoch 루프 구현 및 global step 관리

**수정 파일**:
- `src/components/trainer/base_wmtp_trainer.py`

**구현 내용**:

**3-1. run 메서드 시그니처 수정 (line 411)**:
```python
# 현재
def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
    """
    Args:
        ctx: 'train_dataloader'와 'max_steps' 포함
    """

# 수정 후
def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
    """
    Args:
        ctx: 'train_dataloader', 'num_epochs', 'max_steps' 포함
    """
```

**3-2. Epoch 루프 구현 (line 419-436 교체)**:
```python
# 현재 (lines 419-436)
dataloader = ctx.get("train_dataloader")
if dataloader is None:
    raise ValueError("Trainer.run expects 'train_dataloader' in ctx")
max_steps: int | None = ctx.get("max_steps")

epoch = 0  # 단순화를 위해 epoch=0으로 설정
metrics = {}

log_interval = getattr(self.config, "log_interval", 100) if hasattr(self, "config") and self.config else 100

console.print(f"[green]체크포인트 저장 활성화: 매 {self.save_interval}스텝마다 저장[/green]")
console.print(f"[green]체크포인트 디렉토리: {self.checkpoint_dir}[/green]")
console.print(f"[green]로깅 간격: 매 {log_interval} step마다 출력[/green]")

for step, batch in enumerate(dataloader):
    current_step = step + 1

    # 재개시 이미 완료된 스텝 건너뛰기
    if current_step <= self.start_step:
        continue

# 수정 후
dataloader = ctx.get("train_dataloader")
if dataloader is None:
    raise ValueError("Trainer.run expects 'train_dataloader' in ctx")

num_epochs: int = ctx.get("num_epochs", 1)
max_steps: int | None = ctx.get("max_steps")
metrics = {}

# Global step 관리 (재개 지원)
global_step = self.start_step

log_interval = getattr(self.config, "log_interval", 100) if hasattr(self, "config") and self.config else 100

console.print(f"[green]체크포인트 저장 활성화: 매 {self.save_interval}스텝마다 저장[/green]")
console.print(f"[green]체크포인트 디렉토리: {self.checkpoint_dir}[/green]")
console.print(f"[green]로깅 간격: 매 {log_interval} step마다 출력[/green]")
console.print(f"[green]Epoch 설정: {num_epochs} epochs[/green]")

# Epoch 루프 추가
for epoch in range(num_epochs):
    console.print(f"\n[bold cyan]Epoch {epoch + 1}/{num_epochs}[/bold cyan]")

    for step, batch in enumerate(dataloader):
        global_step += 1

        # 재개시 이미 완료된 스텝 건너뛰기
        if global_step <= self.start_step:
            continue
```

**3-3. 전체 루프에서 max_steps 체크 수정 (line 437 이후)**:
```python
# 모든 current_step을 global_step으로 교체
# 예: line 448, 500, 523 등

# 현재 (line 448)
if current_step % log_interval == 0 or current_step == 1:

# 수정 후
if global_step % log_interval == 0 or global_step == 1:

# 현재 (line 455)
f"[cyan]Step {current_step:>5}[/cyan] │ "

# 수정 후
f"[cyan]Epoch {epoch+1}/{num_epochs} Step {global_step:>5}[/cyan] │ "

# 현재 (line 500)
if current_step % self.save_interval == 0:

# 수정 후
if global_step % self.save_interval == 0:

# 현재 (line 523)
if max_steps is not None and current_step >= max_steps:
    break

# 수정 후
if max_steps is not None and global_step >= max_steps:
    break  # inner loop 종료

# Outer loop에도 추가
if max_steps is not None and global_step >= max_steps:
    break  # outer loop 종료
```

**3-4. 체크포인트 저장 시 epoch 정보 포함 (line 502-504)**:
```python
# 현재
checkpoint_path = self._save_checkpoint(epoch, current_step, metrics)

# 수정 후
checkpoint_path = self._save_checkpoint(epoch, global_step, metrics)
```

**원칙 준수**:
- ✅ [원칙 1] critic_head_pretrainer.py의 epoch 루프 참조 (lines 219-224)
- ✅ [원칙 2] Stage 1과 일관된 패턴 (단, max_steps를 전역 리미터로 변경)
- ✅ [원칙 4-1] `current_step` → `global_step` 일관된 네이밍
- ✅ [원칙 4-2] 단순 wrapper 없이 직접 구현
- ✅ [원칙 4-3] "epoch loop", "global step" 등 핵심 설명만 주석

**검증 방법**:
```bash
# 실제 학습으로 epoch 동작 확인
PYTHONPATH=. python -m src.cli.train \
  --config tests/configs/config.local_test.yaml \
  --recipe tests/configs/recipe.mtp_baseline.yaml \
  --run-name test_num_epochs \
  --tags test,num_epochs \
  --verbose

# 로그 확인:
# - "Epoch 1/3", "Epoch 2/3", "Epoch 3/3" 출력 확인
# - global_step이 데이터셋 크기 * num_epochs까지 증가하는지 확인
```

**예상 영향**:
- 모든 trainer (baseline, critic, rho1)가 자동으로 epoch 지원 (base class 수정)
- 체크포인트에 epoch 정보 저장되어 재개 시 올바른 epoch부터 시작

---

### Phase 4: Recipe 파일 업데이트

**목표**: 모든 recipe에 `num_epochs` 명시적 설정

**수정 파일**:
- `tests/configs/recipe.mtp_baseline.yaml`
- `tests/configs/recipe.critic_wmtp.yaml`
- `tests/configs/recipe.rho1_wmtp_weighted.yaml`
- `tests/configs/recipe.rho1_wmtp_tokenskip.yaml`
- `configs/recipe.mtp_baseline.yaml`
- `configs/recipe.critic_wmtp.yaml`
- `configs/recipe.rho1_wmtp_weighted.yaml`
- `configs/recipe.rho1_wmtp_tokenskip.yaml`

**구현 내용**:

**4-1. 테스트 환경 (tests/configs/)**:
```yaml
# train 섹션에 num_epochs 추가
train:
  algo: "baseline-mtp"  # or critic-wmtp, rho1-wmtp
  full_finetune: true
  num_epochs: 3         # 추가: 테스트 데이터 3회 반복
  max_steps: 30         # 안전장치 (3 epochs * 10 samples = 30 steps 예상)
```

**4-2. 프로덕션 환경 (configs/)**:
```yaml
# train 섹션에 num_epochs 추가
train:
  algo: "critic-wmtp"   # or baseline-mtp, rho1-wmtp
  full_finetune: true
  num_epochs: 3         # 추가: SFT 표준
  max_steps: 30000      # 안전장치 (10K samples * 3 epochs = 30K steps 예상)
  eval_interval: 500
  save_interval: 1000
```

**원칙 준수**:
- ✅ [원칙 1] 각 recipe의 현재 max_steps 확인 후 적절한 값 설정
- ✅ [원칙 2] Stage 1 (pretrain) 설정과 일관성 유지
- ✅ [원칙 4] 기존 설정 삭제 없이 추가만 (max_steps는 보조 리미터로 유지)

**검증 방법**:
```bash
# 각 recipe 별로 검증
for recipe in mtp_baseline critic_wmtp rho1_wmtp_weighted rho1_wmtp_tokenskip; do
  echo "Testing $recipe..."
  PYTHONPATH=. python -m src.cli.train \
    --config tests/configs/config.local_test.yaml \
    --recipe tests/configs/recipe.$recipe.yaml \
    --run-name test_${recipe}_epochs \
    --tags test,epochs,$recipe \
    --verbose
done
```

**예상 영향**:
- **테스트**: 10 samples * 3 epochs = 30 steps (현재 대비 3배 학습)
- **프로덕션**: 10K samples * 3 epochs = 30K steps (현재 대비 3배 학습)
- 데이터 활용률 100% 달성

---

## 구현 순서 및 검증

### 단계별 검증 체크리스트

**Phase 1 완료 후**:
- [ ] Recipe schema에 num_epochs 필드 추가 확인
- [ ] Pydantic 검증 통과 확인 (ge=1)
- [ ] 기본값 3 동작 확인

**Phase 2 완료 후**:
- [ ] training_pipeline.py에서 num_epochs 전달 확인
- [ ] optimizer.setup()에 올바른 num_training_steps 계산 확인
- [ ] dry-run 성공 확인

**Phase 3 완료 후**:
- [ ] base_wmtp_trainer.py에서 epoch 루프 동작 확인
- [ ] global_step 관리 확인
- [ ] 로그에 "Epoch N/M" 출력 확인
- [ ] max_steps에서 정상 종료 확인
- [ ] 체크포인트에 epoch 저장 확인

**Phase 4 완료 후**:
- [ ] 8개 recipe 파일 모두 num_epochs 설정 확인
- [ ] 4개 알고리즘 각각 테스트 통과 확인
- [ ] MLflow에 epoch별 메트릭 기록 확인

### 전체 통합 테스트

```bash
# 1. Baseline MTP
PYTHONPATH=. python -m src.cli.train \
  --config tests/configs/config.local_test.yaml \
  --recipe tests/configs/recipe.mtp_baseline.yaml \
  --run-name final_test_baseline \
  --tags final,baseline,epochs

# 2. Critic WMTP (Stage 1 + Stage 2)
PYTHONPATH=. python -m src.cli.train \
  --config tests/configs/config.local_test.yaml \
  --recipe tests/configs/recipe.critic_wmtp.yaml \
  --run-name final_test_critic \
  --tags final,critic,epochs

# 3. Rho1 WMTP (weighted)
PYTHONPATH=. python -m src.cli.train \
  --config tests/configs/config.local_test.yaml \
  --recipe tests/configs/recipe.rho1_wmtp_weighted.yaml \
  --run-name final_test_rho1_weighted \
  --tags final,rho1,weighted,epochs

# 4. Rho1 WMTP (tokenskip)
PYTHONPATH=. python -m src.cli.train \
  --config tests/configs/config.local_test.yaml \
  --recipe tests/configs/recipe.rho1_wmtp_tokenskip.yaml \
  --run-name final_test_rho1_tokenskip \
  --tags final,rho1,tokenskip,epochs
```

---

## 계획 대비 검증 항목

구현 완료 후 다음 항목들을 계획서와 비교하여 검증:

### 기능 검증
- [ ] Stage 2가 num_epochs 지원
- [ ] 4개 알고리즘 모두 동작 (baseline, critic, rho1 weighted, rho1 tokenskip)
- [ ] 테스트/프로덕션 환경 모두 적용
- [ ] 데이터 활용률 100% (N epochs * dataset_size steps)

### 구조 검증
- [ ] Recipe schema에 num_epochs 추가
- [ ] Training pipeline에서 전달
- [ ] Base trainer에서 epoch 루프 구현
- [ ] Optimizer setup 수정
- [ ] 8개 recipe 파일 업데이트

### 원칙 준수 검증
- [ ] [원칙 1] 기존 구조 분석 후 구현
- [ ] [원칙 2] Stage 1 패턴 존중 (단, max_steps 의미 변경)
- [ ] [원칙 4-1] 일관된 네이밍 (global_step)
- [ ] [원칙 4-2] 불필요한 wrapper 없음
- [ ] [원칙 4-3] 핵심 주석만 작성
- [ ] [원칙 5] 계획서와 객관적 비교

### 성능 검증
- [ ] 학습 시간이 num_epochs 배수만큼 증가 (예상 동작)
- [ ] 메모리 사용량 변화 없음 (동일한 배치 크기)
- [ ] MLflow 로깅 정상 동작
- [ ] Early stopping 정상 동작 (global_step 기준)

---

## 예상 결과

### Before (현재)
```
테스트 환경:
- 10 samples, max_steps=10
- 실제 학습: 10 steps (1 epoch)
- 데이터 활용률: 100% (단일 패스)

프로덕션:
- 10K samples, max_steps=10000
- 실제 학습: 10K steps (1 epoch)
- 데이터 활용률: 100% (단일 패스)
```

### After (구현 후)
```
테스트 환경:
- 10 samples, num_epochs=3, max_steps=30
- 실제 학습: 30 steps (3 epochs)
- 데이터 활용률: 100% (3회 반복)

프로덕션:
- 10K samples, num_epochs=3, max_steps=30000
- 실제 학습: 30K steps (3 epochs)
- 데이터 활용률: 100% (3회 반복)
```

### 학습 효과
- LLM SFT 표준 패턴 준수
- 데이터를 여러 번 반복하여 수렴성 향상
- max_steps는 안전장치 역할 (무한 학습 방지)

---

## 잠재적 위험 및 대응

### 위험 1: max_steps 의미 변경
- **문제**: Stage 1은 "epoch당 max_steps", Stage 2는 "전역 max_steps"
- **대응**: 문서화 명확히 하고, recipe 파일에 주석 추가

### 위험 2: 기존 학습 재개 호환성
- **문제**: 이전 체크포인트에는 epoch 정보 없음
- **대응**: epoch 기본값 0으로 처리, 새 학습부터 epoch 저장

### 위험 3: Optimizer warmup 계산
- **문제**: num_training_steps 변경으로 warmup 영향
- **대응**: Phase 2에서 정확한 계산 로직 구현

### 위험 4: Early stopping 동작
- **문목**: global_step 기준으로 변경됨
- **대응**: 기존 early_stopping 로직이 이미 step 기반이므로 영향 없음 (확인 필요)

---

## 참고 자료

### Stage 1 (Pretraining) Epoch 구현
- 파일: `src/components/trainer/critic_head_pretrainer.py`
- Lines: 219-224 (epoch 루프)

### LLM SFT 표준
- Llama2: num_epochs=3
- Alpaca: num_epochs=3
- HuggingFace: num_epochs=3

### 관련 이슈
- 초기 분석: "Stage 2에 num_epochs 없음"
- 사용자 지적: "LLM SFT는 여러 epoch 돌리는데 왜 효과 없다고 했나?"
- 결론: num_epochs는 필수 파라미터

---

## 구현 시작 승인 대기

이 계획서를 검토한 후, 구현 시작 승인을 받으면 Phase 1부터 순차적으로 진행합니다.

**승인 시 확인 사항**:
1. Phase 순서가 적절한가?
2. 각 Phase의 검증 방법이 충분한가?
3. 원칙 준수 항목이 명확한가?
4. 예상 결과가 요구사항과 일치하는가?
