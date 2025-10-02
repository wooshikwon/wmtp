# Rich Console 출력 일관화 전략

## 문서 개요

### 목적
WMTP 훈련 파이프라인의 Rich Console 출력 방식과 정책을 일관화하여 사용자 경험을 개선하고 디버깅을 용이하게 한다.

### 배경
`test_baseline_output.log`와 `test_critic_output.log` 분석 결과, Stage 1/2 학습 및 모델 로드 결과 출력에서 형식과 계층, 출력 정책이 일관성 없이 난립하고 있음을 확인했다.

### 적용 범위
- 파이프라인 전체 흐름 출력
- 컴포넌트 작업 출력 (모델/데이터 로딩, 학습)
- Stage 1 (Value Head 사전학습) 출력
- Stage 2 (메인 학습) 출력

---

## 현재 상태 분석

### 1. 출력 방식 혼재

**문제점:**
- `model_loader.py`: 일반 `print()` 사용
- `training_pipeline.py`, `trainer` 클래스들: `console.print()` 사용
- MLflow, PyTorch warnings: raw stdout/stderr 출력

**코드 위치:**
```
src/components/loader/model_loader.py:74    print(f"\n🚀 모델 로딩 시작: {model_path}")
src/components/loader/model_loader.py:96    print("✅ 모델 로딩 완료\n")
```

### 2. 이모지 사용 불일치

**현재 사용 중인 이모지:**
- 🔍: 파이프라인 단계 추적 (training_pipeline.py 전반)
- 🚀: 작업 시작 (model_loader.py, data_loader.py)
- ✅: 작업 완료 (model_loader.py, critic_head_pretrainer.py)
- ✓: 완료 표시 (training_pipeline.py)
- 🔬: Stage 1 시작 (training_pipeline.py:273)
- 📎: Stage 간 데이터 전달 (training_pipeline.py:228)
- ⚠: 경고 메시지

**문제점:**
- 동일한 의미(완료)에 ✅, ✓ 혼용
- 🔍가 너무 많은 곳에 사용되어 의미 희석
- Stage 1 전용 이모지 🔬 사용으로 일관성 저해

### 3. 메시지 중복

**중복 1: Stage 1 시작 메시지**
```
training_pipeline.py:273  "[cyan]🔬 Starting Critic-WMTP Stage 1: Value Head Pretraining[/cyan]"
critic_head_pretrainer.py:197  "[cyan]Starting Stage 1: Value Head Pretraining[/cyan]"
```

**중복 2: Stage 1 완료 메시지**
```
critic_head_pretrainer.py:369  "[green]✅ Stage 1 Training Complete[/green]"
training_pipeline.py:294       "[green]✅ Stage 1 complete, Value Head saved at: ...[/green]"
training_pipeline.py:307       "[dim]🔍 Stage1 사전훈련 완료: critic-wmtp[/dim]"
```

**중복 3: 분산 샘플러 설정**
```
training_pipeline.py:198  "[dim]🔍 분산 훈련용 데이터 샘플러 설정 완료: {algo}[/dim]"  # 잘못된 위치
training_pipeline.py:215  "[dim]🔍 분산 훈련용 데이터 샘플러 설정 완료: {algo}[/dim]"  # 올바른 위치
```

### 4. 메시지 계층 구조 불명확

**현재 상태:**
- 모든 메시지가 동일한 레벨처럼 출력됨
- 파이프라인 주요 전환점과 세부 작업이 구분되지 않음
- `[dim]` 태그 사용이 불규칙함

### 5. 용어 혼재

**한글/영어 혼용:**
- "완료" vs "Complete" vs "complete"
- "시작" vs "Starting"
- "로딩 완료" vs "loaded"

**문제 예시:**
```
critic_head_pretrainer.py:369  "✅ Stage 1 Training Complete"  # 영어
training_pipeline.py:307       "🔍 Stage1 사전훈련 완료"        # 한글
```

### 6. Stage 1/2 출력 형식 차이

**Stage 1 (critic_head_pretrainer.py):**
```
Step     1 │ Loss: 19.5954 │ PPL: 323718400.00 │ Grad:  89.28 │ LR: 1.00e-04 │ Avg: 19.5954
```

**Stage 2 (base_wmtp_trainer.py):**
```
Epoch 1/3 Step     1 │ Loss: 0.0109 │ PPL: 1.01 │ Grad:   0.03 │ LR: 5.00e-05
```

**차이점:**
- Stage 1: `Avg` 메트릭 포함, Progress bar 표시
- Stage 2: `Avg` 메트릭 없음, Epoch prefix 포함

---

## 출력 표준 정의

### 3-Tier 계층 구조

**Tier 1: Pipeline Level (파이프라인 주요 단계)**
- 파이프라인 시작/완료
- Stage 전환 (Stage 1 → Stage 2)
- 최종 결과 출력

**Tier 2: Component Level (컴포넌트 작업)**
- 모델 로딩
- 데이터셋 로딩
- Trainer 생성
- Epoch 진행

**Tier 3: Detail Level (세부 진행 상황)**
- 로딩 단계 표시 ([1/4], [2/4] 등)
- Step 로그 및 메트릭
- 설정 정보 출력

### 이모지 표준

| 이모지 | 의미 | 사용 위치 | 예시 |
|--------|------|-----------|------|
| 🚀 | 시작 | Tier 1, 2 | `🚀 파이프라인 실행 시작` |
| ✅ | 완료 | Tier 1, 2 | `✅ 모델 로딩 완료` |
| 📊 | 진행 중 | Tier 2, 3 | `📊 Epoch 1/3` |
| ⚠️ | 경고 | 전체 | `⚠️ Early stopping triggered` |
| ❌ | 오류 | 전체 | `❌ 모델 로딩 실패` |

**제거할 이모지:**
- 🔍: 과도하게 사용되어 의미 희석 → 제거
- 🔬: Stage 1 전용 이모지 → 🚀로 통일
- 📎: 드물게 사용, 불필요 → 제거
- ✓: ✅와 중복 → ✅로 통일

### 색상 및 스타일 표준

**Tier 1 (Pipeline Level):**
```python
console.print(f"[bold cyan]🚀 {메시지}[/bold cyan]")  # 시작
console.print(f"[bold green]✅ {메시지}[/bold green]")  # 완료
```

**Tier 2 (Component Level):**
```python
console.print(f"\n🚀 {메시지}")  # 시작 (개행 포함)
console.print(f"[green]✅ {메시지}[/green]")  # 완료
console.print(f"\n[bold cyan]📊 Epoch {n}/{total}[/bold cyan]")  # 진행
```

**Tier 3 (Detail Level):**
```python
console.print(f"  [{단계}] {메시지}")  # 단계별 진행
console.print(f"  - {설정명}: {값}")  # 설정 정보
```

**경고 및 오류:**
```python
console.print(f"[yellow]⚠️ {메시지}[/yellow]")
console.print(f"[red]❌ {메시지}[/red]")
```

### 메시지 포맷 가이드라인

#### 1. 언어 규칙
- **주요 메시지**: 한글 사용
- **기술 용어**: 영어 유지 (Epoch, Step, Loss, PPL, Gradient 등)
- **통일 용어**:
  - "완료" (× Complete, × complete)
  - "시작" (× Starting, × started)

#### 2. 메시지 구조

**파이프라인 시작:**
```python
console.print("[bold cyan]🚀 파이프라인 실행 시작[/bold cyan]")
```

**컴포넌트 작업 시작:**
```python
console.print(f"\n🚀 모델 로딩 시작: {model_path}")
```

**단계별 진행 (선택적):**
```python
console.print(f"  [1/4] 메타데이터 로드 중...")
console.print(f"  [2/4] S3 다운로드 확인 중...")
```

**컴포넌트 작업 완료:**
```python
console.print("[green]✅ 모델 로딩 완료[/green]\n")
```

**설정 정보 출력:**
```python
console.print(f"  - Hidden size: {hidden_size}")
console.print(f"  - Learning rate: {lr}")
```

**Step 로그 (표준 포맷):**
```python
# Tier 3: Step 메트릭 출력
log_msg = (
    f"[cyan]Step {step:>5}[/cyan] │ "
    f"Loss: [yellow]{loss:.4f}[/yellow] │ "
    f"PPL: [yellow]{ppl:>7.2f}[/yellow] │ "
    f"Grad: [green]{grad_norm:>6.2f}[/green] │ "
    f"LR: [dim]{lr:.2e}[/dim]"
)
console.print(log_msg)
```

**Epoch 표시:**
```python
console.print(f"\n[bold cyan]📊 Epoch {epoch}/{num_epochs}[/bold cyan]")
```

**파이프라인 완료:**
```python
console.print("[bold green]✅ 파이프라인 실행 완료[/bold green]")
console.print(f"최종 메트릭: {metrics}")
```

---

## Phase별 구현 전략

### Phase 1: 표준 문서화 및 분석 (현재 문서)

**목표:**
- 현재 상태 분석 완료
- 출력 표준 정의 완료
- 구현 전략 수립 완료

**산출물:**
- `docs/console_output_standardization_plan.md` (본 문서)

**개발 원칙 적용:**
- ✅ **원칙 1**: 코드 흐름 분석 완료
  - `training_pipeline.py`, `critic_head_pretrainer.py`, `base_wmtp_trainer.py`, `model_loader.py` 분석
  - 로그 파일 2개 분석으로 실제 출력 패턴 확인
- ✅ **원칙 2**: 기존 구조 존중
  - Rich Console 패턴 유지
  - 이모지 사용은 정리하되 완전히 제거하지 않음
  - 계층 구조는 기존 패턴 기반으로 체계화
- ✅ **원칙 5**: 계획 수립 및 검토
  - 전면 삭제가 아닌 점진적 개선 방향 수립
  - Phase별 단계적 적용 계획

---

### Phase 2: 컴포넌트별 순차 적용

#### Phase 2-1: model_loader.py 수정

**목표:** `print()` → `console.print()` 전환, 표준 포맷 적용

**수정 대상:**
- `src/components/loader/model_loader.py`
- `src/components/loader/data_loader.py` (유사한 패턴)

**Before:**
```python
print(f"\n🚀 모델 로딩 시작: {model_path}")
print(f"  [1/4] 메타데이터 로드 중...")
print("✅ 모델 로딩 완료\n")
```

**After:**
```python
from rich.console import Console
console = Console()

console.print(f"\n🚀 모델 로딩 시작: {model_path}")
console.print(f"  [1/4] 메타데이터 로드 중...")
console.print("[green]✅ 모델 로딩 완료[/green]\n")
```

**개발 원칙 적용:**
- ✅ **원칙 1**: model_loader.py의 load_model() 메서드 흐름 확인
- ✅ **원칙 2**: 기존 4단계 구조 유지, 출력 방식만 개선
- ✅ **원칙 4-2**: 단순 wrapper 생성하지 않고 직접 console.print() 사용

---

#### Phase 2-2: training_pipeline.py 중복 제거 및 표준 적용

**목표:** 중복 메시지 제거, Tier 구분 명확화

**수정 사항:**

**1. 분산 샘플러 중복 제거:**
```python
# Before (training_pipeline.py:198-200) - 삭제
console.print(
    f"[dim]🔍 분산 훈련용 데이터 샘플러 설정 완료: {recipe.train.algo}[/dim]"
)

# After (training_pipeline.py:198-200) - 데이터셋 토크나이징 완료로 교체
console.print(f"[green]✅ 데이터셋 토크나이징 완료[/green]")
```

**2. Stage 1 시작 메시지 단일화:**
```python
# Before (training_pipeline.py:272-274)
console.print(
    "[cyan]🔬 Starting Critic-WMTP Stage 1: Value Head Pretraining[/cyan]"
)

# After
console.print("[bold cyan]🚀 Stage 1 시작: Value Head 사전학습[/bold cyan]")
```

**3. Stage 1 완료 메시지 정리:**
```python
# Before (training_pipeline.py:293-295) - 삭제
console.print(
    f"[green]✅ Stage 1 complete, Value Head saved at: {value_head_path}[/green]"
)

# Before (training_pipeline.py:307) - 삭제
console.print(f"[dim]🔍 Stage1 사전훈련 완료: {recipe.train.algo}[/dim]")

# After (training_pipeline.py에는 출력 없음, pretrainer에서만 출력)
# critic_head_pretrainer.py에서 완료 메시지 출력
```

**4. 🔍 이모지 제거 및 Tier 구분:**
```python
# Before
console.print(f"[dim]🔍 Base 모델 로딩 완료: {path}[/dim]")
console.print(f"[dim]🔍 토크나이저 생성 완료: {path}[/dim]")

# After - [dim] 제거, 간결화 또는 삭제
# (model_loader.py에서 이미 출력하므로 pipeline에서는 불필요)
```

**개발 원칙 적용:**
- ✅ **원칙 1**: training_pipeline.py의 전체 흐름 재확인
- ✅ **원칙 2**: 파이프라인 구조는 유지, 중복만 제거
- ✅ **원칙 4**: 중복 메시지는 전격 삭제
- ✅ **원칙 4-3**: "Stage1", "stage 1" 등 불일치 제거

---

#### Phase 2-3: critic_head_pretrainer.py 수정

**목표:** Stage 1 출력을 표준에 맞게 조정

**수정 사항:**

**1. 시작 메시지 제거 (pipeline에서 출력):**
```python
# Before (critic_head_pretrainer.py:197) - 삭제
console.print("[cyan]Starting Stage 1: Value Head Pretraining[/cyan]")

# After - 설정 정보만 출력
console.print("Stage 1 설정:")
console.print(f"  - Hidden size: {hidden_size}")
console.print(f"  - Learning rate: {self.lr}")
```

**2. Epoch 표시 표준화:**
```python
# Before
console.print(f"\n[bold]Epoch {epoch + 1}/{self.num_epochs}[/bold]")

# After
console.print(f"\n[bold cyan]📊 Epoch {epoch + 1}/{self.num_epochs}[/bold cyan]")
```

**3. Step 로그 포맷 통일 (Stage 2와 동일):**
```python
# Before - Avg 메트릭 포함
log_msg = (
    f"[cyan]Step {current_step:>5}[/cyan] │ "
    f"Loss: [yellow]{loss.item():.4f}[/yellow] │ "
    f"PPL: [yellow]{perplexity:>7.2f}[/yellow] │ "
    f"Grad: [green]{grad_norm:>6.2f}[/green] │ "
    f"LR: [dim]{self.lr:.2e}[/dim] │ "
    f"Avg: [dim]{avg_loss:.4f}[/dim]"  # 제거
)

# After - Avg 제거 (선택적으로 summary에만 표시)
log_msg = (
    f"[cyan]Step {current_step:>5}[/cyan] │ "
    f"Loss: [yellow]{loss.item():.4f}[/yellow] │ "
    f"PPL: [yellow]{perplexity:>7.2f}[/yellow] │ "
    f"Grad: [green]{grad_norm:>6.2f}[/green] │ "
    f"LR: [dim]{self.lr:.2e}[/dim]"
)
```

**4. 완료 메시지 표준화:**
```python
# Before
console.print("\n[green]✅ Stage 1 Training Complete[/green]")

# After
console.print("\n[bold green]✅ Stage 1 완료[/bold green]")
console.print("결과 요약:")
console.print(f"  - 평균 Loss: {avg_final_loss:.4f}")
console.print(f"  - 총 Step 수: {step_count}")
console.print(f"  - Value Head 저장 위치: {save_location}")
```

**개발 원칙 적용:**
- ✅ **원칙 1**: critic_head_pretrainer.py의 run() 메서드 흐름 확인
- ✅ **원칙 2**: 기존 학습 로직은 유지, 출력만 개선
- ✅ **원칙 4-3**: "Stage 1 Training Complete" → "Stage 1 완료"

---

#### Phase 2-4: base_wmtp_trainer.py 수정

**목표:** Stage 2 (메인 학습) 출력을 표준에 맞게 조정

**수정 사항:**

**1. Epoch 표시 통일:**
```python
# Before
console.print(f"\n[bold cyan]Epoch {epoch + 1}/{num_epochs}[/bold cyan]")

# After (이미 표준, 유지)
console.print(f"\n[bold cyan]📊 Epoch {epoch + 1}/{num_epochs}[/bold cyan]")
```

**2. Step 로그 포맷 유지 (이미 표준):**
```python
# 현재 형식이 표준이므로 유지
log_msg = (
    f"Epoch {epoch + 1}/{num_epochs} "
    f"[cyan]Step {step:>5}[/cyan] │ "
    f"Loss: [yellow]{loss:.4f}[/yellow] │ "
    f"PPL: [yellow]{ppl:>5.2f}[/yellow] │ "
    f"Grad: [green]{grad_norm:>6.2f}[/green] │ "
    f"LR: [dim]{lr:.2e}[/dim]"
)
```

**3. 체크포인트 저장 메시지:**
```python
# Before - print() 사용
print(f"Checkpoint saved locally: {checkpoint_path}")

# After - console.print() 사용
console.print(f"[green]✅ 체크포인트 저장 완료: {checkpoint_path}[/green]")
```

**개발 원칙 적용:**
- ✅ **원칙 1**: base_wmtp_trainer.py의 train() 메서드 확인
- ✅ **원칙 2**: 기존 출력 패턴이 이미 표준에 가까우므로 최소 수정

---

### Phase 3: 검증 및 정리

**목표:** 표준 적용 후 일관성 검증

**검증 항목:**

1. **로그 파일 재생성 및 비교**
   ```bash
   PYTHONPATH=. python -m src.cli.train \
     --config tests/configs/config.local_test.yaml \
     --recipe tests/configs/recipe.mtp_baseline.yaml > test_baseline_output_v2.log

   PYTHONPATH=. python -m src.cli.train \
     --config tests/configs/config.local_test.yaml \
     --recipe tests/configs/recipe.critic_wmtp.yaml > test_critic_output_v2.log
   ```

2. **체크리스트:**
   - [ ] print() 사용 제거됨
   - [ ] 이모지가 표준(🚀, ✅, 📊, ⚠️, ❌)만 사용됨
   - [ ] 🔍, 🔬, 📎, ✓ 제거됨
   - [ ] 중복 메시지 없음
   - [ ] Stage 1/2 Step 로그 포맷 통일됨
   - [ ] 한글/영어 용어 통일됨 ("완료", "시작")
   - [ ] Tier 구분이 명확함

3. **회귀 테스트:**
   - 기존 기능 정상 작동 확인
   - MLflow 로깅 정상 작동
   - 체크포인트 저장/로드 정상 작동

**개발 원칙 적용:**
- ✅ **원칙 5**: 결과를 계획과 비교하여 객관적으로 평가
- ✅ **원칙 6**: 기능 변경 없음, 출력만 개선

---

## 예시 코드 (Before/After)

### 예시 1: 파이프라인 시작

**Before:**
```python
console.print("🚀 파이프라인 실행 시작")
console.print("🔍 파이프라인 단계 추적 시작...")
```

**After:**
```python
console.print("[bold cyan]🚀 파이프라인 실행 시작[/bold cyan]")
```

---

### 예시 2: 모델 로딩

**Before (`model_loader.py`):**
```python
print(f"\n🚀 모델 로딩 시작: {model_path}")
print(f"  [1/4] 메타데이터 로드 중...")
print(f"  [2/4] 로컬 모델 사용 (다운로드 스킵)")
print(f"  [3/4] 로딩 전략 결정 중...")
print(f"      → {strategy} 전략 사용")
print(f"  [4/4] 커스텀 MTP 모델 로드 중...")
print("✅ 모델 로딩 완료\n")

# training_pipeline.py에서 중복 출력
console.print(f"[dim]🔍 Base 모델 로딩 완료: {config.paths.models.base}[/dim]")
```

**After (`model_loader.py`):**
```python
from rich.console import Console
console = Console()

console.print(f"\n🚀 모델 로딩 시작: {model_path}")
console.print(f"  [1/4] 메타데이터 로드 중...")
console.print(f"  [2/4] 로컬 모델 사용 (다운로드 스킵)")
console.print(f"  [3/4] 로딩 전략 결정 중...")
console.print(f"      → {strategy} 전략 사용")
console.print(f"  [4/4] 커스텀 MTP 모델 로드 중...")
console.print("[green]✅ 모델 로딩 완료[/green]\n")

# training_pipeline.py에서는 출력하지 않음 (중복 제거)
```

---

### 예시 3: Stage 1 시작/완료

**Before:**
```python
# training_pipeline.py
console.print("[cyan]🔬 Starting Critic-WMTP Stage 1: Value Head Pretraining[/cyan]")

# critic_head_pretrainer.py
console.print("[cyan]Starting Stage 1: Value Head Pretraining[/cyan]")
console.print(f"  - Hidden size: {hidden_size}")
# ... 학습 진행 ...
console.print("\n[green]✅ Stage 1 Training Complete[/green]")

# training_pipeline.py (다시)
console.print(f"[green]✅ Stage 1 complete, Value Head saved at: {path}[/green]")
console.print(f"[dim]🔍 Stage1 사전훈련 완료: critic-wmtp[/dim]")
```

**After:**
```python
# training_pipeline.py
console.print("[bold cyan]🚀 Stage 1 시작: Value Head 사전학습[/bold cyan]")

# critic_head_pretrainer.py
console.print("Stage 1 설정:")
console.print(f"  - Hidden size: {hidden_size}")
console.print(f"  - Learning rate: {self.lr}")
# ... 학습 진행 ...
console.print("\n[bold green]✅ Stage 1 완료[/bold green]")
console.print("결과 요약:")
console.print(f"  - 평균 Loss: {avg_final_loss:.4f}")
console.print(f"  - Value Head 저장 위치: {save_location}")

# training_pipeline.py에는 추가 출력 없음
```

---

### 예시 4: Epoch 및 Step 로그

**Before (Stage 1):**
```python
console.print(f"\n[bold]Epoch {epoch + 1}/{self.num_epochs}[/bold]")
for step, batch in enumerate(track(train_loader, description="Training")):
    # ...
    if step % log_interval == 0:
        log_msg = (
            f"[cyan]Step {step:>5}[/cyan] │ "
            f"Loss: [yellow]{loss:.4f}[/yellow] │ "
            f"PPL: [yellow]{ppl:>7.2f}[/yellow] │ "
            f"Grad: [green]{grad_norm:>6.2f}[/green] │ "
            f"LR: [dim]{lr:.2e}[/dim] │ "
            f"Avg: [dim]{avg_loss:.4f}[/dim]"  # Stage 1만 있음
        )
        console.print(log_msg)
```

**Before (Stage 2):**
```python
console.print(f"\n[bold cyan]Epoch {epoch + 1}/{num_epochs}[/bold cyan]")
for step, batch in enumerate(dataloader):
    # ...
    if step % log_interval == 0:
        log_msg = (
            f"Epoch {epoch + 1}/{num_epochs} "
            f"[cyan]Step {step:>5}[/cyan] │ "
            f"Loss: [yellow]{loss:.4f}[/yellow] │ "
            f"PPL: [yellow]{ppl:>5.2f}[/yellow] │ "
            f"Grad: [green]{grad_norm:>6.2f}[/green] │ "
            f"LR: [dim]{lr:.2e}[/dim]"
        )
        console.print(log_msg)
```

**After (Stage 1, 2 통일):**
```python
# 공통 포맷
console.print(f"\n[bold cyan]📊 Epoch {epoch + 1}/{num_epochs}[/bold cyan]")

for step, batch in enumerate(dataloader):
    # ...
    if step % log_interval == 0:
        log_msg = (
            f"[cyan]Step {step:>5}[/cyan] │ "
            f"Loss: [yellow]{loss:.4f}[/yellow] │ "
            f"PPL: [yellow]{ppl:>7.2f}[/yellow] │ "
            f"Grad: [green]{grad_norm:>6.2f}[/green] │ "
            f"LR: [dim]{lr:.2e}[/dim]"
        )
        console.print(log_msg)

# Avg는 Epoch 완료 후 summary에만 표시 (선택적)
```

---

### 예시 5: 파이프라인 완료

**Before:**
```python
console.print("🏁 파이프라인 실행 완료")
console.print(f"🔍 파이프라인 실행 결과: {metrics}")
console.print(f"🎉 훈련 완료! 최종 메트릭: {metrics}")
```

**After:**
```python
console.print("[bold green]✅ 파이프라인 실행 완료[/bold green]")
console.print(f"최종 메트릭: {metrics}")
```

---

## 체크리스트

구현 완료 후 다음 항목을 확인한다:

### Phase 2-1: model_loader.py
- [ ] `print()` → `console.print()` 전환 완료
- [ ] 색상 태그 적용 (`[green]✅...[/green]`)
- [ ] training_pipeline.py의 중복 출력 제거

### Phase 2-2: training_pipeline.py
- [ ] 분산 샘플러 중복 제거 (line 198)
- [ ] Stage 1 시작 메시지 표준화 (🔬 → 🚀)
- [ ] Stage 1 완료 중복 메시지 제거
- [ ] 🔍 이모지 제거 또는 최소화

### Phase 2-3: critic_head_pretrainer.py
- [ ] 시작 메시지 제거 (pipeline에서 출력)
- [ ] Epoch 표시 통일 (📊 추가)
- [ ] Step 로그 `Avg` 메트릭 제거
- [ ] 완료 메시지 표준화 ("완료")

### Phase 2-4: base_wmtp_trainer.py
- [ ] Epoch 표시 확인 (이미 표준)
- [ ] Step 로그 확인 (이미 표준)
- [ ] print() 사용 제거

### Phase 3: 검증
- [ ] 로그 재생성 및 비교
- [ ] 이모지 표준 준수 확인
- [ ] 중복 메시지 없음 확인
- [ ] 용어 통일 확인
- [ ] 회귀 테스트 통과

---

## 참고 사항

### Rich Console 기본 사용법

```python
from rich.console import Console

console = Console()

# 기본 출력
console.print("메시지")

# 색상 및 스타일
console.print("[bold cyan]강조 메시지[/bold cyan]")
console.print("[green]성공 메시지[/green]")
console.print("[yellow]경고 메시지[/yellow]")
console.print("[red]오류 메시지[/red]")
console.print("[dim]흐릿한 메시지[/dim]")

# 진행 표시
from rich.progress import track
for item in track(items, description="처리 중"):
    process(item)
```

### 주의 사항

1. **외부 라이브러리 경고는 제어 불가**
   - MLflow, PyTorch의 UserWarning은 표준 stderr로 출력됨
   - 필요 시 Python warnings 필터로 억제 가능

2. **로그 파일 저장 시 색상 태그**
   - 파일로 리다이렉트하면 ANSI 코드가 포함될 수 있음
   - 필요 시 `Console(force_terminal=False)` 사용

3. **과도한 wrapper 지양**
   - 원칙 4-2: 단순 wrapper 함수 생성하지 말 것
   - 직접 `console.print()` 호출 권장

---

## 개발 원칙 준수 체크

### ✅ 원칙 1: 앞/뒤 흐름 분석
- training_pipeline.py 전체 흐름 분석 완료
- trainer 클래스들의 출력 패턴 분석 완료
- 로그 파일 2개 분석으로 실제 동작 확인 완료

### ✅ 원칙 2: 기존 구조 존중 및 일관된 흐름
- Rich Console 패턴 유지
- 3-Tier 계층은 기존 패턴 체계화
- 중복 제거, 필수 로직 유지

### ✅ 원칙 3: 삭제/재생성 검토
- 전면 삭제 불필요, 점진적 개선 방향 수립
- 중복 메시지는 삭제, 핵심 출력은 개선

### ✅ 원칙 4: 깨끗한 코드 생성
- 원칙 4-1: 변수명/메시지 통일 ("완료", "시작")
- 원칙 4-2: 단순 wrapper 생성하지 않음
- 원칙 4-3: 불필요한 버전 주석 없음

### ✅ 원칙 5: 계획과 비교하여 객관적 기술
- 본 문서가 계획서 역할
- Phase 3에서 구현 결과를 본 문서와 비교 예정

### ✅ 원칙 6: 의존성 도구 활용
- Rich 라이브러리 활용 (기존 의존성)
- 추가 의존성 불필요

---

## 구현 우선순위

1. **High Priority (Phase 2-1, 2-2):**
   - model_loader.py: print() 제거
   - training_pipeline.py: 중복 제거

2. **Medium Priority (Phase 2-3, 2-4):**
   - critic_head_pretrainer.py: Stage 1 표준화
   - base_wmtp_trainer.py: 미세 조정

3. **Low Priority (Phase 3):**
   - 검증 및 문서 업데이트

---

## 버전 이력

- **v1.0.0 (2025-10-02)**: 초기 문서 작성
  - 현재 상태 분석 완료
  - 출력 표준 정의 완료
  - Phase별 구현 전략 수립 완료
