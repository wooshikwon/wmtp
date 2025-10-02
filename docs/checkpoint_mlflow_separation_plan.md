# Checkpoint-MLflow 역할 분리 구현 계획서

**작성일**: 2025-10-02
**목적**: 중복 저장 제거를 통한 비용 절감 및 역할 명확화
**전략**: 역할 기반 분리 (훈련 재개 vs 실험 추적)
**승인**: Option 3 - 최소 변경, 즉시 적용

---

## 📋 Executive Summary

### 핵심 문제
- **현상**: 동일 checkpoint가 paths.checkpoints와 MLflow에 중복 저장
- **원인**: `distribute_manager.py` Line 412-416에서 무조건 MLflow 업로드
- **영향**: 저장 비용 2배 (Production 기준 490GB → 70GB 가능)

### 해결 전략
**역할 기반 명시적 분리**:
- **paths.checkpoints**: 훈련 재개 전용 (주기적, keep_last 관리)
- **MLflow artifacts**: 최종 모델 전용 (영구, 버전 관리)

### 변경 범위
- **2개 파일 수정**: `distribute_manager.py`, `base_wmtp_trainer.py`
- **변경 라인**: 약 30줄 (삭제 15 + 추가 15)
- **적용 시간**: 즉시 (하위 호환성 유지)

### 기대 효과
- ✅ 저장 비용 86% 절감 (420GB 중복 제거)
- ✅ 역할 명확화 (재개 vs 추적)
- ✅ 설정 간소화 (테스트 환경)

---

## 🎯 개발 원칙 준수 체크리스트

### 원칙 1: 앞/뒤 흐름 확인 및 현재 구조 파악
- [x] **Phase 0 완료**: 저장 흐름 전체 분석
  - base_wmtp_trainer.py → _save_checkpoint → distribute_manager.save_checkpoint
  - MLflow 업로드 지점 파악 (distribute_manager.py Line 412-416)
  - 최종 모델 저장 흐름 파악 (_save_final_checkpoint)

**적용 방안**:
- Phase 1 전: distribute_manager.py save_checkpoint 메서드 전체 읽기
- Phase 2 전: base_wmtp_trainer.py _save_final_checkpoint 메서드 전체 읽기
- 각 Phase에서 수정 전 해당 메서드 호출/피호출 관계 확인

### 원칙 2: 기존 구조 존중 및 중복 제거
- [x] **구조 존중**: 기존 checkpoint 저장 로직 유지
- [x] **중복 제거**: MLflow 자동 업로드만 제거 (역할 분리)

**적용 방안**:
- 기존 save_checkpoint 시그니처 유지 (mlflow_manager 파라미터 유지)
- 로컬/S3 저장 로직 변경 없음
- 최종 모델만 MLflow 처리 (명시적 분리)

### 원칙 3: 삭제 vs 수정 검토 및 승인
- [x] **승인 완료**: Option 3 (역할 분리) 선택
- [x] **삭제 대상**: MLflow 자동 업로드 코드만 (Line 412-416)
- [x] **수정 대상**: _save_final_checkpoint에 명시적 MLflow 처리 추가

**적용 방안**:
- distribute_manager.py: 중간 checkpoint MLflow 업로드 삭제
- base_wmtp_trainer.py: 최종 모델 MLflow 등록 강화
- 하위 호환: mlflow_manager 파라미터 유지 (향후 확장 가능)

### 원칙 4: 깨끗한 코드 생성
- [x] **하위 호환 무시**: 중복 업로드 완전 제거
- [x] **통일성**: 주석 및 변수명 일관성 유지
- [x] **단순성**: wrapper 메서드 없이 직접 구현
- [x] **주석**: 불필요한 phase 번호 제거, 동작 설명만

**적용 방안** (원칙 4-1, 4-2, 4-3):
- 변수명: checkpoint_path, final_path (일관성)
- 주석: "역할 분리: 훈련 재개 vs 실험 추적" (핵심만)
- Phase 번호 주석 제거 (ex. "Phase 3:" → 삭제)
- wrapper 없이 직접 mlflow.log_model, log_artifact 호출

### 원칙 5: 계획 대비 검토 및 객관적 보고
- [ ] **Phase 1 완료 후**: 계획서 대비 검증 및 보고
- [ ] **Phase 2 완료 후**: 계획서 대비 검증 및 보고
- [ ] **Phase 4 완료 후**: 최종 성과 객관적 기술

**적용 방안**:
- 각 Phase 완료 시 체크리스트 기반 검증
- 계획서와 실제 변경 사항 비교 보고
- 예상 효과와 실제 효과 비교 (저장 시간, 파일 크기)

### 원칙 6: 패키지 의존성 도구 활용
- [x] **uv 환경 사용**: 모든 테스트에서 `uv run` 활용
- [x] **의존성 변경 없음**: 기존 패키지만 사용 (torch, mlflow)

**적용 방안**:
- 검증 스크립트: `uv run python -m src.cli.train ...`
- 의존성 추가 없음 (기존 코드만 수정)

---

## 1. 문제 정의

### 1.1. 중복 저장 코드 위치

**distribute_manager.py (Line 405-417)**:
```python
else:
    # 로컬 저장
    torch.save(checkpoint, checkpoint_path)
    console.print(
        f"[green]Checkpoint saved locally: {checkpoint_path}[/green]"
    )

    # MLflow에도 기록 (있는 경우)  ← 중복 발생 지점!
    if mlflow_manager:
        mlflow_manager.log_artifact(
            local_path=checkpoint_path, artifact_path="checkpoints"
        )
```

**결과**: 매 save_interval마다 로컬 + MLflow 2곳에 저장

### 1.2. 저장 비용 계산

**Production 예시** (7B 모델, 30K steps, save_interval=1000):
```
Checkpoint 저장 횟수: 30회

paths.checkpoints (keep_last=5):
- 저장량: 14GB × 5 = 70GB

MLflow artifacts (영구 보존):
- 저장량: 14GB × 30 = 420GB

총 저장량: 490GB
중복 제거 후: 70GB (86% 절감)
```

---

## 2. 해결 전략: 역할 기반 분리

### 2.1. 설계 원칙

| 시스템 | 목적 | 저장 대상 | 관리 정책 | 생명주기 |
|--------|------|-----------|-----------|----------|
| **paths.checkpoints** | 훈련 재개 | 주기적 checkpoint | keep_last | Ephemeral |
| **MLflow artifacts** | 실험 추적 | 최종 모델만 | 영구 보존 | Persistent |

### 2.2. 변경 최소화

**변경 파일**:
1. `src/utils/distribute_manager.py` (Line 405-417)
2. `src/components/trainer/base_wmtp_trainer.py` (Line 151-218)

**변경 내용**:
- distribute_manager: MLflow 자동 업로드 제거
- base_wmtp_trainer: _save_final_checkpoint에서 MLflow 명시적 처리

---

## 📝 Phase 0: 사전 분석 ✅

**목표**: 코드 흐름 완전 파악 (원칙 1)

**완료 사항**:
- [x] 저장 흐름 전체 분석
  - base_wmtp_trainer.py (Line 559-602): _save_checkpoint 메서드
  - distribute_manager.py (Line 310-417): save_checkpoint 메서드
  - base_wmtp_trainer.py (Line 151-218): _save_final_checkpoint 메서드

- [x] 중복 발생 지점 파악
  - distribute_manager.py Line 412-416: 무조건 MLflow 업로드
  - S3 경로인 경우 Line 388도 MLflow 업로드

- [x] 영향 범위 확정
  - 2개 파일 수정
  - 기존 API 변경 없음 (하위 호환)
  - Config/Recipe 변경 없음

**원칙 준수**:
- ✅ 원칙 1: 앞/뒤 흐름 완전 분석 완료
- ✅ 원칙 2: 기존 구조 파악 (존중 대상 확인)

---

## 📝 Phase 1: distribute_manager.py 중복 업로드 제거

**목표**: 중간 checkpoint MLflow 자동 업로드 제거

### 원칙 적용 체크리스트

- [ ] **원칙 1**: save_checkpoint 메서드 전체 읽기 (Line 310-419)
- [ ] **원칙 2**: 로컬/S3 저장 로직 유지, MLflow 부분만 제거
- [ ] **원칙 3**: 승인된 삭제 대상 (Line 412-416, Line 388)
- [ ] **원칙 4-1**: 파라미터명 유지 (mlflow_manager), 주석 정리
- [ ] **원칙 4-2**: wrapper 없이 직접 삭제
- [ ] **원칙 4-3**: "Phase 3" 같은 주석 제거, 역할 설명 추가
- [ ] **원칙 5**: 변경 후 계획 대비 검증
- [ ] **원칙 6**: uv run으로 테스트

### 1.1. 코드 변경

**파일**: `src/utils/distribute_manager.py`

#### 변경 1: Docstring 업데이트 (Line 320-340)

**Before**:
```python
"""
체크포인트 저장 (FSDP/non-FSDP 모델 모두 지원).

WMTP 맥락:
학습 중간 상태를 저장하여 재개 가능하게 합니다.
특히 장시간 학습이 필요한 대규모 모델에서 중요합니다.
S3 경로 지원 및 MLflow 자동 업로드 기능이 추가되었습니다.

매개변수:
    model: FSDP 래핑된 모델 또는 일반 torch.nn.Module
    optimizer: 옵티마이저
    checkpoint_path: 저장 경로 (로컬 또는 s3://)
    epoch: 현재 에폭
    step: 현재 스텝
    mlflow_manager: MLflow 매니저 (선택적)
    **kwargs: 추가 저장 데이터 (loss, metrics 등)

주의사항:
    - rank0_only=True로 메인 프로세스만 저장
    - offload_to_cpu=True로 GPU 메모리 절약
    - S3 경로시 직접 업로드, 로컬 경로시 파일 저장 후 MLflow 업로드
"""
```

**After**:
```python
"""
체크포인트 저장 (FSDP/non-FSDP 모델 모두 지원).

역할 분리:
- 주기적 checkpoint: paths.checkpoints에만 저장 (훈련 재개용)
- 최종 모델: base_wmtp_trainer._save_final_checkpoint에서 MLflow 처리

매개변수:
    model: FSDP 래핑된 모델 또는 일반 torch.nn.Module
    optimizer: 옵티마이저
    checkpoint_path: 저장 경로 (로컬 또는 s3://)
    epoch: 현재 에폭
    step: 현재 스텝
    mlflow_manager: MLflow 매니저 (최종 모델용, 중간 checkpoint는 미사용)
    **kwargs: 추가 저장 데이터 (loss, metrics 등)

주의사항:
    - rank0_only=True로 메인 프로세스만 저장
    - offload_to_cpu=True로 GPU 메모리 절약
    - 중간 checkpoint는 MLflow에 업로드하지 않음 (비용 절감)
"""
```

#### 변경 2: S3 경로 MLflow 업로드 제거 (Line 372-404)

**Before**:
```python
# S3 또는 로컬 저장 처리
if checkpoint_path.startswith("s3://"):
    # S3에 직접 저장
    import io
    import tempfile
    from pathlib import Path

    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    buffer.seek(0)

    if mlflow_manager:
        # MLflow를 통해 아티팩트로 업로드 (임시 파일 경유)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / f"checkpoint_step_{step}.pt"
            with open(tmp_path, "wb") as f:
                f.write(buffer.getvalue())
            mlflow_manager.log_artifact(
                local_path=str(tmp_path),
                artifact_path=f"checkpoints/step_{step}",
            )
        console.print(
            f"[green]Checkpoint uploaded to MLflow: step_{step}[/green]"
        )
    else:
        # S3Manager를 사용하여 직접 저장
        from src.utils.s3 import S3Manager

        s3_manager = S3Manager()
        s3_key = checkpoint_path.replace("s3://wmtp/", "")
        s3_manager.upload_from_bytes(buffer.getvalue(), s3_key)
        console.print(
            f"[green]Checkpoint saved to S3: {checkpoint_path}[/green]"
        )
```

**After**:
```python
# S3 저장 (MLflow 우회)
if checkpoint_path.startswith("s3://"):
    import io
    from src.utils.s3 import S3Manager

    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    buffer.seek(0)

    s3_manager = S3Manager()
    s3_key = checkpoint_path.replace("s3://wmtp/", "")
    s3_manager.upload_from_bytes(buffer.getvalue(), s3_key)
    console.print(
        f"[green]Checkpoint saved to S3: {checkpoint_path}[/green]"
    )
```

**변경 요약**:
- 삭제: MLflow 업로드 로직 (tmpdir 생성 및 log_artifact)
- 삭제: if mlflow_manager 조건문
- 유지: S3 직접 업로드 로직
- 간소화: import 정리 (tempfile, Path 제거)

#### 변경 3: 로컬 경로 MLflow 업로드 제거 (Line 405-417)

**Before**:
```python
else:
    # 로컬 저장
    torch.save(checkpoint, checkpoint_path)
    console.print(
        f"[green]Checkpoint saved locally: {checkpoint_path}[/green]"
    )

    # MLflow에도 기록 (있는 경우)
    if mlflow_manager:
        mlflow_manager.log_artifact(
            local_path=checkpoint_path, artifact_path="checkpoints"
        )
```

**After**:
```python
else:
    # 로컬 저장 (훈련 재개용)
    torch.save(checkpoint, checkpoint_path)
    console.print(
        f"[green]Checkpoint saved locally: {checkpoint_path}[/green]"
    )
```

**변경 요약**:
- 삭제: MLflow 업로드 전체 (Line 412-416)
- 주석 수정: "로컬 저장" → "로컬 저장 (훈련 재개용)"

### 1.2. 검증 방법

#### 1.2.1. 코드 리뷰
```bash
# 변경 사항 확인
git diff src/utils/distribute_manager.py

# 기대:
# - Line 320-340: Docstring 업데이트
# - Line 372-404: S3 경로 MLflow 코드 삭제
# - Line 405-417: 로컬 경로 MLflow 코드 삭제
```

#### 1.2.2. 단위 테스트 (Dry-run)
```bash
# 설정 검증만 (실제 훈련 X)
export PYTHONPATH=. && uv run python -m src.cli.train \
    --config tests/configs/config.local_test.yaml \
    --recipe tests/configs/recipe.mtp_baseline.yaml \
    --run-name test_phase1_dryrun \
    --tags test,phase1 \
    --dry-run \
    --verbose
```

**기대 결과**:
- ✅ 설정 검증 통과
- ✅ 에러 없음

#### 1.2.3. 실제 훈련 (10 step)
```bash
# 짧은 훈련으로 checkpoint 저장 확인
export PYTHONPATH=. && uv run python -m src.cli.train \
    --config tests/configs/config.local_test.yaml \
    --recipe tests/configs/recipe.mtp_baseline.yaml \
    --run-name test_phase1_training \
    --tags test,phase1 \
    --verbose 2>&1 | tee /tmp/phase1_training.log
```

**기대 결과**:
- ✅ 훈련 정상 완료
- ✅ 로컬 checkpoint 저장 확인: `ls -lh ./test_checkpoints/*/checkpoint_*.pt`
- ✅ MLflow artifacts 없음 확인: `ls -lh /tmp/mlflow_m3/{run_id}/artifacts/checkpoints/` → 없어야 함

### 1.3. 완료 기준

- [ ] Docstring 업데이트 완료
- [ ] S3 경로 MLflow 업로드 제거 완료
- [ ] 로컬 경로 MLflow 업로드 제거 완료
- [ ] Dry-run 테스트 통과
- [ ] 실제 훈련 테스트 통과
- [ ] MLflow artifacts/checkpoints 디렉토리 없음 확인
- [ ] 로컬 checkpoint 정상 저장 확인

### 1.4. 원칙 5: 계획 대비 검증 및 보고

**계획 목표**:
```
✅ distribute_manager.py 중복 업로드 제거
✅ Docstring 업데이트
✅ S3/로컬 모두 MLflow 우회
```

**실제 달성 (보고 형식)**:
```
[Phase 1 완료 보고]

변경 파일: src/utils/distribute_manager.py
변경 라인: 3개 섹션 (Docstring, S3 로직, 로컬 로직)

계획 대비:
✅ Docstring 업데이트 (역할 분리 명시)
✅ S3 경로 MLflow 업로드 제거 (Line 383-393 삭제)
✅ 로컬 경로 MLflow 업로드 제거 (Line 412-416 삭제)

검증 결과:
✅ Dry-run 테스트 통과
✅ 실제 훈련 10 step 정상 완료
✅ 로컬 checkpoint 정상 저장 (checkpoint_step_1.pt 존재)
✅ MLflow artifacts/checkpoints 없음 (중복 제거 확인)

예상 효과 달성:
✅ 중간 checkpoint MLflow 업로드 0회 (기존 30회)
✅ 저장 시간 단축 (MLflow 업로드 제거)

번외 발견사항: 없음
```

---

## 📝 Phase 2: base_wmtp_trainer.py 최종 모델 MLflow 등록

**목표**: _save_final_checkpoint에서 MLflow 명시적 처리

### 원칙 적용 체크리스트

- [ ] **원칙 1**: _save_final_checkpoint 메서드 전체 읽기 (Line 151-218)
- [ ] **원칙 2**: 기존 저장 로직 유지, MLflow 처리만 강화
- [ ] **원칙 3**: 승인된 수정 (최종 모델만 MLflow)
- [ ] **원칙 4-1**: 변수명 통일 (final_path), 주석 일관성
- [ ] **원칙 4-2**: log_model, log_artifact 직접 호출 (wrapper 없음)
- [ ] **원칙 4-3**: "Phase 3" 제거, 역할 설명만
- [ ] **원칙 5**: 변경 후 계획 대비 검증
- [ ] **원칙 6**: uv run으로 테스트

### 2.1. 코드 변경

**파일**: `src/components/trainer/base_wmtp_trainer.py`

#### 변경: _save_final_checkpoint 수정 (Line 151-218)

**Before** (Line 151-218):
```python
def _save_final_checkpoint(self, epoch: int, step: int, metrics: dict) -> str:
    """
    최종 모델 저장 (Phase 3: S3/로컬 자동 판단)

    Args:
        epoch: 최종 에폭
        step: 최종 스텝
        metrics: 최종 메트릭

    Returns:
        저장된 최종 모델 경로 (문자열)
    """
    # S3/로컬 자동 판단하여 최종 모델 경로 생성
    if self.is_s3_checkpoint:
        # S3 경로: 문자열 결합
        final_path = f"{self.checkpoint_dir}/final_model.pt"
    else:
        # 로컬 경로: Path 객체 사용
        final_path = str(self.checkpoint_dir / "final_model.pt")

    # 최종 체크포인트 저장 (MLflow 통합)
    self.dist_manager.save_checkpoint(
        model=self.model,
        optimizer=self.optimizer,
        checkpoint_path=final_path,
        epoch=epoch,
        step=step,
        mlflow_manager=self.mlflow,  # MLflow 매니저 전달
        metrics=metrics,
        algorithm=getattr(self, "algorithm", "wmtp"),
        final_model=True,
        mlflow_run_id=self.mlflow.get_run_id() if self.mlflow else None,
    )

    # MLflow 모델 레지스트리 등록 및 아티팩트 업로드
    if self.mlflow is not None:
        try:
            # 모델 이름 생성 (recipe에서 알고리즘 정보 사용)
            model_name = f"wmtp-{self.algorithm}"

            # 모델 레지스트리 등록
            self.mlflow.log_model(
                model=self.model,
                name="final_model",
                registered_model_name=model_name,
            )

            # 체크포인트 파일 업로드 (로컬 경로만 지원)
            if not self.is_s3_checkpoint:
                self.mlflow.log_artifact(
                    local_path=final_path, artifact_path="final_checkpoint"
                )
            else:
                console.print(
                    "[blue]S3 체크포인트는 MLflow artifact 업로드 생략[/blue]"
                )

            console.print(f"[green]MLflow 모델 등록 완료: {model_name}[/green]")
        except Exception as e:
            console.print(
                f"[yellow]MLflow model registration warning: {e}[/yellow]"
            )

    storage_type = "S3" if self.is_s3_checkpoint else "로컬"
    console.print(
        f"[green]{storage_type} 최종 모델 저장 완료: {final_path}[/green]"
    )
    return final_path
```

**After**:
```python
def _save_final_checkpoint(self, epoch: int, step: int, metrics: dict) -> str:
    """
    최종 모델 저장 및 MLflow 등록

    역할:
    - paths.checkpoints에 final_model.pt 저장 (훈련 재개용)
    - MLflow에 모델 등록 및 artifact 업로드 (실험 추적용)

    Args:
        epoch: 최종 에폭
        step: 최종 스텝
        metrics: 최종 메트릭

    Returns:
        저장된 최종 모델 경로
    """
    # 1. paths.checkpoints에 저장
    if self.is_s3_checkpoint:
        final_path = f"{self.checkpoint_dir}/final_model.pt"
    else:
        final_path = str(self.checkpoint_dir / "final_model.pt")

    # Early stopping 상태 수집
    es_state = self.early_stopping.get_state() if self.early_stopping else None

    # 저장 (MLflow 전달하지 않음 - 아래에서 별도 처리)
    self.dist_manager.save_checkpoint(
        model=self.model,
        optimizer=self.optimizer,
        checkpoint_path=final_path,
        epoch=epoch,
        step=step,
        mlflow_manager=None,  # 중복 방지
        metrics=metrics,
        algorithm=self.algorithm,
        mlflow_run_id=self.mlflow.get_run_id() if self.mlflow else None,
        early_stopping_state=es_state,
    )

    storage_type = "S3" if self.is_s3_checkpoint else "로컬"
    console.print(
        f"[green]{storage_type} 최종 모델 저장 완료: {final_path}[/green]"
    )

    # 2. MLflow에 모델 등록 및 artifact 업로드
    if self.mlflow:
        try:
            # 2-1. PyTorch 모델 등록 (Model Registry)
            model_name = f"wmtp_{self.algorithm}"
            self.mlflow.log_model(
                model=self.model,
                name="final_model",
                registered_model_name=model_name,
            )
            console.print(
                f"[cyan]MLflow 모델 등록 완료: {model_name}[/cyan]"
            )

            # 2-2. Checkpoint artifact 업로드 (로컬인 경우만)
            if not self.is_s3_checkpoint:
                self.mlflow.log_artifact(
                    local_path=final_path,
                    artifact_path="checkpoints",
                )
                console.print(
                    "[cyan]MLflow artifact 업로드: checkpoints/final_model.pt[/cyan]"
                )
            else:
                # S3 경로는 참조만 기록
                self.mlflow.log_param("final_checkpoint_s3_path", final_path)
                console.print(
                    f"[cyan]MLflow에 S3 경로 기록: {final_path}[/cyan]"
                )

            # 2-3. 최종 메트릭 기록
            self.mlflow.log_metrics(
                {
                    "final/epoch": epoch,
                    "final/step": step,
                    **{f"final/{k}": v for k, v in metrics.items()},
                }
            )
        except Exception as e:
            console.print(
                f"[yellow]MLflow 등록 실패 (체크포인트는 저장됨): {e}[/yellow]"
            )

    return final_path
```

**변경 요약**:
1. **Docstring 간소화**: "Phase 3" 제거, 역할 명시
2. **mlflow_manager=None 전달**: distribute_manager에서 중복 업로드 방지
3. **명시적 MLflow 처리**: log_model, log_artifact, log_metrics 직접 호출
4. **S3 경로 처리**: log_param으로 경로만 기록 (중복 방지)
5. **Early stopping 상태**: 누락되었던 es_state 추가
6. **주석 정리**: 단계별 설명 (1. 저장, 2. MLflow)
7. **에러 핸들링**: MLflow 실패해도 checkpoint 저장 보장

### 2.2. 검증 방법

#### 2.2.1. 코드 리뷰
```bash
# 변경 사항 확인
git diff src/components/trainer/base_wmtp_trainer.py

# 기대:
# - Docstring 간소화
# - mlflow_manager=None 전달
# - 명시적 MLflow 처리 추가
# - early_stopping_state 추가
```

#### 2.2.2. 전체 훈련 (최종 모델까지)
```bash
# 전체 훈련 실행 (30 step, save_final=true)
export PYTHONPATH=. && uv run python -m src.cli.train \
    --config tests/configs/config.local_test.yaml \
    --recipe tests/configs/recipe.mtp_baseline.yaml \
    --run-name test_phase2_final \
    --tags test,phase2 \
    --verbose 2>&1 | tee /tmp/phase2_final.log
```

**기대 결과**:
- ✅ 훈련 정상 완료
- ✅ 로컬 checkpoint 확인:
  ```bash
  ls -lh ./test_checkpoints/*/
  # checkpoint_step_1.pt (중간)
  # final_model.pt (최종)
  ```
- ✅ MLflow 모델 등록 확인:
  ```bash
  mlflow ui --backend-store-uri file:///tmp/mlflow_m3
  # Models 탭에 "wmtp_baseline-mtp" 존재
  ```
- ✅ MLflow artifacts 확인:
  ```bash
  ls -lh /tmp/mlflow_m3/{run_id}/artifacts/checkpoints/
  # final_model.pt만 존재 (checkpoint_step_*.pt 없음)
  ```

### 2.3. 완료 기준

- [ ] Docstring 업데이트 완료 ("Phase 3" 제거)
- [ ] mlflow_manager=None 전달 (중복 방지)
- [ ] 명시적 MLflow 처리 구현 (log_model, log_artifact, log_metrics)
- [ ] Early stopping 상태 저장 추가
- [ ] 전체 훈련 테스트 통과
- [ ] MLflow Model Registry에 모델 등록 확인
- [ ] MLflow artifacts에 final_model.pt만 존재 확인
- [ ] 중간 checkpoint는 MLflow에 없음 확인

### 2.4. 원칙 5: 계획 대비 검증 및 보고

**계획 목표**:
```
✅ _save_final_checkpoint 수정
✅ mlflow_manager=None 전달
✅ 최종 모델만 MLflow 등록
```

**실제 달성 (보고 형식)**:
```
[Phase 2 완료 보고]

변경 파일: src/components/trainer/base_wmtp_trainer.py
변경 라인: _save_final_checkpoint 메서드 전체 (Line 151-218)

계획 대비:
✅ Docstring 간소화 (역할 명시, Phase 번호 제거)
✅ mlflow_manager=None 전달 (중복 방지)
✅ 명시적 MLflow 처리 (log_model, log_artifact, log_metrics)
✅ S3 경로 처리 (log_param으로 참조 기록)

검증 결과:
✅ 전체 훈련 (30 step) 정상 완료
✅ 로컬 checkpoint 정상 저장 (checkpoint_step_*.pt, final_model.pt)
✅ MLflow Model Registry 등록 확인 (wmtp_baseline-mtp)
✅ MLflow artifacts에 final_model.pt만 존재
✅ 중간 checkpoint MLflow 업로드 0회

예상 효과 달성:
✅ 최종 모델만 MLflow 등록 (1회)
✅ 중복 저장 완전 제거 (86% 절감)

번외 발견사항:
⚠️ Early stopping 상태가 누락되어 있었음 → 추가 완료
```

---

## 📝 Phase 3: 문서화 및 주석 정리

**목표**: 역할 분리 문서화 및 불필요한 주석 제거 (원칙 4-3)

### 원칙 적용 체크리스트

- [ ] **원칙 4-3**: Phase 번호 주석 완전 제거
- [ ] **원칙 4-3**: 코드 동작 핵심 설명만 유지
- [ ] **원칙 5**: 문서 품질 검증

### 3.1. 아키텍처 문서 업데이트

**파일**: `docs/WMTP_시스템_아키텍처.md`

**추가 섹션**:
```markdown
### Checkpoint 관리 시스템

WMTP는 역할 기반 이중 저장 시스템을 사용합니다:

#### 1. Training Checkpoints (paths.checkpoints)
**목적**: 훈련 재개 (Resume Training)

- 저장 대상: 주기적 checkpoint (save_interval마다)
- 관리 정책: keep_last (오래된 자동 삭제)
- 저장 위치: Config 설정 (S3 또는 로컬)
- 접근 방법: 파일 시스템 직접
- 생명주기: Ephemeral (훈련 완료 후 삭제 가능)

#### 2. MLflow Artifacts
**목적**: 실험 추적 및 모델 배포

- 저장 대상: 최종 모델만 (save_final=true)
- 관리 정책: 영구 보존 (버전 관리)
- 저장 위치: MLflow tracking_uri/artifacts
- 접근 방법: MLflow API/UI
- 생명주기: Persistent (영구 보존)

#### 설계 철학
- Separation of Concerns: 훈련 재개 vs 실험 추적 역할 분리
- Cost Optimization: 중간 checkpoint MLflow 업로드 제거 (86% 절감)
- Flexibility: 각 시스템 독립적 설정 가능
```

### 3.2. Config 주석 개선

**파일**: `tests/configs/config.local_test.yaml`

**Before**:
```yaml
paths:
  checkpoints:
    base_path: "file://./test_checkpoints"
    save_interval: 100
    keep_last: 1
    save_final: true

mlflow:
  experiment: "wmtp/m3_test_critic"
  tracking_uri: "file:///tmp/mlflow_m3"
  registry_uri: "file:///tmp/mlflow_m3"
```

**After**:
```yaml
paths:
  checkpoints:
    base_path: "file://./test_checkpoints"
    save_interval: 100    # 테스트: 자주 저장하여 재개 로직 검증
    keep_last: 1          # 테스트: 디스크 공간 절약
    save_final: true      # 최종 모델은 MLflow에도 등록

mlflow:
  experiment: "wmtp/m3_test_critic"
  tracking_uri: "file:///tmp/mlflow_m3"
  registry_uri: "file:///tmp/mlflow_m3"
  # Note: 중간 checkpoint는 MLflow에 업로드되지 않음 (final_model만)
```

### 3.3. 완료 기준

- [ ] 아키텍처 문서 업데이트 완료
- [ ] Config 주석 개선 완료
- [ ] 문서 링크 확인 (checkpoint, mlflow 검색)
- [ ] 주석 일관성 확인 (configs/*.yaml, tests/configs/*.yaml)

---

## 📝 Phase 4: 최종 검증 및 성과 보고

**목표**: 통합 테스트 및 계획 대비 100% 달성 검증 (원칙 5)

### 원칙 적용 체크리스트

- [ ] **원칙 5**: 계획서 대비 전체 검증
- [ ] **원칙 5**: 성과 객관적 기술
- [ ] **원칙 5-1**: 번외 발견사항 보고
- [ ] **원칙 6**: uv 환경 사용 확인

### 4.1. 통합 테스트

#### 테스트 시나리오 1: 로컬 훈련 전체 흐름
```bash
# 1. 전체 훈련 실행
export PYTHONPATH=. && uv run python -m src.cli.train \
    --config tests/configs/config.local_test.yaml \
    --recipe tests/configs/recipe.critic_wmtp.yaml \
    --run-name integration_test_final \
    --tags test,integration,final \
    --verbose 2>&1 | tee /tmp/integration_test.log

# 2. 저장 확인
tree ./test_checkpoints/
# 기대:
# test_checkpoints/
#   └── {run_id}/
#       ├── checkpoint_step_1.pt
#       ├── checkpoint_step_2.pt
#       └── final_model.pt

tree /tmp/mlflow_m3/{run_id}/artifacts/
# 기대:
# artifacts/
#   └── checkpoints/
#       └── final_model.pt  (이것만!)

# 3. 훈련 재개 테스트
export PYTHONPATH=. && uv run python -m src.cli.train \
    --config tests/configs/config.local_test.yaml \
    --recipe tests/configs/recipe.critic_wmtp.yaml \
    --resume-checkpoint ./test_checkpoints/{run_id}/checkpoint_step_5.pt \
    --run-name integration_test_resume \
    --verbose
```

**기대 결과**:
- ✅ 전체 훈련 정상 완료
- ✅ 로컬에 모든 checkpoint 존재
- ✅ MLflow에 final_model.pt만 존재
- ✅ 훈련 재개 정상 동작

#### 테스트 시나리오 2: S3 Dry-run
```bash
# S3 경로 설정 검증
export PYTHONPATH=. && uv run python -m src.cli.train \
    --config configs/config.vessl.yaml \
    --recipe configs/recipe.critic_wmtp.yaml \
    --run-name integration_test_s3 \
    --tags test,s3,dryrun \
    --dry-run \
    --verbose
```

**기대 결과**:
- ✅ Dry-run 통과
- ✅ S3 경로 해석 정상

### 4.2. 성능 벤치마크

**저장 시간 측정**:
```bash
# Before vs After 비교
grep "Checkpoint saved" /tmp/integration_test.log

# 기대:
# - "Checkpoint saved locally" 출력만 (MLflow 업로드 메시지 없음)
# - 저장 시간 단축 (MLflow 업로드 제거)
```

**저장량 측정**:
```bash
# 로컬 checkpoint 크기
du -sh ./test_checkpoints/*/

# MLflow artifacts 크기
du -sh /tmp/mlflow_m3/{run_id}/artifacts/checkpoints/

# 기대:
# - MLflow는 final_model.pt만 (1개)
# - 로컬은 모든 checkpoint (N개)
```

### 4.3. 회귀 테스트 체크리스트

- [ ] **기존 기능 보존**
  - [ ] 훈련 정상 완료
  - [ ] Checkpoint 저장 정상
  - [ ] 훈련 재개 정상
  - [ ] Early stopping 동작
  - [ ] FSDP 분산 훈련 정상

- [ ] **MLflow 통합**
  - [ ] 최종 모델 등록 확인
  - [ ] Metrics 로깅 정상
  - [ ] Artifacts 업로드 정상 (final만)
  - [ ] Model Registry 정상

- [ ] **저장 최적화**
  - [ ] 중간 checkpoint MLflow 업로드 없음
  - [ ] 저장 시간 단축 확인
  - [ ] 저장량 감소 확인

### 4.4. Rollback 전략

**Rollback 트리거**:
- 훈련 재개 실패
- MLflow 모델 등록 실패
- 성능 저하 (저장 시간 증가)

**Rollback 절차**:
```bash
# Git revert
git revert HEAD~2  # Phase 1, 2 모두 revert

# 또는 수동 복원
# distribute_manager.py Line 412-416 복원
# base_wmtp_trainer.py mlflow_manager 전달 복원
```

### 4.5. 원칙 5: 최종 성과 보고

**계획 목표**:
```
✅ 중복 저장 제거
✅ 저장 비용 86% 절감
✅ 역할 명확화
✅ 설정 간소화
```

**실제 달성 (최종 보고)**:
```
[최종 성과 보고]

변경 파일: 2개
- src/utils/distribute_manager.py (Line 320-417)
- src/components/trainer/base_wmtp_trainer.py (Line 151-218)

변경 라인: 약 30줄
- 삭제: 15줄 (MLflow 자동 업로드 로직)
- 추가: 15줄 (명시적 MLflow 처리)

계획 대비 달성:
✅ Phase 0: 사전 분석 완료 (코드 흐름 파악)
✅ Phase 1: distribute_manager.py 중복 업로드 제거
✅ Phase 2: base_wmtp_trainer.py 최종 모델 MLflow 등록
✅ Phase 3: 문서화 및 주석 정리
✅ Phase 4: 통합 테스트 통과

검증 결과:
✅ 로컬 훈련 전체 흐름 정상
✅ 훈련 재개 정상 동작
✅ MLflow 모델 등록 정상
✅ 중간 checkpoint MLflow 업로드 0회
✅ 최종 모델만 MLflow 등록 (1회)

정량적 효과:
✅ 저장 비용 86% 절감 (420GB → 0GB, MLflow 중간 checkpoint)
✅ 저장 시간 단축 (MLflow 업로드 제거)
✅ 네트워크 비용 50% 절감 (업로드 대역폭)

정성적 효과:
✅ 역할 명확화 (훈련 재개 vs 실험 추적)
✅ 설정 간소화 (테스트 환경)
✅ 코드 가독성 향상 (명시적 처리)

번외 발견사항:
⚠️ Early stopping 상태가 _save_final_checkpoint에 누락
   → Phase 2에서 추가 완료 (early_stopping_state 파라미터)
✅ S3 경로 처리 개선
   → log_param으로 참조만 기록 (중복 방지)
```

**개발 원칙 준수 평가**:
```
원칙 1 (앞/뒤 흐름 확인): ✅ Phase 0에서 전체 흐름 분석 완료
원칙 2 (기존 구조 존중): ✅ 저장 로직 유지, MLflow 부분만 수정
원칙 3 (삭제 vs 수정 검토): ✅ 승인된 Option 3 적용
원칙 4 (깨끗한 코드): ✅ Phase 번호 제거, 통일성 유지
원칙 5 (계획 대비 검증): ✅ 각 Phase마다 객관적 보고
원칙 6 (패키지 의존성): ✅ uv run 활용, 의존성 변경 없음
```

---

## ✅ 완료 기준

### Phase별 완료 조건

- [ ] **Phase 0**: 사전 분석 완료 ✅
- [ ] **Phase 1**: distribute_manager.py 수정 완료
  - [ ] Docstring 업데이트
  - [ ] S3/로컬 MLflow 업로드 제거
  - [ ] Dry-run 테스트 통과
  - [ ] 실제 훈련 테스트 통과

- [ ] **Phase 2**: base_wmtp_trainer.py 수정 완료
  - [ ] _save_final_checkpoint 수정
  - [ ] mlflow_manager=None 전달
  - [ ] 명시적 MLflow 처리 구현
  - [ ] 전체 훈련 테스트 통과
  - [ ] MLflow 모델 등록 확인

- [ ] **Phase 3**: 문서화 완료
  - [ ] 아키텍처 문서 업데이트
  - [ ] Config 주석 개선
  - [ ] 문서 품질 검증

- [ ] **Phase 4**: 최종 검증 완료
  - [ ] 통합 테스트 통과
  - [ ] 회귀 테스트 통과
  - [ ] 성과 보고 작성

### 전체 완료 조건

1. ✅ 모든 Phase 완료
2. ✅ 중복 저장 완전 제거
3. ✅ 역할 분리 명확화
4. ✅ 회귀 테스트 통과
5. ✅ 계획서 대비 100% 달성

---

## 📚 참고 자료

### 내부 문서
- `docs/WMTP_시스템_아키텍처.md` - 전체 아키텍처
- `docs/checkpoint_mlflow_integration_analysis.md` - 상세 분석 (참고용)

### 코드 위치
- `src/components/trainer/base_wmtp_trainer.py` (Line 151-218, 559-602)
- `src/utils/distribute_manager.py` (Line 310-417)
- `src/utils/mlflow.py` (Line 234-287)

---

**작성자**: Claude Code (개발 원칙 기반)
**승인 대기**: Phase 1-4 구현 전 사용자 최종 승인 필요
**최종 수정일**: 2025-10-02
