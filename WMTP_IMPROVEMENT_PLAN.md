# WMTP 시스템 개선 계획서
## 2025년 1월 - Phase별 상세 실행 계획

---

## 📋 Executive Summary

### 발견된 핵심 이슈
1. **[Critical]** 체크포인트 저장 실패 - KeyError(None)
2. **[High]** MPS/FSDP 호환성 문제
3. **[Medium]** API deprecation 경고
4. **[Low]** 리소스 누수 (세마포어 15개)

### 개선 목표
- **즉시 목표**: 핵심 기능 복구 (체크포인트 저장)
- **단기 목표**: MPS 환경 최적화 및 경고 제거
- **중기 목표**: 기술 부채 해결 및 현대적 API 도입

---

## 🔍 현황 분석

### 1. 체크포인트 저장 실패 분석

#### 코드 흐름
```
BaseWmtpTrainer._save_checkpoint()
    ↓
DistributedManager.save_checkpoint(model, ...)
    ↓
FSDP.state_dict_type(model, ...)  # ← 여기서 실패
```

#### 근본 원인
- `save_checkpoint()` 메서드가 FSDP 래핑된 모델만 처리 가능
- 테스트 환경: `fsdp.enabled: false` → 모델이 일반 `nn.Module`
- `isinstance(model, FSDP)` 체크 없이 무조건 FSDP 로직 실행
- 결과: `KeyError(None)` 발생

#### 영향 범위
- 모든 non-FSDP 환경에서 체크포인트 저장 불가
- 학습 중단 시 복구 불가능
- MLflow 아티팩트 업로드 실패

### 2. MPS 호환성 문제 분석

#### 발생 경고들
```python
# 1. CPU autocast 경고
"In CPU autocast, but the target dtype is not supported"

# 2. FSDP API deprecation
"FSDP.state_dict_type() is being deprecated"

# 3. 리소스 누수
"resource_tracker: 15 leaked semaphore objects"
```

#### 원인 분석
1. **Autocast 문제**: MPS는 fp32만 지원, bf16/fp16 미지원
2. **FSDP API**: 구버전 API 사용 중
3. **리소스 누수**: DataLoader `num_workers=4`가 MPS에서 문제 유발

---

## 📈 Phase별 개선 계획

## Phase 1: Critical Fix - 체크포인트 저장 수정
**목표**: FSDP/non-FSDP 모델 모두에서 체크포인트 저장 가능
**일정**: 즉시 실행
**위험도**: 낮음 (조건 분기만 추가)

### 1.1 구현 내용

#### src/utils/dist.py 수정
```python
def save_checkpoint(
    self,
    model: Union[FSDP, nn.Module],  # 타입 힌트 수정
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
    epoch: int,
    step: int,
    mlflow_manager=None,
    **kwargs,
) -> None:
    """체크포인트 저장 (FSDP/non-FSDP 모델 모두 지원)"""

    if self.is_main_process():
        # 모델 타입에 따른 분기 처리
        if isinstance(model, FSDP):
            # 기존 FSDP 로직 유지
            save_policy = FullStateDictConfig(
                offload_to_cpu=True,
                rank0_only=True,
            )
            with FSDP.state_dict_type(
                model, StateDictType.FULL_STATE_DICT, save_policy
            ):
                state_dict = model.state_dict()
        else:
            # 일반 모델 처리 (신규 추가)
            state_dict = model.state_dict()
            # CPU로 이동 (메모리 효율성)
            if hasattr(model, 'device') and model.device.type != 'cpu':
                state_dict = {k: v.cpu() for k, v in state_dict.items()}

        # 공통 체크포인트 구성 (기존 로직 유지)
        checkpoint = {
            "model": state_dict,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            **kwargs,
        }

        # 저장 로직 (기존 유지)
        self._save_checkpoint_to_storage(checkpoint, checkpoint_path, mlflow_manager)
```

### 1.2 테스트 계획
- [ ] FSDP enabled 환경 테스트
- [ ] FSDP disabled 환경 테스트
- [ ] MPS 환경 테스트
- [ ] 체크포인트 로드 검증
- [ ] MLflow 업로드 검증

### 1.3 롤백 계획
- Git 커밋 단위 관리
- 실패 시 이전 커밋으로 즉시 롤백

---

## Phase 2: MPS Optimization - Apple Silicon 최적화
**목표**: MPS 환경 최적화 및 모든 경고 제거
**일정**: Phase 1 완료 후 1주일 내
**위험도**: 중간 (설정 변경 영향)

### 2.1 구현 내용

#### 2.1.1 Autocast 조건부 처리
```python
# src/components/trainer/base_wmtp_trainer.py
def train(self, ...):
    # MPS 감지
    device_type = str(self.model.device.type) if hasattr(self.model, 'device') else 'cpu'
    use_autocast = device_type in ['cuda'] and self.mixed_precision != 'fp32'

    # 조건부 autocast
    if use_autocast:
        with torch.autocast(device_type=device_type, dtype=self.get_autocast_dtype()):
            loss = self._compute_loss(batch)
    else:
        loss = self._compute_loss(batch)  # MPS는 autocast 스킵
```

#### 2.1.2 DataLoader 자동 최적화
```python
# src/factory/component_factory.py
def create_data_loader(self, ...):
    # MPS 환경 감지
    is_mps = (
        hasattr(torch.backends, 'mps') and
        torch.backends.mps.is_available() and
        self.config.devices.compute_backend == 'mps'
    )

    # num_workers 자동 조정
    if is_mps and num_workers > 0:
        console.print(
            f"[yellow]MPS 환경 감지: num_workers를 {num_workers} → 0으로 자동 조정[/yellow]"
        )
        num_workers = 0

    # DataLoader 생성 (기존 로직 유지)
    return DataLoader(..., num_workers=num_workers)
```

#### 2.1.3 MPS 프로파일 생성
```yaml
# configs/profiles/mps_optimized.yaml
devices:
  compute_backend: "mps"
  mixed_precision: "fp32"  # MPS는 fp32만 지원
  fsdp:
    enabled: false  # MPS에서 FSDP 비활성화

data:
  train:
    num_workers: 0  # 멀티프로세싱 비활성화
  eval:
    num_workers: 0

optim:
  grad_accumulation_steps: 4  # 메모리 효율성

# MPS 전용 최적화 플래그
mps_optimizations:
  use_graph_mode: false  # 안정성 우선
  fallback_to_cpu: true  # 지원 안되는 연산은 CPU로
```

### 2.2 테스트 계획
- [ ] MPS autocast 스킵 검증
- [ ] DataLoader 세마포어 누수 해결 확인
- [ ] 성능 벤치마크 (vs CPU)
- [ ] 메모리 사용량 모니터링

---

## Phase 3: API Modernization - 기술 부채 해결
**목표**: Deprecated API 제거 및 현대적 패턴 도입
**일정**: Phase 2 완료 후 2주 내
**위험도**: 높음 (API 마이그레이션)

### 3.1 구현 내용

#### 3.1.1 FSDP API 현대화
```python
# src/utils/dist.py - 새로운 API 도입
def save_checkpoint_v2(self, model, optimizer, ...):
    """새로운 FSDP API 사용"""
    from torch.distributed.checkpoint import (
        get_state_dict,
        StateDictOptions,
    )

    if isinstance(model, FSDP):
        # 새로운 API 사용
        options = StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
        )
        state_dict = get_state_dict(model, options)
    else:
        state_dict = model.state_dict()

    # ... 저장 로직
```

#### 3.1.2 MLflow 키 관리 강화
```python
# src/utils/mlflow.py
class MLflowManager:
    def get_run_id(self) -> Optional[str]:
        """안전한 run ID 반환"""
        if self.run is None:
            return None
        try:
            return self.run.info.run_id
        except AttributeError:
            console.print("[yellow]MLflow run ID 조회 실패[/yellow]")
            return None

    def log_metrics_safe(self, metrics: dict, step: int):
        """실패 시 graceful degradation"""
        try:
            if self.run is not None:
                mlflow.log_metrics(metrics, step)
        except Exception as e:
            console.print(f"[yellow]MLflow 메트릭 로깅 실패 (계속 진행): {e}[/yellow]")
```

#### 3.1.3 리소스 관리 강화
```python
# src/components/trainer/base_wmtp_trainer.py
class BaseWmtpTrainer:
    def __enter__(self):
        """컨텍스트 매니저 진입"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """리소스 정리"""
        # DataLoader 정리
        if hasattr(self, 'train_loader'):
            if hasattr(self.train_loader, '_iterator'):
                del self.train_loader._iterator

        # 세마포어 명시적 해제
        import multiprocessing
        multiprocessing.resource_tracker.ensure_running()

        # GPU 캐시 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, 'mps'):
            # MPS 캐시 정리 (있을 경우)
            pass

# 사용 예시
with BaseWmtpTrainer(...) as trainer:
    trainer.train()
```

### 3.2 마이그레이션 전략
1. **Phase 3.1**: 새 API와 구 API 병행 지원
2. **Phase 3.2**: Deprecation 경고 추가
3. **Phase 4**: 구 API 완전 제거 (3개월 후)

### 3.3 테스트 계획
- [ ] 새 FSDP API 동작 검증
- [ ] MLflow 실패 시 학습 계속 진행 확인
- [ ] 리소스 누수 완전 해결 검증
- [ ] 성능 regression 테스트

---

## 🎯 성공 지표

### 정량적 지표
- ✅ 체크포인트 저장 성공률: 100%
- ✅ MPS 경고 수: 0
- ✅ 리소스 누수: 0
- ✅ 테스트 통과율: 100%

### 정성적 지표
- 코드 가독성 향상
- 유지보수성 개선
- 개발자 경험 향상

---

## ⚠️ 위험 관리

### Phase 1 위험
- **위험**: 최소 (조건 분기만 추가)
- **완화**: 충분한 테스트 커버리지

### Phase 2 위험
- **위험**: 설정 변경으로 인한 성능 영향
- **완화**: 프로파일 기반 점진적 적용

### Phase 3 위험
- **위험**: API 마이그레이션 중 호환성 문제
- **완화**: 병행 지원 기간 제공

---

## 📅 타임라인

| Phase | 작업 내용 | 예상 기간 | 우선순위 |
|-------|-----------|-----------|----------|
| **Phase 1** | 체크포인트 저장 수정 | 1일 | **Critical** |
| **Phase 2** | MPS 최적화 | 1주일 | **High** |
| **Phase 3** | API 현대화 | 2주일 | **Medium** |
| **Phase 4** | 구 API 제거 | 3개월 후 | **Low** |

---

## 🔄 개발 원칙 준수 체크리스트

- [x] **[필수1]** 현재 구조 완전 분석 완료
- [x] **[필수2]** 기존 구조 최대한 유지, 중복 제거
- [x] **[필수3]** 기존 코드 유지가 적절 (조건 분기만 추가)
- [x] **[필수4]** Phase 4에서 구버전 완전 제거 계획
- [x] **[필수5]** 계획서 작성 완료, 객관적 기술
- [x] **[필수6]** uv 기반 패키지 의존성 활용

---

## 📝 다음 단계

1. **즉시**: Phase 1 구현 시작 (체크포인트 저장 수정)
2. **Phase 1 완료 후**: 테스트 및 검증
3. **검증 완료 후**: Phase 2 진행 결정

**작성일**: 2025-01-27
**작성자**: WMTP 개발팀
**버전**: 1.0