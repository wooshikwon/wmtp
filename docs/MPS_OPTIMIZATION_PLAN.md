# WMTP MPS (Metal Performance Shaders) 최적화 계획서
## Apple Silicon MacBook 지원을 위한 4D 텐서 아키텍처 개선

---

## 1. 배경 및 현황 분석

### 1.1 문제 정의
- **현상**: MacBook M3에서 WMTP 실제 학습 시 무한 블로킹 (Dry-run은 정상)
- **원인**: MPS 백엔드의 4D 텐서 [B, S, H, V] 처리 미성숙
- **영향**: 개발자의 로컬 테스트/디버깅 불가능

### 1.2 현재 아키텍처 구조 (필수1: 구조 파악)

```
┌─────────────────┐
│   modeling.py   │ ← torch.stack(dim=2)로 4D 텐서 생성
└────────┬────────┘
         ↓
┌─────────────────┐
│ base_trainer.py │ ← compute_weighted_mtp_loss()에서 4D→2D flatten
└────────┬────────┘
         ↓
┌─────────────────┐
│ Algorithm       │
│ Trainers        │ ← baseline/critic/rho1이 공통 함수 사용
└─────────────────┘
```

### 1.3 핵심 블로킹 지점 식별

| 파일 | 위치 | 문제 연산 | 심각도 |
|------|------|-----------|--------|
| modeling.py | L94 | `torch.stack(all_logits, dim=2)` | 🔴 Critical |
| base_wmtp_trainer.py | L78-87 | `view(B*S*H, V)` → CE → `view(B,S,H)` | 🔴 Critical |
| 다수 trainer | 여러 곳 | `unsqueeze(-1).expand(-1,-1,H)` | 🟡 Medium |

---

## 2. 개발 원칙 준수 검토

### 2.1 기존 구조 존중 (필수2)
- ✅ `compute_weighted_mtp_loss()`는 모든 알고리즘의 핵심 → 유지
- ✅ 4D 텐서 [B,S,H,V]는 WMTP의 정체성 → 유지
- ✅ 중복 제거: MPS/CUDA 로직을 단일 함수로 통합

### 2.2 삭제/재작성 판단 (필수3-4)
- ❌ 전면 재작성 불필요: 조건부 분기로 해결 가능
- ✅ 하위 호환성 무시: MPS 최적화는 새로운 경로로 추가

### 2.3 의존성 활용 (필수6)
- PyTorch ≥ 2.0 required (torch.compile 지원)
- MPS availability check via `torch.backends.mps.is_available()`

---

## 3. Phase별 수정 계획

### 🎯 Phase 0: 현황 확인 및 테스트 인프라 구축 (2시간)

#### 목표
- MPS 블로킹 재현 가능한 최소 테스트 케이스 작성
- 성능 측정 베이스라인 확립

#### 작업 내용
1. **MPS 테스트 스크립트 생성**: `tests/test_mps_compatibility.py`
```python
# MPS 블로킹 재현 테스트
def test_mps_4d_tensor_operations():
    """MPS에서 4D 텐서 연산 블로킹 테스트"""
    device = torch.device("mps")
    # 4D tensor operations...
    assert execution_time < 5.0  # 5초 이내 완료
```

2. **벤치마크 스크립트**: `benchmarks/mps_vs_cpu.py`
```python
# MPS vs CPU 성능 비교
# 현재 상태 (블로킹) vs 수정 후 측정
```

#### 성공 기준
- [ ] MPS 블로킹 100% 재현
- [ ] CPU 대비 성능 측정 완료

---

### 🔧 Phase 1: 최소 침습 수정 - Contiguous 메모리 보장 (2시간)

#### 목표
- 최소한의 코드 변경으로 MPS 작동 확인
- 기존 CUDA 성능에 영향 최소화

#### 수정 파일 및 내용

**1. modeling.py 수정**
```python
# tests/tiny_models/distilgpt2-mtp/modeling.py L94
# 변경 전:
mtp_logits = torch.stack(all_logits, dim=2)

# 변경 후:
mtp_logits = torch.stack(all_logits, dim=2).contiguous()
```

**2. base_wmtp_trainer.py 수정**
```python
# src/components/trainer/base_wmtp_trainer.py L78
# 변경 전:
logits_flat = logits.view(B * S * H, V)

# 변경 후:
logits_flat = logits.contiguous().view(B * S * H, V)
```

#### 성공 기준
- [ ] MPS에서 1 step 학습 완료
- [ ] CUDA 성능 저하 < 1%

---

### 🚀 Phase 2: 핵심 블로킹 해소 - 조건부 최적화 경로 (4시간)

#### 목표
- MPS 전용 최적화 경로 구현
- 설정 기반 자동 경로 선택

#### 수정 내용

**1. MPS 최적화 유틸리티 생성**: `src/utils/mps_optimizer.py`
```python
"""MPS 백엔드 최적화 유틸리티"""

class MPSOptimizer:
    """MPS 특화 텐서 연산 최적화"""

    @staticmethod
    def is_mps_available() -> bool:
        """MPS 사용 가능 여부 확인"""
        return (
            torch.backends.mps.is_available() and
            torch.backends.mps.is_built()
        )

    @staticmethod
    def optimize_4d_stack(tensors: List[torch.Tensor], dim: int) -> torch.Tensor:
        """MPS 최적화된 4D 스택 연산"""
        if dim == 2 and MPSOptimizer.is_mps_available():
            # MPS 최적화: cat + reshape
            B, S, V = tensors[0].shape
            H = len(tensors)

            # Method 1: Direct assignment (most MPS-friendly)
            result = torch.zeros(B, S, H, V, device=tensors[0].device)
            for i, t in enumerate(tensors):
                result[:, :, i, :] = t
            return result.contiguous()
        else:
            # CUDA/CPU: 기존 방식
            return torch.stack(tensors, dim=dim)
```

**2. compute_weighted_mtp_loss 개선**
```python
# src/components/trainer/base_wmtp_trainer.py

def compute_weighted_mtp_loss(
    logits: torch.Tensor,
    target_labels: torch.Tensor,
    head_weights: torch.Tensor,
    ignore_index: int = -100,
    selection_mask: torch.Tensor | None = None,
    use_mps_optimization: bool | None = None,  # 새 파라미터
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MPS 최적화 옵션이 추가된 WMTP 손실 계산"""

    from src.utils.mps_optimizer import MPSOptimizer

    # MPS 최적화 자동 감지
    if use_mps_optimization is None:
        use_mps_optimization = MPSOptimizer.is_mps_available()

    B, S, H, V = logits.shape

    if use_mps_optimization:
        # MPS 최적화 경로: 헤드별 3D 처리
        ce_list = []
        for h in range(H):
            logits_h = logits[:, :, h, :].contiguous()
            labels_h = target_labels[:, :, h].contiguous()

            ce_h = F.cross_entropy(
                logits_h.transpose(1, 2),
                labels_h,
                ignore_index=ignore_index,
                reduction='none'
            )
            ce_list.append(ce_h)

        ce_per_head = torch.stack(ce_list, dim=2).contiguous()
    else:
        # 기존 CUDA 최적화 경로 (4D flatten)
        logits_flat = logits.view(B * S * H, V)
        target_flat = target_labels.view(B * S * H)

        ce_flat = F.cross_entropy(
            logits_flat, target_flat,
            ignore_index=ignore_index,
            reduction='none'
        )
        ce_per_head = ce_flat.view(B, S, H)

    # 이후 로직은 동일...
    return final_loss, token_valid_mask, ce_per_head
```

**3. modeling.py MPS 최적화**
```python
# tests/tiny_models/distilgpt2-mtp/modeling.py
def forward(self, ...):
    # ...
    from src.utils.mps_optimizer import MPSOptimizer

    # MPS 최적화된 스택
    mtp_logits = MPSOptimizer.optimize_4d_stack(all_logits, dim=2)
    # ...
```

#### 성공 기준
- [ ] MPS에서 전체 학습 완료 (max_steps=100)
- [ ] MPS 성능: CPU 대비 2x 이상
- [ ] CUDA 성능 영향: ±1% 이내

---

### ⚙️ Phase 3: 전체 최적화 및 설정 통합 (4시간)

#### 목표
- 설정 파일 기반 MPS 최적화 토글
- torch.compile() 통합
- 성능 모니터링 추가

#### 수정 내용

**1. 설정 스키마 확장**: `src/settings/config_schema.py`
```python
class DeviceConfig(BaseModel):
    compute_backend: Literal["cuda", "mps", "cpu"] = "cuda"
    mixed_precision: str = "bf16"
    mps_optimization: bool = True  # 새 필드
    use_torch_compile: bool = False  # 새 필드
```

**2. torch.compile 통합** (PyTorch 2.0+)
```python
# src/components/trainer/base_wmtp_trainer.py
if self.config.get("use_torch_compile", False):
    compute_weighted_mtp_loss = torch.compile(
        compute_weighted_mtp_loss,
        backend="inductor",
        mode="reduce-overhead"
    )
```

**3. 성능 모니터링 추가**
```python
# MLflow 메트릭에 MPS 성능 지표 추가
if self.device.type == "mps":
    metrics["device/mps_memory_allocated"] = torch.mps.current_allocated_memory()
    metrics["device/mps_optimization_enabled"] = use_mps_optimization
```

#### 성공 기준
- [ ] 설정 기반 MPS 최적화 on/off 가능
- [ ] torch.compile 적용 시 추가 10% 성능 향상
- [ ] MLflow에서 MPS 메트릭 확인 가능

---

## 4. 검증 계획

### 4.1 단위 테스트
```bash
# MPS 호환성 테스트
pytest tests/test_mps_compatibility.py -v

# 성능 회귀 테스트
pytest tests/test_performance_regression.py -v
```

### 4.2 통합 테스트
```bash
# M3 MacBook 전체 파이프라인 테스트
python tests/script/test_m3_pipeline.py \
    --config tests/configs/config.local_test.yaml \
    --recipe tests/configs/recipe.mtp_baseline.yaml
```

### 4.3 벤치마크
```bash
# MPS vs CPU vs CUDA 성능 비교
python benchmarks/compare_backends.py --report
```

---

## 5. 위험 요소 및 대응

| 위험 | 확률 | 영향 | 대응 방안 |
|------|------|------|-----------|
| CUDA 성능 저하 | 낮음 | 높음 | 조건부 분기로 격리 |
| MPS 버그 | 중간 | 중간 | PyTorch 업데이트 추적 |
| 코드 복잡도 증가 | 높음 | 낮음 | 유틸리티 클래스로 캡슐화 |

---

## 6. 일정 및 마일스톤

| Phase | 예상 시간 | 완료 기준 | 담당자 |
|-------|-----------|-----------|--------|
| Phase 0 | 2시간 | 테스트 인프라 구축 | - |
| Phase 1 | 2시간 | 최소 작동 확인 | - |
| Phase 2 | 4시간 | MPS 최적화 완료 | - |
| Phase 3 | 4시간 | 전체 통합 완료 | - |
| **합계** | **12시간** | **MPS 완전 지원** | - |

---

## 7. 성과 측정 (필수5: 객관적 평가)

### 7.1 정량적 지표
- **MPS 작동률**: 0% → 100% (블로킹 해소)
- **MPS 성능**: CPU 대비 2-3x 향상 목표
- **CUDA 영향**: ±1% 이내 유지
- **코드 증가량**: +200줄 이내

### 7.2 정성적 성과
- MacBook 개발자의 로컬 테스트 가능
- 오픈소스 접근성 향상
- PyTorch MPS 생태계 기여

---

## 8. 결론

본 계획은 WMTP의 4D 텐서 아키텍처를 유지하면서 MPS 호환성을 확보하는 실용적 접근법입니다. 기존 CUDA 성능에 영향 없이 Apple Silicon 지원을 추가하여 개발자 경험을 크게 향상시킬 수 있습니다.

**핵심 성공 요소**:
1. 조건부 최적화로 위험 최소화
2. 단계적 접근으로 안정성 확보
3. 기존 구조 존중으로 유지보수성 유지

---

*작성일: 2024-09-27*
*버전: 1.0*
*상태: 검토 대기*
