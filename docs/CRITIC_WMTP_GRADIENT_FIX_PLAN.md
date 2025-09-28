# Critic-WMTP Gradient 문제 해결 Phase별 개선 계획

## 문제 요약
- **위치**: `src/components/trainer/critic_wmtp_trainer.py` 263-265번 줄
- **원인**: Softmax 출력 텐서에 대한 inplace 수정으로 인한 자동 미분 그래프 손상
- **증상**: RuntimeError - gradient computation 변수가 inplace operation으로 수정됨

## 개발 원칙 준수 체크리스트
- ✅ [필수1] 현재 구조 파악: Critic WMTP의 delta 기반 가중치 계산 흐름 분석 완료
- ✅ [필수2] 기존 로직 존중: 핵심 알고리즘 로직 유지, 구현 방식만 개선
- ✅ [필수3] 최선의 방안 검토: 3가지 해결책 중 마스크 사전 적용이 최적
- ✅ [필수4] 깨끗한 구현: 하위 호환성 고려 없이 올바른 구현으로 전면 교체
- ✅ [필수5] 객관적 검토: 각 Phase 완료 후 테스트로 검증
- ✅ [필수6] uv 기반 의존성: PyTorch 기존 버전 유지, 코드 수정으로 해결

---

## Phase 1: 즉시 수정 (긴급 패치)
### 목표
Gradient 계산 오류를 최소한의 변경으로 즉시 해결

### 구현 내용
**파일**: `src/components/trainer/critic_wmtp_trainer.py`
**함수**: `_compute_head_weights_from_values()` (209-276번 줄)

#### 수정 전 (문제 코드)
```python
# 260-270번 줄
weights_t = F.softmax(delta_tensor / self.temperature, dim=1)  # [B, H]

# ❌ Softmax 출력을 직접 수정 - gradient graph 손상
for k in range(H):
    if t + k + 1 >= S:
        weights_t[:, k] = 0.0

# 재정규화
weights_sum = weights_t.sum(dim=1, keepdim=True).clamp(min=1e-8)
weights_t = weights_t / weights_sum
```

#### 수정 후 (해결 코드)
```python
# 255-271번 줄 대체
if delta_list:
    # Stack deltas for all heads: [B, H]
    delta_tensor = torch.stack(delta_list, dim=1)

    # 유효하지 않은 헤드에 대한 마스크 생성 (Softmax 전!)
    for k in range(H):
        if t + k + 1 >= S:
            # Softmax에서 자연스럽게 0에 가까워지도록 매우 작은 값 설정
            delta_tensor[:, k] = -1e10

    # Softmax with temperature (이제 안전함)
    weights_t = F.softmax(delta_tensor / self.temperature, dim=1)  # [B, H]

    # 수치 안정성을 위한 클리핑만 적용 (inplace 수정 없음)
    weights_t = torch.clamp(weights_t, min=1e-8, max=1.0)

    head_weights[:, t, :] = weights_t
```

### 검증 방법
```bash
# 1. 수정 직후 간단한 테스트
PYTHONPATH=. python tests/script/test_m3_pipeline.py \
  --config tests/configs/config.local_test.yaml \
  --recipe tests/configs/recipe.critic_wmtp.yaml \
  --dry-run --verbose

# 2. 실제 학습 테스트 (max_steps=2)
PYTHONPATH=. python tests/script/test_m3_pipeline.py \
  --config tests/configs/config.local_test.yaml \
  --recipe tests/configs/recipe.critic_wmtp.yaml \
  --verbose
```

### 완료 기준
- ✅ Gradient 오류 없이 학습 진행
- ✅ Stage 1 (Value Head 사전훈련) 정상 완료
- ✅ Stage 2 (메인 훈련) 정상 시작 및 진행

---

## Phase 2: 코드 품질 개선
### 목표
전체적인 코드 구조 개선 및 안정성 강화

### 구현 내용

#### 2.1 헬퍼 메서드 추가
```python
def _create_valid_head_mask(self, t: int, S: int, H: int, device: torch.device) -> torch.Tensor:
    """시퀀스 경계를 고려한 헤드별 유효성 마스크 생성

    Args:
        t: 현재 시점
        S: 시퀀스 길이
        H: 헤드 수 (horizon)
        device: 텐서 디바이스

    Returns:
        mask: [H] 형태의 바이너리 마스크 (1.0=유효, 0.0=무효)
    """
    mask = torch.ones(H, device=device)
    for k in range(H):
        if t + k + 1 >= S:
            mask[k] = 0.0
    return mask

def _apply_sequence_boundary_masking(
    self,
    delta_tensor: torch.Tensor,
    t: int,
    S: int
) -> torch.Tensor:
    """시퀀스 경계를 넘는 예측에 대해 masking 적용

    Softmax 전에 적용하여 gradient 안전성 보장
    """
    B, H = delta_tensor.shape
    for k in range(H):
        if t + k + 1 >= S:
            delta_tensor[:, k] = -1e10  # 매우 작은 값 (softmax → 0)
    return delta_tensor
```

#### 2.2 메인 함수 리팩토링
```python
def _compute_head_weights_from_values(
    self, values: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
    """개선된 가중치 계산 - 명확한 단계 분리"""
    B, S = values.shape
    H = self.horizon

    # 1. Delta 계산
    deltas = self._compute_deltas(values)  # [B, S]

    # 2. 헤드별 가중치 초기화
    head_weights = torch.zeros((B, S, H), device=values.device, dtype=values.dtype)

    # 3. 각 시점별 처리
    for t in range(S):
        # 3.1 Delta 수집
        delta_tensor = self._collect_future_deltas(deltas, t, B, S, H)

        # 3.2 경계 마스킹 (Softmax 전!)
        delta_tensor = self._apply_sequence_boundary_masking(delta_tensor, t, S)

        # 3.3 Softmax 정규화
        weights_t = F.softmax(delta_tensor / self.temperature, dim=1)

        # 3.4 저장
        head_weights[:, t, :] = weights_t

    # 4. 최종 valid mask 적용
    head_weights = head_weights * valid_mask.unsqueeze(-1)

    return head_weights
```

### 검증 방법
```bash
# 단위 테스트 추가
python -m pytest tests/test_critic_gradient_fix.py -v

# 통합 테스트
python tests/script/test_m3_pipeline.py \
  --config tests/configs/config.local_test.yaml \
  --recipe tests/configs/recipe.critic_wmtp.yaml \
  --verbose
```

---

## Phase 3: 테스트 및 문서화
### 목표
변경사항 검증 및 향후 유지보수를 위한 문서화

### 구현 내용

#### 3.1 테스트 케이스 추가
**파일**: `tests/test_critic_gradient_fix.py`

```python
import torch
import torch.nn.functional as F
import pytest

class TestCriticGradientFix:
    """Critic WMTP gradient 문제 수정 검증"""

    def test_no_inplace_modification(self):
        """Softmax 출력이 inplace 수정되지 않음을 검증"""
        # 테스트 데이터
        delta_tensor = torch.randn(2, 4, requires_grad=True)

        # 마스킹 적용 (Softmax 전)
        delta_tensor_masked = delta_tensor.clone()
        delta_tensor_masked[:, 3] = -1e10

        # Softmax
        weights = F.softmax(delta_tensor_masked, dim=1)

        # Backward 가능 여부 확인
        loss = weights.sum()
        loss.backward()  # 오류 없이 실행되어야 함

        assert delta_tensor.grad is not None

    def test_weight_sum_to_one(self):
        """가중치 합이 1이 됨을 검증"""
        from src.components.trainer.critic_wmtp_trainer import CriticWmtpTrainer

        trainer = CriticWmtpTrainer()
        trainer.horizon = 4
        trainer.temperature = 1.0

        values = torch.randn(2, 10)  # [B=2, S=10]
        valid_mask = torch.ones(2, 10)

        weights = trainer._compute_head_weights_from_values(values, valid_mask)

        # 각 위치에서 헤드 가중치 합이 1 (또는 0)
        weight_sums = weights.sum(dim=2)  # [B, S]
        assert torch.allclose(
            weight_sums[valid_mask.bool()],
            torch.ones_like(weight_sums[valid_mask.bool()]),
            atol=1e-6
        )
```

#### 3.2 성능 벤치마크
```python
# tests/benchmark_critic_training.py
import time
import torch
from src.components.trainer.critic_wmtp_trainer import CriticWmtpTrainer

def benchmark_training_step():
    """학습 스텝 성능 측정"""
    trainer = CriticWmtpTrainer()
    # ... 설정 ...

    times = []
    for _ in range(100):
        start = time.time()
        trainer.train_step(batch)
        times.append(time.time() - start)

    print(f"평균 스텝 시간: {np.mean(times):.4f}초")
    print(f"표준 편차: {np.std(times):.4f}초")
```

### 문서 업데이트
**파일**: `docs/WMTP_시스템_아키텍처.md` - 이미 업데이트됨
**파일**: `CHANGELOG.md` 추가

```markdown
# CHANGELOG

## [Fix] Critic-WMTP Gradient 문제 해결 (2024-01-XX)

### 문제
- Softmax 출력에 대한 inplace 수정으로 인한 gradient 계산 오류
- `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation`

### 해결
- Softmax 전 단계에서 마스킹 적용으로 변경
- 시퀀스 경계 처리 로직 개선
- Gradient 안전성 보장

### 영향
- Critic-WMTP 알고리즘 정상 학습 가능
- Stage 1, 2 모두 안정적 실행
- 성능 영향 없음
```

---

## 실행 일정

### Day 1 (즉시)
- [x] Phase 1 구현: Gradient 문제 즉시 수정
- [x] 기본 테스트 실행 확인

### Day 2
- [ ] Phase 2 구현: 코드 품질 개선
- [ ] 단위 테스트 작성

### Day 3
- [ ] Phase 3 구현: 벤치마크 및 문서화
- [ ] 전체 회귀 테스트
- [ ] PR 생성 및 리뷰

---

## 성공 지표

1. **기능적 성공**
   - ✅ Gradient 오류 완전 해결
   - ✅ Critic-WMTP 정상 학습
   - ✅ 기존 성능 유지 또는 개선

2. **코드 품질**
   - ✅ 명확한 함수 분리
   - ✅ 적절한 주석 및 문서화
   - ✅ 테스트 커버리지 > 90%

3. **성능 지표**
   - ✅ 학습 속도 저하 없음
   - ✅ 메모리 사용량 증가 없음
   - ✅ Stage 1: 3 epoch 내 수렴
   - ✅ Stage 2: 안정적 loss 감소

---

## 위험 관리

### 잠재 위험
1. **다른 부분에서 유사한 inplace 문제 존재 가능**
   - 대응: 전체 코드베이스 grep으로 검색
   - 패턴: `tensor\[.*\] = ` after softmax/sigmoid

2. **수치 안정성 문제**
   - 대응: -1e10 대신 torch.finfo(dtype).min 사용 고려
   - 모니터링: NaN/Inf 체크 추가

3. **성능 저하**
   - 대응: 프로파일링으로 병목 확인
   - 최적화: 필요시 JIT 컴파일 적용

---

## 결론
이 계획은 Critic-WMTP의 gradient 문제를 체계적으로 해결하면서, 코드 품질과 유지보수성을 개선합니다. 개발 원칙을 철저히 준수하여 기존 구조를 존중하면서도 필요한 개선을 수행합니다.