# WMTP 코드베이스 리팩토링 계획서

## 🎯 리팩토링 목표

**연구 철학**: "Not All Tokens Are What You Need"를 명확하고 효율적으로 구현

### 핵심 원칙
- **[필수4] 완전 분화**: 하위 호환성 무시, 깨끗한 재구성
- **알고리즘별 격리**: mtp-baseline, critic-wmtp, rho1-wmtp 완전 분리
- **중복 제거**: 공통 로직을 BaseWmtpTrainer로 추상화
- **Rho-1 핵심화**: Reference vs Base 모델 CE 비교로 단순화

---

## 📊 현재 상태 분석

### 기존 구조의 문제점

| 파일 | 크기 | 문제점 | 해결책 |
|------|------|--------|--------|
| `mtp_weighted_ce_trainer.py` | 900줄 | 3개 알고리즘 혼재 | **3개 독립 클래스로 분화** |
| `rho1_excess.py` | 480줄 | Mock 로직, 과도한 복잡성 | **핵심 로직만 남기고 전격 삭제** |

### 레지스트리 현황
```python
@trainer_registry.register("mtp-baseline", ...)    # 1개 클래스가
@trainer_registry.register("critic-wmtp", ...)     # 3개 알고리즘을
@trainer_registry.register("rho1-wmtp", ...)       # 모두 처리
```

---

## 🏗️ 새로운 아키텍처

### 계층 구조
```
BaseWmtpTrainer (추상)
├── MtpBaselineTrainer    # 균등 가중치
├── CriticWmtpTrainer     # Critic 기반 가중치
└── Rho1WmtpTrainer       # Reference 모델 기반 가중치
```

### 파일 구조
```
src/components/trainer/
├── base_wmtp_trainer.py      # 공통 로직 (NEW)
├── mtp_baseline_trainer.py   # Baseline 구현 (NEW)
├── critic_wmtp_trainer.py    # Critic 구현 (NEW)
├── rho1_wmtp_trainer.py      # Rho-1 구현 (NEW)
└── mtp_weighted_ce_trainer.py (DELETE)

src/components/scorer/
└── rho1_excess.py            # 대폭 단순화
```

---

## 📋 Phase별 구현 계획

## **Phase 1: 공통 기반 클래스 생성**

### 1.1 BaseWmtpTrainer 설계
```python
class BaseWmtpTrainer(BaseComponent):
    """WMTP 알고리즘 공통 기능 제공"""

    # 공통 메서드들 (기존에서 추출)
    def setup(self, ctx) -> None: ...           # 모델/옵티마이저 초기화
    def run(self, ctx) -> dict: ...             # 체크포인트 관리 훈련 루프
    def _save_checkpoint(...) -> Path: ...      # 체크포인트 저장
    def _manage_checkpoints(...) -> list: ...   # 체크포인트 관리
    def _save_final_checkpoint(...) -> Path: ... # 최종 모델 저장

    # 추상 메서드 (알고리즘별 구현 필요)
    @abstractmethod
    def compute_head_weights(self, logits, target_ids, **kwargs) -> torch.Tensor:
        """각 알고리즘별 헤드 가중치 계산"""
        pass

    @abstractmethod
    def train_step(self, batch) -> dict:
        """알고리즘별 훈련 스텝 구현"""
        pass
```

### 1.2 공통 손실 함수 개선
```python
def compute_weighted_mtp_loss(
    logits: torch.Tensor,        # [B, S, H, V]
    target_ids: torch.Tensor,    # [B, S]
    head_weights: torch.Tensor,  # [B, S, H]
    horizon: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        weighted_loss: 가중 MTP 손실 (scalar)
        valid_mask: 유효 토큰 마스크 [B, S]
        ce_per_head: 헤드별 CE [B, S, H] ← Rho-1을 위한 핵심 변경
    """
```

---

## **Phase 2: 알고리즘별 트레이너 구현**

### 2.1 MtpBaselineTrainer
**특징**: 가장 단순, 균등 가중치
```python
@trainer_registry.register("mtp-baseline", category="trainer", version="2.0.0")
class MtpBaselineTrainer(BaseWmtpTrainer):
    def compute_head_weights(self, logits, target_ids, **kwargs):
        B, S, H, V = logits.shape
        return torch.ones((B, S, H), device=logits.device)

    def train_step(self, batch):
        # 가장 단순한 구현 - Scorer 없음
        logits = self.model(**batch)
        head_weights = self.compute_head_weights(logits, batch["labels"])
        loss, valid_mask, ce_per_head = compute_weighted_mtp_loss(...)
        return {"loss": loss.item()}
```

### 2.2 CriticWmtpTrainer
**특징**: Critic 기반 가중치, Value Head 활용
```python
@trainer_registry.register("critic-wmtp", category="trainer", version="2.0.0")
class CriticWmtpTrainer(BaseWmtpTrainer):
    def compute_head_weights(self, logits, target_ids, **kwargs):
        # Critic scorer를 활용한 가중치 계산
        hidden_states = kwargs.get("hidden_states")
        score_out = self.scorer.run({
            "hidden_states": hidden_states,
            "target_ids": target_ids
        })
        return score_out["weights"]  # [B, S, H]
```

### 2.3 Rho1WmtpTrainer ⭐ **핵심 구현**
**특징**: Reference 모델과의 CE 비교, 가장 효과적
```python
@trainer_registry.register("rho1-wmtp", category="trainer", version="2.0.0")
class Rho1WmtpTrainer(BaseWmtpTrainer):
    def setup(self, ctx):
        super().setup(ctx)
        self.ref_model = ctx.get("ref_model")  # Reference 모델 필수

    def compute_reference_ce(self, input_ids, target_ids):
        """효율적 Reference CE 계산 (한 번의 forward pass)"""
        with torch.no_grad():
            ref_logits = self.ref_model(input_ids)  # [B, S, V]
            ref_ce_all = F.cross_entropy(
                ref_logits[:, :-1].transpose(1, 2),  # [B, V, S-1]
                target_ids[:, 1:],                   # [B, S-1]
                reduction='none'
            )  # [B, S-1] - 각 위치의 next token CE
        return ref_ce_all

    def align_ref_ce_to_mtp(self, ref_ce_all, mtp_ce_heads):
        """Reference CE를 MTP 헤드와 정렬"""
        B, S, H = mtp_ce_heads.shape
        aligned_ref_ce = torch.zeros_like(mtp_ce_heads)

        for k in range(H):
            if k < ref_ce_all.size(1):
                valid_len = min(S - k - 1, ref_ce_all.size(1) - k)
                if valid_len > 0:
                    aligned_ref_ce[:, :valid_len, k] = ref_ce_all[:, k:k+valid_len]
        return aligned_ref_ce

    def compute_head_weights(self, logits, target_ids, ce_per_head, **kwargs):
        """Rho-1 방식: |CE^ref - CE^base| 기반 가중치"""
        # Reference CE 계산
        input_ids = kwargs.get("input_ids")
        ref_ce_all = self.compute_reference_ce(input_ids, target_ids)

        # MTP 헤드와 정렬
        aligned_ref_ce = self.align_ref_ce_to_mtp(ref_ce_all, ce_per_head)

        # Excess loss 계산: |CE^ref - CE^base|
        excess_loss = torch.abs(ce_per_head - aligned_ref_ce)  # [B, S, H]

        # Rho-1 가중치 변환 (단순화됨)
        weights = F.softmax(excess_loss / self.temperature, dim=-1)
        return weights  # [B, S, H]
```

---

## **Phase 3: Rho1 Scorer 대폭 정리**

### 3.1 삭제 대상 (전격적 제거)
```python
# ❌ 완전 삭제
- Mock random CE generation (50줄)
- apply_percentile_emphasis() (45줄)
- _compute_head_weights() 복잡 로직 (66줄)
- 불필요한 emphasis_type들 (30줄)

# 총 191줄 → 30줄로 축소 (84% 감소)
```

### 3.2 핵심만 남긴 새로운 구조
```python
@scorer_registry.register("rho1-wmtp", category="scorer", version="2.0.0")
class Rho1ExcessScorer(BaseComponent):
    """Reference vs Base 모델 CE 비교만 담당 (초경량화)"""

    def __init__(self, config=None):
        super().__init__(config)
        self.temperature = self.config.get("temperature", 0.7)

    def compute_excess_loss(self, base_ce, ref_ce):
        """핵심: |CE^ref - CE^base|"""
        return torch.abs(ref_ce - base_ce)

    def run(self, ctx) -> dict:
        """단순화된 실행: excess loss만 계산"""
        base_ce = ctx["base_ce"]      # [B, S, H]
        ref_ce = ctx["ref_ce"]        # [B, S, H]

        excess = self.compute_excess_loss(base_ce, ref_ce)
        weights = F.softmax(excess / self.temperature, dim=-1)

        return {"weights": weights, "excess": excess}

# 480줄 → 30줄 (94% 감소!)
```

---

## **Phase 4: 레지스트리 업데이트 및 통합**

### 4.1 레지스트리 리매핑
```python
# OLD (1개 클래스가 3개 처리)
"mtp-baseline"  → MTPWeightedCETrainer
"critic-wmtp"   → MTPWeightedCETrainer
"rho1-wmtp"     → MTPWeightedCETrainer

# NEW (각자 독립)
"mtp-baseline"  → MtpBaselineTrainer
"critic-wmtp"   → CriticWmtpTrainer
"rho1-wmtp"     → Rho1WmtpTrainer
```

### 4.2 기존 파일 삭제
```python
# 전격적 삭제 (필수4 원칙)
rm src/components/trainer/mtp_weighted_ce_trainer.py  # 900줄 삭제
```

---

## **Phase 5: 테스트 및 검증**

### 5.1 기능 검증
```bash
# 각 알고리즘별 테스트
python test_m3_pipeline.py --algo mtp-baseline --tiny
python test_m3_pipeline.py --algo critic-wmtp --tiny
python test_m3_pipeline.py --algo rho1-wmtp --tiny
```

### 5.2 성능 비교
```python
# 기대 결과
- mtp-baseline: 기존과 동일
- critic-wmtp: 기존과 동일
- rho1-wmtp: Reference CE 비교로 향상된 토큰 선별
```

---

## 📈 기대 효과

### 코드 품질 향상
| 지표 | Before | After | 개선도 |
|------|--------|-------|--------|
| **총 라인 수** | 1,380줄 | ~600줄 | **57% 감소** |
| **파일 복잡도** | 900줄 단일 | 150줄×4개 | **가독성 향상** |
| **알고리즘 격리** | 혼재 | 완전 분리 | **유지보수성 향상** |
| **중복 코드** | 높음 | 제거 | **재사용성 향상** |

### 연구 효과 향상
- **Rho-1 정확성**: Reference vs Base 모델 정확한 CE 비교
- **실험 효율성**: 알고리즘별 독립 실행 및 비교
- **확장성**: 새로운 알고리즘 추가 용이

---

## ⚠️ 리스크 및 대응

### 개발 리스크
1. **레지스트리 호환성**: 기존 설정 파일 업데이트 필요
2. **테스트 커버리지**: 각 알고리즘별 충분한 검증 필요
3. **성능 회귀**: 리팩토링으로 인한 성능 저하 방지

### 대응책
- Phase별 점진적 검증
- 기존 테스트 케이스 모두 통과 확인
- 성능 벤치마크 비교

---

## 🚀 실행 일정

| Phase | 작업 내용 | 예상 시간 | 우선순위 |
|-------|-----------|-----------|----------|
| **Phase 1** | BaseWmtpTrainer 생성 | 2시간 | 🔥 최고 |
| **Phase 2** | 3개 트레이너 분화 | 3시간 | 🔥 최고 |
| **Phase 3** | Rho1Scorer 정리 | 1시간 | 🔥 최고 |
| **Phase 4** | 레지스트리 업데이트 | 1시간 | 🔥 최고 |
| **Phase 5** | 테스트 및 검증 | 2시간 | ⚠️ 필수 |

**총 예상 시간**: 9시간

---

## 💡 구현 시작 준비

**다음 즉시 실행할 작업**:
1. BaseWmtpTrainer 클래스 생성
2. 공통 로직 추출 및 추상화
3. MtpBaselineTrainer부터 구현 시작

**준비 완료!** 🚀

---

*"Not All Tokens Are What You Need - 이제 코드도 마찬가지입니다"* ✨