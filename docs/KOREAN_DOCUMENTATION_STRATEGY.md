# WMTP 한글 문서화 전략

## 📖 문서화 철학

### 핵심 가치
> **"파이썬 초보자도 우리 연구를 이해하고 확장할 수 있도록"**

우리의 WMTP(Weighted Multi-Token Prediction) 연구는 단순한 코드가 아닌 **혁신적인 학습 방법론**입니다. 모든 코드가 연구 철학과 수학적 기초를 반영해야 하며, 한글 문서화를 통해 다음을 달성합니다:

1. **연구 맥락 이해**: "왜 이 코드가 필요한가?"
2. **구현 의도 파악**: "어떻게 논문 공식이 코드가 되었나?"
3. **확장 가능성**: "새로운 아이디어를 어떻게 추가할까?"

## 🧠 연구 철학과 구현 의도

### WMTP의 핵심 아이디어
```
"모든 토큰이 동등하지 않다" (Not All Tokens Are What You Need)

기존: w₁ = w₂ = w₃ = w₄ = 1 (균등한 중요도)
WMTP: w₁ ≠ w₂ ≠ w₃ ≠ w₄     (차등적 중요도)

L_WMTP = Σ(k=0 to 3) wₜ₊ₖ × CE(yₜ₊ₖ, ŷₜ₊ₖ)
```

### 세 가지 알고리즘 철학
1. **MTP-Baseline**: 균등 가중치 (기준점)
2. **Critic-WMTP**: 강화학습 가치함수로 미래 중요도 계산
3. **Rho1-WMTP**: 참조모델과의 차이로 어려운 토큰 발견

## 📊 시스템 파이프라인 전체 흐름

### Phase 1: 시작 - 연구 설정 선택
```
사용자 입력 → CLI → 설정 파싱 → 알고리즘 선택
              ↓
         ComponentFactory
         "레고 블록 조립기"
```

### Phase 2: 재료 준비 - 모델과 데이터
```
Facebook Native MTP → ModelLoader → 4개 헤드 모델 준비
MBPP/CodeContests  → DataLoader  → 코딩 문제 데이터
S3/Local 캐시      → 자동 다운로드  → 효율적 저장
```

### Phase 3: 핵심 - 토큰 중요도 계산
```
알고리즘 선택:
├─ mtp-baseline  → Scorer = None        → uniform weights
├─ critic-wmtp   → CriticDeltaScorer   → δₜ = V(sₜ) - V(sₜ₋₁)
└─ rho1-wmtp     → Rho1ExcessScorer    → |CE_ref - CE_base|
```

### Phase 4: 학습 엔진 - 가중 손실 적용
```
MTPWeightedCETrainer (통합 엔진)
├─ 토큰 가중치 × Cross Entropy
├─ 4개 헤드별 개별 손실 계산
└─ 가중 평균으로 최종 손실
```

### Phase 5: 성능 검증 - Meta 논문과 동일한 평가
```
MetaMTPEvaluator
├─ HumanEval: pass@1, pass@10, pass@100
├─ MBPP: 동일한 평가 지표
└─ 통계적으로 유의한 성능 비교
```

## 🎨 한글 Docstring 및 주석 작성 규칙

### Docstring 구조 (함수/클래스)
```python
def example_function(param1: int, param2: str) -> dict:
    """
    [1줄 요약] 이 함수는 ~를 담당합니다.

    [연구 맥락] WMTP에서 이것이 왜 필요한지 설명:
    - 연구 논문의 어떤 공식/개념과 연결되는가
    - 세 알고리즘(baseline/critic/rho1) 중 어디에 사용되는가

    [구체적 동작] 함수가 실제로 수행하는 작업:
    - 입력을 어떻게 처리하는가
    - 어떤 계산 과정을 거치는가
    - 출력이 무엇을 의미하는가

    매개변수:
        param1: 설명 (구체적인 예시 포함)
        param2: 설명 (가능한 값들 나열)

    반환값:
        dict: 키별 의미와 활용 방법

    예시:
        >>> # 실제 사용 예시
        >>> result = example_function(4, "critic")
        >>> print(result["weights"])  # 토큰별 가중치 출력

    주의사항:
        - 알려진 제약사항이나 주의점
        - 디버깅 시 확인할 포인트
    """
```

### 라인 주석 패턴
```python
# 1. 연구 맥락 설명
# 논문의 w_t = softmax(δ_t/T) 공식을 구현

# 2. 복잡한 로직 해설
# 여기서 4개 MTP 헤드별로 각각 다른 가중치를 적용합니다
for k in range(4):  # k=0,1,2,3 (t+1, t+2, t+3, t+4 시점)

# 3. 조건별 분기 설명
if self.scorer is None:
    # baseline 알고리즘: 모든 토큰에 동일한 가중치
    weights = torch.ones_like(hidden_states)
else:
    # critic 또는 rho1: scorer가 토큰 중요도 계산
    weights = self.scorer.compute_weights(...)

# 4. 주의사항과 디버깅 팁
# 주의: 가중치 합이 1이 아닐 수 있으므로 정규화 필요
# 디버깅: weights.sum() 확인으로 이상값 감지 가능
weights = F.softmax(weights / temperature, dim=-1)
```

### 용어 통일 사전
| 영어 용어 | 한글 표기 | 설명 |
|-----------|-----------|------|
| Multi-Token Prediction | 다중토큰예측 (MTP) | 여러 미래 토큰 동시 예측 |
| Cross Entropy | 교차엔트로피 (CE) | 예측 확률과 실제값의 차이 |
| Token weights | 토큰가중치 | 각 토큰의 중요도 점수 |
| Value head | 가치헤드 | 상태 가치를 예측하는 신경망 |
| Reference model | 참조모델 | 비교 기준이 되는 모델 |
| Sequence reward | 시퀀스보상 | 전체 문장에 대한 보상 점수 |
| Hidden states | 은닉상태 | 모델 내부의 표현 벡터 |
| Logits | 로짓 | Softmax 적용 전 점수 |

## 📁 파일별 문서화 전략

### Phase A: 핵심 파이프라인 (우선순위 1)

#### 1. `src/cli/train.py` - "연구 시작점"
```python
"""
WMTP 연구 실험을 시작하는 진입점입니다.

이 CLI 도구는 세 가지 알고리즘으로 실험할 수 있습니다:
1. mtp-baseline: 기존 MTP 방식 (균등 가중치)
2. critic-wmtp: 강화학습 가치함수 기반 가중치
3. rho1-wmtp: 참조모델 차이 기반 가중치

사용법:
    uv run python -m src.cli.train --config configs/config.local.yaml --recipe configs/recipe.critic.yaml
"""
```

#### 2. `src/factory/component_factory.py` - "레고 블록 시스템"
```python
"""
WMTP 시스템의 핵심: 설정에 따라 필요한 구성요소를 자동으로 조립합니다.

마치 레고처럼, recipe.yaml 파일의 설정만 바꾸면:
- 다른 알고리즘 (critic ↔ rho1 ↔ baseline)
- 다른 모델 (Facebook MTP ↔ HuggingFace)
- 다른 데이터셋 (MBPP ↔ CodeContests)
으로 실험할 수 있습니다.

핵심 아이디어: "설정 파일 하나로 모든 실험 변경"
"""
```

#### 3. `src/pipelines/training_pipeline.py` - "학습 전체 흐름"
```python
"""
WMTP 학습의 전체 과정을 조율하는 지휘자입니다.

학습 단계:
1. 모델/데이터 로딩 (Facebook MTP + 코딩 데이터)
2. Scorer 생성 (알고리즘별 토큰 중요도 계산기)
3. [Critic만] Stage1: 가치헤드 사전 학습
4. Stage2: 가중치 적용한 전체 모델 학습
5. 평가 및 결과 저장

모든 알고리즘이 동일한 파이프라인을 거치지만,
Step 2에서 다른 Scorer를 사용하여 차별화됩니다.
"""
```

#### 4. `src/components/trainer/mtp_weighted_ce_trainer.py` - "가중 학습 엔진"
```python
"""
WMTP 연구의 핵심 엔진: 토큰별 가중치를 손실함수에 적용합니다.

연구 공식 구현:
L_WMTP = Σ(k=0→3) w_{t+k} × CE(y_{t+k}, ŷ_{t+k})

MTP 모델의 4개 헤드 각각에 대해:
- Head 0: t+1 시점 예측 → weight[t+1] 적용
- Head 1: t+2 시점 예측 → weight[t+2] 적용
- Head 2: t+3 시점 예측 → weight[t+3] 적용
- Head 3: t+4 시점 예측 → weight[t+4] 적용

최종 손실 = 4개 헤드의 가중 평균
"""
```

### Phase B: 알고리즘별 구현 (우선순위 2)

#### 5. `src/components/scorer/critic_delta.py` - "가치함수 방식"
```python
"""
Critic-WMTP 알고리즘: 강화학습의 가치함수로 토큰 중요도를 계산합니다.

핵심 아이디어:
"더 큰 가치 증가 = 더 중요한 토큰"

수학적 원리:
δ_t = V(s_t) - V(s_{t-1})  # 가치 증가량
w_t = softmax(δ_t / T)     # 가중치로 변환

2단계 학습:
Stage 1: RM 보상으로 가치헤드 사전 학습
Stage 2: 학습된 가치헤드로 토큰 가중치 계산
"""
```

#### 6. `src/components/scorer/rho1_excess.py` - "참조모델 비교 방식"
```python
"""
Rho1-WMTP 알고리즘: 참조모델과의 차이로 어려운 토큰을 찾습니다.

핵심 아이디어:
"참조모델도 어려워하는 토큰 = 중요한 토큰"

수학적 원리:
s_t = |CE_ref(t) - CE_base(t)|  # 교차엔트로피 차이
w_t = percentile_filter(s_t)     # 상위 20% 토큰만 선택

장점: Critic 학습 불안정성 없이 바로 가중치 계산 가능
"""
```

#### 7. `src/components/reward/sequence_reward.py` - "시퀀스 보상 계산"
```python
"""
Critic Stage1에서 사용: 전체 문장의 품질을 점수로 계산합니다.

역할:
1. RM(Reward Model)이 문장 전체를 평가
2. 하나의 스칼라 점수 반환 (예: 3.7점)
3. CriticDeltaScorer가 이를 토큰별로 분배

RM이 없을 때 fallback:
평균 교차엔트로피의 음수 사용 (-mean_CE)
"더 자연스러운 문장 = 더 높은 점수"
"""
```

### Phase C: 지원 시스템 (우선순위 3)

#### 8. `src/components/loader/` - "데이터 준비 시스템"
```python
"""
ModelLoader: Facebook Native MTP vs HuggingFace 모델 지원
- MTP Native: consolidated.pth 직접 로딩 (4개 헤드 내장)
- HuggingFace: 표준 모델 + 변환 레이어 추가

DatasetLoader: 코딩 평가 데이터셋
- MBPP: 974개 Python 기초 문제
- CodeContests: 복잡한 알고리즘 문제
- HumanEval: 164개 손수 검증된 문제 (추후 추가)
"""
```

#### 9. `src/components/evaluator/` - "성능 측정 시스템"
```python
"""
Meta MTP 논문과 동일한 평가 방식으로 공정한 성능 비교

평가 지표:
- Pass@1, Pass@10, Pass@100 (k번 시도 중 1번 성공률)
- Chen et al. (2021) unbiased estimator 사용
- 문제당 200개 샘플 생성 (통계적 신뢰도)

목표: "우리 WMTP가 정말 더 좋은지 과학적으로 증명"
"""
```

#### 10. `src/utils/` - "유틸리티 함수들"
```python
"""
S3Utils: VESSL GPU 클러스터에서 모델 자동 다운로드
HFUtils: HuggingFace 모델 안전 로딩 (Local→S3→Hub 순서)
EvalUtils: Pass@k 계산, 코드 실행 등 평가 보조
"""
```

## 🔍 품질 관리 가이드라인

### 문서화 체크리스트
- [ ] **연구 맥락**: 논문의 어떤 부분과 연결되는가?
- [ ] **수식 연결**: 수학 공식이 코드로 어떻게 구현되었나?
- [ ] **초보자 친화**: 파이썬 초보자가 읽고 이해할 수 있나?
- [ ] **예시 제공**: 실제 입력/출력 예시가 있나?
- [ ] **디버깅 팁**: 문제 상황 해결 방법 제시했나?
- [ ] **용어 일관성**: 통일된 한글 용어 사용했나?

### 리뷰 프로세스
1. **자체 검토**: 작성자가 위 체크리스트 확인
2. **동료 리뷰**: 다른 팀원이 이해 가능성 검증
3. **초보자 테스트**: 실제 파이썬 초보자에게 피드백 요청
4. **연구진 검토**: 연구 내용 정확성 최종 확인

### 유지보수 원칙
- **새 코드 추가시**: 동일한 문서화 스타일 적용
- **알고리즘 변경시**: 관련된 모든 주석 함께 업데이트
- **정기 점검**: 월 1회 문서 품질 검토
- **피드백 반영**: 사용자 의견을 바탕으로 지속 개선

## 🎯 예상 효과

### 단기 효과 (1-2개월)
- ✅ 새로운 연구원도 빠른 코드 이해
- ✅ 디버깅과 확장 작업 효율성 증대
- ✅ 연구 재현성 향상

### 장기 효과 (6개월-1년)
- ✅ 다른 연구자들의 WMTP 연구 참여 촉진
- ✅ 오픈소스 프로젝트로서의 완성도 증대
- ✅ 연구 성과의 더 넓은 확산과 인용

---

> **"코드는 컴퓨터가 실행하지만, 문서는 사람이 읽습니다."**
> **좋은 문서화는 연구의 지속가능성을 보장합니다.**

이 전략에 따라 단계적으로 전체 코드베이스를 한글로 문서화하여,
WMTP 연구를 누구나 이해하고 확장할 수 있는 오픈소스 프로젝트로 발전시키겠습니다.
