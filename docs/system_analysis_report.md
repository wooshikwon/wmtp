# WMTP 시스템 분석 보고서
**Weighted Multi-Token Prediction 연구 의도 및 코드베이스 구현 분석**

---

## 1. 연구 의도 및 핵심 철학 분석

### 1.1 연구 제안서의 핵심 아이디어

**원래 연구 제안 (연구제안서.md):**
- **핵심 철학**: "Not All Tokens Are What You Need" - 모든 토큰이 동등하게 중요하지 않다
- **문제 정의**: 기존 MTP(Multi-Token Prediction)는 N개의 미래 토큰을 균등하게 학습하여 비효율성 발생
- **해결 방안**: Critic Function을 활용한 토큰 중요도 기반 가중치 부여

**수학적 공식:**
```
L_WMTP = Σ(k=0 to 3) w_{t+k} × CE(y_{t+k}, ŷ_{t+k})
여기서 w_{t+k} = softmax(δ_{t+k})
δ_t = V_θ(s_t) - V_θ(s_{t-1}) (Critic의 가치함수 차분)
```

### 1.2 연구 개선안의 핵심 수정사항

**교수님 피드백 반영 (연구개선안.md):**
- **Critic 방식의 한계 인식**: Value estimation의 난이도와 불안정성 문제
- **대안 제시**:
  1. **Reference Model 기반 방법** (Rho-1 스타일)
  2. **직접적 선호 최적화** (DPO/SimPO 계열)
  3. **Gradient 기반 중요도 산출**
- **GRPO/DPO 등 최신 연구 반영**: Critic 없이도 효과적인 토큰 가중화 가능

**개선된 수학적 접근:**
```
Rho-1 방식: w_t = |CE^ref(x_t) - CE^base(x_t)|
TI-DPO 방식: w_t = |∇_θ R(x_t)| (Gradient-based importance)
```

## 2. 현재 코드베이스 아키텍처 분석

### 2.1 전체 시스템 구조

```
WMTP 시스템 = Factory Pattern + Registry Pattern + Pipeline Architecture

src/
├── pipelines/training_pipeline.py    # 통합 실행 엔진
├── factory/component_factory.py      # 알고리즘별 컴포넌트 생성
├── components/
│   ├── registry.py                   # 통합 레지스트리
│   ├── scorer/                       # 토큰 중요도 계산
│   │   ├── critic_delta.py          # Critic-WMTP 구현
│   │   └── rho1_excess.py           # Rho1-WMTP 구현
│   └── trainer/                      # 가중치 적용 훈련
│       └── mtp_weighted_ce_trainer.py
└── settings/recipe_schema.py         # 알고리즘 설정 정의
```

### 2.2 핵심 설계 패턴

**1. "One Pipeline, Multiple Algorithms"**
- `training_pipeline.py`: 모든 알고리즘이 동일한 파이프라인 사용
- `ComponentFactory`: 설정에 따라 다른 컴포넌트 조합 생성
- **장점**: 알고리즘 간 공정한 비교, 코드 중복 제거

**2. Factory Pattern + Registry**
```python
# 알고리즘별 컴포넌트 조합
scorer = ComponentFactory.create_scorer(recipe)
trainer = ComponentFactory.create_trainer(recipe, config)

# 내부적으로 Registry에서 적합한 구현체 선택
if algo == "critic-wmtp":
    scorer = CriticDeltaScorer(config)
elif algo == "rho1-wmtp":
    scorer = Rho1ExcessScorer(config)
else:  # mtp-baseline
    scorer = None  # 균등 가중치
```

### 2.3 구현된 알고리즘 상세

#### 2.3.1 MTP Baseline (`mtp-baseline`)
```python
# 균등 가중치: 모든 토큰에 동일한 중요도
w_{t+k} = 1.0 for all k
L = Σ(k=0 to 3) 1.0 × CE_k
```

#### 2.3.2 Critic-WMTP (`critic-wmtp`)
```python
# 2단계 학습
# Stage 1: Value Head 사전학습
L_critic = Σ_t (V_θ(h_t) - R̂_t)²

# Stage 2: Delta 기반 가중치
δ_t = V_θ(h_t) - λ × V_θ(h_{t-1})
w_{t+k} = softmax([δ_{t+1}, δ_{t+2}, δ_{t+3}, δ_{t+4}])
```

#### 2.3.3 Rho1-WMTP (`rho1-wmtp`) - **권장 방식**
```python
# Reference Model 차이 기반
score_t = |CE^ref(x_t) - CE^base(x_t)|
w_{t+k} = f(score_{t+k})  # 변환 함수 적용

# 구현 세부사항:
def compute_weights(ref_ce, base_ce):
    excess = torch.abs(ref_ce - base_ce)
    # Percentile 기반 강조
    top_p_threshold = torch.quantile(excess, 1 - percentile_top_p)
    weights = torch.where(excess >= top_p_threshold, excess * 2.0, excess)
    return F.softmax(weights / temperature, dim=-1)
```

## 3. 연구 의도와 구현의 매핑

### 3.1 핵심 철학 구현도

| 연구 의도 | 코드 구현 | 구현도 |
|-----------|-----------|--------|
| "Not All Tokens Are What You Need" | 세 가지 다른 가중치 계산 방식 | ✅ 완전 구현 |
| MTP 기반 병렬 학습 | `n_heads=4`, horizon=4 | ✅ 완전 구현 |
| 토큰별 차등 가중치 | `w_{t+k}` 헤드별 적용 | ✅ 완전 구현 |
| 교수님 피드백 반영 | Critic 대신 Rho-1 권장 | ✅ 완전 구현 |

### 3.2 수식과 코드의 대응

**연구제안서 공식:**
```
L_total = L_WMTP + λ × L_VF
L_WMTP = Σ(k=0 to 3) w_{t+k} × CE_k
```

**실제 구현 (mtp_weighted_ce_trainer.py):**
```python
def compute_weighted_loss(self, logits, labels, weights):
    """WMTP 공식 직접 구현"""
    head_losses = []
    for k in range(self.n_heads):
        head_logits = logits[:, :, k, :]  # [batch, seq, vocab]
        head_labels = labels[:, k:]       # t+k 토큰
        ce_loss = F.cross_entropy(head_logits, head_labels, reduction='none')
        weighted_loss = ce_loss * weights[:, :, k]  # w_{t+k} 적용
        head_losses.append(weighted_loss.mean())

    return sum(head_losses)  # Σ w_{t+k} × CE_k
```

### 3.3 개선안 반영도

| 개선안 권장사항 | 현재 구현 | 상태 |
|-----------------|-----------|------|
| Critic 복잡성 해결 | Rho-1 방식 기본 채택 | ✅ 반영됨 |
| Reference Model 활용 | `Rho1ExcessScorer` 구현 | ✅ 반영됨 |
| DPO/SimPO 고려 | 미구현 (향후 과제) | ❌ 미반영 |
| Gradient 기반 중요도 | 미구현 (TI-DPO 스타일) | ❌ 미반영 |

## 4. 시스템의 확장성 및 연구 지원도

### 4.1 새로운 알고리즘 추가 용이성

**현재 시스템의 확장 방법:**
```python
# 1. Scorer 구현
@scorer_registry.register("new-algorithm", category="scorer")
class NewAlgorithmScorer(BaseComponent):
    def compute_weights(self, batch_data):
        # 새로운 가중치 계산 로직
        return weights

# 2. Recipe 스키마 확장
class Train(BaseModel):
    algo: Literal["mtp-baseline", "critic-wmtp", "rho1-wmtp", "new-algorithm"]

# 3. Factory 매핑 추가 (자동)
# Registry 패턴으로 인해 추가 코드 최소화
```

### 4.2 실험 추적 및 재현성

```python
# MLflow 기반 실험 관리
mlflow.start_run(run_name=recipe.run.name)
mlflow.log_params({
    "algorithm": recipe.train.algo,
    "lambda": recipe.loss.lambda_weight,
    "temperature": recipe.loss.temperature
})
mlflow.log_metrics(trainer_metrics)
```

### 4.3 설정 기반 실험 관리

```yaml
# configs/recipe.rho1.yaml
train:
  algo: "rho1-wmtp"
loss:
  lambda: 0.5
  temperature: 0.5
rho1:
  score: "abs_excess_ce"
  percentile_top_p: 0.15
```

## 5. 현재 구현의 강점과 한계

### 5.1 강점

1. **연구 철학 충실 구현**: "Not All Tokens Are What You Need" 완전 실현
2. **교수님 피드백 반영**: Critic 대신 안정적인 Rho-1 방식 채택
3. **확장성**: Registry + Factory 패턴으로 새 알고리즘 추가 용이
4. **공정한 비교**: 동일 파이프라인에서 알고리즘 간 성능 비교 가능
5. **산업 표준**: HuggingFace, MLflow 등 검증된 도구 활용

### 5.2 한계 및 개선 방향

1. **개선안 미반영 부분**:
   - DPO/SimPO 스타일 구현 없음
   - TI-DPO의 gradient 기반 중요도 미구현

2. **실험 범위**:
   - 코딩 도메인 중심 (MBPP, CodeContests)
   - 자연어 도메인 확장 필요

3. **평가 메트릭**:
   - Pass@k 중심
   - Perplexity, BLEU 등 추가 메트릭 필요

## 6. 결론

### 6.1 연구 의도 구현도 평가

**전체 평가: 85/100점**
- ✅ 핵심 철학 구현: 완벽
- ✅ 수식-코드 대응: 정확
- ✅ 교수님 피드백 반영: 양호
- ⚠️ 개선안 완전 반영: 부분적
- ⚠️ 확장성 입증: 필요

### 6.2 연구 기여도

1. **이론적 기여**: MTP + Token Weighting 최초 결합
2. **실용적 기여**: 안정적이고 확장 가능한 구현 제공
3. **방법론적 기여**: Factory + Registry 기반 알고리즘 비교 프레임워크

### 6.3 향후 발전 방향

1. **단기 (1-2개월)**:
   - TI-DPO 스타일 gradient 기반 scorer 추가
   - 자연어 도메인 확장 (일반 텍스트, 수학)

2. **중기 (3-6개월)**:
   - DPO/SimPO 기반 선호 최적화 통합
   - 더 정교한 평가 프로토콜 개발

3. **장기 (6개월+)**:
   - 대규모 모델 (13B+) 지원
   - 다국어 확장 및 도메인별 특화

---

**작성일**: 2025년 9월 25일
**분석 대상**: WMTP 연구 문서 및 코드베이스
**분석자**: Claude Code System Analysis