# Weighted Multi-Token Prediction (WMTP) 학술 연구제안서
## — "Not All Tokens Are What You Need"의 이론·실증 통합 —

---

## 1. 초록 (Abstract)

대규모 언어모델(LLM)의 Multi-Token Prediction(MTP)은 동일 연산 예산에서 성능과 효율을 개선하는 패러다임으로 주목받고 있으나, 표준 MTP는 예측하는 모든 미래 토큰에 균등 가중을 부여하여 비핵심 토큰에도 학습 자원을 낭비할 수 있다. 본 제안은 “모든 토큰이 동등하게 중요하지 않다(Not All Tokens Are What You Need)”는 통찰을 바탕으로, 토큰별 중요도를 동적으로 반영하는 Weighted MTP(WMTP)를 정식화하고 실증한다. 우리는 (i) 참조-비교 기반(Rho‑1식), (ii) Critic(가치) 기반, (iii) 그래디언트 기반, (iv) GRPO 기반 총 네 가지 가중화 계열을 제안·분석하고, 정보 이론·중요도 샘플링·정규화 관점에서 WMTP의 이점을 논증한다. 코드 생성·추론·일반 언어 이해 벤치마크에서 WMTP의 성능·효율·안정성 이득과 규모에 따른 스케일링 특성을 체계적으로 검증한다.

---

## 2. 배경과 문제 정의

- 배경: NTP(Next-Token Prediction) 대비 MTP는 한 시점에서 다수의 미래 토큰을 병렬 예측해 수렴 가속과 다운스트림 성능 향상을 보고하였다. 또한 병렬 생성·자기추측 디코딩과의 결합으로 추론 속도 이점도 가능하다.
- 문제: 표준 MTP는 모든 미래 토큰을 균등 가중으로 다루어, 쉬운/비핵심 토큰에도 동일한 학습 자원을 배분한다. 이는 고비용 데이터 구간·장기 의존·결정적 토큰에서 비효율·불안정성을 유발할 수 있다.
- 핵심 가설: 중요 토큰에 계산을 집중하는 WMTP는 동일 FLOPs에서 더 높은 성능과 안정적 수렴을 달성한다.

---

## 3. 관련 연구

- MTP 계열: 표준 MTP는 병렬 예측을 통해 효율을 높였고, 도약(leap) 예측 등은 장거리 의존·병렬성을 확장했다.
- 선택적/가중 학습: 참조 모델 기반 선택적 학습(Selective LM)은 어려운 토큰에 집중해 효율적 성능 향상을 보여주었다(예: Rho‑1, NeurIPS 2024, [OpenReview 링크](https://openreview.net/forum?id=0NMzBwqaAJ); [NeurIPS 포스터](https://neurips.cc/virtual/2024/poster/96931); [HF 모델 카드](https://huggingface.co/microsoft/rho-math-7b-v0.1)).
- 선호 최적화·정책경사: Actor‑Critic·Policy Gradient 정론(Policy Gradient Theorem, Sutton et al. 1999, [NIPS 논문](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)), GAE(Schulman et al. 2015, [arXiv](https://arxiv.org/abs/1506.02438)), Off‑policy Actor‑Critic(Degris et al. 2012, [arXiv](https://arxiv.org/abs/1205.4839)).
- GRPO: Critic‑free 그룹 정규화 보상 최적화(DeepSeek 계열). 이론적 정식화·수렴·손실 해석이 최근 제시됨([“What is the Alignment Objective of GRPO?”](https://arxiv.org/abs/2502.18548), [“GRPO’s Effective Loss…”](https://arxiv.org/html/2503.06639v1)).

요약: WMTP는 MTP의 병렬 예측 이점을 토큰‑수준 중요도 가중과 결합해 기존 방법을 일반화하며, Critic 의존을 줄이거나(참조/GRPO/그래디언트) 안정화 기법을 병행해 재현성과 안정성을 강화한다.

---

## 4. 제안 방법: Weighted Multi-Token Prediction

### 4.1 목적함수 정식화(네 가지 가중화 계열)

입력 x, 시점 t, 예측 범위 H에 대해 WMTP 손실은 다음과 같다.

```
L_WMTP = E_x [ Σ_t Σ_{k=1..H} w_{t,k} · CE( P_θ(x_{t+k} | x_{<t}), x_{t+k} ) ]
```

- 제약: w_{t,k} ≥ 0, Σ_k w_{t,k} = 1.
- 전제: 동일 토큰 공간(토크나이저 일치), 동일 태스크의 시점 정렬(MTP k‑헤드 ↔ t+k 예측).

가중치 `w_{t,k}` 산출의 네 계열과 이론적 배경:

- Critic(가치) 기반: 정책경사 정론에 따르면 이득(Advantage)으로 로그우도 가중 시 분산 감소와 수렴 보장이 가능(Sutton et al. 1999). WMTP에서는 각 미래 단계의 상태 가치/TD‑오차를 토큰 가중으로 사용한다. 분산 저감을 위해 GAE(λ)·Baseline·KL 정규화 등을 병행(Schulman et al. 2015). Off‑policy 시 중요도 보정·안정화가 필요(Degris et al. 2012).
- 참조‑비교(Rho‑1) 기반: 참조 모델 손실과 기준 손실의 초과 손실(excess loss)을 점수화하고 온도 T로 정규화해 w를 산출(Selective LM; NeurIPS 2024). 전제는 동일 토큰화·예측 태스크의 시점 정렬이며, 참조 비용은 캐시/샘플링/오프라인 스코어링으로 완화.
- 그래디언트 기반: 최종 목적(선호/정답/가능도)에 대한 각 토큰 표현·로짓의 기여도(∥∇·∥)를 중요도로 사용. 목표‑그래디언트 중요도(Goal‑Gradient) 계열과 유사한 동인 연구가 보고됨(CoT 압축/중요도, 2025, [arXiv](https://arxiv.org/abs/2505.08392)). 장점은 Critic/참조 없이 동적·세밀 신호를 반영하는 점이며, 노이즈 완화(EMA, 윈도우 누적)와 스케일 정규화가 중요.
- GRPO 기반: 그룹 내 샘플의 보상(검증 가능 보상 포함)을 표준화(whitening)해 상대 이득을 형성, KL‑정규화 하에 정책을 업데이트하는 critic‑free 최적화(DeepSeek; [arXiv 2502.18548](https://arxiv.org/abs/2502.18548), [arXiv 2503.06639](https://arxiv.org/html/2503.06639v1)). WMTP에서는 토큰‑수준 손실에 GRPO의 그룹 상대 가중을 결합해 중요 토큰에 더 큰 학습 비중을 부여할 수 있다.

이론적 정당성(공통):
- 정보 이론: 토큰별 정보량이 상이하며, 정보 이득이 큰 토큰에 계산을 배분하면 샘플 효율이 개선.
- 중요도 샘플링: w는 중요도 가중 역할을 수행하며, 적절한 정규화·클리핑 하에 분산 감소와 안정 추정에 기여.
- 최적화: 불필요한 손실 기여 억제로 유효 그래디언트 증폭, 수렴 변동성 감소.

예상 이슈·유의점(계열별 요약):
- Critic: 가치 표류로 가중 왜곡 위험 → KL/Trust‑region, GAE(λ), 보상 정규화, 보조 손실(표현 고정화) 병행.
- 참조: 토큰화/시점 불일치 시 비교 왜곡 → 동일 토크나이저 강제, 정렬 자동 검증, 실패 시 배치 제외.
- 그래디언트: 고분산·노이즈 → 스케일 정규화, EMA, 윈도우 누적, outlier 클리핑.
- GRPO: 그룹 크기·보상 스케일·KL 조절 실패 시 붕괴/해킹 → 그룹 표준화, KL β 스윕, verifiable reward 채택, 클리핑.

### 4.2 실무적 고려(원칙)

- 토크나이저 일치: 어휘·특수 토큰·분절 정책 불일치는 확률 비교를 왜곡한다. 최소한의 샘플링 검증으로 일치성을 담보.
- 시점 정렬: 참조/선호/보상 신호는 동일 예측 태스크로 정렬되어야 하며, 헤드 k와 t+k 예측이 정확히 대응되도록 설계.
- 수치 안정성: 온도 T, 가중치 엔트로피 범위, 가중치 클리핑, outlier 억제, 최소 엔트로피 제약 등을 적용.

---

## 6. 실험 설계

### 6.1 데이터셋·벤치마크

- 코드 생성: MBPP, HumanEval
- 수학/추론: MATH, GSM8K
- 일반 이해: HellaSwag (필요시 추가)

### 6.2 모델·스케일

- 아키텍처: MTP(H=4) 기반.
- 규모: ≈1B·≈7B 두 스케일로 비교해 스케일링 법칙 분석.

### 6.3 비교 조건·소거(Ablations)

- 균등 MTP vs WMTP(참조‑비교 / Critic / 그래디언트 / GRPO / 하이브리드)
- 온도 T, 가중치 엔트로피 범위, 스파르시티(톱‑p/톱‑k 가중) 스윕
- 시점 정렬 오차·토큰화 불일치 민감도
- 참조 모델 비용‑효과(빈도·샘플링·캐시) 분석

### 6.4 지표·통계·재현성

- 성능: Exact Match, Pass@K, 과제별 정답률
- 효율: 스텝당 FLOPs, 벽시계 시간, 수렴 스텝 수
- 안정: 수렴 분산, 그래디언트 놈·가중치 엔트로피 분포
- 분석: 가중치 분포/엔트로피, 토큰 난이도별 오류율, 어텐션 상관
- 통계: 비모수 검정(윌콕슨 등), 95% CI 부트스트랩, 시드≥5 반복
- 재현성: 설정·로그·체크포인트 공개, 시드·데이터 스플릿 고정

### 6.5 핵심 설계 제약·권고(토큰 일치·표현 안정화)

- 토크나이저 강제 일치: 학습·참조·정책(및 보상평가)이 모두 동일 토크나이저·특수토큰 체계를 사용하도록 강제. 최소 샘플 토큰 세트에 대한 ID 일치 자동 점검을 루프에 포함.
- 시점 정렬 유효성: MTP 헤드 k ↔ t+k 예측이 항상 일치하도록 라벨 시프트·마스킹을 자동 검증.
- Critic 표현 안정화: 히든 상태 분포 변화를 억제하기 위해 auxiliary loss를 병행(예: (a) 히든 스테이트 이동평균 앵커 MSE, (b) 가치‑헤드 출력 분산/스케일 정규화, (c) LM 보조 CE로 trunk 표현 보존). KL/Trust‑region과 병행해 가치 표류 완화.
- 그래디언트 중요도 안정화: 토큰별 ∥∇·∥ 스케일 정규화, EMA 누적, 상위‑p 절단, 이상치 클리핑.
- GRPO 안정화: 그룹 크기·보상 스케일 표준화, KL β 스윕, 클리핑 창, verifiable reward(정답/실행 검증) 우선 채택.

---

## 7. 리스크와 완화(4.1 기반 구체 보완)

- 토큰화/정렬 불일치: w 오산정·손실 왜곡 → 동일 토크나이저 강제, 정렬 자동 검증, 실패 시 배치 제외.
- Critic 표류·고분산: 가치 편향·분산 증가 → GAE(λ), 보상/이득 정규화, KL/Trust‑region, auxiliary 표현 고정화, 학습‑평가 스케줄 분리.
- 참조 비용·편향: 참조 과도 사용 시 비용 급증·편향 → 샘플링/주기화, 캐시·오프라인 스코어링, 온도 T 스윕으로 집중도 제어.
- 그래디언트 노이즈: 중요도 과대·불안정 → 스케일/엔트로피 가드, EMA, 윈도우 누적, 상위‑p 제한, 클리핑.
- GRPO 붕괴/해킹: KL 약화·보상 왜곡 시 분포 붕괴 → β 스윕, 그룹 표준화, verifiable reward, 클리핑·early‑stop.

---

## 8. 윤리·안전·사회적 영향

- 가중치 편향: 드문/민감 토큰의 과소·과대 가중을 모니터링하고, 그룹 공정성·유해성 지표를 병행 보고한다.
- 능력 증대와 오용: 강화된 코드/추론 능력의 오용 가능성에 대비해 책임 있는 공개·제한적 배포·안전 가드레일을 명시한다.
- 데이터·프라이버시: 참조/선호 신호 추출 시 데이터 정책과 프라이버시를 준수한다.

---

## 10. 기대 효과·기여

- 이론: MTP에 대한 정보 이론·중요도 샘플링 관점의 가중화 프레임워크 확립
- 방법: Critic/참조/그래디언트/GRPO 기반 WMTP의 정식화 및 안정화 기법(온도·정규화·검증)
- 실증: 코드·추론·일반 과제에서 성능·효율·안정성 동시 개선 증거 제공
- 실용: 구성 주도형 설계를 통한 확장성·재현성·비용 효율성

---

## 참고문헌(발췌)

- Glöckle et al. (2024). Better & Faster LLM via Multi‑Token Prediction.
- Lin et al. (2024). Rho‑1: Not All Tokens Are What You Need. (NeurIPS 2024 Oral) — [OpenReview](https://openreview.net/forum?id=0NMzBwqaAJ), [NeurIPS 포스터](https://neurips.cc/virtual/2024/poster/96931), [HF 모델 카드](https://huggingface.co/microsoft/rho-math-7b-v0.1)
- Sutton et al. (1999). Policy Gradient Methods… — [NIPS 논문](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)
- Schulman et al. (2015). Generalized Advantage Estimation — [arXiv](https://arxiv.org/abs/1506.02438)
- Degris et al. (2012). Off‑Policy Actor‑Critic — [arXiv](https://arxiv.org/abs/1205.4839)
- Vojnovic & Yun (2025). What is the Alignment Objective of GRPO? — [arXiv](https://arxiv.org/abs/2502.18548)
- Mroueh (2025). GRPO’s Effective Loss, Dynamics… — [arXiv](https://arxiv.org/html/2503.06639v1)
- Zhuang et al. (2025). Goal‑Gradient Importance(CoT) — [arXiv](https://arxiv.org/abs/2505.08392)
