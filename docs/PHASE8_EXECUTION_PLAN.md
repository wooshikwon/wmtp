# Phase 8 — Pipeline Orchestration: Execution Plan

본 문서는 Phase 8 개선 과제를 코드베이스 기존 구조(Registry/Factory/Utils/Settings)를 존중하여 구현하기 위한 구체 실행 계획이다. 모든 변경은 uv 기반 의존성을 전제로 하며, 기능은 코드로 해결하고 환경/의존성 문제는 환경으로 해결한다.

---

## 0) 범위와 원칙
- 기존 구조 존중: `src/utils/*`, `src/components/*`(registry), `src/factory/component_factory.py`, `src/settings/*`, `src/pipelines/*`, `src/cli/*`
- Registry/Factory 일관: 컴포넌트는 레지스트리에 등록, 팩토리에서 선택/조립
- 유틸 단일화: 분산(FSDP), MLflow, HF/S3, 토크나이즈/데이터로더는 각각 `utils` 또는 기존 로더 사용
- 토큰 가중치 철학: 스코어러가 per-token 가중치 w_t(softmax 포함, mean≈1.0, clip)를 산출, 트레이너는 이를 그대로 사용

---

## 1) 트레이너 MTP CE 정렬 및 마스킹

### 목표
- CE_k(t) ↔ 레이블 y[t+k] 정렬 보장
- 각 t에 대해 유효한 k만 평균(|K(t)|로 나눔)
- 이후 토큰 가중치 w_t 적용(헤드별 가중치 아님)

### 구현 포인트 (`src/components/trainer/mtp_weighted_ce_trainer.py`)
- [ ] `_compute_mtp_ce_loss` 개선:
  - 입력: logits `[B,S,H,V]`, labels `[B,S]`
  - 각 k에 대해 `labels_k = labels[:, k:]`, `logits_k = logits[:, :S-k, k, :]`
  - CE_k를 길이 정렬 후 pad/mask 처리하여 `[B,S]`로 재투영(미유효 구간 0, mask로 평균 분모 제외)
  - `CE_avg(t) = mean_k∈K(t) CE_k(t)`
- [ ] 마스킹 로직: 시퀀스 실제 길이(있다면) 반영. 없으면 S 기준 `t+k < S`로 계산
- [ ] 손실: `loss = λ * mean_b,t(w_t * CE_avg(t))`
- [ ] 검증: logits.ndim==4, `H == horizon`

---

## 2) 스코어러 출력 torch 표준화

### 목표
- Numpy↔Torch 왕복 제거, device/dtype 정렬

### 구현 포인트
- [ ] `critic_delta.py`, `rho1_excess.py`의 `run()` 반환에서 `weights`를 `torch.Tensor`(model/device와 동일 dtype/device)로 선택적 지원
- [ ] 트레이너에서는 `weights`가 numpy면 tensor로 변환하되, 가능하면 scorer가 torch로 반환
- [ ] 평균≈1.0/클립 범위/finite 불변식을 유지하는 단위 테스트 강화

---

## 3) 모델 어댑터: [B,S,H,V] 미충족 시 폴백/래퍼

### 목표
- 표준 CausalLM([B,S,V]) 사용 시 H=1 폴백 또는 teacher-forcing 기반 k-step 로짓 생성 래퍼 제공

### 구현 포인트
- [ ] `src/utils/hf.py` 또는 `src/components/loader/hf_local_s3_loader.py` 인근에 **선택적 래퍼** 추가:
  - `MTPWrapper(model, horizon=H)` → `forward`에서 k-step 로짓 생성(teacher forcing, 효율 고려 시 옵션)
  - 최소 폴백: H>1인데 [B,S,V] 모델이면 경고 후 H=1 사용
- [ ] 팩토리/파이프라인에서 `recipe.model.mtp.horizon`과 모델 출력 형상 검사, 불일치 시 폴백/에러처리

---

## 4) DataLoader 분산/캐시/마스킹 정비

### 목표
- 분산 환경에서 올바른 샘플링/시드, 토크나이즈 캐시, 마스킹 일관성 확보

### 구현 포인트 (`src/pipelines/training.py`)
- [ ] `DistributedSampler` 적용 및 `set_seed` 후 각 rank별 시드 고정
- [ ] `dataset.map` 토크나이즈 결과를 캐시 디렉토리에 저장(HF datasets 캐시)
- [ ] padding/eos 마스크와 k-step 경계 마스크를 트레이너에 전달(가능 시 `labels` 생성 시 pad=-100 사용)

---

## 5) MLflow 메트릭 확장 및 실패 게이트

### 목표
- 학습 관찰성 강화, 실패 시 조기 중단/로깅

### 구현 포인트
- [ ] 트레이너에서 로깅 확장: `train/loss`, `train/ce_mean`, `train/ce_head_i`, `train/weight_mean/min/max`, `train/valid_token_ratio`
- [ ] NaN/Inf 감지 시 학습 중단 및 MLflow에 상태 기록
- [ ] 가중치 통계(평균≈1.0, 범위) 불변식도 로깅

---

## 6) 테스트 강화

### 목표
- 정렬·마스킹·가중치 불변식 보장

### 구현 포인트 (`tests/`)
- [ ] MTP CE 정렬 테스트: 더미 logits/labels로 k-step 평균이 기대대로 나오는지
- [ ] 가중치 불변식: finite, mean≈1.0, [ε,Wmax] 범위
- [ ] 스모크: H=1과 H>1, scorer 두 종류(Rho-1/Critic), 드라이런(max_steps small) 경로

---

## 7) 파이프라인 연결(현 구조 유지)

- `src/cli/train.py` → `src/pipelines/training.py: TrainingPipeline`
- `ComponentFactory`로 scorer/trainer/optimizer/loader 생성(이미 구현)
- 모델/데이터 로딩은 기존 로더 사용. 변경은 트레이너 내부 정렬/마스킹과 선택적 모델 래퍼로 국한

---

## 8) 위험요인 및 완충책
- 성능 부담: teacher-forcing k-step 래퍼는 비용↑ → 기본은 H=1 폴백, 고급 옵션으로 래퍼 제공
- 메모리: H 증가 시 VRAM↑ → grad-accum/token-budget 조정
- 불일치: 레이블/마스크 정렬 오류 → 단위테스트로 조기 검출

---

## 9) 산출물 체크리스트
- [ ] 트레이너 MTP CE 정렬/마스킹 수정
- [ ] 스코어러 torch 출력 표준화(또는 트레이너 변환 최소화)
- [ ] 모델 어댑터(래퍼/폴백) 택1 적용
- [ ] DataLoader 분산/캐시/마스킹 보강
- [ ] MLflow 메트릭/실패 게이트 확장
- [ ] 유닛/스모크 테스트 추가 및 통과

---

## 10) 적용 순서(작업 세트)
1) 트레이너 `_compute_mtp_ce_loss` 정렬/마스킹 구현 → 테스트 추가
2) 스코어러 출력 torch 정렬 → 트레이너 변환 제거
3) (옵션) MTP 래퍼 도입, 기본은 H=1 폴백 유지
4) 파이프라인 DataLoader 분산/캐시/마스킹 전달 강화
5) MLflow 확장 메트릭/게이트 추가
6) 통합 스모크(H=1/H>1, critic/rho1) 실행 및 문서 갱신

---

본 계획은 현 구조를 그대로 활용하며(Registry/Factory/Utils/Loaders/Trainer), 변경은 **트레이너 내부의 CE 정렬/마스킹 정확화**, **스코어러 출력 표준화**, **선택적 모델 래퍼**에 집중한다. 이를 통해 “토큰 중요도 확률화 → CE 가중”의 핵심 요구와 MTP의 k-step 의미를 모두 충족한다.
