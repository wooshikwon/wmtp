# Weighted Multi-Token Prediction (WMTP) 파인튜닝 프레임워크

## 1. 시스템 아키텍처

### 파이프라인 흐름

WMTP 프레임워크는 CLI에서 실행까지 깔끔한 모듈식 아키텍처를 따릅니다:

```
CLI 진입점 → 파이프라인 오케스트레이션 → 팩토리 생성 → 레지스트리 조회 → 컴포넌트 실행
```

#### 상세 흐름

```python
# 1. CLI 진입점
src/cli/train.py --config configs/config.yaml --recipe configs/recipe.yaml
         ↓
# 2. 파이프라인 오케스트레이션
TrainingPipeline.run()
    - Pydantic 스키마를 통한 설정 로드 및 검증
    - MLflow 실험 생성
    ↓
# 3. 팩토리 패턴
ComponentFactory.create_*()
    - create_model_loader()
    - create_scorer(algo="critic-wmtp" 또는 "rho1-wmtp")
    - create_trainer()
    - create_optimizer()
    ↓
# 4. 레지스트리 시스템
@scorer_registry.register("critic-delta-v1")
@scorer_registry.register("rho1-excess-v1")
@trainer_registry.register("mtp-weighted-ce-trainer")
    ↓
# 5. 컴포넌트 실행
component.setup(ctx) → component.run(ctx) → outputs
```

#### 핵심 설계 원칙

- **레지스트리 패턴**: 모든 컴포넌트는 동적 조회를 위해 고유 키로 등록
- **팩토리 패턴**: `create_*` 메서드만을 통한 중앙화된 생성 로직
- **컨텍스트 전달**: 컴포넌트는 컨텍스트 딕셔너리를 통해 통신
- **설정 기반**: `recipe.train.algo` 필드를 통한 알고리즘 선택

## 2. 학습 방법론

이 프레임워크는 Multi-Token Prediction 모델을 위한 세 가지 학습 접근법을 구현하고 평가합니다:

### 2.1 기준선 MTP 파인튜닝

가중치 없는 표준 Multi-Token Prediction으로 기준선 역할을 합니다.

**손실 함수:**
```
L_MTP = -E[Σ(k=1 to H) log P_θ(x_{t+k} | x_{<t})]
```

여기서:
- `H`: 예측 범위 (예측 헤드 수, 기본값=4)
- `P_θ`: 모델 확률 분포
- `x_{t+k}`: 위치 t+k의 토큰

### 2.2 Critic 가중 MTP (2단계 접근법)

학습된 가치 함수를 사용하여 토큰 중요도를 결정하는 2단계 접근법입니다.

#### 1단계: 가치 함수 학습
```
L_VF = Σ_t (V_θ(h_t) - V̂_t)²
```

여기서:
- `V_θ(h_t)`: 은닉 상태 h_t에서 예측된 가치
- `V̂_t`: GAE를 통한 보상 모델의 목표 가치

#### 2단계: Delta를 통한 가중 CE
```
δ_t = V_t - V_{t-1}  (V_{-1} = 0)
w_t = softmax(δ_t / T)
L_WMTP = Σ(k=0 to H-1) w_{t+k} × CE(y_{t+k}, ŷ_{t+k})
```

여기서:
- `δ_t`: 시간차 (중요도 신호)
- `T`: 소프트맥스 온도 (기본값=0.7)
- `w_{t+k}`: 토큰 t+k의 정규화된 가중치
- `CE`: 교차 엔트로피 손실

### 2.3 Rho-1 참조 가중 MTP

Critic 없이 참조 모델 기반 접근법입니다 (권장).

**점수 계산:**
```
s_t = |CE^{ref}_t - CE^{base}_t|
```

**가중치 생성:**
```
w_t = normalize(s_t) × I(s_t > percentile_p) + α
w_t = softmax(w_t / T) × (1.0 / mean(w_t))
```

**최종 손실:**
```
L_Rho1 = Σ(k=0 to H-1) w_{t+k} × CE_k(y_{t+k}, ŷ_{t+k})
```

여기서:
- `CE^{ref}`: 참조 모델(CodeLlama-7B)의 교차 엔트로피
- `CE^{base}`: 기본 MTP 모델의 교차 엔트로피
- `percentile_p`: 강조할 상위 백분위수 (기본값=0.15)
- `α`: 제로 가중치 방지를 위한 기본 가중치

#### 가중치 정규화 파이프라인

모든 방법은 다음 정규화를 적용합니다:
1. Z-점수: `(w - μ) / σ`
2. 소프트맥스: `softmax(w / T)`
3. 평균 정규화: `w × (1.0 / mean(w))`
4. 클리핑: `clip(w, ε, W_max)` (ε=0.05, W_max=3.0)
5. 평균=1.0 유지를 위한 재정규화

## 3. VESSL 배포 가이드

### 3.1 사전 준비

1. **Docker 이미지 빌드**
```bash
# Docker 이미지 빌드
cd docker
docker build -t wmtp:latest -f Dockerfile ..

# 레지스트리용 태그 지정
docker tag wmtp:latest <your-registry>/wmtp:latest

# 레지스트리에 푸시
docker push <your-registry>/wmtp:latest
```

2. **VESSL 시크릿 설정**

VESSL UI에서 다음 시크릿 추가:
- `AWS_ACCESS_KEY_ID`: AWS 액세스 키
- `AWS_SECRET_ACCESS_KEY`: AWS 시크릿 키
- `HF_TOKEN`: 모델 다운로드용 HuggingFace 토큰

### 3.2 학습 실행

#### 단계 1: VESSL YAML 설정

`docker/vessl.yaml` 업데이트:
```yaml
image: <your-registry>/wmtp:latest
resources:
  cluster: vessl-gcp-oregon  # 클러스터 이름
  preset: v1-a100-4-pod      # 4x A100 GPU
```

#### 단계 2: 학습 방법 선택

레시피 파일 선택:
- **기준선 MTP**: 표준 설정으로 레시피 생성
- **Critic 가중**: `configs/recipe.critic.yaml` 사용
- **Rho-1 (권장)**: `configs/recipe.rho1.yaml` 사용

#### 단계 3: 학습 시작

```bash
# VESSL에 제출
vessl run create -f docker/vessl.yaml

# 또는 vessl.yaml의 command 수정:
command: |
  # Critic 가중 학습
  uv run python -m src.cli.train \
    --config configs/config.vessl.yaml \
    --recipe configs/recipe.critic.yaml

  # Rho-1 학습 (권장)
  uv run python -m src.cli.train \
    --config configs/config.vessl.yaml \
    --recipe configs/recipe.rho1.yaml
```

### 3.3 CLI 명령어

#### 학습
```bash
uv run python -m src.cli.train \
  --config configs/config.yaml \
  --recipe configs/recipe.yaml \
  --run-name "실험명" \
  --tags "tag1,tag2" \
  --max-steps 10000 \
  --dry-run  # 선택사항: 학습 없이 검증만
```

#### 평가
```bash
uv run python -m src.cli.eval \
  --config configs/config.yaml \
  --recipe configs/recipe.yaml \
  --checkpoint /path/to/checkpoint.pt \
  --datasets "mbpp,contest"
```

### 3.4 설정 파일

#### config.vessl.yaml
```yaml
storage:
  mode: "s3"
  s3:
    bucket: "wmtp"
    region: "eu-north-1"

mlflow:
  tracking_uri: "s3://wmtp/mlflow"
  registry_uri: "s3://wmtp/mlflow"

launcher:
  target: "vessl"
  resources:
    gpus: 4
    gpu_type: "A100"
```

#### recipe.rho1.yaml (권장 설정)
```yaml
train:
  algo: "rho1-wmtp"

model:
  base_id: "facebook/multi-token-prediction"
  ref_id: "codellama/CodeLlama-7b-Python-hf"

loss:
  lambda: 0.5       # 가중치 강도
  temperature: 0.5  # 더 날카로운 분포

rho1:
  percentile_top_p: 0.15  # 상위 15% 강조
```

### 3.5 모니터링

MLflow에서 실험 추적:
```python
# 실험은 다음 체계를 따름
experiment_name = "mtp/{algo}/{dataset}"
# 예: "mtp/rho1-wmtp/mbpp"

# 기록되는 주요 메트릭
- train/loss
- val/loss
- mbpp_exact_match
- contest_pass@1
- contest_pass@5
- weight_stats/mean
- weight_stats/std
```

### 3.6 예상 학습 흐름

1. **초기화** (5-10분)
   - HuggingFace/S3에서 모델 다운로드
   - 분산 학습 설정
   - MLflow 실험 초기화

2. **학습 루프**
   - Critic: 1단계 (가치 학습) → 2단계 (가중 CE)
   - Rho-1: 직접 가중 CE 학습
   - N 스텝마다 체크포인트

3. **평가** (학습 후 자동)
   - MBPP 정확 일치 평가
   - CodeContests pass@k 메트릭
   - 마크다운 보고서 생성

4. **저장된 아티팩트**
   - 모델 체크포인트: `s3://wmtp/models/`
   - MLflow 메트릭: `s3://wmtp/mlflow/`
   - 평가 보고서: `s3://wmtp/reports/`

## 빠른 시작

```bash
# 1. 저장소 복제
git clone <repo-url>
cd wmtp

# 2. 의존성 설치
uv sync --frozen

# 3. 환경 설정
cp configs/config.example.yaml configs/config.local.yaml
# config.local.yaml 편집

# 4. 로컬 실행 (테스트)
uv run python -m src.cli.train \
  --config configs/config.local.yaml \
  --recipe configs/recipe.rho1.yaml \
  --dry-run

# 5. VESSL 배포 (프로덕션)
vessl run create -f docker/vessl.yaml
```

## 모델 다운로드

필요한 모델은 자동으로 다운로드됩니다:
- **기본 MTP**: `facebook/multi-token-prediction` (7B, 4 헤드)
- **참조 모델**: `codellama/CodeLlama-7b-Python-hf` (Rho-1용)
- **보상 모델**: `sfair/Llama-3-8B-RM-Reward-Model` (Critic용)

## 라이선스

MIT 라이선스 - 자세한 내용은 LICENSE 파일 참조
