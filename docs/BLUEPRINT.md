# BLUEPRINT.md

본 문서는 **Meta의 Multi-Token Prediction(MTP)** 기반 7B 모델을 **두 가지 실험 파이프라인(critic-weighted, Rho-1-weighted)** 으로 파인튜닝/평가하기 위한 **개발 청사진**입니다. 사용자는 **Python 3.11 + uv**, **MLflow(원격 S3 백엔드)**, **Docker 기반 VESSL GPU 클러스터(A100 호환)**, **로컬/ S3 데이터·모델 양방향 로딩**을 요구하며, **레지스트리·팩토리 구조**로 컴포넌트를 느슨 결합합니다.

---

## 0) 핵심 결정 요약

* **Python/패키지**: Python 3.11, `uv`로 고정(locked)·재현성 보장

  * 핵심 스택: `torch==2.4.*`(CUDA 12.1), `transformers>=4.42,<5`, `accelerate>=0.33`, `peft>=0.11`(LoRA 옵션), `bitsandbytes`(옵션), `flash-attn>=2.5`(옵션), `mlflow>=2.14`, `boto3`, `pydantic>=2.6`, `pyyaml`
* **모델**:

  * Base(MTP): **facebook/multi-token-prediction**(7B\_1T\_4 계열; n\_heads=4 기본), tokenizer 공유
  * Reward Model: **sfair/Llama-3-8B-RM-Reward-Model** (시퀀스 보상 → 토큰 분배 필요)
  * Reference(Rho-1): **Sheared LLaMA 1.3B**(head=1, CE 기준점)
* **파인튜닝 방식**: **Full fine-tuning 기본**, `recipe.yaml`로 **LoRA/QLoRA 옵션** 토글
* **데이터·평가**: **MBPP, CodeContests** 사용. Meta MTP 논문과 동일 프로토콜/지표(예: MBPP exact-match, CodeContests pass\@k) 재현
* **분산/정밀도**: A100(40/80GB) 기준 **bf16** 우선, **FSDP**(Auto Wrap + Activation Checkpointing) 기본. 대규모 시 **ZeRO-3** 대안
* **학습 안정화**: 가중 CE 클립/정규화, warmup cosine 스케줄러, grad clip, loss-scaler(amp)
* **MLflow**: **S3 트래킹/아티팩트** (예: `s3://<bucket>/mlflow`), 실험명 규약 `{proj}/{algo}/{data}`
* **S3 레이아웃**:

  * `s3://<bucket>/datasets/{mbpp,contest}/…`
  * `s3://<bucket>/models/{7b_1t_4,llama3_8b_rm,sheared_llama_1.3b}/…`
  * `s3://<bucket>/mlflow/` (tracking/registry/artifacts)
* **로컬 우선 정책**: 로컬 경로 **존재 시 로컬 사용**, 없으면 S3에서 **미러링 후 캐시**
* **보안/시크릿**: VESSL Secret + 런타임 env(ACCESS\_KEY/SECRET, MLFLOW\_\*), 코드 내 하드코딩 금지

---

## 1) 리포 구조

```
project/
├─ models/                       # (옵션) 로컬 캐시/직접 배치 경로
│  ├─ 7b_1t_4/
│  ├─ Llama_3_8B_RM/
│  └─ sheared_llama_1.3B/
├─ dataset/
│  ├─ mbpp/
│  └─ contest/
├─ src/
│  ├─ cli/                       # uv entrypoint 스크립트 (train/eval)
│  ├─ pipelines/                 # 파이프라인 오케스트레이션 (critic-wmtp / rho1-wmtp)
│  ├─ components/                # 레지스트리 패턴 구성요소
│  │  ├─ loader/
│  │  ├─ scorer/                 # critic / rho-1 등 “토큰 중요도”
│  │  ├─ trainer/
│  │  ├─ optimizer/
│  │  └─ evaluator/
│  ├─ factory/                   # settings→components 빌더
│  ├─ settings/                  # pydantic 스키마 + 로더
│  └─ utils/                     # s3, mlflow, hf, fsdp 등 공통 유틸
├─ configs/
│  ├─ config.example.yaml        # 환경 설정(스토리지, 경로, mlflow, launcher)
│  └─ recipe.example.yaml        # 모델/학습 하이퍼/옵션
├─ docker/
│  ├─ Dockerfile
│  └─ vessl.yaml                 # VESSL 스펙
├─ tests/                        # 단위/스모크/E2E
└─ README.md
```

---

## 2) 설정 스키마 (pydantic) & YAML 예시

### 2.1 `config.yaml` (환경)

* 필수: `storage.mode` in {`local`,`s3`}, `paths.*`, `mlflow.tracking_uri`, `mlflow.registry_uri`, `launcher`(local/vessl), `devices`
* 제약: MLflow는 **S3 백엔드** 사용(트래킹/아티팩트 모두 S3). S3 사용 시 `aws.region`·버킷 필수

**예시**

```yaml
project: "mtp_ft"
seed: 42

storage:
  mode: "local"                # local | s3
  s3:
    bucket: "wmtp-artifacts"
    region: "ap-northeast-2"
    prefix: "mtpfw/"
paths:
  models:
    base_local: "models/7b_1t_4"
    rm_local: "models/Llama_3_8B_RM"
    ref_local: "models/sheared_llama_1.3B"
  datasets:
    mbpp_local: "dataset/mbpp"
    contest_local: "dataset/contest"
  cache: ".cache"

mlflow:
  experiment: "mtp/wmtp"
  tracking_uri: "s3://wmtp-artifacts/mlflow"
  registry_uri: "s3://wmtp-artifacts/mlflow"

launcher:
  target: "vessl"              # local | vessl
  resources:
    gpus: 4
    gpu_type: "A100"
    cpus: 32
    memory_gb: 256
    disk_gb: 500

devices:
  mixed_precision: "bf16"
  fsdp:
    enabled: true
    auto_wrap: true
    activation_ckpt: true
    sharding: "full"           # full | shard-grad-op (대안)
```

### 2.2 `recipe.yaml` (모델/학습)

* 필수: 모델 식별자, MTP 설정, 학습/평가 하이퍼, 스코어러(critic|rho1), 로라 옵션
* 제약: **base는 MTP 사전학습 모델이어야 함**, `mtp.n_heads=4` 기본

**예시**

```yaml
run:
  name: "critic-wmtp_mbpp"
  tags: ["critic", "wmtp", "mbpp"]

model:
  base_id: "facebook/multi-token-prediction"
  rm_id: "sfair/Llama-3-8B-RM-Reward-Model"
  ref_id: "ShearedLlama-1.3B"         # 정확 HF id로 교체
  tokenizer_pad_side: "right"
  mtp:
    n_heads: 4
    horizon: 4                          # 예: k=1..4

train:
  algo: "critic-wmtp"                   # critic-wmtp | rho1-wmtp
  full_finetune: true
  lora:
    enabled: false                      # true시 아래 적용
    r: 16
    alpha: 32
    dropout: 0.05
    target_modules: ["q_proj","v_proj","o_proj","k_proj","gate_proj","up_proj","down_proj"]

optim:
  optimizer: "adamw"
  lr: 1.2e-5
  weight_decay: 0.1
  betas: [0.9, 0.95]
  grad_clip: 1.0
  scheduler: "cosine"
  warmup_ratio: 0.03

data:
  train:
    sources: ["mbpp"]
    max_length: 2048
    pack_sequences: true
  eval:
    sources: ["mbpp","contest"]
    batch_size: 8

batching:
  global_batch_tokens: 4_000_000        # token-based batching 권장
  micro_batch_size: 1
  grad_accum_steps: 64

loss:
  weight_norm: "mean1.0_clip"           # 가중치 평균 1.0 유지+클립
  lambda: 0.3                           # 가중치 강도
  temperature: 0.7                      # softmax 온도 (critic/rho1 공통)

critic:                                  # critic 단계 세부
  target: "rm_sequence"                  # RM 시퀀스 점수
  token_spread: "gae"                    # rm→토큰 분배: {uniform, length, attention, gae}
  delta_mode: "td"                       # δ_t = V_t - V_{t-1}
  normalize: "zscore"                    # 가중치 안정화

rho1:
  score: "abs_excess_ce"                 # |CE_ref - CE_base|
  percentile_top_p: 0.2                  # 상위 20% 강조(soft)
  refresh_per_epoch: false               # 대규모면 false 권장

eval:
  protocol: "meta-mtp"                   # 고정
  sampling:
    temperature: 0.2
    top_p: 0.95
    n: 1
  metrics:
    - "mbpp_exact"
    - "contest_pass@1"
    - "contest_pass@5"
```

---

## 3) 레지스트리·팩토리 설계

### 3.1 공통 인터페이스

```python
class Component(Protocol):
    def setup(self, ctx: dict) -> None: ...
    def run(self, ctx: dict) -> dict: ...   # 입력 ctx → 산출물 dict

class Registry:
    def __init__(self): self._reg = {}
    def register(self, name): ...
    def get(self, name): ...
```

* `components/loader`: 데이터·모델 로더(로컬 우선 → S3 미러링)
* `components/scorer`: **critic\_delta\_v1**, **rho1\_excess\_v1** 등 토큰 가중치 산출
* `components/trainer`: **mtp\_weighted\_ce\_trainer** (FSDP/AMP/grad-accum 포함)
* `components/optimizer`: AdamW, Lion(옵션) 등
* `components/evaluator`: MBPP/CodeContests 프로토콜

### 3.2 팩토리

* `factory/component_factory.py`: Pydantic 검증된 설정 → 레지스트리 키 기반 컴포넌트 인스턴스화
* **선택 규칙**: `recipe.train.algo`에 따라 scorer 선택, `full_finetune/lora.enabled`에 따라 트레이너 내부 경로 분기

---

## 4) 파이프라인 정의

### 4.1 Critic-Weighted MTP (2-Stage, 선택/옵션)

1. **Critic 학습(값함수 회귀)**

* 목적: RM의 시퀀스 보상 $R$을 토큰 단계 가치 $V_t$로 근사
* 분배: `critic.token_spread`(권장: **GAE** 스타일)로 $R\to r_t$ 분해
* 학습: $\min \sum_t (V_\theta(h_t) - \hat{V}_t)^2$
* 산출: 각 토큰 t에 대해 **δ\_t = V\_t − V\_{t−1}** (문장 시작은 $V_{-1}=0$)

2. **MTP Weighted CE**

* 각 **미래 k(1..H)** 에 대해 CE\_k(t) 계산 (MTP 4-head)
* **가중치 $w_t = \mathrm{softmax}(\delta_t / T)$**, 안정화로 z-score → softmax, **평균 1.0 정규화 + \[ε, W\_max] 클립**
* 최종손실: $\mathcal{L} = \lambda \cdot \sum_t w_t \cdot \frac{1}{H}\sum_{k=1}^H CE_k(t)$
* 기본: $H=4$, $\lambda=0.3$, $T=0.7$, **bf16**, **FSDP**, **cosine + warmup 3%**

### 4.2 Rho-1 (Reference-Weighted MTP, 기본/Default)

* **점수**: $s_t = |CE^{ref}_t - CE^{base}_t|$ (abs\_excess\_ce)
* **정규화**: z-score → softmax(T) → 평균 1.0 정규화 + 클립
* **선택 강화**: 상위 p(예: 20%)의 상대 가중 상승(연속 가중; hard drop은 비권장)
* **손실**: critic-WMTP와 동일 형태로 가중 CE 적용
* **리프레시**: 데이터 거대 시 **사전 프리컴퓨트 고정**, 소규모면 epoch-별 리프레시 옵션

> 두 파이프라인 모두 **동일 Trainer**를 재사용하고 **Scorer**만 교체되도록 설계(모듈 동형).

---

## 5) 학습·분산·안정화 정책

* **분산**: 기본 **FSDP(full sharding)** + Activation Checkpointing; 대체로 **ZeRO-3**도 선택 가능
* **정밀도**: **bf16**(A100 최적), 필요 시 fp16로 폴백
* **배치**: **token-based global budget**(권장) + grad accumulation로 VRAM 경계 내 운용
* **스케줄러**: `cosine` + warmup ratio 0.03
* **Grad Clip**: 1.0
* **정규화**: **가중치 평균 1.0 유지**, ε=0.05\~0.1 클립, W\_max=3.0 권장
* **검증**: step/epoch 마다 eval, **early stopping** 선택
* **체크포인트**: last + best(val metric) + periodic(k steps) 보존

---

## 6) 데이터 로딩 & 캐싱

* **로컬 우선**: `paths.datasets.*` 존재하면 사용, 아니면 **S3 → 로컬 캐시(.cache)**
* **해시 키**: (데이터 버전 + 전처리 옵션 + 스플릿 시드) → 캐시 디렉터리 이름화
* **스플릿**: 고정 시드(42), train/val/test 비중 논문 재현 기준
* **디컨태미네이션**: 중복/리드미/해설 누출 제거(간단 룰 포함)

---

## 7) 평가 하네스 (Meta MTP 프로토콜 재현)

* **MBPP**: prompt 규격화 → 단일/다중 시도, **exact-match** 기준
* **CodeContests**: pass\@k(k∈{1,5}) 및 표준 샘플링 파라미터(**T=0.2, top-p=0.95**)
* **로그**: MLflow에 샘플 결과(정답/예측) 일부, pass/fail 분포, head별 CE 통계
* **추가**: MTP-head별 에러 기여도 히스토그램

---

## 8) CLI

```
uv run python -m src.cli.train --config configs/config.yaml --recipe configs/recipe.yaml \
  --run-name wmtp_critic_mbpp

uv run python -m src.cli.eval  --config configs/config.yaml --recipe configs/recipe.yaml \
  --checkpoint runs/<id>/best.pt
```

* `train` 옵션: `--resume`, `--tags`, `--dry-run`
* `eval` 옵션: `--datasets mbpp,contest`, `--save-report`

> 구현 매핑: `src/cli/train.py`는 `src/pipelines/training.py:TrainingPipeline`을 사용하고, `src/settings` 로더로 YAML을 검증한다. `src/cli/eval.py`는 `ComponentFactory`를 통해 `meta-mtp-evaluator`를 구성하도록 연결(개선 과제)하며, `dataset-mbpp-loader`/`dataset-contest-loader`와 `hf-local-s3-loader`를 재사용한다.

---

## 9) MLflow 표준

* **experiment**: `mtp/{critic-wmtp|rho1-wmtp}/{mbpp|contest}`
* **params**: 모델 id, n\_heads, λ, T, optimizer, lr, batch, FSDP 설정, data hash 등
* **metrics**: train/loss, val/loss, **mbpp\_exact**, **contest\_pass\@1,5**, throughput(tok/s)
* **artifacts**: 체크포인트, 샘플 예측, 중요토큰 가중치 통계, `reports/{run_id}.md`
* **registry**: **“staging”→“production”** 전환 가능(모델 카드 기록)

---

## 10) S3 & 시크릿

* **자격증명**: VESSL Secret → 런타임 env(`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`)
* **엔드포인트**: MLflow `tracking_uri/registry_uri = s3://…`
* **폴더**: 위 “S3 레이아웃” 고정. **코드 내 키 하드코딩 금지**

---

## 11) Docker & VESSL 스펙

### 11.1 Dockerfile (요지)

```dockerfile
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9

RUN apt-get update && apt-get install -y git gcc && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace

# uv 설치
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

COPY . .
ENV PYTHONUNBUFFERED=1 TRANSFORMERS_NO_ADVISORY_WARNINGS=1
```

### 11.2 VESSL (요지)

```yaml
run:
  gpu: { type: A100, count: 4 }
  cpu: 32
  memory: 256Gi
  disk: 500Gi
  image: <built-image>
  env:
    - name: AWS_DEFAULT_REGION
      value: ap-northeast-2
    - name: MLFLOW_TRACKING_URI
      value: s3://wmtp-artifacts/mlflow
    - name: MLFLOW_S3_ENDPOINT_URL
      value: https://s3.<region>.amazonaws.com
  secrets:   # VESSL secret 이름 참조
    - AWS_ACCESS_KEY_ID
    - AWS_SECRET_ACCESS_KEY
  command:
    - uv run python -m src.cli.train --config configs/config.yaml --recipe configs/recipe.yaml
```

---

## 12) 유틸 모듈 원칙 (`src/utils/`)

* **s3.py**: download\_if\_missing(path\_spec) / upload\_artifact / exists / etag
* **mlflow\.py**: init(experiment, run\_name, tags), log\_params/metrics/artifacts, auto-resume
* **hf.py**: 안전한 `from_pretrained`(로컬 우선→S3→HF), tokenizer resize 규칙, (선택) **MTPWrapper** 설계 지침: [B,S,V] 출력 모델에 대해 H>1 요청 시 teacher-forcing 기반 k-step 로짓 생성. 기본은 H=1 폴백.
* **dist.py**: FSDP 초기화/auto wrap/ckpt, seed 설정, throughput 측정
* **eval.py**: MBPP/Contest 드라이버(프로토콜 캡슐화)

> *금지*: 다른 디렉터리에서 임시 유틸 정의 금지. 모든 공통 함수를 **utils** 로 집중.

### 구현 매핑 요약

* Registry 키는 `kebab-case`로 통일: 예) `critic-delta-v1`, `rho1-excess-v1`, `mtp-weighted-ce-trainer`
* Factory: `factory/component_factory.py`에서 알고리즘→스코어러, 스케줄러/옵티마이저/로더 선택 및 Mock 폴백
* Pipeline: `pipelines/training.py` 단일 파이프라인이 critic/rho1 모두 지원 (algo에 의해 scorer 교체)
* Evaluator: `components/evaluator/{mbpp_eval.py,codecontests.py,meta_mtp.py}` 구현됨. `cli/eval.py` 연결 작업 예정

---

## 13) 최소 테스트 & 재현성

* **스모크 E2E**: tiny 모델(예: 100M dummy) + 100 스텝, 두 파이프라인이 끝까지 돈다
* **단위**: pydantic 스키마, scorer(critic/rho-1) 산출 통계, 가중치 정규화 불변식(평균=1.0±ε)
* **결정론**: seed=42, `torch.use_deterministic_algorithms(False)` (성능 선호), 주요 난수 생성기 고정

---

## 14) 위험요인 & 완충책

* **RM→토큰 분배**: GAE 가중 시 길이 의존성 과도 증가 → **온도/클립**으로 제어
* **가중 CE 폭주**: 상위 토큰 과도 가중 → **mean-norm + \[ε,W\_max] 클립 + λ 스윕**
* **OOM**: FSDP + activation ckpt + token-budget 감소, grad-accum 증가로 대응
* **데이터 누출**: 평가셋 파일명/정답 텍스트 학습 제외 룰 체크

---

## 15) 로드맵 (권장 순서)

1. **스켈레톤 생성**(레지스트리·팩토리·pydantic·CLI)
2. **로컬 tiny 스모크**(critic/rho-1 scorer mock → weighted CE Trainer → MLflow 로깅)
3. **S3/MLflow/캐시** 안정화
4. **실데이터(MBPP) 단일 GPU 실험**으로 end-to-end
5. **VESSL A100×4** 스케일링(FSDP)
6. **평가 하네스** 정확도 점검, Meta 프로토콜 매칭
7. **하이퍼 스윕**: λ∈{0.1,0.3,1.0}, T∈{0.5,0.7,1.0}, percentile∈{0.1,0.2,0.3}

---

## 부록 A) 내부 컴포넌트 이름 규약(권장)

* loader: `hf-local-s3-loader`, `dataset-mbpp-loader`, `dataset-contest-loader`
* scorer: `critic-delta-v1`, `rho1-excess-v1`
* trainer: `mtp-weighted-ce-trainer`
* optimizer: `adamw-bf16-fused`
* evaluator: `mbpp-v1`, `codecontests-v1`, `meta-mtp-evaluator`

---

## 부록 B) Pydantic(요지)

```python
class Config(BaseModel):
    project: str
    seed: int = 42
    storage: Storage
    paths: Paths
    mlflow: Mlflow
    launcher: Launcher
    devices: Devices
    model_config = ConfigDict(extra="forbid")

class Recipe(BaseModel):
    run: Run
    model: Model
    train: Train
    optim: Optim
    data: Data
    batching: Batching
    loss: Loss
    critic: Critic
    rho1: Rho1
    eval: Eval
    @field_validator("model")
    def validate_base_is_mtp(cls, v): ...
```

---

## 마무리

이 청사진은 **연구 제안서(critic-weighted MTP)** 와 **연구 개선안(Rho-1 기반 critic-free 가중)** 두 축을 **동일 파이프라인 프레임**에서 스위치로 운용하도록 설계했습니다.

* **Full FT** 기본, **LoRA 옵션**으로 비용 완화
* **MLflow on S3**, **A100·FSDP**, **Meta MTP 평가 재현**
* **로컬/S3 이중 소스**와 **유틸 일원화**
