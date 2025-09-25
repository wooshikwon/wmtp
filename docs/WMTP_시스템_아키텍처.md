# WMTP 시스템 아키텍처 개요

본 문서는 `WMTP_학술_연구제안서.md`의 방법론을 실제 코드베이스에 구현한 파이프라인을 아키텍처 관점에서 요약합니다. CLI → 설정 로더 → 파이프라인 → 컴포넌트 팩토리 → 레지스트리/트레이너 → 유틸(MLflow, 데이터) 순으로 흐름을 설명합니다.

---

## 1) CLI 엔트리포인트 → 파이프라인 실행

- `wmtp train`/`wmtp eval` 하위 커맨드로 구동되며, 각각 설정을 로드해 통합 파이프라인을 호출합니다.

주요 지점:

```145:153:/Users/wesley/Desktop/wooshikwon/wmtp/src/cli/train.py
        from src.pipelines import run_training  # 통합 훈련 파이프라인
        from src.settings import load_config, load_recipe  # Pydantic 기반 설정 로더

        # YAML 파일들을 Pydantic 모델로 로드 및 검증
        cfg = load_config(config, verbose=verbose)
        rcp = load_recipe(
            recipe, verbose=verbose
        )  # 훈련 레시피 (알고리즘, 하이퍼파라미터)
```

파이프라인 alias:

```6:9:/Users/wesley/Desktop/wooshikwon/wmtp/src/pipelines/__init__.py
# Use run_training_pipeline for all algorithms (mtp-baseline, critic-wmtp, rho1-wmtp)
# The pipeline handles all three cases internally
run_training = run_training_pipeline
run_evaluation = run_evaluation_pipeline
```

---

## 2) 설정 로더와 스키마 검증

- `src/settings/loader.py`가 YAML 로딩, 환경변수 치환, Pydantic 검증을 수행합니다.
- 스키마는 `config_schema.py`(환경/리소스/경로/MLflow/디바이스)와 `recipe_schema.py`(모델/알고리즘/데이터/손실/평가)로 분리.
- 알고리즘 선택은 레시피의 `train.algo`로 제어되며 유효 값은 `baseline-mtp | critic-wmtp | rho1-wmtp` 입니다.

알고리즘 스키마:

```245:247:/Users/wesley/Desktop/wooshikwon/wmtp/src/settings/recipe_schema.py
    algo: Literal["baseline-mtp", "critic-wmtp", "rho1-wmtp"] = Field(
        ..., description="Training algorithm"
    )
```

---

## 3) 통합 훈련 파이프라인(run_training_pipeline)

- 공통 엔진이 단계적으로 컴포넌트를 조립해 학습을 수행합니다.
- 재개 체크포인트, MLflow 추적, 모델/토크나이저/데이터 로딩, 알고리즘별 추가 모델(ref/rm), 옵티마이저, 분산 샘플러, Stage1(critic 전용), 메인 트레이너 실행까지 일관된 순서로 구성됩니다.

시그니처/결과:

```53:63:/Users/wesley/Desktop/wooshikwon/wmtp/src/pipelines/training_pipeline.py
def run_training_pipeline(
    config: Config,
    recipe: Recipe,
    dry_run: bool = False,
    resume_checkpoint: str | Path | None = None,
) -> RunOutputs:
    """WMTP 통합 훈련 파이프라인 - 모든 알고리즘의 메인 실행 함수.

    Returns:
        RunOutputs: 훈련 메트릭이 포함된 결과 객체
```

핵심 단계(요약):
- Step 0~1: 시드 고정, MLflow 실행 시작
- Step 2~4: Base 모델, 토크나이저, 알고리즘별 보조 모델(ref/rm) 로딩
- Step 5: 옵티마이저 구성(학습률/스케줄/클리핑)
- Step 6~9: 데이터 로딩→토크나이징→분산 샘플러→DataLoader
- Step 10: Critic 전용 Stage1(Value Head 사전훈련) 조건 실행
- Step 11~13: 알고리즘별 트레이너 생성/설정/실행 → WMTP 손실 학습
- Step 14: MLflow 종료, 메트릭 반환

---

## 4) 컴포넌트 팩토리와 레지스트리

- 팩토리(`ComponentFactory`)가 설정 기반으로 트레이너/로더/토크나이저/옵티마이저/체크포인트 로더를 생성합니다.
- 실제 구현 클래스 조회는 통합 레지스트리(`UnifiedRegistry`)를 통해 키 기반으로 수행됩니다.

트레이너 생성:

```130:132:/Users/wesley/Desktop/wooshikwon/wmtp/src/factory/component_factory.py
        # Registry에서 Trainer 인스턴스 생성 및 반환
        return trainer_registry.create(recipe.train.algo, trainer_config)
```

통합 레지스트리:

```20:33:/Users/wesley/Desktop/wooshikwon/wmtp/src/components/registry.py
class UnifiedRegistry:
    """
    단일 레지스트리로 모든 카테고리의 컴포넌트를 관리한다.
    
    - 내부 구조: category -> { key -> class }
    - 메타데이터는 (category, key) 단위로 저장
``` 

카테고리 어댑터(호환 인터페이스): `loader_registry`, `trainer_registry`, `optimizer_registry`, `evaluator_registry`, `pretrainer_registry`, `tokenizer_registry`.

---

## 5) 알고리즘별 트레이너(핵심 구현)

세 알고리즘은 공통 베이스 트레이너를 상속하고, 헤드 가중치 계산 방식만 다릅니다. 가중 손실은 공통 함수로 계산합니다(`compute_weighted_mtp_loss`).

- Baseline MTP(균등 가중치):

```33:41:/Users/wesley/Desktop/wooshikwon/wmtp/src/components/trainer/baseline_mtp_trainer.py
@trainer_registry.register("baseline-mtp", category="trainer", version="2.0.0")
class BaselineMtpTrainer(BaseWmtpTrainer):
    """Baseline MTP 트레이너 - 균등 가중치 WMTP 알고리즘.
```

- Critic WMTP(가치 기반): Value Head 예측→TD(δ)→softmax 가중치, Stage1 사전훈련 지원

```42:50:/Users/wesley/Desktop/wooshikwon/wmtp/src/components/trainer/critic_wmtp_trainer.py
@trainer_registry.register("critic-wmtp", category="trainer", version="2.1.0")
class CriticWmtpTrainer(BaseWmtpTrainer):
    """Critic WMTP 트레이너 - 가치함수 델타 기반 WMTP 알고리즘.

    연구 철학 "Not All Tokens Are What You Need"의 강화학습 구현:
```

- Rho‑1 WMTP(참조 비교): Reference CE vs Base CE 정렬→excess loss→softmax 가중치

```36:44:/Users/wesley/Desktop/wooshikwon/wmtp/src/components/trainer/rho1_wmtp_trainer.py
@trainer_registry.register("rho1-wmtp", category="trainer", version="2.0.0")
class Rho1WmtpTrainer(BaseWmtpTrainer):
    """Rho-1 WMTP 트레이너 - Reference 모델 비교 기반 WMTP 알고리즘.

    연구 철학 "Not All Tokens Are What You Need"의 핵심 구현:
```

공통 손실 계산(요지):
- 모델 로짓 `logits[B,S,H,V]`와 타깃 `labels[B,S]`에서 헤드별 CE를 구하고, 헤드 가중치 `[B,S,H]`를 곱해 평균.
- 베이스라인은 상수 1.0 가중, Rho‑1/Critic은 동적으로 산출.

---

## 6) 데이터·토크나이저·분산·체크포인트

- 데이터: `ComponentFactory.create_data_loader()`가 소스에 맞는 경로를 주입하고 통합 로더를 생성합니다.
- 토크나이저: `create_tokenizer()`가 `hf`/`raw`를 선택, 파이프라인에서 데이터 전처리 후 `DataLoader` 구성.
- 분산: `DistributedSampler`를 조건적으로 적용해 멀티 GPU에 배분.
- 체크포인트: `create_checkpoint_loader()`가 재개용 메타데이터(epoch/step/run_id)까지 일괄 로드.

---

## 7) MLflow 추적과 메트릭 로깅

- 파이프라인 시작 시 `create_mlflow_manager`로 run을 시작하고 파라미터/메트릭을 기록합니다.
- 트레이너는 주기적으로 손실/가중치 통계/알고리즘 특화 지표(Critic value, Rho‑1 excess 등)를 로깅합니다.

---

## 8) 평가 파이프라인(run_evaluation_pipeline)

- 체크포인트 로드 → 토크나이저 생성 → 평가 타입 선택 → 타입별 평가기 생성/실행 → 메트릭 집계/로깅.
- 기본 타입은 `meta-mtp`이며 pass@k와 코드 실행 기반 정확도 등을 수집합니다.

```55:66:/Users/wesley/Desktop/wooshikwon/wmtp/src/pipelines/evaluation_pipeline.py
def run_evaluation_pipeline(
    config: Config,
    recipe: Recipe,
    checkpoint_path: Path,
    eval_types: list[str] | None = None,
    dry_run: bool = False,
) -> EvaluationOutputs:
    """WMTP 통합 평가 파이프라인 - Meta 2024 논문 기준 메인 평가 함수.

    training_pipeline.py와 동일한 구조로 설계되어 일관성을 유지합니다.
```

---

## 9) 파이프라인 상의 방법론-구현 정렬(핵심 포인트)

- 토크나이저 일치: Base/Ref/RM와 동일 토크나이저 사용(동일 vocab/특수토큰)을 전제로 비교/가중치를 계산.
- 시점 정렬(Rho‑1): MTP 헤드 k(t→t+k+1) ↔ 참조 t+k(t+k→t+k+1)로 정확히 매칭해 CE 비교.
- Critic 안정화: Stage1(Value Head) 사전훈련 지원 및 Stage2에서 auxiliary value loss(MSE/GAE 기반)를 통해 표현 표류 완화.
- 공통 손실: `L_WMTP = Σ w_{t,k} · CE_k`를 일관된 방식으로 벡터화 계산.

---

## 10) 구성 요약(개념 지도)

- CLI: Typer 기반 `wmtp train|eval` → 설정 로딩 → 파이프라인 호출
- Settings: `load_config`/`load_recipe` + Pydantic 스키마(`Config`, `Recipe`)
- Pipelines: 통합 엔진(훈련/평가) — 단계별 컴포넌트 조립 및 실행
- Factory: 설정 기반 컴포넌트 생성(모델/데이터/토크나이저/옵티마이저/트레이너/프리트레이너)
- Registry: 카테고리별 키 기반 구현 클래스 조회/생성
- Trainers: 알고리즘별 `compute_head_weights` 차별화, `compute_weighted_mtp_loss` 공통 사용
- Utils: MLflow 추적, 시드 고정, 보상 계산 등 보조 기능

이 문서는 구현 흐름을 빠르게 파악하도록 아키텍처 중심으로 요약했으며, 세부 동작은 각 파일의 docstring과 주석(위 인용 코드 위치)에서 바로 확인 가능합니다.