# PHASED\_DELIVERY\_PLAN.md

*Zero-to-Production Plan for WMTP Fine-Tuning Framework (Git · uv · Docker/VESSL · MLflow/S3)*

본 문서는 **제로베이스에서 레포 생성 → 인프라/코드 스켈레톤 → 파이프라인 완성 → 스케일/운영**까지의 전 과정을 **페이즈(Phase)** 단위로 쪼개어 실행·검증·전달물을 정의합니다. 모든 항목은 이미 합의된 **BLUEPRINT**와 **DEV PRINCIPLES**를 기준으로 정합성을 유지합니다. &#x20;

---

## Phase 0 — Repository & Governance Bootstrap

**목표**: Git 규약·브랜치 전략·기본 자동화 장치 수립

* **작업**

  * 초기 레포 생성: `git init && git add -A && git commit -m "chore: bootstrap repo"`
  * 브랜치 전략: `main`(보호) / `develop` / `feature/*` / `exp/*`
  * 템플릿 추가

    * `README.md`, `CONTRIBUTING.md`, `CODEOWNERS`, `LICENSE`
    * 이 문서(`PHASED_DELIVERY_PLAN.md`), `BLUEPRINT.md`, `DEV_PRINCIPLES.md`
  * 린트/포맷: `ruff` + `pyproject.toml` 규칙 고정 (pre-commit 훅)
  * GitHub Actions (선택): Lint/Test 스모크 워크플로우
* **전달물**

  * 보호 브랜치 정책, 기본 워크플로우(YAML), 프로젝트 규약 문서
* **Exit Criteria (DoD)**

  * PR이 아닌 직접 push 차단(보호 규칙)
  * CI에서 ruff + pytest 스모크 성공
* **근거**: 명명·스타일·테스트 규율은 DEV PRINCIPLES와 정합&#x20;

---

## Phase 1 — Environment & Package Management (uv) + Project Skeleton

**목표**: Python 3.11 + uv 고정, 디렉토리 트리/엔트리포인트 구성

* **작업**

  * `pyproject.toml`/`uv.lock` 생성 후 의존성 핀:
    `torch 2.4.*, transformers 4.42+, accelerate, peft, mlflow, boto3, pydantic v2, pyyaml` 등
  * 디렉토리 스켈레톤 생성 (BLUEPRINT 기준):

    ```
    src/{cli,pipelines,components/{loader,scorer,trainer,optimizer,evaluator},factory,settings,utils}
    configs/, docker/, tests/
    ```
  * 엔트리포인트: `src/cli/train.py`, `src/cli/eval.py` (argparse/typer)
  * 러너 통일: `uv run python -m src.cli.train ...`
* **전달물**

  * `pyproject.toml`, `uv.lock`, 기본 `__init__.py`들, 빈 CLI 스텁
* **DoD**

  * `uv sync --frozen` 성공, `uv run`으로 CLI 헬프 출력
* **근거**: 언어/패키지 스택·트리 구조는 BLUEPRINT 합치&#x20;

---

## Phase 2 — Settings Schema (Pydantic) & YAML Contracts

**목표**: `config.yaml`(환경)·`recipe.yaml`(학습) 스키마 및 검증

* **작업**

  * `src/settings/{config_schema.py,recipe_schema.py,loader.py}`

    * 제약: MTP base·algo 값·가중치 정규화 정책 등 검증
  * 샘플: `configs/{config.example.yaml,recipe.example.yaml}`
* **전달물**

  * 스키마/밸리데이터 테스트: `tests/settings/test_schemas.py`
* **DoD**

  * 잘못된 YAML 실패 시 명확한 에러 메시지
* **근거**: 설정 주도·검증 규율은 두 문서 공통 원칙 &#x20;

---

## Phase 3 — Registry & Factory

**목표**: 컴포넌트 플러그인 구조와 빌더 도입(교체 가능성 확보)

* **작업**

  * `src/components/registry.py` (register/get)
  * 공통 인터페이스(Protocol) 정의, `factory/`에서 settings→instances
* **전달물**

  * 유닛테스트: 등록/미등록 키, 인터페이스 준수
* **DoD**

  * YAML 키 변경만으로 컴포넌트 교체 가능
* **근거**: 레지스트리·팩토리 설계와 정합&#x20;

---

## Phase 4 — Utils Consolidation (S3 · HF · MLflow · Dist/Eval)

**목표**: 모든 외부 IO/런타임 유틸을 `src/utils/`로 집약

* **작업**

  * `utils/{s3.py, hf.py, mlflow.py, dist.py, eval.py}`
  * 기능: 로컬우선→S3 미러링, 안전한 `from_pretrained`, MLflow 초기화/로그, FSDP init, 평가 드라이버
* **전달물**

  * 목킹 기반 유닛테스트(네트워크 없이), 로컬 캐시 동작 검증
* **DoD**

  * **utils 외부에서** S3/HF/MLflow 호출 없음(검색 검사)
* **근거**: 유틸 단일화 원칙과 일치&#x20;

---

## Phase 5 — Data & Model Loaders

**목표**: 로컬/캐시/S3 모델·데이터 로더 구현

* **작업**

  * `components/loader/{dataset_mbpp_loader.py, dataset_contest_loader.py}`
  * `components/loader/hf_local_s3_loader.py` (base/rm/ref 모델)
  * 데이터 해시 키·캐시 경로 규약
* **전달물**

  * 스플릿/캐시/버전 키 테스트
* **DoD**

  * 로컬만/ S3만/ 혼합 상황 모두 통과
* **근거**: 로딩 정책·해시·스플릿 시드 규정 준수&#x20;

---

## Phase 6 — Scorers (Critic & Rho-1)

**목표**: 토큰 중요도 산출기 2종 구현(동일 인터페이스)

* **작업**

  * `scorer/critic_delta_v1.py`: RM 시퀀스 보상→토큰 분배(GAE)→V\_t 회귀→δ\_t→정규화(softmax T, mean=1, clip)
  * `scorer/rho1_excess_v1.py`: |CE\_ref−CE\_base|→정규화→상위 p% 연속 가중 강화
  * 가중 평균 1.0±ε 불변식 보장
* **전달물**

  * 통계 테스트: finite/mean/clip 범위 검증
* **DoD**

  * 두 스코어러 교체해도 Trainer에 무변경 적용
* **근거**: 알고리즘·정규화·상한/하한 규정에 합치&#x20;

---

## Phase 7 — Trainer (MTP Weighted CE) + Optimizer

**목표**: 단일 Trainer가 여러 스코어러와 동작, FSDP·AMP·스케줄러·Grad Clip 포함

* **작업**

  * `trainer/mtp_weighted_ce_trainer.py`: H=4 head CE 평균×토큰가중
  * Optimizer/Scheduler 팩토리: AdamW, cosine, warmup 3%, grad clip 1.0
  * Full FT 기본, `recipe.train.lora.enabled` 시 PEFT 분기
* **전달물**

  * 단계적 손실·가중치/헤드별 로깅(MLflow)
* **DoD**

  * NaN/OOM 방지 및 폴백(배치/정밀도/accum) 로직 동작
* **근거**: 손실식·안정화·스케줄러 규정 준수&#x20;

---

## Phase 8 — Pipelines Orchestration

**목표**: `pipelines/{critic_wmtp.py, rho1_wmtp.py}`로 엔드투엔드 조립

* **작업**

  * settings→factory→{loaders, scorer, trainer, evaluator} 연결
  * 공통 컨텍스트(ctx) 규약, 단계별 MLflow 로깅
* **전달물**

  * 파이프라인 스모크: tiny 모델로 100 step 학습/평가
* **DoD**

  * 두 파이프라인 모두 스모크 통과
* **근거**: 오케스트레이션 책임 범위 원칙&#x20;

---

## Phase 9 — Evaluation Harness (Meta MTP Protocol)

**목표**: MBPP exact, CodeContests pass\@k 지표 재현

* **작업**

  * `evaluator/{mbpp_eval_v1.py, contest_eval_v1.py}`
  * 샘플링 파라미터 고정(T=0.2, top-p=0.95)
* **전달물**

  * 예측/정답 일부, head별 CE 통계, 가중치 분포 아티팩트
* **DoD**

  * 리그레션 테스트(샘플 세트 기준 동일 결과 재현)
* **근거**: BLUEPRINT 평가 규정 준수&#x20;

---

## Phase 10 — MLflow (S3) Integration & Experiment Taxonomy

**목표**: 실험 규약/아티팩트 구조/레지스트리 플로우 확립

* **작업**

  * `utils/mlflow.py` 통합: experiment `mtp/{algo}/{dataset}`, params/metrics/artifacts 기록
  * 아티팩트 표준: `checkpoints/{best,last,periodic}`, `reports/{run_id}.md`
* **전달물**

  * MLflow UI 기준 실험 트리 확인 스크린샷(문서화)
* **DoD**

  * 로컬·VESSL 실행 모두 동일 구조로 기록
* **근거**: MLflow 표준/아티팩트 규정 합치&#x20;

---

## Phase 11 — Dockerization (CUDA 12.1) & VESSL Spec

**목표**: A100 호환 이미지·런 배포 스펙 확정

* **작업**

  * `docker/Dockerfile`(pytorch/pytorch:2.4.0-cuda12.1-cudnn9), uv 설치, frozen sync
  * `docker/vessl.yaml`: GPU/CPU/Mem/Disk/Secrets/command
  * 이미지 빌드/푸시 스크립트
* **전달물**

  * `make image && make push` 또는 `scripts/build_push.sh`
* **DoD**

  * VESSL에서 `uv run python -m src.cli.train ...` 정상 동작
* **근거**: Docker/VESSL 요건 준수&#x20;

---

## Phase 12 — Test Pyramid & Quality Gates

**목표**: 유닛·통합·스모크·(선택)E2E 구축, 속도/결정론성 가이드 준수

* **작업**

  * `tests/` 구성: settings/scorer/utils/trainer/pipelines/evaluator
  * 마커: `@pytest.mark.slow` / 기본 3분 내 테스트
  * 가중치 정규화 불변식 테스트, OOM/NaN 시뮬레이션
* **전달물**

  * CI 연동, 커버리지 리포트
* **DoD**

  * main 합류 전 필수 테스트·린트 통과
* **근거**: 테스트 전략·불변식 규정 합치&#x20;

---

## Phase 13 — First Real Run (Local → Single GPU → VESSL x4 A100)

**목표**: 실제 MBPP subset → MBPP full → CodeContests까지 점진적 확장

* **작업**

  * `exp/*` 레시피로 작은 러닝부터 시작 (critic→rho-1 순)
  * 성능·메모리 프로파일, 토큰 예산/accum 최적화
* **전달물**

  * run 링크(MLflow), 리포트, 이슈/교훈 정리
* **DoD**

  * 두 파이프라인 모두 end-to-end 성공, 지표 기록/재현 확인
* **근거**: 로드맵의 스케일링 순서 준수&#x20;

---

## Phase 14 — Hyperparameter Sweep & Ablations

**목표**: λ/T/percentile 등 핵심 하이퍼 탐색, LoRA 옵션 비교

* **작업**

  * 스윕: λ∈{0.1,0.3,1.0}, T∈{0.5,0.7,1.0}, p∈{0.1,0.2,0.3}
  * Full FT vs LoRA 비교(비용 대비 성능)
* **전달물**

  * 아블레이션 표/플롯(MLflow → 보고서 자동 생성)
* **DoD**

  * 최선 조합과 근거 정리, 다음 실험 계획 업데이트
* **근거**: BLUEPRINT 로드맵 7단계와 합치&#x20;

---

## Phase 15 — Hardening & Release

**목표**: 운영 안전장치, 문서 완성, “staging→production” 승급

* **작업**

  * 실패 대응(중단 기준, 폴백 로직) 재검토, 비밀/권한 점검
  * 문서: README, 운영 가이드, Trouble-shooting, 모델 카드
  * 모델 레지스트리 승급 플로우(MLflow)
* **전달물**

  * `v1.0.0` 릴리스 태그, 릴리스 노트, 모델 카드
* **DoD**

  * 재현 가능한 end-to-end, 문서/테스트/CI/이미지 모두 green
* **근거**: MLflow 레지스트리·운영 규율 일치 &#x20;

---

## Git & Workflow Cheatsheet

```bash
# Bootstrap
git init && git add -A && git commit -m "chore: bootstrap"
git checkout -b develop
git checkout -b feature/skeleton

# Dev loop
uv sync --frozen
uv run python -m src.cli.train --config configs/config.yaml --recipe configs/recipe.yaml --dry-run

# Test & Lint
uv run pytest -q
uv run ruff check .

# PR
git push origin feature/skeleton
# → Open PR to develop (CI must pass)

# Tag & Release
git tag v1.0.0 && git push origin v1.0.0
```

---

## Docker & VESSL Cheatsheet

```bash
# Build image
docker build -t ghcr.io/<org>/<repo>:cuda12.1-a100 -f docker/Dockerfile .

# Run local (GPU)
docker run --gpus all -it --shm-size=16g \
  -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_DEFAULT_REGION \
  -v $PWD:/workspace ghcr.io/<org>/<repo>:cuda12.1-a100 \
  uv run python -m src.cli.train --config configs/config.yaml --recipe configs/recipe.yaml

# VESSL spec (docker/vessl.yaml) → submit via UI/CLI
```

---

## Risk Ledger & Mitigations

* **OOM/NaN**: token budget↓, grad-accum↑, bf16→fp16 폴백, loss-scaler/grad-clip 강화 (Trainer 내 자동 폴백)&#x20;
* **가중 CE 폭주**: mean-norm + \[ε,Wmax] 클립, λ/T 스윕으로 완충, 상위 p% 과도강조 금지&#x20;
* **데이터 누출**: 평가셋 격리/마스킹, 디컨태미네이션 룰 CI 체크&#x20;
* **환경 드리프트**: uv lock·Docker frozen sync, CI에서 `--frozen` 강제&#x20;
* **운영 보안**: Secrets는 VESSL/ENV로만, 코드 하드코딩 금지&#x20;

---

## Definition of Done (Global)

* 모든 Phase DoD 충족 + CI green
* **BLUEPRINT**와 **DEV PRINCIPLES**의 요구사항과 완전 정합 (구조/설정/유틸/로깅/평가/운영) &#x20;
* 로컬→VESSL A100 환경에서 **두 파이프라인(critic, rho-1)** 이 **Meta MTP 평가 프로토콜**로 end-to-end 완주

> 이 계획대로 진행하면, 제로베이스에서 **코드/인프라/운영**까지 끊김 없이 구축하고, 빠른 반복 실험과 안전한 운영을 모두 달성할 수 있습니다.
