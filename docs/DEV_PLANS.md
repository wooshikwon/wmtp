# DEV_PLANS.md (Revised)

*Zero-to-Production Plan for WMTP Fine-Tuning Framework (Git · uv · Docker/VESSL · MLflow/S3)*

본 문서는 BLUEPRINT, 연구제안서/연구개선안, PHASE8 실행계획을 반영하여, 현재 구현 상태와 GAP 중심으로 Phase별 목표·작업·DoD를 재정의합니다. 완료된 항목은 간략히 요약하고, 미비/미완 항목은 실행 과제로 구체화합니다.

---

## Phase 0 — Repository & Governance Bootstrap
- 상태: 부분 완료
- 요약: 레포/라이선스/README/문서/린트 규칙 존재. CI/브랜치 보호/pre-commit/코드오너 부재.
- 남은 작업
  - **브랜치 보호**: `main` 보호 규칙 + 필수 CI 통과 필요
  - **CI**: GitHub Actions 워크플로우(`lint+tests`) 추가
  - **pre-commit**: `ruff check/format` 훅 구성, `uv run` 사용
  - **CODEOWNERS**: 리뷰 경로 규정 추가
- DoD
  - PR 아닌 직접 push 차단, CI green, pre-commit 훅 동작, CODEOWNERS 반영

## Phase 1 — Environment & Package Management (uv) + Project Skeleton
- 상태: 완료
- 요약: `pyproject.toml`/`uv.lock` 고정, 디렉토리/엔트리/러너 구성 완료.
- DoD: `uv sync --frozen` 성공 및 CLI 헬프 출력 완료

## Phase 2 — Settings Schema (Pydantic) & YAML Contracts
- 상태: 완료
- 요약: `src/settings/{config_schema,recipe_schema,loader}.py` 스키마/로더 구현, `configs/config.example.yaml`, `configs/recipe.example.yaml` 제공.
- DoD: 부적합 YAML에 명확한 에러 메시지, 스키마 검증 통과

## Phase 3 — Registry & Factory
- 상태: 완료
- 요약: `components/registry.py` 통합 레지스트리(카테고리 어댑터) 정비, `factory/component_factory.py`는 `create_*` 생성자만 유지(표준화). Mock 경로 제거.
- DoD: 모든 컴포넌트는 레지스트리 등록 + Factory `create_*`로만 생성(일괄 빌더 제거), 설정 변경만으로 컴포넌트 교체 가능

## Phase 4 — Utils Consolidation (S3 · HF · MLflow · Dist/Eval)
- 상태: 완료
- 요약: `utils/{s3,hf,mlflow,dist,eval}.py` 일원화 구현, 외부 호출 억제 원칙 충족.
- DoD: utils 외 직접 호출 금지 준수, 로컬 캐시/미러링 동작

## Phase 5 — Data & Model Loaders
- 상태: 완료
- 요약: `loader/{dataset_mbpp_loader,dataset_contest_loader,hf_local_s3_loader}.py` 구현, 로컬우선→S3 캐시.
- DoD: 로컬/S3/혼합 시나리오에서 정상 로드

## Phase 6 — Scorers (Critic & Rho-1)
 - 상태: 완료
 - 요약: `scorer/{critic_delta.py,rho1_excess.py}` 구현. Rho-1 경로는 트레이너가 `ref_logits` 전달로 실가중 적용. Critic 경로는 Stage1(가치헤드 회귀·캐시) 프리트레이너와 Stage2 자동 로드(`value_head.pt`)가 연결되어 Δ 가중이 실제 적용됨.
 - DoD: 트레이너 무변경 적용 + (Rho-1: ref CE 기반 실제 가중) 또는 (Critic: RM→가치헤드 학습→Δ 가중, 저장된 `value_head` 자동 로드) 동작, 통계 불변식 테스트 통과

## Phase 7 — Trainer (MTP Weighted CE) + Optimizer
- 상태: 완료(기본), 개선 과제는 Phase 8로 이관
- 요약: `_compute_mtp_ce_loss`로 k-step 정렬/마스킹, 토큰 가중 CE, AMP/FSDP 훅, AdamW+스케줄러 지원.
- DoD: NaN/OOM 게이트, 로깅 기본 지표 기록

## Phase 8 — Pipelines Orchestration (및 정합성 강화)
- 상태: 핵심 작업 완료 (3/6 완료, 나머지는 선택적 구현)
- 요약: `pipelines/training_pipeline.py` 신설로 MLflow/로더/트레이너 조립을 `create_*` 표준 방식으로 수행. 레거시 `pipelines/training.py` 제거. 분기는 Factory/레지스트리·모듈 내부로 위임.
- **완료된 작업 (PHASE8_EXECUTION_PLAN.md 준수)**
  - ✅ **스코어러 출력 표준화**: `weights`를 torch.Tensor(device/dtype 정렬)로 반환 완료, 트레이너 변환 최소화
  - ✅ **트레이너 정렬/마스킹 강화**: `_compute_mtp_ce_loss` 입력검증·개별마스킹·수치안정성·valid_mask 보완 완료
  - ✅ **MLflow 확장 지표**: weight 분포(p25/p75/p95), 방식별 특화지표(rho1_usage_ratio/critic_delta_mean), 실패게이트(nan/extreme_weights) 완료
  - ~~**Critic Stage2 연결**: Stage1에서 저장한 `value_head.pt`를 학습 단계에서 자동 로딩하도록 스코어러/트레이너 연결 보강~~ (완료)
- 남은 작업 (선택적 구현)
  - **모델 폴백/래퍼**: [B,S,V]만 제공 모델 시 H>1 요청이면 경고 후 H=1 폴백; 선택적 `MTPWrapper`(teacher-forcing k-step) 제공
  - **DataLoader 품질**: `DistributedSampler`/시드 고정, HF datasets 캐시, `labels=-100` 마스킹 일관화
  - **테스트 강화**: 정렬/마스킹, 가중치 불변식, 스모크(H=1/H>1, critic/rho1, dry-run) 추가
  - **스코어러 컨텍스트 표준화**: `base_logits/ref_logits/base_ce/ref_ce/rewards/hidden_states` 키 합의 및 문서화(현행 유지)
- DoD
  - ✅ 핵심 기능 테스트 통과 및 MLflow 지표/로그 확인 완료 (두 방식 비교 관측성 확보)

## Phase 9 — Evaluation Harness (Meta MTP Protocol)
- 상태: 부분 완료
- 요약: `evaluator/{mbpp_eval.py,codecontests.py,meta_mtp.py}` 구현되어 단독 사용 가능. 다만 `src/cli/eval.py`는 스텁 상태로 파이프라인 미연결.
- 남은 작업
  - **CLI 연결**: `cli/eval.py` → 설정 로드 → 팩토리로 evaluator 생성 → 모델/데이터 로더 통해 실제 평가 실행 → 결과 테이블/MLflow 기록
  - **데이터 연동**: 평가 시 `dataset_mbpp_loader`/`dataset_contest_loader` 재사용 (샘플링 파라미터 고정)
  - **아티팩트**: 예측/정답 일부, 가중치/CE 통계 아티팩트 저장
- DoD
  - `uv run python -m src.cli.eval ...`로 MBPP exact, CodeContests pass@k 결과 산출 및 MLflow 기록

## Phase 10 — MLflow (S3) Integration & Experiment Taxonomy
- 상태: 완료
- 요약: `utils/mlflow.py` 매니저/자동 파라미터 로깅, 파이프라인에서 실사용.
- DoD: 로컬/S3 동일 구조 기록, 실험 트리 규약 충족

## Phase 11 — Dockerization (CUDA 12.1) & VESSL Spec
- 상태: 미완료
- 남은 작업
  - **Dockerfile**: `pytorch/pytorch:2.4.0-cuda12.1-cudnn9`, uv 설치, frozen sync, 환경 변수
  - **vessl.yaml**: GPU/CPU/Mem/Disk/Secrets/command, S3/MLflow env
  - **빌드/푸시 스크립트**: `make image && make push` 또는 `scripts/build_push.sh`
- DoD
  - 로컬 GPU 및 VESSL에서 `uv run python -m src.cli.train ...` 정상 동작

## Phase 12 — Test Pyramid & Quality Gates
- 상태: 완료(기본), 확장 테스트는 Phase 8과 연동
- 요약: 스코어러/트레이너 정렬/가중치/평가자 기본 테스트 존재.
- 남은 작업
  - Phase 8 개선에 따른 추가 단위/스모크 테스트 보강 및 커버리지 유지(≥80)
- DoD: CI 통합, 커버리지 리포트, 핵심 불변식 테스트 green

## Phase 13 — First Real Run (Local → Single GPU → VESSL x4 A100)
- 상태: 미완료
- 실행 계획
  - **단일 GPU 스모크**: tiny 레시피(dry-run→소규모 step)로 critic/rho1 각각 End-to-End
  - **실데이터**: MBPP subset → MBPP full → Contest로 확대, 토큰 예산/accum 최적화
  - **관측/보고**: MLflow 링크/지표, 성능·메모리 프로파일, 이슈/교훈 기록
- DoD: 두 파이프라인 end-to-end 성공 및 지표 재현 확인

## Phase 14 — Hyperparameter Sweep & Ablations
- 상태: 미완료
- 실행 계획 (연구개선안 취지 반영: Rho-1 우선, critic-free 경로 우선 실험)
  - λ∈{0.1,0.3,1.0}, T∈{0.5,0.7,1.0}, percentile∈{0.1,0.2,0.3}
  - Full FT vs LoRA, H=1 vs H>1(MTPWrapper 사용 시) 비교
  - 선택: gradient 기반 토큰 중요도(향후) 대비 Rho-1 기반 가중의 효과 비교 설계
- DoD: 최선 조합/근거 정리, 자동 보고서(MLflow 아티팩트) 생성

## Phase 15 — Hardening & Release
- 상태: 미완료
- 실행 계획
  - 실패 대응(중단 기준/폴백) 재점검, 비밀/권한 점검, 운영 가이드/트러블슈팅/모델 카드 작성
  - MLflow 레지스트리: staging→production 승급 플로우 마련
- DoD: `v1.0.0` 릴리스 태그, 문서/테스트/CI/이미지 green 상태, 재현성 보장

---

## 연구 취지 정렬 (요지)
- **토큰 가중 철학**: “모든 토큰이 동일하지 않다”를 반영하여, Rho-1(critic-free) 기반 연속 가중을 기본값으로 운용하고 Critic 경로는 선택(리스크 관리).
- **MTP 결합**: MTP의 효율성과 토큰 가중의 효과를 결합. [B,S,V] 모델의 경우 H=1 폴백 또는 선택적 MTP 래퍼 제공.
- **평가 프로토콜**: Meta MTP 프로토콜 재현(MBPP exact, CodeContests pass@k), 관측성(MLflow) 강화.

---

## 실행 우선순위(Next)
1) Phase 9 CLI 평가 파이프라인 연결 → 2) Phase 11 Docker/VESSL → 3) Phase 13 실런 → 4) Phase 14 스윕/아블레이션 → 5) Phase 15 하드닝/릴리스 → 6) [선택적] Phase 8 잔여 작업(MTPWrapper, DataLoader 품질 개선)

---

## Cheatsheet (유지)
```bash
# Dev loop
uv sync --frozen
uv run python -m src.cli.train --config configs/config.yaml --recipe configs/recipe.example.yaml --dry-run

# Tests & Lint
uv run pytest -q
uv run ruff check .
```
