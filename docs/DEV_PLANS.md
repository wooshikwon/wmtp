# DEV_PLANS.md (Revised)

*Zero-to-Production Plan for WMTP Fine-Tuning Framework (Git · uv · Docker/VESSL · MLflow/S3)*

본 문서는 BLUEPRINT, 연구제안서/연구개선안, PHASE8 실행계획을 반영하여, 현재 구현 상태와 GAP 중심으로 Phase별 목표·작업·DoD를 재정의합니다. 완료된 항목은 간략히 요약하고, 미비/미완 항목은 실행 과제로 구체화합니다.

---

## Phase 0 — Repository & Governance Bootstrap
- **상태**: ⚠️ 부분 완료
- **요약**: 레포/라이선스/README/문서/린트 규칙 존재. CI/브랜치 보호/pre-commit/코드오너 부재.
- **남은 작업**:
  - **브랜치 보호**: `main` 보호 규칙 + 필수 CI 통과 필요
  - **CI**: GitHub Actions 워크플로우(`lint+tests`) 추가
  - **pre-commit**: `ruff check/format` 훅 구성, `uv run` 사용
  - **CODEOWNERS**: 리뷰 경로 규정 추가
- **DoD**: PR 아닌 직접 push 차단, CI green, pre-commit 훅 동작, CODEOWNERS 반영

## Phase 1 — Environment & Package Management (uv) + Project Skeleton
- **상태**: ✅ 완료
- **요약**: `pyproject.toml`/`uv.lock` 고정, 디렉토리/엔트리/러너 구성 완료.
- **DoD**: `uv sync --frozen` 성공 및 CLI 헬프 출력 완료

## Phase 2 — Settings Schema (Pydantic) & YAML Contracts
- **상태**: ✅ 완료
- **요약**: `src/settings/{config_schema,recipe_schema,loader}.py` 스키마/로더 구현, `configs/config.example.yaml`, `configs/recipe.example.yaml` 제공.
- **최근 개선**: Recipe 스키마에서 `critic`/`rho1` 필드를 Optional로 변경, 각 알고리즘별 깔끔한 설정 구조 확보
- **DoD**: 부적합 YAML에 명확한 에러 메시지, 스키마 검증 통과

## Phase 3 — Registry & Factory
- **상태**: ✅ 완료
- **요약**: `components/registry.py` 통합 레지스트리(카테고리 어댑터) 정비, `factory/component_factory.py`는 `create_*` 생성자만 유지(표준화). Mock 경로 제거.
- **DoD**: 모든 컴포넌트는 레지스트리 등록 + Factory `create_*`로만 생성(일괄 빌더 제거), 설정 변경만으로 컴포넌트 교체 가능

## Phase 4 — Utils Consolidation (S3 · HF · MLflow · Dist/Eval)
- **상태**: ✅ 완료
- **요약**: `utils/{s3,hf,mlflow,dist,eval}.py` 일원화 구현, 외부 호출 억제 원칙 충족.
- **DoD**: utils 외 직접 호출 금지 준수, 로컬 캐시/미러링 동작

## Phase 5 — Data & Model Loaders
- **상태**: ✅ 완료
- **요약**: `loader/{dataset_mbpp_loader,dataset_contest_loader,hf_local_s3_loader}.py` 구현, 로컬우선→S3 캐시.
- **최근 개선**: Reference 모델을 CodeLlama-7B-Python으로 표준화 (tokenizer/vocab 호환성 확보)
- **DoD**: 로컬/S3/혼합 시나리오에서 정상 로드

## Phase 6 — Scorers (Critic & Rho-1)
- **상태**: ✅ 완료
- **요약**: `scorer/{critic_delta.py,rho1_excess.py}` 구현. Rho-1 경로는 트레이너가 `ref_logits` 전달로 실가중 적용. Critic 경로는 Stage1(가치헤드 회귀·캐시) 프리트레이너와 Stage2 자동 로드(`value_head.pt`)가 연결되어 Δ 가중이 실제 적용됨.
- **DoD**: 트레이너 무변경 적용 + (Rho-1: ref CE 기반 실제 가중) 또는 (Critic: RM→가치헤드 학습→Δ 가중, 저장된 `value_head` 자동 로드) 동작, 통계 불변식 테스트 통과

## Phase 7 — Trainer (MTP Weighted CE) + Optimizer
- **상태**: ✅ 완료
- **요약**: `_compute_mtp_ce_loss`로 k-step 정렬/마스킹, 토큰 가중 CE, AMP/FSDP 훅, AdamW+스케줄러 지원.
- **DoD**: NaN/OOM 게이트, 로깅 기본 지표 기록

## Phase 8 — Pipelines Orchestration (및 정합성 강화)
- **상태**: ✅ 완료 (핵심 작업)
- **요약**: `pipelines/training_pipeline.py` 신설로 MLflow/로더/트레이너 조립을 `create_*` 표준 방식으로 수행. 레거시 `pipelines/training.py` 제거. 분기는 Factory/레지스트리·모듈 내부로 위임.
- **완료된 작업** (PHASE8_EXECUTION_PLAN.md 준수):
  - ✅ 스코어러 출력 표준화: `weights`를 torch.Tensor(device/dtype 정렬)로 반환 완료
  - ✅ 트레이너 정렬/마스킹 강화: `_compute_mtp_ce_loss` 입력검증·개별마스킹·수치안정성 보완
  - ✅ MLflow 확장 지표: weight 분포(p25/p75/p95), 방식별 특화지표 로깅
  - ✅ Critic Stage2 연결: `value_head.pt` 자동 로딩 구현
  - ✅ MTPWrapper 구현: H>1 요청 시 teacher-forcing k-step 에뮬레이션
- **선택적 구현** (향후 개선):
  - DataLoader 품질: `DistributedSampler`/시드 고정
  - 테스트 강화: 추가 스모크/불변식 테스트
- **DoD**: 핵심 기능 테스트 통과 및 MLflow 지표/로그 확인 완료

## Phase 9 — Evaluation Harness (Meta MTP Protocol)
- **상태**: ✅ 완료
- **요약**: 완전한 평가 파이프라인 구현 완료
- **구현 완료**:
  - ✅ **EvaluationPipeline 생성**: `src/pipelines/evaluation_pipeline.py` 구현
  - ✅ **CLI 연결**: `cli/eval.py`에서 실제 파이프라인 호출 구현
  - ✅ **MLflow 통합**: 메트릭 로깅, 파라미터 기록, 아티팩트 저장 구현
  - ✅ **Stage 3 구현**: 예측 샘플, 가중치 통계, Markdown 보고서 생성 기능 추가
- **테스트 완료**: `test_evaluation_mlflow.py`로 모든 기능 검증
- **DoD**: `uv run python -m src.cli.eval`로 MBPP exact, CodeContests pass@k 결과 산출 및 MLflow 기록 ✅

## Phase 10 — MLflow (S3) Integration & Experiment Taxonomy
- **상태**: ✅ 완료
- **요약**: `utils/mlflow.py` 매니저/자동 파라미터 로깅, 파이프라인에서 실사용.
- **DoD**: 로컬/S3 동일 구조 기록, 실험 트리 규약 충족

## Phase 11 — Dockerization (CUDA 12.1) & VESSL Spec
- **상태**: ⬜ 미완료
- **남은 작업**:
  - Dockerfile: `pytorch/pytorch:2.4.0-cuda12.1-cudnn9`, uv 설치, frozen sync
  - vessl.yaml: GPU/CPU/Mem/Disk/Secrets/command, S3/MLflow env
  - 빌드/푸시 스크립트: `make image && make push` 또는 `scripts/build_push.sh`
- **DoD**: 로컬 GPU 및 VESSL에서 `uv run python -m src.cli.train` 정상 동작

## Phase 12 — Test Pyramid & Quality Gates
- **상태**: ✅ 완료 (기본)
- **요약**: 스코어러/트레이너 정렬/가중치/평가자 기본 테스트 존재.
- **완료 항목**:
  - ✅ Recipe 스키마 검증 테스트
  - ✅ MLflow 통합 테스트 (`test_evaluation_mlflow.py`)
  - ✅ Scorer 통합 테스트
- **DoD**: 핵심 불변식 테스트 green

## Phase 13 — First Real Run (Local → Single GPU → VESSL x4 A100)
- **상태**: ⬜ 미완료
- **실행 계획**:
  - 단일 GPU 스모크: tiny 레시피(dry-run→소규모 step)로 critic/rho1 각각 End-to-End
  - 실데이터: MBPP subset → MBPP full → Contest로 확대
  - 관측/보고: MLflow 링크/지표, 성능·메모리 프로파일
- **DoD**: 두 파이프라인 end-to-end 성공 및 지표 재현 확인

## Phase 14 — Hyperparameter Sweep & Ablations
- **상태**: ⬜ 미완료
- **실행 계획** (Rho-1 우선):
  - λ∈{0.1,0.3,1.0}, T∈{0.5,0.7,1.0}, percentile∈{0.1,0.2,0.3}
  - Full FT vs LoRA, H=1 vs H>1(MTPWrapper) 비교
  - Rho-1 기반 가중 효과 정량화
- **DoD**: 최선 조합/근거 정리, 자동 보고서(MLflow 아티팩트) 생성

## Phase 15 — Hardening & Release
- **상태**: ⬜ 미완료
- **실행 계획**:
  - 실패 대응/폴백 재점검
  - 운영 가이드/트러블슈팅/모델 카드 작성
  - MLflow 레지스트리: staging→production 승급
- **DoD**: `v1.0.0` 릴리스 태그, 문서/테스트/CI green, 재현성 보장

---

## 연구 취지 정렬 (요지)
- **토큰 가중 철학**: "모든 토큰이 동일하지 않다"를 반영하여, Rho-1(critic-free) 기반 연속 가중을 기본값으로 운용하고 Critic 경로는 선택(리스크 관리).
- **MTP 결합**: MTP의 효율성과 토큰 가중의 효과를 결합. [B,S,V] 모델의 경우 H=1 폴백 또는 선택적 MTP 래퍼 제공.
- **평가 프로토콜**: Meta MTP 프로토콜 재현(MBPP exact, CodeContests pass@k), 관측성(MLflow) 강화.
- **Reference 모델 선택**: Rho-1 방식에서 CodeLlama-7B-Python을 reference 모델로 사용하여 MTP 모델과 tokenizer/vocab 호환성 확보 및 코드 도메인 CE 분포 일치도 극대화.

---

## 실행 우선순위 (Next Steps)
1. **Phase 11**: Docker/VESSL 환경 구축
2. **Phase 13**: 첫 실제 실행 (Local → GPU)
3. **Phase 14**: 하이퍼파라미터 스윕
4. **Phase 15**: 하드닝 및 릴리스
5. **Phase 0**: CI/CD 완성 (선택적)

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
