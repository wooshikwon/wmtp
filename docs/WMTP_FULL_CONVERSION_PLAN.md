# WMTP 완전 전환 계획 (환경 일치 · Phase ≤ 6)

본 문서는 현재 코드베이스를 사용자의 운영 환경에 맞게 일관된 방식으로 “완전 전환(기존 구현 삭제/치환)” 하기 위한 실행 계획이다. 목표는 Phase 6 범위에서 즉시 실행 가능한 수준으로 불일치를 제거하고, 이후 단계(Phase 7+)의 구현을 방해하지 않는 견고한 기반을 마련하는 것이다.

---

## 1) 목표와 범위
- 목표: 아래 개선 항목의 “완전 전환” 수행
  - S3 설정 주입 규약 통일(중복/평탄화 키 제거)
  - Evaluator 매핑 정합성 확보(`meta-mtp-evaluator` 도입)
  - 테스트를 새 규약으로 전면 갱신
  - 기본 HF 모델 ID를 청사진과 정합화
- 범위: Phase 6까지의 컴포넌트/유틸/테스트 및 문서. 학습 파이프라인(Phase 7+)과 오케스트레이션(Phase 8+)은 본 계획의 후속 항목으로 별도 문서에서 다룬다.

---

## 2) 변경 요약(Destructive Changes 포함)
- S3 설정 주입 규약 통일
  - 변경: 로더에 전달되는 설정은 반드시 중첩 구조(`storage: {mode, s3:{...}}`)만 허용
  - 삭제: 평탄화 키(`storage_mode`, `s3_config`) 지원 코드 및 테스트 전면 제거
- `meta-mtp-evaluator` 추가 및 팩토리 매핑 정정
  - 추가: `components/evaluator/meta_mtp.py` 구현 및 `@evaluator_registry.register("meta-mtp-evaluator")`
  - 변경: `ComponentFactory.EVALUATOR_MAP["meta-mtp"] = "meta-mtp-evaluator"`
  - 역할: 레시피의 `eval.metrics`·샘플링 설정을 사용해 MBPP/CodeContests evaluator를 내부 호출(프로토콜 오케스트레이터)
- 테스트 전면 갱신
  - 기존 평탄화 S3 키 사용 테스트 제거·수정
  - `meta-mtp-evaluator`의 등록/오케스트레이션 테스트 추가
- HF 기본 모델 ID 정합화
  - 기본값을 BLUEPRINT와 일치시키되, 레시피/로컬 경로/환경 값이 항상 우선하도록 명시

---

## 3) 상세 작업 지시(파일별)

### A. Factory → 로더 설정 주입 규약 통일
- 파일: `src/factory/component_factory.py`
- 편집:
  - `create_data_loader()`에서 생성하는 `loader_config`를 아래 형태로 통일
    - `loader_config["storage"] = {"mode": config.storage.mode, "s3": config.storage.s3.model_dump() if config.storage.s3 else None}`
    - 기존 `storage_mode`, `s3_config` 키 제거
  - 데이터·모델 경로 키는 현행 그대로 유지(`local_path`, `model_paths`, `cache_dir`)
- 수용 기준(Acceptance): 로더 내부에서 `create_s3_manager(config)`가 정상적으로 S3 매니저를 생성하고, 로컬→캐시→S3→HF 순서로 동작

### B. Utils.S3 → 평탄화 키 제거 보장(옵션)
- 파일: `src/utils/s3.py`
- 원칙: 팩토리에서 이미 중첩 구조를 강제하므로, 유틸 측에 평탄화 호환 코드는 유지하지 않음(완전 전환)
- 수용 기준: `create_s3_manager()`가 중첩 구조만 수용하고, 잘못된 입력 시 명확한 경고/None 반환

### C. Evaluator 매핑/구현 정합화
- 파일(신규): `src/components/evaluator/meta_mtp.py`
- 내용(요지):
  - 등록 키: `meta-mtp-evaluator`
  - 역할: `recipe.eval.metrics`를 확인해 필요한 evaluator를 내부 생성/호출
    - 예: `mbpp_exact` 포함 시 `mbpp-v1` 실행, `contest_pass@{1,5}` 포함 시 `codecontests-v1` 실행
    - 결과는 metric 키로 병합 후 리턴, 샘플링 설정은 공통 적용
- 파일(수정): `src/factory/component_factory.py`
  - `EVALUATOR_MAP["meta-mtp"] = "meta-mtp-evaluator"`
- 수용 기준: `ComponentFactory.create_evaluator()`가 `meta-mtp`에서 신규 evaluator를 반환하고, 단일/복수 메트릭 구성이 모두 정상 작동

### D. 로더/스코어러 테스트 갱신
- 파일: `tests/test_loaders.py`
  - 변경: 설정 구성 시 `{"storage": {"mode": "s3", "s3": {...}}, "paths": {"cache": ...}}` 형태 사용
  - `loader.s3_manager` 직접 패치가 필요한 경우, 인스턴스 생성 후 `loader.s3_manager = MagicMock()`로 명시 주입
  - 로컬 우선·캐시·S3 폴백 경로를 모두 커버하도록 케이스 정리
- 파일(신규): `tests/test_evaluator_meta_mtp.py`
  - 신규 evaluator 등록 확인
  - MBPP/CodeContests 조합별 metric 집계 검증
- 수용 기준: 변경 후 `pytest -q` 통과, 가중치/정규화/로더 불변식 유지

### E. HF 기본 모델 ID 정합화
- 파일: `src/components/loader/hf_local_s3_loader.py`
  - `default_model_ids["base"] = "facebook/multi-token-prediction"`로 정렬(문서와 일치)
  - 주석으로 “레시피/로컬 경로/환경이 항상 우선” 명시
- 수용 기준: 레시피로 명시 시 기본값이 사용되지 않으며, 기본값만으로도 허브에서 정상 로드

### F. 문서 반영
- 파일: `docs/BLUEPRINT.md`, `docs/DEV_PRINCIPLE.md`
  - 설정/평가 파트에 “중첩 S3 설정만 지원”과 `meta-mtp-evaluator` 오케스트레이션 간단 추가

---

## 4) 마이그레이션 순서(권장)
1. 브랜치 생성: `git checkout -b refactor/full-conversion-phase6`
2. A(Factory) → B(S3 유틸 확인) 순으로 설정 경로 정리
3. C(Evaluator) 구현 및 팩토리 매핑 교체
4. D(테스트) 업데이트 및 전 실행: `uv run pytest -q`
5. E(HF 기본 ID) 정렬, 문서(F) 동기화
6. 리뷰/머지: `develop` → `main`(CI green 후)

---

## 5) 검증 체크리스트(Definition of Done)
- 설정
  - [ ] 로더 구성에 평탄화 키 미사용(검색 시 0건)
  - [ ] `create_s3_manager()`가 중첩 구조 입력에서만 동작
- 평가
  - [ ] `meta-mtp-evaluator` 등록 및 팩토리 매핑 동작
  - [ ] MBPP/CodeContests 단독/동시 메트릭 집계 정상
- 테스트/품질
  - [ ] `pytest -q` 전체 green, 가중치 평균 1.0±ε 불변식 유지
  - [ ] 린트/포맷 CI green
- 문서
  - [ ] 설정 규약/평가 오케스트레이션 변경점 반영

---

## 6) 롤백 전략
- 문제가 발생하면 브랜치를 롤백하고, 임시로 `create_s3_manager()`에 평탄화 키 호환 분기를 한시 추가(문서에 EOL 날짜 명시) 후 재시도.

---

## 7) 리스크 & 완충책
- 리스크: 외부 스크립트가 평탄화 키에 의존했을 가능성
  - 대책: 해당 사용처를 PR에서 검색·주석 가이드 제공, 릴리스 노트에 Breaking Change 표기
- 리스크: evaluator 오케스트레이션의 메트릭 키 충돌
  - 대책: 메트릭 키 네임스페이스 고정(`mbpp_exact`, `contest_pass@k`), 충돌 시 Prefix 부여 옵션 준비

---

## 8) 후속(Phase 7+) 제안(참고)
- Trainer 구현 고도화 및 파이프라인 오케스트레이션(`pipelines/`) 도입
- CLI에서 실제 파이프라인 호출 연결 및 MLflow 로깅 일원화
- E2E 스모크(작은 모델·100 step) 자동화

위 계획대로 이행하면, 사용자의 현재 환경에서 Phase 6까지 요구되는 핵심 불일치를 해소하고, 이후 단계 구현을 위한 안정적 기반을 확보할 수 있다.
