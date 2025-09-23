# Phase 9 Implementation Plan: Evaluation Harness (Meta MTP Protocol)

## 개요

Phase 9는 WMTP(Weighted Multi-Token Prediction) 연구의 핵심 검증 도구인 평가 하네스 구현을 목표로 합니다. 본 계획서는 연구제안서와 연구개선안의 의도를 반영하여, 기존 코드베이스와 정합하는 안정적이고 확장 가능한 평가 파이프라인을 구축합니다.

## 연구 의도와의 정합성

### 연구제안서 (Critic-based WMTP) 지원
- **토큰별 중요도 평가**: Critic 함수를 통한 가중치 부여 효과 측정
- **MTP 성능 검증**: 4-토큰 예측 vs 1-토큰 예측 성능 비교
- **코딩 벤치마크**: CodeContests 데이터셋을 통한 정량적 평가

### 연구개선안 (Rho-1-based WMTP) 지원
- **Reference 모델 기반**: Critic 없는 안정적 토큰 가중 방식 검증
- **Meta MTP 프로토콜**: MBPP exact-match, CodeContests pass@k 재현
- **비교 분석**: Rho-1 vs Critic 방식 효과 정량화

## 기존 코드베이스와의 정합성

### 아키텍처 일관성 ✅
- **Factory 패턴**: `ComponentFactory.create_evaluator()` 활용
- **Registry 시스템**: `evaluator_registry` 기반 컴포넌트 생성
- **Pipeline 구조**: `training_pipeline.py`와 동일한 패턴 적용

### 컴포넌트 재사용 ✅
- **Evaluator**: `meta_mtp.py`, `mbpp_eval.py`, `codecontests.py` 완전 활용
- **Data Loader**: `dataset_mbpp_loader`, `dataset_contest_loader` 재사용
- **Model Loader**: `hf_local_s3_loader` 체크포인트 로딩에 활용
- **MLflow**: 기존 `utils/mlflow.py` 매니저 활용

### 설정 시스템 통합 ✅
- **Config/Recipe**: 기존 Pydantic 스키마 완전 호환
- **S3/로컬**: 기존 storage 설정 그대로 활용
- **BLUEPRINT 준수**: 305라인 구현 매핑 원칙 완전 따름

## 개발 원칙 준수

### [필수1] 구조 파악 및 존중
- 기존 evaluator 컴포넌트들의 인터페이스와 구현 방식 완전 분석
- `training_pipeline.py`의 패턴을 평가 파이프라인에 적용

### [필수2] 중복 방지 및 일관성
- 새로운 메서드나 기능 중복 생성 금지
- 기존 Factory/Registry 패턴 일관성 유지
- 데이터 로더와 MLflow 매니저 재사용

### [필수3] 깨끗한 구현
- `cli/eval.py`의 스텁 코드 완전 제거
- TODO 주석과 mock 결과 삭제
- 실제 파이프라인 호출로 전면 교체

### [필수4] 하위 호환성 무시
- 기존 컴포넌트는 잘 작동하므로 보존
- 스텁 코드만 과감히 삭제하고 새로 구현

## 구현 계획

### 1단계: Evaluation Pipeline 생성

**파일**: `src/pipelines/evaluation_pipeline.py` (신규 생성)

**주요 기능**:
- Config/Recipe 검증 및 로딩
- Factory를 통한 컴포넌트 생성 (evaluator, model_loader, data_loader)
- 체크포인트 로딩 및 모델 초기화
- 데이터셋 로딩 (MBPP/CodeContests)
- 평가 실행 및 결과 수집
- MLflow 로깅 및 아티팩트 저장

**구조**:
```python
class EvaluationPipeline:
    def __init__(self, config: Config, recipe: Recipe)
    def run(self, checkpoint: Path, datasets: list[str] | None = None) -> dict[str, Any]
```

**패턴**: `training_pipeline.py`와 동일한 구조로 일관성 확보

### 2단계: CLI 구현 완성

**파일**: `src/cli/eval.py` (기존 파일 수정)

**변경 사항**:
- TODO 주석 완전 제거
- mock results 삭제
- EvaluationPipeline 호출 로직 구현
- 실제 결과 테이블 출력
- 아티팩트 저장 기능 구현

**보존 요소**:
- 기존 CLI 인터페이스 (typer 기반)
- Rich 테이블 출력 구조
- 에러 핸들링 패턴

### 3단계: MLflow 연동 및 아티팩트

**기능**:
- 평가 결과 metrics 로깅 (MBPP exact-match, CodeContests pass@k)
- 예측/정답 샘플 아티팩트 저장
- 가중치 분포 통계 저장 (토큰별 중요도 분석용)
- 평가 보고서 자동 생성

## 검증 방법

### DoD (Definition of Done)
1. `uv run python -m src.cli.eval --config configs/config.yaml --recipe configs/recipe.yaml --checkpoint <path>` 정상 실행
2. MBPP exact-match 결과 산출 및 출력
3. CodeContests pass@1, pass@5 결과 산출 및 출력
4. MLflow에 평가 결과 기록 확인
5. 아티팩트 저장 (예측 샘플, 가중치 통계) 확인

### 테스트 시나리오
- **기본 평가**: critic-wmtp 체크포인트로 MBPP+CodeContests 평가
- **Rho-1 평가**: rho1-wmtp 체크포인트로 동일 데이터셋 평가
- **비교 분석**: 두 방식의 성능 차이 정량화
- **아티팩트 검증**: MLflow UI에서 저장된 결과 확인

## 예상 효과

### 연구 관점
- **토큰 가중 효과 정량화**: WMTP 방법론의 실제 성능 향상 검증
- **방법론 비교**: Critic vs Rho-1 방식의 장단점 객관적 평가
- **Meta MTP 재현**: 기존 연구와의 직접 비교 가능

### 개발 관점
- **실험 자동화**: 체크포인트별 평가 결과 자동 생성
- **관측성 향상**: MLflow를 통한 체계적 실험 추적
- **재현성 확보**: 동일 설정으로 평가 결과 재현 보장

## 위험 요소 및 대응

### 위험 요소
- 체크포인트 로딩 실패
- 데이터셋 접근 오류 (S3/로컬)
- 평가 중 메모리 부족

### 대응 방안
- 체크포인트 검증 로직 추가
- 기존 data_loader의 캐시 메커니즘 활용
- 배치 크기 조정 옵션 제공

## 결론

Phase 9 구현은 WMTP 연구의 핵심 검증 도구로서, 기존 코드베이스의 아키텍처를 완전히 존중하면서도 연구 의도를 충실히 반영합니다. Factory/Registry 패턴의 일관성을 유지하고, 기존 컴포넌트를 최대한 재사용하여 안정적이고 확장 가능한 평가 시스템을 구축합니다.

이를 통해 Weighted-MTP의 실제 성능 효과를 정량적으로 검증하고, 연구 결과의 신뢰성을 확보할 수 있습니다.