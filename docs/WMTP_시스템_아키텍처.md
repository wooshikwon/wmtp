# WMTP 시스템 아키텍처
## "Not All Tokens Are What You Need" 연구 철학의 시스템 구현

본 문서는 WMTP(Weighted Multi-Token Prediction) 연구 제안서의 핵심 아이디어를 구현한 시스템 아키텍처를 설명합니다. 토큰별 중요도를 동적으로 계산하여 학습 효율성을 높이는 세 가지 핵심 알고리즘(baseline-mtp, critic-wmtp, rho1-wmtp)과 추가 변형(rho1-tokenskip)을 통합 파이프라인으로 지원하는 현대적인 시스템입니다.

---

## 시스템 설계 철학

### 핵심 원칙
**통합성**: 서로 다른 세 알고리즘이 동일한 파이프라인을 통해 실행되어 공정한 성능 비교를 보장합니다.

**모듈성**: 각 계층이 독립적인 책임을 가지며, 컴포넌트 간 결합도를 최소화하여 확장성을 확보했습니다.

**타입 안전성**: Pydantic 스키마 기반의 설정 검증으로 런타임 오류를 사전에 방지합니다.

**재현성**: MLflow 통합과 고정 시드를 통해 실험 결과의 재현성을 보장합니다.

### 연구 수식 구현
`L_WMTP = E_x [ Σ_t Σ_{k=1..H} w_{t,k} · CE( P_θ(x_{t+k} | x_{<t}), x_{t+k} ) ]`

이 수식의 핵심인 토큰별 가중치 w_{t,k} 계산을 세 가지 방식으로 구현:
- **Baseline**: 균등 가중치 (w = 1)
- **Critic**: Value Head 기반 동적 가중치
- **Rho1**: 참조 모델과의 CE 차이 기반 가중치

---

## 계층별 아키텍처 분석

### 1계층: configs/ - 실험 설정 관리

**핵심 책임**: 알고리즘별 하이퍼파라미터 및 실험 메타데이터 정의

**주요 기능**:
- **알고리즘 선택**: train.algo 필드로 baseline-mtp, critic-wmtp, rho1-wmtp 중 선택
- **모드 선택**: rho1-wmtp의 경우 weighted/token_skip 모드 지원
- **손실함수 파라미터**: lambda(정규화 강도), temperature(소프트맥스 온도), epsilon(수치 안정성) 설정
- **훈련 메타데이터**: run.name과 tags를 통한 실험 추적 정보
- **데이터 소스 지정**: MBPP, CodeContests, HumanEval 등 벤치마크 데이터셋 선택

**설계 특징**:
- YAML 형태의 가독성 높은 설정 파일
- 각 알고리즘별 독립된 recipe 파일로 실험 격리
- MLflow 태그 시스템과 연동한 실험 분류

### 2계층: src/settings/ - 설정 스키마 및 검증

**핵심 책임**: Pydantic 기반 타입 안전 설정 시스템 제공

**주요 컴포넌트**:
- **config_schema.py**: 환경 설정 스키마 (GPU, 분산학습, S3, MLflow, 경로 등)
- **recipe_schema.py**: 실험 설정 스키마 (알고리즘, 데이터, 손실함수, 평가 등)
- **loader.py**: YAML 로딩, 환경변수 치환, 검증 엔진

**핵심 기능**:
- **프로토콜 기반 경로 시스템**: s3://, file:// 프리픽스로 저장소 자동 감지
- **환경변수 치환**: ${VAR_NAME:-default} 패턴으로 환경별 설정 차별화
- **알고리즘별 필수 설정 검증**: critic-wmtp는 critic 섹션, rho1-wmtp는 rho1 섹션 필수
- **하드웨어 호환성 자동 조정**: MPS + bf16 → fp16 자동 변경 등

**Phase 2 현대화**:
- 기존 storage.mode 방식을 프로토콜 기반으로 전환
- S3 인증 정보를 s3_auth 섹션으로 분리
- 더 직관적인 필드명으로 개선 (auxiliary_loss_coef 등)

### 3계층: src/pipelines/ - 통합 실행 엔진

**핵심 책임**: 모든 알고리즘을 위한 단일 실행 파이프라인 제공

**training_pipeline.py - 통합 훈련 엔진**:
- **14단계 구조**: 시드 고정부터 결과 반환까지 체계적 실행 흐름
- **알고리즘별 조건부 로딩**: rho1은 ref 모델, critic은 rm 모델만 선택적 로딩
- **Stage 분리 훈련**: Critic 알고리즘의 2단계 학습 (Value Head 사전훈련 → 메인 훈련)
- **분산 학습 통합**: FSDP와 DistributedSampler를 통한 다중 GPU 지원
- **체크포인트 재개**: 훈련 중단점에서 안전한 재개 기능

**evaluation_pipeline.py - 평가 엔진**:
- **Meta 논문 호환**: 2024 Meta MTP 논문의 모든 평가 메트릭 재현
- **다중 평가 타입**: meta-mtp, inference-speed, per-head-analysis, token-accuracy
- **벤치마크 통합**: MBPP, CodeContests, HumanEval 자동 로딩 및 평가

**설계 우수성**:
- 어셈블리 전용 구조로 복잡한 로직을 Factory와 Registry에 위임
- 알고리즘 간 공정한 비교를 위한 동일한 실행 환경
- MLflow 통합으로 모든 실험 자동 추적

### 4계층: src/factory/ - 컴포넌트 생성 관리

**핵심 책임**: 설정 기반 적합한 컴포넌트 자동 선택 및 생성

**ComponentFactory 주요 메서드**:
- **create_trainer()**: 알고리즘별 독립 트레이너 생성 (BaselineMtpTrainer, CriticWmtpTrainer, Rho1WmtpTrainer)
- **create_aux_model_loader()**: 알고리즘별 보조 모델 로더 (ref/rm 모델 전용)
- **create_tokenizer()**: 환경 기반 토크나이저 자동 선택 (test: hf-transformers, prod: hf-sentencepiece)
- **create_evaluator_by_type()**: 평가 타입별 특화된 평가기 생성

**핵심 특징**:
- **설정 주도 생성**: recipe.yaml의 값이 컴포넌트 선택을 완전히 결정
- **타입 안전 검증**: 잘못된 알고리즘-보조모델 조합 사전 차단
- **하드코딩 방지**: MTP_CONFIG 상수로 고정값 관리 (n_heads=4, horizon=4)
- **Registry 패턴**: 실제 구현체는 UnifiedRegistry에서 키 기반 조회

### 5계층: src/components/ - 플러그인 아키텍처

**핵심 책임**: 확장 가능한 컴포넌트 시스템 제공

**base.py - 컴포넌트 인터페이스**:
- **Protocol 기반 설계**: Loader, Trainer, Optimizer, Evaluator 등 표준 인터페이스
- **setup()/run() 패턴**: 모든 컴포넌트가 동일한 생명주기 관리
- **BaseComponent**: 공통 기능을 제공하는 추상 기반 클래스

**registry.py - 컴포넌트 등록/조회 시스템**:
- **UnifiedRegistry**: 단일 레지스트리로 모든 카테고리 통합 관리
- **kebab-case 키 시스템**: 일관된 명명 규칙 (baseline-mtp, critic-wmtp 등)
- **카테고리별 어댑터**: 기존 인터페이스 호환성 유지 (trainer_registry, loader_registry 등)
- **메타데이터 관리**: 버전, 설명, 모듈 정보 등 컴포넌트별 상세 정보

**Phase 2 통합 혁신**:
- Scorer 컴포넌트 제거하고 모든 로직을 Trainer에 통합
- 각 알고리즘별 독립 트레이너로 책임 분리
- 더 명확한 컴포넌트 경계와 역할 정의

---

## 알고리즘별 구현 전략

### 현재 구현된 알고리즘

#### Baseline MTP (baseline-mtp)
**철학**: 표준 MTP의 균등 가중치 기준선 제공
**구현**: BaselineMtpTrainer에서 모든 토큰에 w=1 적용
**목적**: 가중치 기법의 효과를 측정하기 위한 대조군

#### Critic WMTP (critic-wmtp)
**철학**: 강화학습 가치 함수 기반 토큰 중요도 계산
**구현**: CriticWmtpTrainer + Stage1 Value Head 사전훈련
**핵심**: TD 오차와 GAE를 통한 동적 가중치, auxiliary loss로 안정성 확보

#### Rho1 WMTP (rho1-wmtp)
**철학**: 참조 모델과의 비교를 통한 선택적 학습
**구현**: Rho1WmtpTrainer + 참조 모델 CE 차이 계산
**핵심**: |CE_ref - CE_base| 기반 가중치, 노이즈 필터링 지원
**모드**:
- **Weighted 모드**: CE 차이를 softmax 온도로 정규화하여 연속적 가중치 생성
- **Token Skip 모드**: 하위 일정 비율(기본 30%) 토큰을 완전히 제외하는 이진 선택

### 미구현 알고리즘 (향후 확장 계획)

#### Gradient-based WMTP
**철학**: 최종 목적에 대한 각 토큰의 기여도(gradient norm) 기반 가중치
**계획**: 목표 gradient 중요도 계산, EMA 스무딩, 스케일 정규화 적용
**참조**: 연구제안서 4.1절 그래디언트 기반 가중화 계열

#### GRPO-based WMTP
**철학**: 그룹 내 상대 보상 기반 critic-free 최적화
**계획**: 토큰 수준 GRPO 가중치, 그룹 표준화, KL 정규화 통합
**참조**: 연구제안서 4.1절 GRPO 기반 가중화 계열

---

## 시스템 통합 특징

### 데이터 플로우
1. **설정 로딩**: configs/ YAML → settings/ 스키마 검증
2. **파이프라인 실행**: pipelines/ 통합 엔진 → factory/ 컴포넌트 생성
3. **컴포넌트 조합**: components/ 레지스트리 → 알고리즘별 트레이너 실행
4. **결과 추적**: MLflow 자동 로깅 → 실험 관리

### 확장성 메커니즘
- **새 알고리즘 추가**: Trainer 구현 + Registry 등록만으로 완료
- **새 데이터셋 지원**: Loader 구현 + 경로 설정으로 확장
- **새 평가 방식**: Evaluator 구현 + 평가 파이프라인 등록

### 운영 안정성
- **오류 처리**: 각 계층별 명확한 에러 메시지와 복구 가이드
- **재현성**: 시드 고정, 설정 파일 보관, MLflow 추적
- **성능 최적화**: BF16 혼합 정밀도, FSDP 분산 학습, 메모리 효율적 체크포인트

---

## 시스템 진화 방향

### 현재 달성사항
- **통합 아키텍처**: 세 알고리즘의 완전한 통합 실행
- **타입 안전성**: Pydantic 기반 설정 검증 시스템
- **확장 가능성**: Registry 패턴 기반 플러그인 아키텍처
- **운영 완성도**: MLflow 추적, 분산 학습, 체크포인트 재개

### 미래 확장 가능성
- **새로운 WMTP 변형**:
  - 그래디언트 기반 알고리즘 (연구제안서 4.1절 참조)
  - GRPO 기반 알고리즘 (DeepSeek 스타일 critic-free 최적화)
  - 하이브리드 가중화 방식 (여러 방법론 결합)
- **추가 Token Selection 전략**:
  - 적응형 threshold (데이터셋별 자동 조정)
  - 다단계 선택 (coarse-to-fine selection)
- **다양한 모델 아키텍처**: 현재 MTP 외 다른 아키텍처 지원
- **고급 평가 메트릭**: 더 정교한 성능 분석 도구
- **클라우드 네이티브**: Kubernetes 기반 대규모 실험 관리

이 아키텍처는 WMTP 연구의 핵심 아이디어를 효과적으로 구현하면서도, 미래의 연구 방향을 위한 확장성을 충분히 확보한 현대적인 시스템입니다.
