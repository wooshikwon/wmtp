"""
WMTP 환경 설정 스키마: config.yaml 구조 정의 및 검증

WMTP 연구 맥락:
WMTP(Weighted Multi-Token Prediction)는 모든 토큰이 동일하게 중요하지 않다는
"Not All Tokens Are What You Need" 철학을 MTP 구조에 적용한 혁신적 방법론입니다.
이 모듈은 WMTP 학습 환경의 모든 인프라 설정을 관리합니다.

핵심 기능:
- 환경 설정 검증: S3/로컬 스토리지, MLflow, GPU 설정의 타입 안정성 보장
- 알고리즘별 최적화: Critic-WMTP와 Rho1-WMTP의 리소스 요구사항 차별화
- 분산 학습 지원: FSDP를 통한 대규모 모델(7B+) 효율적 학습
- 재현성 보장: 시드 고정 및 결정론적 연산 설정

WMTP 알고리즘과의 연결:
- Baseline MTP: 표준 MTP 설정으로 비교 기준 제공
- Critic-WMTP: 보상 모델(rm_local) 경로 설정 필수
- Rho1-WMTP: 참조 모델(ref_local) 경로 설정 필수

사용 예시:
    >>> from src.settings.config_schema import Config
    >>> from src.settings.loader import load_config
    >>>
    >>> # 설정 파일 로드 및 검증
    >>> config = load_config("configs/config.yaml")
    >>> print(f"프로젝트: {config.project}")
    프로젝트: wmtp_experiment
    >>> print(f"GPU 설정: {config.devices.mixed_precision}")
    GPU 설정: bf16
    >>>
    >>> # 알고리즘별 모델 경로 확인
    >>> if recipe.train.algo == "critic-wmtp":
    >>>     print(f"보상 모델: {config.paths.models.rm_local}")
    >>> elif recipe.train.algo == "rho1-wmtp":
    >>>     print(f"참조 모델: {config.paths.models.ref_local}")

성능 최적화:
- bf16 사용 시 메모리 50% 절감 (A100/RTX30xx+ 필수)
- FSDP full sharding으로 7B 모델을 4x A6000에서 학습 가능
- S3 스토리지 사용 시 로컬 디스크 I/O 병목 해소
- 캐시 디렉토리를 SSD에 위치시켜 토크나이저 로딩 속도 향상

디버깅 팁:
- ValidationError 발생: config.yaml의 타입과 필수 필드 확인
- S3 접근 오류: AWS_PROFILE 환경변수 및 버킷 권한 확인
- CUDA OOM: mixed_precision=bf16, FSDP.enabled=true 확인
- 재현성 문제: seed 값이 모든 실험에서 동일한지 확인
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class S3Config(BaseModel):
    """S3 스토리지 설정: WMTP 대규모 실험을 위한 클라우드 스토리지

    WMTP 연구 맥락:
    7B 모델의 체크포인트는 약 14GB이며, 학습 중 다수의 체크포인트가 생성됩니다.
    S3를 사용하면 팀원 간 모델 공유와 실험 결과 보존이 용이합니다.
    특히 Critic-WMTP는 value_head.pt 등 추가 파일이 생성되어 S3 관리가 중요합니다.

    구체적 활용:
    - 체크포인트 저장: checkpoint_epoch_*.pt (각 14GB)
    - MLflow 아티팩트: 메트릭, 그래프, 로그 파일
    - 평가 결과: MBPP/CodeContests 생성 코드 및 점수
    - Value Head 파일: critic-wmtp의 stage1 학습 결과

    Attributes:
        bucket: S3 버킷 이름 (필수)
            예: "wmtp-models", "ml-experiments"
            권장: 프로젝트별 전용 버킷 사용
        region: AWS 리전 (기본값: ap-northeast-2 서울)
            다른 리전: "us-east-1", "eu-west-1"
            팁: 학습 서버와 같은 리전 선택 시 전송 속도 향상
        prefix: 모든 객체에 적용할 접두사 (선택)
            예: "experiments/2024/" → s3://bucket/experiments/2024/...
            권장: "algo/date/" 형식 (예: "rho1/20241225/")

    WMTP 알고리즘별 활용:
        - Baseline: 표준 체크포인트만 저장
        - Critic: value_head.pt 추가 저장 필요
        - Rho1: 참조 모델 CE 점수 캐시 저장 권장

    Example:
        # 개발 환경
        s3:
          bucket: wmtp-dev
          region: ap-northeast-2
          prefix: dev/rho1/

        # 프로덕션
        s3:
          bucket: wmtp-prod
          region: ap-northeast-2
          prefix: prod/critic/20241225/

    디버깅 팁:
        - boto3 ImportError: pip install boto3
        - AccessDenied: AWS IAM 정책에 s3:PutObject 권한 확인
        - NoSuchBucket: 버킷 생성 필요 (aws s3 mb s3://버킷명)
    """

    bucket: str = Field(..., description="S3 버킷 이름")
    region: str = Field(default="ap-northeast-2", description="AWS 리전")
    prefix: str = Field(default="", description="모든 S3 객체의 접두사")

    @field_validator("bucket")
    @classmethod
    def validate_bucket(cls, v: str) -> str:
        """S3 버킷 이름 검증.

        AWS S3 버킷 명명 규칙을 검증합니다:
        - 비어있으면 안됨
        - 최대 63자
        - 소문자로 자동 변환

        Args:
            v: 입력된 버킷 이름

        Returns:
            str: 검증되고 정규화된 버킷 이름 (소문자)

        Raises:
            ValueError: 버킷 이름이 규칙에 맞지 않을 때
        """
        if not v:
            raise ValueError("S3 버킷 이름은 비어있을 수 없습니다")
        if len(v) > 63:
            raise ValueError("S3 버킷 이름은 63자 이하여야 합니다")
        # AWS는 버킷 이름을 소문자로 요구
        return v.lower()


class Storage(BaseModel):
    """스토리지 설정.

    WMTP 학습 결과물 저장 방식을 설정합니다.
    'auto' 모드는 경로에 따라 자동으로 로컬/S3를 판별합니다.

    Attributes:
        mode: 스토리지 모드
            - "auto": 경로에 따라 자동 판별 (s3:// 프리픽스 확인)
            - "local": 로컬 파일시스템 사용 (개발/테스트용)
            - "s3": AWS S3 사용 (프로덕션/협업용)
        s3: S3 설정 (S3 경로 사용 시 필요)

    Example:
        # 자동 판별 (권장)
        storage:
          mode: auto
          s3:
            bucket: wmtp

        # 로컬 전용
        storage:
          mode: local

        # S3 전용
        storage:
          mode: s3
          s3:
            bucket: wmtp-prod
    """

    mode: Literal["auto", "local", "s3"] = Field(..., description="스토리지 모드")
    s3: S3Config | None = None

    @model_validator(mode="after")
    def validate_s3_required(self):
        """S3 모드 시 S3 설정 필수 검증.

        storage.mode가 's3'로 설정되었을 때
        s3 설정이 제공되었는지 확인합니다.

        Raises:
            ValueError: s3 모드인데 s3 설정이 없을 때
        """
        if self.mode == "s3" and not self.s3:
            raise ValueError(
                "storage.mode가 's3'일 때는 s3 설정이 필요합니다\n"
                "예: storage:\n"
                "      mode: s3\n"
                "      s3:\n"
                "        bucket: my-bucket"
            )
        return self


class ModelPaths(BaseModel):
    """모델 경로 설정: WMTP 알고리즘별 필수 모델 관리

    WMTP 연구 맥락:
    WMTP는 "모든 토큰이 동일하게 중요하지 않다"는 가정 하에,
    토큰별 중요도를 계산하기 위해 추가 모델을 활용합니다.
    Critic-WMTP는 보상 모델로 가치 함수를 학습하고,
    Rho1-WMTP는 참조 모델과의 CE 차이로 중요도를 산출합니다.

    통합 경로 시스템:
    PathResolver가 s3:// 프리픽스를 자동 감지하여
    로컬 경로와 S3 URI를 투명하게 처리합니다.

    구체적 동작:
    1. Base 모델 로드: MTP 사전학습 모델 (필수)
    2. 알고리즘별 추가 모델:
       - critic-wmtp → rm 로드 → value head 학습
       - rho1-wmtp → ref 로드 → CE 차이 계산
    3. 토크나이저 호환성 체크 (자동)
    4. 가중치 적용된 MTP 학습

    Attributes:
        base: 기본 MTP 모델 경로
            - Facebook의 Multi-Token Prediction 모델
            - 7B 파라미터, 4개 토큰 동시 예측 (n=4)
            - 실제 학습이 진행되는 모델
            - 예: "models/7b_1t_4" 또는 "s3://wmtp/models/7b_1t_4"

        rm: 보상 모델 경로 (Critic-WMTP 전용)
            - 생성된 텍스트의 품질을 평가하는 Reward Model
            - Llama-3 8B 기반, 인간 선호도 학습됨
            - Stage1에서 value head 학습의 타겟 제공
            - critic-wmtp 선택 시 필수

        ref: 참조 모델 경로 (Rho1-WMTP 전용)
            - 토큰 중요도 계산의 기준점 제공
            - CodeLlama-7B 또는 ShearedLlama-1.3B 권장
            - CE 차이 계산: |CE_ref - CE_base|
            - 작은 모델 사용 시 메모리 효율적 (1.3B < 7B)

    WMTP 알고리즘별 활용:
        - Baseline MTP: base_local만 사용
        - Critic-WMTP: base_local + rm_local 필요
        - Rho1-WMTP: base_local + ref_local 필요

    예시:
        >>> # Critic-WMTP 설정
        >>> models:
        >>>   base_local: models/facebook_mtp_7b
        >>>   rm_local: models/llama3_8b_rm  # 필수!
        >>>
        >>> # Rho1-WMTP 설정
        >>> models:
        >>>   base_local: models/facebook_mtp_7b
        >>>   ref_local: models/codellama_7b  # 필수!

    주의사항:
        - 경로가 존재하지 않으면 HuggingFace에서 자동 다운로드 시도
        - 토크나이저 호환성: base와 ref/rm 모델 간 vocab 일치 필요
        - 대용량 모델: 7B는 약 14GB, 충분한 디스크 공간 확보

    디버깅 팁:
        - FileNotFoundError: 경로 확인 또는 HF 모델 ID 사용
        - CUDA OOM: ref_local을 더 작은 모델로 변경 (7B → 1.3B)
        - Tokenizer 오류: 모델 간 토크나이저 호환성 확인
    """

    base: str = Field(
        default="models/7b_1t_4",
        description="기본 MTP 모델 경로 (로컬 또는 S3 URI)",
    )
    rm: str = Field(
        default="models/Llama_3_8B_RM",
        description="보상 모델 경로 (Critic용, 로컬 또는 S3 URI)",
    )
    ref: str = Field(
        default="models/sheared_llama_1.3B",
        description="참조 모델 경로 (Rho1용, 로컬 또는 S3 URI)",
    )


class DatasetPaths(BaseModel):
    """데이터셋 경로 설정.

    코드 생성 능력 평가를 위한 벤치마크 데이터셋 경로입니다.

    Attributes:
        mbpp: MBPP 데이터셋 경로
            - Mostly Basic Python Problems
            - 974개의 Python 프로그래밍 문제
            - 초급~중급 난이도
            - Google Research 제작

        contest: CodeContests 데이터셋 경로
            - 경쟁 프로그래밍 문제 모음
            - 높은 난이도의 알고리즘 문제
            - DeepMind 제작
            - pass@k 메트릭 평가용

        humaneval: HumanEval 데이터셋 경로
            - OpenAI의 코드 생성 벤치마크
            - 164개의 Python 함수 작성 문제
            - 프로그래밍 능력 평가의 표준

    Example:
        datasets:
          mbpp: /data/benchmarks/mbpp
          contest: /data/benchmarks/codecontests
          humaneval: /data/benchmarks/humaneval
    """

    mbpp: str = Field(
        default="dataset/mbpp",
        description="MBPP 데이터셋 경로 (로컬 또는 S3 URI)",
    )
    contest: str = Field(
        default="dataset/contest",
        description="CodeContests 데이터셋 경로 (로컬 또는 S3 URI)",
    )
    humaneval: str = Field(
        default="dataset/humaneval",
        description="HumanEval 데이터셋 경로 (로컬 또는 S3 URI)",
    )


class Paths(BaseModel):
    """전체 경로 설정 통합.

    WMTP 프로젝트에서 사용하는 모든 경로를 한 곳에서 관리합니다.
    PathResolver를 통해 로컬 경로와 S3 URI를 자동으로 구분합니다.

    Attributes:
        models: 모델 파일 경로들 (로컬 또는 S3)
        datasets: 데이터셋 경로들 (로컬 또는 S3)

    Note:
        S3 체크포인트 전략 사용 - 모든 중간 결과는 S3에 직접 저장됩니다.
    """

    models: ModelPaths = Field(default_factory=ModelPaths)
    datasets: DatasetPaths = Field(default_factory=DatasetPaths)


class MLflow(BaseModel):
    """MLflow 실험 추적 설정.

    MLflow를 사용한 실험 관리 및 모델 버전 관리 설정입니다.
    WMTP의 모든 학습 실험은 MLflow로 추적됩니다.

    Attributes:
        experiment: 실험 이름/경로
            - 계층 구조 지원: "project/subproject/experiment"
            - 기본값: "mtp/wmtp"
            - 팀별로 다른 namespace 사용 권장

        tracking_uri: 실험 추적 서버 URI (필수)
            - 로컬: "file://./mlflow_runs" 또는 "./mlflow_runs"
            - S3: "s3://bucket/path/mlflow"
            - 원격: "http://mlflow-server:5000"

        registry_uri: 모델 레지스트리 URI (필수)
            - 학습된 모델 버전 관리
            - 보통 tracking_uri와 동일하게 설정
            - 프로덕션 모델 배포 시 사용

        artifact_location: 아티팩트 저장 위치 (선택)
            - MLflow 아티팩트 (모델, 플롯, 파일) 저장 위치
            - S3: "s3://bucket/mlflow-artifacts"
            - 로컬: "./mlflow-artifacts"
            - None이면 tracking_uri 기본 위치 사용

    Example:
        # 로컬 개발
        mlflow:
          experiment: dev/wmtp
          tracking_uri: ./mlflow_runs
          registry_uri: ./mlflow_runs

        # 프로덕션 (아티팩트 별도 저장)
        mlflow:
          experiment: prod/wmtp
          tracking_uri: s3://ml-artifacts/mlflow
          registry_uri: s3://ml-artifacts/mlflow
          artifact_location: s3://ml-artifacts/mlflow-artifacts

    Note:
        환경 변수 사용 가능: ${MLFLOW_TRACKING_URI}
    """

    experiment: str = Field(
        default="mtp/wmtp",
        description="MLflow 실험 이름",
    )
    tracking_uri: str = Field(
        ...,
        description="MLflow 추적 서버 URI (원격의 경우 S3 필수)",
    )
    registry_uri: str = Field(
        ...,
        description="MLflow 모델 레지스트리 URI (원격의 경우 S3 필수)",
    )
    artifact_location: str | None = Field(
        default=None,
        description="MLflow 아티팩트 저장 위치 (None이면 tracking_uri 기본 위치 사용)",
    )

    @field_validator("tracking_uri", "registry_uri")
    @classmethod
    def validate_uri(cls, v: str) -> str:
        """Validate and normalize MLflow URIs for both local and remote environments."""

        # 로컬 개발: 상대 경로를 절대 경로로 변환
        if v.startswith("file://./"):
            # file://./mlflow_runs → file:///absolute/path/to/mlflow_runs
            relative_path = v[7:]  # "./mlflow_runs" 추출
            absolute_path = Path.cwd() / relative_path
            return f"file://{absolute_path.resolve()}"

        # 환경 변수 치환 후 상대 경로 처리 (예: ./mlflow_runs)
        if v.startswith("./") or (
            not v.startswith(("s3://", "file://", "http://", "https://")) and "/" in v
        ):
            # 상대 경로를 file:// URI로 변환
            path = Path(v).resolve()
            return f"file://{path}"

        # 표준 URI 형식 검증
        if not v.startswith(("s3://", "file://", "http://", "https://")):
            raise ValueError(
                f"Invalid MLflow URI: {v}. Must start with s3://, file://, http://, or https://"
            )
        return v


class LauncherResources(BaseModel):
    """런처 리소스 요구사항.

    학습 작업에 필요한 하드웨어 리소스를 명시합니다.
    VESSL 같은 클라우드 플랫폼에서 자동 할당에 사용됩니다.

    Attributes:
        gpus: GPU 개수
            - 0: CPU 전용 모드
            - 4: 7B 모델 full fine-tuning 권장
            - 8: 13B+ 모델 또는 빠른 학습

        gpu_type: GPU 종류
            - "A100": 최고 성능 (80GB VRAM)
            - "A6000": 비용 효율적 (48GB VRAM)
            - "V100": 구세대 (32GB VRAM)
            - "RTX4090": 개인 워크스테이션용

        cpu_limit: CPU 리소스 제한 (VESSL 형식)
            - 문자열 형태: "32" 또는 "16"
            - Kubernetes resource limit 형식

        memory_limit: 메모리 제한 (VESSL 형식)
            - 문자열 형태: "256Gi" 또는 "128Gi"
            - Kubernetes resource limit 형식

        shm_size: 공유 메모리 크기 (VESSL/Docker)
            - 문자열 형태: "32Gi"
            - 분산 훈련에서 중요

    Example:
        # VESSL A100x2 설정
        resources:
          gpus: 2
          gpu_type: A100
          cpu_limit: "32"
          memory_limit: "256Gi"
          shm_size: "32Gi"
    """

    gpus: int = Field(default=2, ge=0, description="GPU 개수")
    gpu_type: str = Field(default="A100", description="GPU 종류")
    cpu_limit: str = Field(default="32", description="CPU 리소스 제한")
    memory_limit: str = Field(default="256Gi", description="메모리 제한")
    shm_size: str = Field(default="32Gi", description="공유 메모리 크기")


class Launcher(BaseModel):
    """실행 환경 설정.

    WMTP 학습을 실행할 환경을 설정합니다.

    Attributes:
        target: 실행 대상 플랫폼
            - "local": 로컬 머신에서 직접 실행
                * 개발 및 디버깅용
                * 자체 GPU 서버 사용
            - "vessl": VESSL 클라우드 플랫폼
                * 대규모 실험용
                * 자동 리소스 할당
                * 실험 관리 UI 제공

        resources: 필요한 하드웨어 리소스
            - local에서는 참고용
            - vessl에서는 자동 할당에 사용

    Example:
        # 로컬 실행
        launcher:
          target: local
          resources:
            gpus: 2  # 실제 사용 가능한 GPU 수

        # VESSL 클라우드
        launcher:
          target: vessl
          resources:
            gpus: 8
            gpu_type: A100
    """

    target: Literal["local", "vessl"] = Field(..., description="실행 대상 플랫폼")
    resources: LauncherResources = Field(default_factory=LauncherResources)


class FSDPConfig(BaseModel):
    """FSDP(Fully Sharded Data Parallel) 설정.

    대규모 모델을 여러 GPU에 분산시켜 학습하는 PyTorch FSDP 설정입니다.
    메모리 효율성과 학습 속도를 최적화합니다.

    Attributes:
        enabled: FSDP 활성화 여부
            - True: 다중 GPU에서 모델 분할 (권장)
            - False: 데이터 병렬 처리만 사용

        auto_wrap: 자동 레이어 래핑
            - True: 트랜스포머 블록 자동 감지 및 분할
            - False: 수동 래핑 (고급 사용자용)

        activation_ckpt: 활성화 체크포인팅
            - True: 메모리 절약 (속도 약간 감소)
            - False: 빠른 속도 (메모리 많이 사용)
            - 7B+ 모델에서는 True 권장

        sharding: 샤딩 전략
            - "full": 파라미터, 그래디언트, 옵티마이저 상태 모두 분할
                * 최대 메모리 효율
                * 통신 오버헤드 있음
            - "shard_grad_op": 그래디언트와 옵티마이저 상태만 분할
                * 중간 메모리 효율
                * 통신 오버헤드 적음

    Note:
        FSDP는 PyTorch 2.0+ 에서 안정적으로 작동합니다.
        단일 GPU에서는 자동으로 비활성화됩니다.

    Example:
        # 메모리 최적화 (큰 모델)
        fsdp:
          enabled: true
          auto_wrap: true
          activation_ckpt: true
          sharding: full

        # 속도 최적화 (작은 모델)
        fsdp:
          enabled: true
          auto_wrap: true
          activation_ckpt: false
          sharding: shard_grad_op
    """

    enabled: bool = Field(default=True, description="FSDP 활성화")
    auto_wrap: bool = Field(default=True, description="자동 래핑 활성화")
    activation_ckpt: bool = Field(default=True, description="활성화 체크포인팅 활성화")
    sharding: Literal["full", "full_shard", "shard_grad_op"] = Field(
        default="full_shard", description="FSDP 샤딩 전략"
    )

    @field_validator("sharding")
    @classmethod
    def validate_sharding_strategy(cls, v: str) -> str:
        """FSDP 샤딩 전략 정규화."""
        # config 파일에서 full_shard로 올 수 있음
        if v == "full_shard":
            return "full"
        return v


class Devices(BaseModel):
    """디바이스 및 분산 학습 설정.

    학습에 사용할 하드웨어와 최적화 설정을 정의합니다.

    Attributes:
        compute_backend: 연산 백엔드
            - "cuda": NVIDIA GPU (가장 일반적)
            - "mps": Apple Silicon GPU (M1/M2/M3)
            - "cpu": CPU 전용 (매우 느림)
            - "auto": 자동 감지 (권장)

        device_ids: 사용할 GPU ID 목록
            - None: 모든 사용 가능한 GPU 자동 사용
            - [0, 1]: GPU 0번과 1번만 사용
            - [2, 3, 4, 5]: GPU 2~5번 사용

        mixed_precision: 혼합 정밀도 모드
            - "bf16": Brain Float 16 (A100, RTX30xx+ 권장)
                * 안정적, 넓은 표현 범위
                * 메모리 50% 절감
            - "fp16": Float 16 (V100, 구형 GPU)
                * 빠르지만 overflow 가능성
                * 메모리 50% 절감
            - "fp32": Float 32 (전체 정밀도)
                * 가장 안정적
                * 메모리 많이 사용

        fsdp: FSDP 분산 학습 설정

    Example:
        # NVIDIA GPU 자동 설정
        devices:
          compute_backend: auto
          mixed_precision: bf16

        # 특정 GPU만 사용
        devices:
          compute_backend: cuda
          device_ids: [0, 1, 2, 3]
          mixed_precision: bf16

        # Apple Silicon
        devices:
          compute_backend: mps
          mixed_precision: fp16  # MPS는 bf16 미지원
    """

    compute_backend: Literal["cuda", "mps", "cpu", "auto"] = Field(
        default="auto", description="연산 백엔드 (auto=런타임 자동 감지)"
    )
    device_ids: list[int] | None = Field(
        default=None, description="사용할 특정 디바이스 ID (None=자동 감지)"
    )
    mixed_precision: Literal["bf16", "fp16", "fp32"] = Field(
        default="bf16", description="혼합 정밀도 모드"
    )
    fsdp: FSDPConfig = Field(default_factory=FSDPConfig)

    @field_validator("mixed_precision")
    @classmethod
    def validate_precision(cls, v: str) -> str:
        """정밀도 모드 지원 검증.

        선택한 혼합 정밀도가 하드웨어에서 지원되는지 확인합니다.

        Args:
            v: 정밀도 모드 (bf16/fp16/fp32)

        Returns:
            str: 검증된 정밀도 모드

        Note:
            - bf16은 Ampere 이상 GPU 필요 (A100, A6000, RTX 30xx+)
            - 실제 검증은 런타임에 하드웨어 확인 후 수행
            - 지원하지 않는 경우 자동으로 fp16 또는 fp32로 폴백
        """
        if v == "bf16":
            # bf16 지원 GPU: A100, A6000, RTX 3090, RTX 4090 등
            # 런타임에 torch.cuda.is_bf16_supported()로 확인
            pass
        return v

    @field_validator("compute_backend")
    @classmethod
    def validate_compute_backend(cls, v: str) -> str:
        """연산 백엔드 호환성 검증.

        선택한 백엔드가 현재 시스템에서 사용 가능한지 확인합니다.

        Args:
            v: 연산 백엔드 (cuda/mps/cpu/auto)

        Returns:
            str: 검증된 백엔드

        Note:
            - 사용 불가능한 백엔드는 경고만 출력
            - 실제 폴백은 런타임에 처리
            - auto 선택 시 우선순위: cuda > mps > cpu
        """
        import torch

        if v == "cuda" and not torch.cuda.is_available():
            # CUDA 사용 불가 - 런타임에 CPU로 폴백
            print("경고: CUDA를 사용할 수 없습니다. 런타임에 CPU로 전환될 수 있습니다.")
        elif v == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            # Apple Silicon MPS 사용 불가
            print(
                "경고: MPS를 사용할 수 없습니다. Apple Silicon이 아니거나 PyTorch 버전이 낮습니다."
            )
        return v

    @model_validator(mode="after")
    def validate_device_compatibility(self):
        """환경별 디바이스 설정 호환성 검증 및 자동 조정.

        하드웨어와 설정 간의 호환성을 확인하고
        필요시 자동으로 최적 설정으로 조정합니다.

        자동 조정 규칙:
        1. MPS + bf16 → fp16으로 변경 (MPS는 bf16 미지원)
        2. 단일 GPU + FSDP → FSDP 비활성화 (불필요)
        3. CPU + mixed_precision → fp32로 변경 (CPU는 혼합정밀도 불필요)

        Returns:
            self: 조정된 설정
        """
        # MPS는 bf16을 지원하지 않음 - fp16으로 자동 변경
        if self.compute_backend == "mps" and self.mixed_precision == "bf16":
            print("정보: MPS는 bf16을 지원하지 않아 fp16으로 자동 변경됩니다.")
            self.mixed_precision = "fp16"

        # 단일 GPU 환경에서는 FSDP 불필요 - 자동 비활성화
        if self.device_ids and len(self.device_ids) == 1:
            if self.fsdp.enabled:
                print("정보: 단일 GPU 환경에서 FSDP를 자동으로 비활성화합니다.")
                self.fsdp.enabled = False

        # CPU 모드에서는 mixed precision 의미 없음
        if self.compute_backend == "cpu" and self.mixed_precision != "fp32":
            print("정보: CPU 모드에서는 fp32 정밀도를 사용합니다.")
            self.mixed_precision = "fp32"

        return self


class Config(BaseModel):
    """환경 설정 루트 스키마.

    config.yaml 파일의 최상위 구조를 정의합니다.
    WMTP 프로젝트의 모든 환경 관련 설정을 포함합니다.

    Attributes:
        project: 프로젝트 이름
            - MLflow와 로그에 사용
            - 여러 실험을 구분하는 식별자
            - 예: "wmtp_dev", "wmtp_prod", "wmtp_ablation"

        seed: 난수 시드
            - 재현 가능한 실험을 위한 고정 시드
            - 42: 과학계의 전통적인 기본값
            - 같은 시드 = 같은 결과

        storage: 스토리지 설정 (local/s3)
        paths: 모델/데이터/캐시 경로
        mlflow: 실험 추적 설정
        launcher: 실행 환경 설정
        devices: GPU/CPU 및 분산 학습 설정

    Example:
        project: wmtp_experiment_v1
        seed: 42
        storage:
          mode: local
        paths:
          models:
            base_local: models/7b_mtp
        mlflow:
          experiment: wmtp/baseline
          tracking_uri: ./mlruns
          registry_uri: ./mlruns
        launcher:
          target: local
        devices:
          compute_backend: auto
          mixed_precision: bf16

    Note:
        필수 필드는 storage, mlflow, launcher입니다.
        나머지는 기본값이 제공됩니다.
    """

    project: str = Field(default="mtp_ft", description="프로젝트 이름")
    seed: int = Field(default=42, description="재현성을 위한 난수 시드")
    storage: Storage = Field(..., description="스토리지 설정")
    paths: Paths = Field(default_factory=Paths, description="경로 설정")
    mlflow: MLflow = Field(..., description="MLflow 설정")
    launcher: Launcher = Field(..., description="런처 설정")
    devices: Devices = Field(default_factory=Devices, description="디바이스 설정")

    model_config = ConfigDict(
        extra="forbid",  # 정의되지 않은 필드 금지 (오타 방지)
        str_strip_whitespace=True,  # 문자열 앞뒤 공백 자동 제거
        validate_assignment=True,  # 할당 시에도 검증 수행
        use_enum_values=True,  # Enum 대신 실제 값 사용
    )

    @field_validator("seed")
    @classmethod
    def validate_seed(cls, v: int) -> int:
        """시드 값 검증.

        난수 시드가 유효한 범위인지 확인합니다.

        Args:
            v: 입력된 시드 값

        Returns:
            int: 검증된 시드 값

        Raises:
            ValueError: 음수 시드인 경우

        Note:
            시드 선택 팁:
            - 0: 가장 단순한 시드
            - 42: 과학계 전통
            - 날짜 기반: 20241225 (재현 가능하면서 구분 가능)
        """
        if v < 0:
            raise ValueError("시드는 0 이상의 정수여야 합니다.\n" "예: seed: 42")
        return v
