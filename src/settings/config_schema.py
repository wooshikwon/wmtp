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
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.utils.path_resolver import PathResolver


class S3AuthConfig(BaseModel):
    """S3 인증 설정: AWS S3 접근을 위한 인증 정보

    Phase 2 리팩토링:
    storage.mode를 제거하고 경로에 직접 프로토콜을 포함하도록 변경했습니다.
    이제 S3AuthConfig는 S3 접근을 위한 인증 정보만 담당합니다.

    사용 예시:
    - s3://wmtp/models/7b_1t_4 경로 사용 시 자동으로 S3 접근
    - 기본 버킷과 리전 정보만 제공 (경로별 버킷은 URI에서 추출)

    Attributes:
        default_bucket: 기본 S3 버킷 이름 (선택)
            경로에 버킷이 명시되지 않은 경우 사용
            예: "wmtp-models"
        region: AWS 리전 (기본값: ap-northeast-2)
            서울 리전이 기본값
        profile: AWS 프로파일 이름 (선택)
            ~/.aws/credentials의 프로파일 사용

    Example:
        # S3 인증 정보만 설정 (경로는 직접 s3:// 사용)
        s3_auth:
          default_bucket: wmtp
          region: ap-northeast-2

    Note:
        환경변수 AWS_PROFILE, AWS_REGION도 자동으로 인식됩니다.
    """

    # AWS 인증 정보 (런타임에 .env에서 주입됨)
    access_key_id: str | None = Field(
        default=None, description="AWS Access Key ID (.env에서 주입)"
    )
    secret_access_key: str | None = Field(
        default=None, description="AWS Secret Access Key (.env에서 주입)"
    )

    # S3 설정
    default_bucket: str | None = Field(
        default=None, description="기본 S3 버킷 (경로에 버킷이 없을 때 사용)"
    )
    region: str = Field(default="ap-northeast-2", description="AWS 리전")
    profile: str | None = Field(
        default=None, description="AWS 프로파일 이름 (None이면 기본 프로파일)"
    )

    @field_validator("default_bucket")
    @classmethod
    def validate_bucket(cls, v: str | None) -> str | None:
        """S3 버킷 이름 검증.

        AWS S3 버킷 명명 규칙을 검증합니다:
        - 최대 63자
        - 소문자로 자동 변환

        Args:
            v: 입력된 버킷 이름

        Returns:
            str | None: 검증되고 정규화된 버킷 이름 (소문자) 또는 None
        """
        if v is None:
            return None
        if len(v) > 63:
            raise ValueError("S3 버킷 이름은 63자 이하여야 합니다")
        # AWS는 버킷 이름을 소문자로 요구
        return v.lower()


# Storage 클래스 제거됨 - Phase 2 리팩토링
# 경로에 직접 프로토콜(s3://, file://)을 포함하는 방식으로 변경
# S3 인증 정보는 Config.s3_auth로 이동


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


class CheckpointConfig(BaseModel):
    """체크포인트 저장 설정

    WMTP 맥락:
    학습 중간 상태를 저장하여 재개 가능하게 하는 체크포인트 관리 설정입니다.
    로컬/S3 경로 모두 지원하며, Phase 1에서 구현한 S3 기능을 활용합니다.

    Attributes:
        base_path: 체크포인트 기본 저장 경로
            - file://./checkpoints: 로컬 상대 경로
            - file:///absolute/path: 로컬 절대 경로
            - s3://bucket/path: S3 경로
            - 기본값: "file://./checkpoints"

        save_interval: 체크포인트 저장 간격 (steps)
            - 매 N 스텝마다 중간 체크포인트 저장
            - 기본값: 500

        keep_last: 보관할 체크포인트 개수
            - 디스크 공간 절약을 위해 오래된 체크포인트 자동 삭제
            - 기본값: 3

        save_final: 최종 모델 저장 여부
            - 학습 완료 시 final_model.pt 저장
            - 기본값: True

    Example:
        checkpoints:
          base_path: "s3://wmtp/checkpoints"
          save_interval: 1000
          keep_last: 5
          save_final: true
    """

    base_path: str = Field(
        default="file://./checkpoints",
        description="체크포인트 기본 저장 경로 (file:// 또는 s3://)",
    )
    save_interval: int = Field(default=500, description="체크포인트 저장 간격 (steps)")
    keep_last: int = Field(default=3, description="보관할 체크포인트 개수")
    save_final: bool = Field(default=True, description="최종 모델 저장 여부")


class Paths(BaseModel):
    """전체 경로 설정 통합.

    Phase 2 리팩토링:
    모든 경로에 프로토콜을 직접 포함할 수 있습니다:
    - s3://bucket/key: S3 경로
    - file:///absolute/path: 명시적 로컬 절대 경로
    - file://./relative/path: 명시적 로컬 상대 경로
    - ./path 또는 /path: 암시적 로컬 경로

    Attributes:
        models: 모델 파일 경로들 (프로토콜 포함 가능)
        datasets: 데이터셋 경로들 (프로토콜 포함 가능)
        checkpoints: 체크포인트 저장 설정 (Phase 1에서 추가)

    Example:
        paths:
          models:
            base: s3://wmtp/models/7b_1t_4/
            rm: s3://wmtp/models/Llama_3_8B_RM/
            ref: file://./local_models/sheared_llama/
          datasets:
            mbpp: s3://wmtp/dataset/mbpp
            contest: file://./dataset/contest
          checkpoints:
            base_path: s3://wmtp/checkpoints
            save_interval: 500
            keep_last: 3
    """

    models: ModelPaths = Field(default_factory=ModelPaths)
    datasets: DatasetPaths = Field(default_factory=DatasetPaths)
    checkpoints: CheckpointConfig = Field(default_factory=CheckpointConfig)

    @field_validator("models", "datasets", "checkpoints")
    @classmethod
    def validate_paths(
        cls, v: ModelPaths | DatasetPaths | CheckpointConfig, info
    ) -> ModelPaths | DatasetPaths | CheckpointConfig:
        """경로 프로토콜 검증 및 정규화.

        Phase 2 핵심 기능:
        PathResolver를 사용하여 모든 경로의 프로토콜을 검증합니다.
        잘못된 프로토콜이나 형식은 에러를 발생시킵니다.

        Phase 1 확장:
        CheckpointConfig의 base_path도 동일한 검증 로직을 적용합니다.

        Args:
            v: ModelPaths, DatasetPaths, 또는 CheckpointConfig 인스턴스
            info: Pydantic validation context

        Returns:
            검증된 경로 객체

        Raises:
            ValueError: 잘못된 경로 형식
        """
        resolver = PathResolver()
        field_name = info.field_name

        # 각 경로 검증 (빈 문자열은 허용)
        for attr_name in v.model_fields:
            path = getattr(v, attr_name)
            # 문자열 타입이고 비어있지 않은 경우만 검증
            if path and isinstance(path, str) and path.strip():
                try:
                    # 경로 해석 시도 (프로토콜 검증)
                    path_type, resolved = resolver.resolve(path)

                    # S3 경로의 경우 버킷과 키 검증
                    if path_type == "s3":
                        bucket, key = resolver.extract_bucket_and_key(path)
                        if not bucket:
                            raise ValueError(
                                f"{field_name}.{attr_name}: S3 경로에 버킷이 없습니다: {path}"
                            )
                        # S3 경로는 최소한 버킷과 키 구조를 가져야 함 (s3://bucket/key)
                        if not path.endswith("/") and not key:
                            raise ValueError(
                                f"{field_name}.{attr_name}: S3 경로가 잘못된 형식입니다. s3://bucket/key 또는 s3://bucket/ 형태여야 합니다: {path}"
                            )
                except Exception as e:
                    raise ValueError(
                        f"{field_name}.{attr_name}: 잘못된 경로 형식: {path}\n"
                        f"오류: {str(e)}"
                    ) from e

        return v


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


class DistributedConfig(BaseModel):
    """분산 학습 환경 설정.

    분산 훈련의 정책 설정만 관리합니다.
    실제 rank/world_size/local_rank는 torchrun이 환경변수로 제공합니다.

    torchrun 사용 예시:
        torchrun --nproc_per_node=4 --nnodes=1 -m src.cli train --config config.yaml

    torchrun이 자동 설정하는 환경변수:
        - RANK: 전체 프로세스 중 순번 (0, 1, 2, 3)
        - WORLD_SIZE: 총 프로세스 수 (4)
        - LOCAL_RANK: 노드 내 GPU 순번 (0, 1, 2, 3)
        - MASTER_ADDR, MASTER_PORT: 마스터 노드 정보
    """

    enabled: bool = Field(
        default=False, description="분산 학습 활성화 (True: 멀티 GPU, False: 단일 GPU)"
    )

    backend: Literal["nccl", "gloo", "auto"] = Field(
        default="auto",
        description="분산 통신 백엔드 (nccl: GPU, gloo: CPU, auto: 자동감지)",
    )

    init_method: str = Field(
        default="env://",
        description="분산 초기화 방법 (env:// 권장, torchrun 환경변수 사용)",
    )

    timeout: int = Field(
        default=1800, description="분산 통신 타임아웃 (초, NCCL 기본값)"
    )

    find_unused_parameters: bool = Field(
        default=False,
        description="사용되지 않는 파라미터 검색 (FSDP 환경에서 False 권장)",
    )

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, v: str) -> str:
        """분산 백엔드 호환성 검증."""
        if v == "auto":
            # 런타임에 자동 감지: cuda 사용가능하면 nccl, 아니면 gloo
            return v
        elif v == "nccl":
            # NVIDIA GPU 전용, 가장 빠른 통신
            return v
        elif v == "gloo":
            # CPU/GPU 범용, 느리지만 안정적
            return v
        else:
            raise ValueError(f"지원하지 않는 백엔드: {v}. nccl, gloo, auto 중 선택")


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

    # 🆕 분산 학습 설정 추가
    distributed: DistributedConfig = Field(
        default_factory=DistributedConfig, description="분산 학습 설정"
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
        if self.device_ids and len(self.device_ids) == 1 and self.fsdp.enabled:
            print("정보: 단일 GPU 환경에서 FSDP를 자동으로 비활성화합니다.")
            self.fsdp.enabled = False

        # CPU 모드에서는 mixed precision 의미 없음
        if self.compute_backend == "cpu" and self.mixed_precision != "fp32":
            print("정보: CPU 모드에서는 fp32 정밀도를 사용합니다.")
            self.mixed_precision = "fp32"

        # 🆕 분산 설정과 FSDP 설정 간 일관성 검증
        from rich.console import Console

        console = Console()

        # 분산이 활성화되지 않으면 FSDP도 단일 GPU 모드로 조정
        if not self.distributed.enabled and self.fsdp.enabled:
            console.print(
                "[yellow]단일 GPU 환경에서 FSDP는 효과가 제한적입니다.[/yellow]"
            )

        # FSDP가 활성화되면 분산도 활성화하는 것이 일반적
        if self.fsdp.enabled and not self.distributed.enabled:
            console.print(
                "[yellow]FSDP 사용 시 분산 학습을 함께 활성화하는 것을 권장합니다.[/yellow]"
            )

        return self


class HFAuthConfig(BaseModel):
    """HuggingFace 인증 설정: HF Hub 접근을 위한 토큰 정보

    사용자가 제한된 모델이나 데이터셋에 접근할 때 필요합니다.

    Attributes:
        token: HuggingFace 액세스 토큰
            - hf_xxx 형태의 토큰
            - 읽기 전용 또는 쓰기 권한 토큰 모두 지원
    """

    token: str = Field(description="HuggingFace 액세스 토큰")


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

        s3_auth: S3 인증 설정 (S3 경로 사용 시)
        hf_auth: HuggingFace 인증 설정 (HF 모델 사용 시)
        paths: 모델/데이터/캐시 경로
        mlflow: 실험 추적 설정
        launcher: 실행 환경 설정
        devices: GPU/CPU 및 분산 학습 설정

    Example:
        project: wmtp_experiment_v1
        seed: 42
        s3_auth:
          default_bucket: wmtp
          region: ap-northeast-2
        hf_auth:
          token: hf_xxxxxxxxxx
        paths:
          models:
            base: s3://wmtp/models/7b_mtp
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
        필수 필드는 mlflow, launcher입니다.
        나머지는 기본값이 제공됩니다.
    """

    project: str = Field(default="mtp_ft", description="프로젝트 이름")
    seed: int = Field(default=42, description="재현성을 위한 난수 시드")
    s3_auth: S3AuthConfig | None = Field(
        default=None, description="S3 인증 설정 (S3 경로 사용 시 필요)"
    )
    hf_auth: HFAuthConfig | None = Field(
        default=None, description="HuggingFace 인증 설정 (HF 모델 사용 시 필요)"
    )
    paths: Paths = Field(default_factory=Paths, description="경로 설정")
    mlflow: MLflow = Field(..., description="MLflow 설정")
    launcher: Launcher = Field(..., description="런처 설정")
    devices: Devices = Field(default_factory=Devices, description="디바이스 설정")

    # 하위 호환성을 위한 storage 필드 (deprecated)
    storage: Any | None = Field(
        default=None,
        description="[Deprecated] Phase 2에서 제거됨. s3_auth와 경로 프로토콜 사용",
    )

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
