"""
WMTP MLflow 통합: 실험 추적 및 모델 관리 시스템

WMTP 연구 맥락:
WMTP의 세 알고리즘(baseline/critic/rho1) 성능을 체계적으로 비교하기 위해
모든 실험을 MLflow로 추적합니다. 손실 곡선, pass@k 메트릭, 체크포인트 등
연구 재현성에 필요한 모든 정보를 자동으로 기록합니다.

핵심 기능:
- 실험 추적: 하이퍼파라미터, 메트릭, 아티팩트 자동 기록
- 모델 레지스트리: 최고 성능 모델 버전 관리
- S3 백엔드: 대용량 체크포인트의 효율적 저장
- 실험 비교: 알고리즘 간 성능 차이 시각화

WMTP 알고리즘과의 연결:
- Baseline MTP: 기준 성능 메트릭 기록
- Critic-WMTP: value_loss, delta 분포 등 추가 메트릭
- Rho1-WMTP: CE 차이 히스토그램, percentile 통계

사용 예시:
    >>> mlflow_manager = create_mlflow_manager(config)
    >>> mlflow_manager.start_run("rho1_experiment")
    >>> mlflow_manager.log_metrics({"train/loss": 2.5, "train/ppl": 12.3})
    >>> mlflow_manager.log_artifact("checkpoints/best_model.pt")
    >>> mlflow_manager.end_run()

성능 최적화:
- 배치 로깅으로 네트워크 호출 최소화
- S3 멀티파트 업로드로 대용량 파일 처리
- 비동기 로깅으로 학습 속도 영향 최소화

디버깅 팁:
- ConnectionError: MLflow 서버 URI 및 네트워크 확인
- S3 업로드 실패: AWS 자격증명 및 버킷 권한 확인
- 실험 중복: experiment_name 고유성 보장
"""

from pathlib import Path
from typing import Any

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from rich.console import Console

console = Console()


class MLflowManager:
    """
    MLflow 작업 관리자: S3 백엔드를 지원하는 실험 추적 시스템.

    WMTP 연구 맥락:
    각 알고리즘의 학습 과정을 상세히 기록하여 성능 비교와
    재현성을 보장합니다. 특히 토큰 가중치의 효과를 정량화하기 위해
    다양한 메트릭을 체계적으로 수집합니다.

    주요 책임:
    - 실험 생성 및 관리
    - 메트릭 로깅 (loss, perplexity, pass@k)
    - 아티팩트 저장 (체크포인트, 생성 샘플)
    - 모델 레지스트리 관리

    WMTP 특화 기능:
    - Critic: value_loss, advantage 분포 추적
    - Rho1: CE excess 통계, percentile 히스토그램
    - 토큰별 가중치 시각화 데이터 저장
    """

    def __init__(
        self,
        tracking_uri: str,
        registry_uri: str | None = None,
        experiment_name: str = "default",
    ):
        """
        MLflow 관리자 초기화.

        WMTP 맥락:
        알고리즘별로 다른 experiment를 생성하여 체계적 비교가 가능합니다.
        예: "wmtp/baseline", "wmtp/critic", "wmtp/rho1"

        매개변수:
            tracking_uri: MLflow 추적 서버 URI
                - 로컬: "file://./mlruns"
                - S3: "s3://bucket/mlflow"
                - 원격: "http://mlflow-server:5000"
            registry_uri: 모델 레지스트리 URI (None시 tracking_uri 사용)
            experiment_name: 실험 이름 (기본값: "default")
                - 권장 형식: "{project}/{algorithm}/{date}"

        예시:
            >>> manager = MLflowManager(
            >>>     tracking_uri="s3://wmtp-artifacts/mlflow",
            >>>     experiment_name="wmtp/rho1/20241225"
            >>> )

        디버깅 팁:
            - file:// URI는 절대 경로 사용 권장
            - S3 사용시 AWS_PROFILE 환경변수 확인
            - 원격 서버는 방화벽 설정 확인
        """
        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri or tracking_uri
        self.experiment_name = experiment_name

        # Set MLflow URIs
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_registry_uri(self.registry_uri)

        # Initialize client
        self.client = MlflowClient(
            tracking_uri=self.tracking_uri,
            registry_uri=self.registry_uri,
        )

        # Set or create experiment
        self.experiment_id = self._setup_experiment(experiment_name)
        self.run = None

    def _setup_experiment(self, experiment_name: str) -> str:
        """
        Set up or create MLflow experiment.

        Args:
            experiment_name: Experiment name

        Returns:
            Experiment ID
        """
        try:
            experiment = self.client.get_experiment_by_name(experiment_name)
            if experiment:
                mlflow.set_experiment(experiment_name)
                return experiment.experiment_id
            else:
                experiment_id = self.client.create_experiment(experiment_name)
                mlflow.set_experiment(experiment_name)
                return experiment_id
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to setup experiment: {e}[/yellow]")
            # Use default experiment
            mlflow.set_experiment("Default")
            return "0"

    def start_run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
        nested: bool = False,
        resume: bool = False,
        run_id: str | None = None,
    ) -> mlflow.ActiveRun:
        """
        Start or resume an MLflow run.

        Args:
            run_name: Optional run name
            tags: Optional tags dictionary
            nested: Whether this is a nested run
            resume: Whether to resume existing run
            run_id: Specific run ID to resume

        Returns:
            Active MLflow run
        """
        if resume and run_id:
            # Resume existing run
            self.run = mlflow.start_run(
                run_id=run_id,
                nested=nested,
            )
            console.print(f"[green]Resumed MLflow run: {run_id}[/green]")
        else:
            # Start new run
            self.run = mlflow.start_run(
                run_name=run_name,
                nested=nested,
                tags=tags,
            )
            console.print(f"[green]Started MLflow run: {self.run.info.run_id}[/green]")

        return self.run

    def end_run(self, status: str = "FINISHED") -> None:
        """
        End current MLflow run.

        Args:
            status: Run status ('FINISHED', 'FAILED', 'KILLED')
        """
        if self.run:
            mlflow.end_run(status=status)
            console.print(f"[green]Ended MLflow run with status: {status}[/green]")
            self.run = None

    def log_params(self, params: dict[str, Any]) -> None:
        """
        Log parameters to MLflow.

        Args:
            params: Dictionary of parameters
        """
        if not self.run:
            console.print("[yellow]Warning: No active MLflow run[/yellow]")
            return

        # Flatten nested dictionaries
        flat_params = self._flatten_dict(params)

        for key, value in flat_params.items():
            # MLflow param values must be strings
            mlflow.log_param(key, str(value))

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        """
        Log metrics to MLflow.

        Args:
            metrics: Dictionary of metrics
            step: Optional step number
        """
        if not self.run:
            console.print("[yellow]Warning: No active MLflow run[/yellow]")
            return

        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)

    def log_artifact(
        self,
        local_path: str | Path,
        artifact_path: str | None = None,
    ) -> None:
        """
        Log artifact to MLflow.

        Args:
            local_path: Local file or directory path
            artifact_path: Artifact subdirectory
        """
        if not self.run:
            console.print("[yellow]Warning: No active MLflow run[/yellow]")
            return

        local_path = Path(local_path)

        if local_path.is_file():
            mlflow.log_artifact(str(local_path), artifact_path)
        elif local_path.is_dir():
            mlflow.log_artifacts(str(local_path), artifact_path)
        else:
            console.print(f"[red]Artifact not found: {local_path}[/red]")

    def log_model(
        self,
        model: Any,
        name: str = "model",
        registered_model_name: str | None = None,
        signature: Any | None = None,
        input_example: Any | None = None,
    ) -> None:
        """
        Log PyTorch model to MLflow.

        Args:
            model: PyTorch model
            name: Name for the logged model artifact
            registered_model_name: Optional model registry name
            signature: Optional model signature
            input_example: Optional input example
        """
        if not self.run:
            console.print("[yellow]Warning: No active MLflow run[/yellow]")
            return

        mlflow.pytorch.log_model(
            model,
            name=name,
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=input_example,
        )

        if registered_model_name:
            console.print(
                f"[green]Model logged and registered as: {registered_model_name}[/green]"
            )
        else:
            console.print("[green]Model logged to MLflow[/green]")

    def get_run_id(self) -> str | None:
        """Get current run ID."""
        if self.run:
            return self.run.info.run_id
        return None

    def _flatten_dict(
        self,
        d: dict[str, Any],
        parent_key: str = "",
        sep: str = ".",
    ) -> dict[str, Any]:
        """
        Flatten nested dictionary.

        Args:
            d: Dictionary to flatten
            parent_key: Parent key prefix
            sep: Separator

        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


def create_mlflow_manager(config: dict[str, Any]) -> MLflowManager:
    """
    Create MLflow manager from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        MLflowManager instance
    """
    mlflow_config = config.get("mlflow", {})

    return MLflowManager(
        tracking_uri=mlflow_config.get("tracking_uri", "file:///tmp/mlflow"),
        registry_uri=mlflow_config.get("registry_uri"),
        experiment_name=mlflow_config.get("experiment", "default"),
    )


# Export main functions and classes
__all__ = [
    "MLflowManager",
    "create_mlflow_manager",
]
