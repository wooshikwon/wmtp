"""
WMTP 알고리즘 공통 기반 클래스

BaseWmtpTrainer는 모든 WMTP 알고리즘(mtp-baseline, critic-wmtp, rho1-wmtp)의
공통 기능을 제공하는 추상 클래스입니다.

공통 기능:
- 모델/옵티마이저 초기화 (setup)
- 체크포인트 관리 훈련 루프 (run)
- 체크포인트 저장/관리 (_save_checkpoint, _manage_checkpoints, _save_final_checkpoint)
- MLflow 통합 및 분산 훈련 지원

각 알고리즘별 구현이 필요한 추상 메서드:
- compute_head_weights: 알고리즘별 헤드 가중치 계산
- train_step: 알고리즘별 훈련 스텝 구현
"""

from __future__ import annotations  # Python 3.10+ 타입 힌트 호환성

from abc import abstractmethod  # 추상 메서드
from contextlib import contextmanager  # Context manager 데코레이터
from pathlib import Path  # 경로 처리
from typing import Any  # 범용 타입 힌트

import torch  # PyTorch 딥러닝 프레임워크
import torch.nn as nn  # 신경망 모듈
import torch.nn.functional as F  # 함수형 API (cross_entropy 등)
from rich.console import Console  # 컬러풀한 콘솔 출력
from rich.progress import track  # Progress bar

from src.components.base import BaseComponent  # WMTP 컴포넌트 베이스 클래스
from src.utils import get_dist_manager  # 분산 훈련 매니저

console = Console()  # 전역 콘솔 객체


def compute_weighted_mtp_loss(
    logits: torch.Tensor,  # [B, S, H, V]
    target_labels: torch.Tensor,  # [B, S, H] - 3D 라벨 (MTPDataCollator에서 생성)
    head_weights: torch.Tensor,  # [B, S, H]
    ignore_index: int = -100,
    selection_mask: torch.Tensor | None = None,  # [B, S, H] - 토큰 선택 마스크
    config: dict | None = None,  # 설정 딕셔너리 (MPS 경로 판단용)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    MTPDataCollator 기반 간단화된 WMTP 손실 계산: L_WMTP = Σ(k=0 to H-1) w_{t+k} × CE_k

    3D 라벨을 직접 받아서 헤드별 CE를 계산하고 가중치를 적용합니다.
    복잡한 shift 연산이 제거되어 성능이 대폭 향상됩니다.

    MPS 최적화: config의 gpu_type이 "mps"면 자동으로 MPS 경로를 사용합니다.

    Args:
        logits: [B, S, H, V] - MTP 모델 출력
        target_labels: [B, S, H] - MTPDataCollator에서 생성된 3D 라벨
        head_weights: [B, S, H] - 헤드별 가중치 매트릭스
        ignore_index: 무시할 라벨 값 (-100)
        selection_mask: [B, S, H] - 토큰 선택 마스크 (None이면 모두 1)
        config: 설정 딕셔너리 (MPS 경로 판단용)

    Returns:
        weighted_loss: 가중 평균 손실 (scalar)
        valid_mask: 유효한 위치 마스크 [B, S]
        ce_per_head: 헤드별 CE 손실 [B, S, H]
    """
    from src.utils.mps_optimizer import MPSOptimizer

    B, S, H, V = logits.shape

    # Input validation (간소화)
    if target_labels.shape != (B, S, H):
        raise ValueError(
            f"Expected target_labels shape [B,S,H], got {target_labels.shape}"
        )
    if head_weights.shape != (B, S, H):
        raise ValueError(
            f"Expected head_weights shape [B,S,H], got {head_weights.shape}"
        )

    # Selection mask 기본값
    if selection_mask is None:
        selection_mask = torch.ones_like(target_labels, dtype=torch.float)

    # 유효 라벨 마스크 생성
    valid_mask = (target_labels != ignore_index).float()  # [B, S, H]

    # MPS 경로 판단
    use_mps_path = False
    if config:
        use_mps_path = MPSOptimizer.should_use_mps_path(config)

    # ===== 핵심 분기: MPS vs CUDA =====
    if use_mps_path:
        # MPS 최적화 경로: 헤드별 3D 처리 (view 없이)
        ce_per_head = MPSOptimizer.compute_ce_per_head_mps(
            logits, target_labels, ignore_index
        )
    else:
        # 기존 CUDA 최적화 경로: 4D→2D flatten
        # logits: [B, S, H, V] -> [B*S*H, V]
        # target_labels: [B, S, H] -> [B*S*H]
        logits_flat = logits.view(B * S * H, V)
        target_flat = target_labels.view(B * S * H)

        ce_flat = F.cross_entropy(
            logits_flat, target_flat, ignore_index=ignore_index, reduction="none"
        )  # [B*S*H]

        ce_per_head = ce_flat.view(B, S, H)  # [B, S, H]

    # 마스킹 적용
    effective_mask = valid_mask * selection_mask  # [B, S, H]

    # 가중 CE 계산
    weighted_ce = head_weights * ce_per_head * effective_mask  # [B, S, H]
    effective_weights = head_weights * effective_mask  # [B, S, H]

    # 토큰별 가중 평균 ([B, S] 차원으로 축약)
    token_weighted_ce = weighted_ce.sum(dim=2)  # [B, S]
    token_weights = effective_weights.sum(dim=2).clamp(min=1e-8)  # [B, S]

    # 토큰별 유효성 (최소 하나 헤드가 유효한 경우)
    token_valid_mask = token_weights > 1e-8  # [B, S]

    # 최종 스칼라 손실
    if token_valid_mask.any():
        weighted_loss_per_token = token_weighted_ce / token_weights
        final_loss = (
            weighted_loss_per_token * token_valid_mask.float()
        ).sum() / token_valid_mask.sum()
    else:
        final_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    return final_loss, token_valid_mask, ce_per_head


class BaseWmtpTrainer(BaseComponent):
    """WMTP 알고리즘 공통 기능을 제공하는 추상 기반 클래스.

    연구 철학 "Not All Tokens Are What You Need"의 구현을 위해
    모든 WMTP 알고리즘(mtp-baseline, critic-wmtp, rho1-wmtp)이
    공유하는 기본 기능을 제공합니다.

    🔬 핵심 동작 원리:
        1. 알고리즘별 토큰 가중치 계산 (compute_head_weights - 추상)
        2. 각 MTP 헤드별로 Cross-Entropy 손실 계산
        3. WMTP 공식 적용: L_WMTP = Σ w_{t+k} × CE_k
        4. 체크포인트 관리와 MLflow 통합된 훈련 루프

    공통 제공 기능:
        - 모델/옵티마이저 초기화 및 분산 훈련 설정 (setup)
        - 체크포인트 관리 훈련 루프 (run)
        - 주기적/최종 체크포인트 저장/관리
        - MLflow 실험 추적 통합
        - 혼합 정밀도 및 그래디언트 클리핑

    필수 구현 메서드 (하위 클래스에서):
        - compute_head_weights: 알고리즘별 헤드 가중치 계산
        - train_step: 알고리즘별 훈련 스텝 구현

    필수 설정 키:
        - horizon: 예측 헤드 수 (일반적으로 4)
        - mixed_precision: 혼합 정밀도 ("bf16"/"fp16"/"fp32")
        - loss_config: 손실 함수 설정 (lambda 등)
        - scorer: 토큰 가중치 계산기 (None이면 baseline)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """WMTP 베이스 트레이너 초기화.

        Args:
            config: 트레이너 설정 딕셔너리 (horizon, 손실 설정, Scorer 등)
        """
        super().__init__(config)

        # 주요 컴포넌트들 (setup에서 초기화됨)
        self.model: nn.Module | None = None  # Facebook MTP 모델
        self.optimizer = None  # AdamW 등 최적화기

        # 훈련 상태 추적
        self.global_step: int = 0  # 전역 훈련 스텝 카운터

        # MTP 설정
        self.horizon: int = int(self.config.get("horizon", 4))  # 예측 헤드 수

        # Scorer 출력 캐싱 (성능 최적화용)
        self._last_score_out: dict[str, Any] | None = None

        # Early stopping (setup에서 초기화됨)
        self.early_stopping = None  # LossEarlyStopping 인스턴스

    def setup(self, ctx: dict[str, Any]) -> None:
        """트레이너 초기화 - 모델, 분산 훈련, Scorer 등 모든 컴포넌트 설정.

        이 메서드는 파이프라인에서 제공받은 컴포넌트들을 연결하고
        WMTP 훈련에 필요한 모든 설정을 완료합니다.

        Args:
            ctx: 컨텍스트 딕셔너리 (model, optimizer, scorers, tokenizers 등)
        """
        super().setup(ctx)
        dm = get_dist_manager()  # 분산 훈련 매니저

        # 필수 컴포넌트 검증 및 설정
        model: nn.Module | None = ctx.get("model")  # Facebook MTP 모델
        optimizer = ctx.get("optimizer")  # AdamW 등 최적화기
        if model is None:
            raise ValueError("트레이너에 'model'이 필요합니다 (ctx에서 누락)")
        if optimizer is None:
            raise ValueError("트레이너에 'optimizer'가 필요합니다 (ctx에서 누락)")

        # 분산 훈련: FSDP 래핑 (선택적)
        fsdp_cfg = self.config.get("fsdp_config")
        if fsdp_cfg:
            # Fully Sharded Data Parallel로 모델 래핑
            model = dm.setup_fsdp(model, fsdp_cfg)

        self.model = model
        self.optimizer = optimizer

        # 디바이스 설정 - 모델의 파라미터로부터 추론
        if hasattr(model, "parameters") and list(model.parameters()):
            self.device = next(model.parameters()).device
        else:
            # 폴백: 사용 가능한 최적 디바이스 자동 선택
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

        # 혼합 정밀도 설정: 메모리와 속도 최적화
        mp = str(self.config.get("mixed_precision", "bf16")).lower()
        if mp not in {"bf16", "fp16", "fp32"}:
            mp = "bf16"  # 기본값: BFloat16 (권장)

        self._amp_dtype = (
            torch.bfloat16  # BF16: 안정성과 성능의 균형
            if mp == "bf16"
            else (torch.float16 if mp == "fp16" else torch.float32)  # FP16 또는 FP32
        )

        # 🎯 핵심: 알고리즘별 토큰 가중치 계산 Scorer 연결
        self.scorer = self.config.get(
            "scorer"
        )  # None(baseline), CriticScorer, Rho1Scorer

        # WMTP 손실 함수 설정
        self.loss_cfg = self.config.get("loss_config", {})

        # MLflow 실험 추적 (선택적)
        self.mlflow = ctx.get("mlflow_manager")

        # 알고리즘별 보조 모델들 (선택적 - 해당 알고리즘에서만 사용)
        self.ref_model: nn.Module | None = ctx.get("ref_model")  # Rho-1용 참조 모델
        self.rm_model: nn.Module | None = ctx.get("rm_model")  # Critic용 보상 모델
        self.base_tokenizer = ctx.get("base_tokenizer")  # 기본 토크나이저
        self.ref_tokenizer = ctx.get("ref_tokenizer")  # 참조 모델용 토크나이저

        # 체크포인트 관리 설정
        self.dist_manager = dm  # 분산 훈련 매니저 (체크포인트 저장/로드용)

        # Config에서 체크포인트 설정 파싱 (Phase 2: Recipe에서 Config로 이동)
        config = ctx.get("config")  # Config 객체 가져오기
        recipe = ctx.get("recipe")  # Recipe 객체 가져오기 (알고리즘 정보용)

        if config and hasattr(config, "paths") and hasattr(config.paths, "checkpoints"):
            checkpoint_config = config.paths.checkpoints
            self.save_interval = checkpoint_config.save_interval
            self.keep_last = checkpoint_config.keep_last
            self.save_final = checkpoint_config.save_final
        else:
            # 기본값 설정 (하위 호환성)
            self.save_interval = 500
            self.keep_last = 3
            self.save_final = True

        # 알고리즘 정보 저장 (recipe에서 추출)
        self.algorithm = (
            getattr(recipe.train, "algo", "wmtp")
            if recipe and hasattr(recipe, "train")
            else "wmtp"
        )

        # 체크포인트 디렉토리 설정 (Phase 3: Config + MLflow run_id 기반)
        checkpoint_path, self.is_s3_checkpoint = self._resolve_checkpoint_path(
            config, recipe, ctx
        )

        if self.is_s3_checkpoint:
            # S3 경로: 문자열로 저장
            self.checkpoint_dir = checkpoint_path
        else:
            # 로컬 경로: Path 객체로 생성 및 디렉토리 생성
            self.checkpoint_dir = Path(checkpoint_path)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 저장된 체크포인트 목록 관리
        self.saved_checkpoints = []

        # Early stopping 초기화 (Stage 2 Main Training)
        if recipe and hasattr(recipe.train, "early_stopping"):
            es_config = recipe.train.early_stopping
            if es_config and es_config.enabled:
                from src.utils.early_stopping import LossEarlyStopping

                es_config_dict = (
                    es_config.model_dump()
                    if hasattr(es_config, "model_dump")
                    else es_config
                )

                self.early_stopping = LossEarlyStopping(es_config_dict)
                console.print(
                    f"[cyan]Early stopping enabled (monitor={es_config.monitor}, patience={es_config.patience})[/cyan]"
                )

        # 재개 처리 로직
        self.start_step = 0
        self.resume_metrics = {}

        resume_checkpoint = ctx.get("resume_checkpoint")
        if resume_checkpoint:
            checkpoint_data = self.dist_manager.load_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                checkpoint_path=str(resume_checkpoint),
            )

            self.start_step = checkpoint_data.get("step", 0)
            self.resume_metrics = checkpoint_data.get("metrics", {})

            # Early stopping 상태 복원
            if self.early_stopping and checkpoint_data:
                es_state = checkpoint_data.get("early_stopping_state")
                if es_state:
                    self.early_stopping.load_state(es_state)
                    console.print("[cyan]Early stopping state restored[/cyan]")

            console.print(
                f"[green]Model and optimizer states restored from step {self.start_step}[/green]"
            )

    def _resolve_checkpoint_path(self, config, recipe, ctx) -> tuple[str, bool]:
        """체크포인트 경로 해석 (Phase 3: Config + MLflow run_id 기반)

        Args:
            config: Config 객체
            recipe: Recipe 객체
            ctx: 컨텍스트 딕셔너리

        Returns:
            (checkpoint_dir, is_s3): 체크포인트 디렉토리와 S3 여부
        """
        from src.utils.path_resolver import resolve_checkpoint_path

        # MLflow run_id 가져오기 (최우선 식별자)
        run_id = self.mlflow.get_run_id() if self.mlflow else None

        # run_id가 없으면 recipe.run.name 사용 (fallback)
        if not run_id:
            run_id = (
                recipe.run.name
                if recipe and hasattr(recipe, "run") and hasattr(recipe.run, "name")
                else "no_mlflow_run"
            )

        # Config에서 체크포인트 설정 가져오기
        if config and hasattr(config, "paths") and hasattr(config.paths, "checkpoints"):
            base_path = config.paths.checkpoints.base_path
            checkpoint_dir, is_s3 = resolve_checkpoint_path(base_path, run_id)
            return checkpoint_dir, is_s3
        else:
            # 기본값 (하위 호환성)
            checkpoint_dir = f"./checkpoints/{run_id}"
            return checkpoint_dir, False

    @contextmanager
    def _get_autocast_context(self):
        """혼합 정밀도 학습을 위한 autocast context manager.

        MPS + fp32 환경에서는 autocast를 비활성화하여 불필요한 warning을 제거합니다.
        CUDA/CPU 환경에서는 설정된 precision에 맞춰 autocast를 적용합니다.

        Yields:
            autocast context 또는 no-op context
        """
        # fp32인 경우 autocast 불필요 (no-op context)
        if self._amp_dtype == torch.float32:
            yield
            return

        # MPS 환경 체크
        is_mps = torch.backends.mps.is_available() and str(self.device).startswith(
            "mps"
        )

        if is_mps:
            # MPS는 autocast 미지원 → no-op context
            yield
        elif torch.cuda.is_available():
            # CUDA: autocast 활성화
            with torch.autocast(device_type="cuda", dtype=self._amp_dtype):
                yield
        else:
            # CPU: autocast 활성화
            with torch.autocast(device_type="cpu", dtype=self._amp_dtype):
                yield

    @abstractmethod
    def compute_head_weights(
        self, logits: torch.Tensor, target_ids: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """각 알고리즘별 헤드 가중치 계산 (필수 구현).

        Args:
            logits: MTP 모델 출력 [B, S, H, V]
            target_ids: 타겟 토큰 ID [B, S]
            **kwargs: 알고리즘별 추가 인자 (hidden_states, ce_per_head 등)

        Returns:
            head_weights: 헤드별 가중치 [B, S, H]
        """
        pass

    @abstractmethod
    def train_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        """알고리즘별 훈련 스텝 구현 (필수 구현).

        Args:
            batch: 훈련 배치 데이터

        Returns:
            메트릭 딕셔너리 (loss, lr 등)
        """
        pass

    def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """
        체크포인트 저장 기능이 포함된 확장된 훈련 루프.
        주기적 체크포인트 저장과 최종 모델 저장을 지원합니다.

        Args:
            ctx: 'train_dataloader', 'num_epochs', 'max_steps' 포함

        Returns:
            훈련 메트릭 딕셔너리
        """
        if self.model is None:
            raise RuntimeError("Trainer not initialized. Call setup() first.")

        dataloader = ctx.get("train_dataloader")
        if dataloader is None:
            raise ValueError("Trainer.run expects 'train_dataloader' in ctx")

        num_epochs: int = ctx.get("num_epochs", 1)
        max_steps: int | None = ctx.get("max_steps")
        metrics = {}

        # Global step 관리 (재개 지원)
        global_step = self.start_step

        # Config에서 log_interval 가져오기 (기본값 100)
        config = ctx.get("config")
        log_interval = getattr(config, "log_interval", 100) if config else 100

        console.print(
            f"[green]체크포인트 저장 활성화: 매 {self.save_interval}스텝마다 저장[/green]"
        )
        console.print(f"[green]체크포인트 디렉토리: {self.checkpoint_dir}[/green]")
        console.print(f"[green]로깅 간격: 매 {log_interval} step마다 출력[/green]")
        console.print(f"[green]Epoch 설정: {num_epochs} epochs[/green]")

        # Epoch 루프
        for epoch in range(num_epochs):
            console.print(f"\n[bold cyan]📊 Epoch {epoch + 1}/{num_epochs}[/bold cyan]")

            for _step, batch in enumerate(track(dataloader, description="Training")):
                global_step += 1

                # 재개시 이미 완료된 스텝 건너뛰기
                if global_step <= self.start_step:
                    continue

                # 각 알고리즘별 훈련 스텝 실행 (추상 메서드)
                out = self.train_step(batch)
                metrics = out

                # Console 로깅 (주기적 출력)
                if global_step % log_interval == 0 or global_step == 1:
                    loss = metrics.get("loss", 0.0)
                    lr = metrics.get("lr", 0.0)
                    grad_norm = metrics.get("grad_norm", 0.0)
                    ppl = metrics.get("perplexity", 0.0)

                    log_msg = (
                        f"[cyan]Epoch {epoch + 1}/{num_epochs} Step {global_step:>5}[/cyan] │ "
                        f"Loss: [yellow]{loss:.4f}[/yellow] │ "
                        f"PPL: [yellow]{ppl:>7.2f}[/yellow] │ "
                        f"Grad: [green]{grad_norm:>6.2f}[/green] │ "
                        f"LR: [dim]{lr:.2e}[/dim]"
                    )

                    # WMTP만: weight_entropy 추가
                    if "weight_entropy" in metrics:
                        w_ent = metrics["weight_entropy"]
                        log_msg += f" │ W_Ent: [magenta]{w_ent:.3f}[/magenta]"

                    console.print(log_msg)

                # Early stopping 체크
                if self.early_stopping:
                    should_stop = self.early_stopping.should_stop(metrics)

                    # 분산 학습: rank 0 결정을 모든 rank에 브로드캐스트
                    if (
                        torch.distributed.is_available()
                        and torch.distributed.is_initialized()
                    ):
                        should_stop_tensor = torch.tensor(
                            [should_stop], dtype=torch.bool, device=self.device
                        )
                        torch.distributed.broadcast(should_stop_tensor, src=0)
                        should_stop = should_stop_tensor.item()

                    if should_stop:
                        reason = self.early_stopping.stop_reason
                        console.print(f"[yellow]⚠ Early stopping: {reason}[/yellow]")

                        # MLflow 로깅
                        if self.mlflow:
                            self.mlflow.log_metrics(
                                {
                                    "early_stopping/final_step": global_step,
                                    "early_stopping/best_value": self.early_stopping.best_value,
                                    "early_stopping/counter": self.early_stopping.counter,
                                }
                            )
                        break

                # 주기적 체크포인트 저장
                if global_step % self.save_interval == 0:
                    try:
                        checkpoint_path = self._save_checkpoint(
                            epoch, global_step, metrics
                        )
                        self.saved_checkpoints = self._manage_checkpoints(
                            self.saved_checkpoints, checkpoint_path
                        )
                    except Exception as e:
                        # 실제 파일 저장 여부 확인
                        checkpoint_path = (
                            self.checkpoint_dir / f"checkpoint_step_{global_step}.pt"
                        )
                        if checkpoint_path.exists():
                            console.print(
                                f"[yellow]체크포인트 저장 완료, 부가 기능 오류 (스텝 {global_step}): {repr(e)}[/yellow]"
                            )
                        else:
                            console.print(
                                f"[red]체크포인트 저장 실패 (스텝 {global_step}): {repr(e)}[/red]"
                            )

                # 최대 스텝 도달 시 종료
                if max_steps is not None and global_step >= max_steps:
                    break

            # Outer loop 종료 체크
            if max_steps is not None and global_step >= max_steps:
                break

        # 최종 체크포인트 저장
        if self.save_final:
            try:
                final_path = self._save_final_checkpoint(epoch, global_step, metrics)
                console.print(f"[green]최종 모델 저장 완료: {final_path}[/green]")
            except Exception as e:
                # 실제 파일 저장 여부 확인
                final_path = self.checkpoint_dir / "final_model.pt"
                if final_path.exists():
                    console.print(
                        f"[yellow]최종 모델 저장 완료, 부가 기능 오류: {repr(e)}[/yellow]"
                    )
                else:
                    console.print(f"[red]최종 모델 저장 실패: {repr(e)}[/red]")

        return metrics

    def _save_checkpoint(self, epoch: int, step: int, metrics: dict) -> str:
        """
        단일 체크포인트 저장 (Phase 3: S3/로컬 자동 판단)

        Args:
            epoch: 현재 에폭
            step: 현재 스텝
            metrics: 훈련 메트릭

        Returns:
            저장된 체크포인트 경로 (문자열)
        """
        # S3/로컬 자동 판단하여 체크포인트 경로 생성
        if self.is_s3_checkpoint:
            # S3 경로: 문자열 결합
            checkpoint_path = f"{self.checkpoint_dir}/checkpoint_step_{step}.pt"
        else:
            # 로컬 경로: Path 객체 사용
            checkpoint_path = str(self.checkpoint_dir / f"checkpoint_step_{step}.pt")

        # Early stopping 상태 수집
        es_state = self.early_stopping.get_state() if self.early_stopping else None

        # FSDP 호환 체크포인트 저장
        self.dist_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            checkpoint_path=checkpoint_path,
            epoch=epoch,
            step=step,
            metrics=metrics,
            algorithm=getattr(self, "algorithm", "wmtp"),
            mlflow_run_id=self.mlflow.get_run_id() if self.mlflow else None,
            early_stopping_state=es_state,
        )

        storage_type = "S3" if self.is_s3_checkpoint else "로컬"
        console.print(
            f"[green]{storage_type} 체크포인트 저장 완료: {checkpoint_path}[/green]"
        )
        return checkpoint_path

    def _manage_checkpoints(self, saved_checkpoints: list, new_checkpoint: str) -> list:
        """
        체크포인트 파일 개수 관리 (keep_last 개만 유지).
        Phase 3: S3/로컬 자동 판단

        Args:
            saved_checkpoints: 기존 체크포인트 목록
            new_checkpoint: 새로 저장된 체크포인트 경로 (문자열)

        Returns:
            업데이트된 체크포인트 목록
        """
        saved_checkpoints.append(new_checkpoint)

        # keep_last 개수 초과 시 오래된 파일 삭제
        while len(saved_checkpoints) > self.keep_last:
            old_checkpoint_path = saved_checkpoints.pop(0)
            try:
                if self.is_s3_checkpoint:
                    # S3 체크포인트 삭제
                    from src.utils.s3 import S3Manager

                    s3_manager = S3Manager()
                    # S3 경로에서 버킷과 키 분리
                    bucket, key = old_checkpoint_path.replace("s3://", "").split("/", 1)
                    s3_manager.delete_object(bucket, key)
                    checkpoint_name = key.split("/")[-1]
                    console.print(
                        f"[blue]이전 S3 체크포인트 삭제: {checkpoint_name}[/blue]"
                    )
                else:
                    # 로컬 체크포인트 삭제
                    old_checkpoint = Path(old_checkpoint_path)
                    if old_checkpoint.exists():
                        old_checkpoint.unlink()
                        console.print(
                            f"[blue]이전 체크포인트 삭제: {old_checkpoint.name}[/blue]"
                        )
            except Exception as e:
                storage_type = "S3" if self.is_s3_checkpoint else "로컬"
                console.print(
                    f"[yellow]{storage_type} 체크포인트 삭제 실패: {e}[/yellow]"
                )

        return saved_checkpoints

    def _save_final_checkpoint(self, epoch: int, step: int, metrics: dict) -> str:
        """
        최종 모델 저장 및 MLflow 등록

        역할:
        - paths.checkpoints에 final_model.pt 저장 (훈련 재개용)
        - MLflow에 모델 등록 및 artifact 업로드 (실험 추적용)

        Args:
            epoch: 최종 에폭
            step: 최종 스텝
            metrics: 최종 메트릭

        Returns:
            저장된 최종 모델 경로
        """
        # 1. paths.checkpoints에 저장
        if self.is_s3_checkpoint:
            final_path = f"{self.checkpoint_dir}/final_model.pt"
        else:
            final_path = str(self.checkpoint_dir / "final_model.pt")

        # Early stopping 상태 수집
        es_state = self.early_stopping.get_state() if self.early_stopping else None

        # 체크포인트 저장 (MLflow는 아래에서 별도 처리)
        self.dist_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            checkpoint_path=final_path,
            epoch=epoch,
            step=step,
            metrics=metrics,
            algorithm=self.algorithm,
            mlflow_run_id=self.mlflow.get_run_id() if self.mlflow else None,
            early_stopping_state=es_state,
        )

        storage_type = "S3" if self.is_s3_checkpoint else "로컬"
        console.print(
            f"[green]{storage_type} 최종 모델 저장 완료: {final_path}[/green]"
        )

        # 2. MLflow에 모델 등록 및 artifact 업로드
        if self.mlflow:
            try:
                # 2-1. PyTorch 모델 등록 (Model Registry)
                model_name = f"wmtp_{self.algorithm}"
                self.mlflow.log_model(
                    model=self.model,
                    name="final_model",
                    registered_model_name=model_name,
                )
                console.print(f"[cyan]MLflow 모델 등록 완료: {model_name}[/cyan]")

                # 2-2. Checkpoint artifact 업로드 (로컬인 경우만)
                if not self.is_s3_checkpoint:
                    self.mlflow.log_artifact(
                        local_path=final_path,
                        artifact_path="checkpoints",
                    )
                    console.print(
                        "[cyan]MLflow artifact 업로드: checkpoints/final_model.pt[/cyan]"
                    )
                else:
                    # S3 경로는 참조만 기록
                    self.mlflow.log_param("final_checkpoint_s3_path", final_path)
                    console.print(f"[cyan]MLflow에 S3 경로 기록: {final_path}[/cyan]")

                # 2-3. 최종 메트릭 기록
                self.mlflow.log_metrics(
                    {
                        "final/epoch": epoch,
                        "final/step": step,
                        **{f"final/{k}": v for k, v in metrics.items()},
                    }
                )
            except Exception as e:
                console.print(
                    f"[yellow]MLflow 등록 실패 (체크포인트는 저장됨): {e}[/yellow]"
                )

        return final_path
