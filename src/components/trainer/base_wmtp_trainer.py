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
from pathlib import Path  # 경로 처리
from typing import Any  # 범용 타입 힌트

import torch  # PyTorch 딥러닝 프레임워크
import torch.nn as nn  # 신경망 모듈
import torch.nn.functional as F  # 함수형 API (cross_entropy 등)
from rich.console import Console  # 컬러풀한 콘솔 출력

from src.components.base import BaseComponent  # WMTP 컴포넌트 베이스 클래스
from src.utils import get_dist_manager  # 분산 훈련 매니저

console = Console()  # 전역 콘솔 객체


def compute_weighted_mtp_loss(
    logits: torch.Tensor,  # [B, S, H, V]
    target_labels: torch.Tensor,  # [B, S, H] - 3D 라벨 (MTPDataCollator에서 생성)
    head_weights: torch.Tensor,  # [B, S, H]
    ignore_index: int = -100,
    selection_mask: torch.Tensor | None = None,  # [B, S, H] - 토큰 선택 마스크
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    MTPDataCollator 기반 간단화된 WMTP 손실 계산: L_WMTP = Σ(k=0 to H-1) w_{t+k} × CE_k

    3D 라벨을 직접 받아서 헤드별 CE를 계산하고 가중치를 적용합니다.
    복잡한 shift 연산이 제거되어 성능이 대폭 향상됩니다.

    Args:
        logits: [B, S, H, V] - MTP 모델 출력
        target_labels: [B, S, H] - MTPDataCollator에서 생성된 3D 라벨
        head_weights: [B, S, H] - 헤드별 가중치 매트릭스
        ignore_index: 무시할 라벨 값 (-100)
        selection_mask: [B, S, H] - 토큰 선택 마스크 (None이면 모두 1)

    Returns:
        weighted_loss: 가중 평균 손실 (scalar)
        valid_mask: 유효한 위치 마스크 [B, S]
        ce_per_head: 헤드별 CE 손실 [B, S, H]
    """
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

    # 헤드별 CE 계산 (벡터화)
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
            ctx: 'train_dataloader'와 'max_steps' 포함

        Returns:
            훈련 메트릭 딕셔너리
        """
        if self.model is None:
            raise RuntimeError("Trainer not initialized. Call setup() first.")

        dataloader = ctx.get("train_dataloader")
        if dataloader is None:
            raise ValueError("Trainer.run expects 'train_dataloader' in ctx")
        max_steps: int | None = ctx.get("max_steps")

        epoch = 0  # 단순화를 위해 epoch=0으로 설정
        metrics = {}

        console.print(
            f"[green]체크포인트 저장 활성화: 매 {self.save_interval}스텝마다 저장[/green]"
        )
        console.print(f"[green]체크포인트 디렉토리: {self.checkpoint_dir}[/green]")

        for step, batch in enumerate(dataloader):
            current_step = step + 1

            # 재개시 이미 완료된 스텝 건너뛰기
            if current_step <= self.start_step:
                continue

            # 각 알고리즘별 훈련 스텝 실행 (추상 메서드)
            out = self.train_step(batch)
            metrics = out

            # 주기적 체크포인트 저장
            if current_step % self.save_interval == 0:
                try:
                    checkpoint_path = self._save_checkpoint(
                        epoch, current_step, metrics
                    )
                    self.saved_checkpoints = self._manage_checkpoints(
                        self.saved_checkpoints, checkpoint_path
                    )
                except Exception as e:
                    # 실제 파일 저장 여부 확인
                    checkpoint_path = (
                        self.checkpoint_dir / f"checkpoint_step_{current_step}.pt"
                    )
                    if checkpoint_path.exists():
                        console.print(
                            f"[yellow]체크포인트 저장 완료, 부가 기능 오류 (스텝 {current_step}): {repr(e)}[/yellow]"
                        )
                    else:
                        console.print(
                            f"[red]체크포인트 저장 실패 (스텝 {current_step}): {repr(e)}[/red]"
                        )

            # 최대 스텝 도달 시 종료
            if max_steps is not None and current_step >= max_steps:
                break

        # 최종 체크포인트 저장
        if self.save_final:
            try:
                final_step = step + 1 if "step" in locals() else 1
                final_path = self._save_final_checkpoint(epoch, final_step, metrics)
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

        # FSDP 호환 체크포인트 저장 (MLflow 통합)
        self.dist_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            checkpoint_path=checkpoint_path,
            epoch=epoch,
            step=step,
            mlflow_manager=self.mlflow,  # MLflow 매니저 전달
            metrics=metrics,
            algorithm=getattr(self, "algorithm", "wmtp"),
            mlflow_run_id=self.mlflow.get_run_id() if self.mlflow else None,
        )

        # MLflow 업로드는 분산 매니저에서 수행함 (중복 제거)

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
        최종 모델 저장 (Phase 3: S3/로컬 자동 판단)

        Args:
            epoch: 최종 에폭
            step: 최종 스텝
            metrics: 최종 메트릭

        Returns:
            저장된 최종 모델 경로 (문자열)
        """
        # S3/로컬 자동 판단하여 최종 모델 경로 생성
        if self.is_s3_checkpoint:
            # S3 경로: 문자열 결합
            final_path = f"{self.checkpoint_dir}/final_model.pt"
        else:
            # 로컬 경로: Path 객체 사용
            final_path = str(self.checkpoint_dir / "final_model.pt")

        # 최종 체크포인트 저장 (MLflow 통합)
        self.dist_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            checkpoint_path=final_path,
            epoch=epoch,
            step=step,
            mlflow_manager=self.mlflow,  # MLflow 매니저 전달
            metrics=metrics,
            algorithm=getattr(self, "algorithm", "wmtp"),
            final_model=True,
            mlflow_run_id=self.mlflow.get_run_id() if self.mlflow else None,
        )

        # MLflow 모델 레지스트리 등록 및 아티팩트 업로드
        if self.mlflow is not None:
            try:
                # 모델 이름 생성 (recipe에서 알고리즘 정보 사용)
                model_name = f"wmtp-{self.algorithm}"

                # 모델 레지스트리 등록
                self.mlflow.log_model(
                    model=self.model,
                    artifact_path="final_model",
                    registered_model_name=model_name,
                )

                # 체크포인트 파일 업로드 (로컬 경로만 지원)
                if not self.is_s3_checkpoint:
                    self.mlflow.log_artifact(
                        local_path=final_path, artifact_path="final_checkpoint"
                    )
                else:
                    console.print(
                        "[blue]S3 체크포인트는 MLflow artifact 업로드 생략[/blue]"
                    )

                console.print(f"[green]MLflow 모델 등록 완료: {model_name}[/green]")
            except Exception as e:
                console.print(
                    f"[yellow]MLflow model registration warning: {e}[/yellow]"
                )

        storage_type = "S3" if self.is_s3_checkpoint else "로컬"
        console.print(
            f"[green]{storage_type} 최종 모델 저장 완료: {final_path}[/green]"
        )
        return final_path
