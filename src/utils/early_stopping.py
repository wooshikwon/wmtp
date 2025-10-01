"""
Early Stopping Utilities for WMTP Training.

연구제안서 기반 조기 종료 전략 구현:
- Stage 1 (Value Head Pretraining): 다중 기준 체크 (loss, gradient, variance)
- Stage 2 (Main Training): Loss convergence 체크

Version 2.0 (Phase 1):
- Flexible mode: "any" (practical) | "all" (conservative) | "loss_only"
- Window-based gradient instability detection
- Improved state management with gradient history

Reference: PPO best practices (TRL/TRLX)
- patience=100 (100 steps without improvement)
- min_delta=1e-4 (minimum improvement threshold)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Any


class BaseEarlyStopping(ABC):
    """조기 종료 전략 베이스 클래스.

    모든 early stopping 전략의 공통 인터페이스를 정의합니다.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Early stopping 초기화.

        Args:
            config: 조기 종료 설정 딕셔너리
        """
        self.config = config or {}
        self.enabled = self.config.get("enabled", False)
        self.best_value: float | None = None
        self.counter = 0
        self.should_stop_flag = False
        self.stop_reason: str | None = None

    @abstractmethod
    def should_stop(self, metrics: dict[str, float]) -> bool:
        """조기 종료 여부를 판단.

        Args:
            metrics: 현재 스텝의 메트릭 딕셔너리

        Returns:
            조기 종료 여부
        """
        pass

    def reset(self) -> None:
        """Early stopping 상태 초기화."""
        self.best_value = None
        self.counter = 0
        self.should_stop_flag = False
        self.stop_reason = None

    def get_state(self) -> dict[str, Any]:
        """현재 상태를 딕셔너리로 반환 (체크포인트 저장용)."""
        return {
            "best_value": self.best_value,
            "counter": self.counter,
            "should_stop_flag": self.should_stop_flag,
            "stop_reason": self.stop_reason,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """체크포인트에서 상태 복원."""
        self.best_value = state.get("best_value")
        self.counter = state.get("counter", 0)
        self.should_stop_flag = state.get("should_stop_flag", False)
        self.stop_reason = state.get("stop_reason")


class LossEarlyStopping(BaseEarlyStopping):
    """Loss convergence 기반 조기 종료 (Stage 2 전용).

    연구제안서 Stage 2 요구사항:
    - Loss가 patience 횟수만큼 개선되지 않으면 조기 종료
    - min_delta 미만의 개선은 무시

    사용 예시:
        >>> early_stopping = LossEarlyStopping({
        ...     "enabled": True,
        ...     "patience": 100,
        ...     "min_delta": 1e-5,
        ...     "monitor": "loss"
        ... })
        >>> for step in range(max_steps):
        ...     metrics = train_step()
        ...     if early_stopping.should_stop(metrics):
        ...         print(f"Early stopping: {early_stopping.stop_reason}")
        ...         break
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Loss early stopping 초기화.

        Args:
            config: 설정 딕셔너리
                - enabled: 조기 종료 활성화 (기본값: False)
                - patience: 개선 없이 허용할 스텝 수 (기본값: 100)
                - min_delta: 개선으로 간주할 최소 변화량 (기본값: 1e-5)
                - monitor: 모니터링할 메트릭 이름 (기본값: "loss")
        """
        super().__init__(config)
        self.patience = self.config.get("patience", 100)
        self.min_delta = self.config.get("min_delta", 1e-5)
        self.monitor = self.config.get("monitor", "loss")

    def should_stop(self, metrics: dict[str, float]) -> bool:
        """Loss 개선 여부를 확인하여 조기 종료 판단.

        Args:
            metrics: 현재 메트릭 딕셔너리 (monitor key 필수)

        Returns:
            조기 종료 여부
        """
        if not self.enabled:
            return False

        # 모니터링 메트릭 추출
        current_value = metrics.get(self.monitor)
        if current_value is None:
            # 메트릭이 없으면 조기 종료하지 않음
            return False

        # 첫 번째 값 저장
        if self.best_value is None:
            self.best_value = current_value
            return False

        # Loss 개선 여부 확인 (작을수록 좋음)
        improvement = self.best_value - current_value

        if improvement > self.min_delta:
            # 개선됨: 카운터 리셋
            self.best_value = current_value
            self.counter = 0
        else:
            # 개선 없음: 카운터 증가
            self.counter += 1

        # Patience 초과 시 조기 종료
        if self.counter >= self.patience:
            self.should_stop_flag = True
            self.stop_reason = (
                f"Loss convergence: no improvement for {self.patience} steps "
                f"(best={self.best_value:.6f}, current={current_value:.6f})"
            )
            return True

        return False


class ValueHeadEarlyStopping(BaseEarlyStopping):
    """Value Head 학습 조기 종료 (Stage 1 전용).

    연구제안서 Stage 1 요구사항 (v2.0):
    1. Value loss convergence
    2. Gradient instability detection (window-based)
    3. Variance out of range detection

    Stopping Modes:
    - "any" (기본값, 실용적): 조건 중 하나라도 만족하면 중단
    - "all" (보수적): 모든 조건이 만족해야 중단
    - "loss_only": Loss convergence만 체크

    사용 예시:
        >>> early_stopping = ValueHeadEarlyStopping({
        ...     "enabled": True,
        ...     "mode": "any",
        ...     "patience": 10,
        ...     "min_delta": 1e-4,
        ...     "monitor": "value_loss",
        ...     "grad_norm_threshold": 50.0,
        ...     "grad_norm_window_size": 10,
        ...     "grad_norm_threshold_ratio": 0.7,
        ...     "variance_min": 0.1,
        ...     "variance_max": 5.0
        ... })
        >>> for step in range(max_steps):
        ...     metrics = {
        ...         "value_loss": loss.item(),
        ...         "grad_norm": total_norm,
        ...         "value_variance": pred_variance
        ...     }
        ...     if early_stopping.should_stop(metrics):
        ...         reason = early_stopping.stop_reason
        ...         print(f"Early stopping: {reason}")
        ...         break
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Value Head early stopping 초기화.

        Args:
            config: 설정 딕셔너리
                - enabled: 조기 종료 활성화 (기본값: False)
                - mode: 중단 모드 "any" | "all" | "loss_only" (기본값: "any")
                - patience: Loss 개선 없이 허용할 스텝 수 (기본값: 10)
                - min_delta: Loss 개선 최소 임계값 (기본값: 1e-4)
                - monitor: 모니터링할 loss 메트릭 (기본값: "value_loss")
                - grad_norm_threshold: Gradient norm 임계값 (기본값: 50.0)
                - grad_norm_window_size: Gradient 체크 윈도우 크기 (기본값: 10)
                - grad_norm_threshold_ratio: 윈도우 내 초과 비율 임계값 (기본값: 0.7)
                - variance_min: Value 분산 최소값 (기본값: 0.1)
                - variance_max: Value 분산 최대값 (기본값: 5.0)
        """
        super().__init__(config)

        # Mode 설정
        self.mode = self.config.get("mode", "any")
        if self.mode not in ["any", "all", "loss_only"]:
            raise ValueError(
                f"Invalid mode: {self.mode}. Must be 'any', 'all', or 'loss_only'"
            )

        # Loss convergence 설정
        self.patience = self.config.get("patience", 10)
        self.min_delta = self.config.get("min_delta", 1e-4)
        self.monitor = self.config.get("monitor", "value_loss")

        # Gradient instability 설정 (window-based)
        self.grad_norm_threshold = self.config.get("grad_norm_threshold", 50.0)
        self.grad_norm_window_size = self.config.get("grad_norm_window_size", 10)
        self.grad_norm_threshold_ratio = self.config.get(
            "grad_norm_threshold_ratio", 0.7
        )
        self.grad_norm_history: deque[bool] = deque(maxlen=self.grad_norm_window_size)

        # Variance range 설정
        self.variance_min = self.config.get("variance_min", 0.1)
        self.variance_max = self.config.get("variance_max", 5.0)

    def should_stop(self, metrics: dict[str, float]) -> bool:
        """종합적인 조기 종료 판단 (mode에 따라 유연하게 처리).

        Args:
            metrics: 현재 메트릭 딕셔너리
                - {monitor}: Value loss (예: "value_loss")
                - "grad_norm": Gradient norm (optional)
                - "value_variance": Value 예측 분산 (optional)

        Returns:
            조기 종료 여부
        """
        if not self.enabled:
            return False

        # 필수 메트릭 확인
        value_loss = metrics.get(self.monitor)
        grad_norm = metrics.get("grad_norm")
        value_variance = metrics.get("value_variance")

        if value_loss is None:
            # Loss 메트릭이 없으면 조기 종료하지 않음
            return False

        # 🎯 각 조건 독립적으로 체크
        loss_converged = self._check_loss_convergence(value_loss)
        grad_unstable = self._check_gradient_instability(grad_norm)
        variance_invalid = self._check_variance_invalid(value_variance)

        # 🎯 Mode별 중단 결정 및 이유 수집
        should_stop = False
        reasons = []

        if self.mode == "any":
            # 하나라도 만족하면 중단 (실용적)
            if loss_converged:
                reasons.append(
                    f"loss converged ({value_loss:.6f}, patience={self.patience})"
                )
            if grad_unstable:
                reasons.append(
                    f"gradient unstable (threshold={self.grad_norm_threshold}, ratio={self.grad_norm_threshold_ratio})"
                )
            if variance_invalid:
                reasons.append(
                    f"variance out of range ({value_variance:.4f}, range=[{self.variance_min}, {self.variance_max}])"
                )

            should_stop = len(reasons) > 0

        elif self.mode == "all":
            # 모두 만족해야 중단 (보수적)
            if loss_converged and not grad_unstable and not variance_invalid:
                reasons.append(
                    f"all conditions met: loss={value_loss:.6f}, grad stable, variance valid"
                )
                should_stop = True

        else:  # "loss_only"
            # Loss convergence만 체크
            if loss_converged:
                reasons.append(
                    f"loss converged ({value_loss:.6f}, patience={self.patience})"
                )
                should_stop = True

        # 중단 결정
        if should_stop:
            self.should_stop_flag = True
            self.stop_reason = (
                f"Stage 1 early stop ({self.mode} mode): {'; '.join(reasons)}"
            )
            return True

        return False

    def _check_loss_convergence(self, current_loss: float) -> bool:
        """Loss convergence 체크 (LossEarlyStopping과 동일 로직).

        Args:
            current_loss: 현재 loss 값

        Returns:
            Loss가 수렴했는지 여부
        """
        # 첫 번째 값 저장
        if self.best_value is None:
            self.best_value = current_loss
            return False

        # Loss 개선 여부 확인
        improvement = self.best_value - current_loss

        if improvement > self.min_delta:
            # 개선됨: 카운터 리셋
            self.best_value = current_loss
            self.counter = 0
            return False
        else:
            # 개선 없음: 카운터 증가
            self.counter += 1

        # Patience 초과 시 수렴으로 판단
        return self.counter >= self.patience

    def _check_gradient_instability(self, grad_norm: float | None) -> bool:
        """Gradient 불안정성 체크 (window-based).

        윈도우 내 일정 비율 이상이 threshold를 초과하면 불안정으로 판단.

        Args:
            grad_norm: Gradient L2 norm

        Returns:
            Gradient가 불안정한지 여부 (True면 조기 종료 사유)
        """
        if grad_norm is None or self.grad_norm_threshold is None:
            # Gradient norm이 없으면 체크하지 않음 (불안정하지 않음)
            return False

        # Window에 초과 여부 기록
        exceeds_threshold = grad_norm > self.grad_norm_threshold
        self.grad_norm_history.append(exceeds_threshold)

        # Window가 차기 전엔 판단하지 않음
        if len(self.grad_norm_history) < self.grad_norm_window_size:
            return False

        # 최근 window 내 초과 비율 계산
        unstable_ratio = sum(self.grad_norm_history) / len(self.grad_norm_history)

        # 일정 비율 이상 초과하면 불안정
        return unstable_ratio >= self.grad_norm_threshold_ratio

    def _check_variance_invalid(self, variance: float | None) -> bool:
        """Variance 범위 이탈 체크.

        Args:
            variance: Value 예측 분산

        Returns:
            분산이 범위를 벗어났는지 여부 (True면 조기 종료 사유)
        """
        if variance is None:
            # Variance가 없으면 체크하지 않음 (유효함)
            return False

        # 범위를 벗어났으면 True (invalid)
        return not (self.variance_min <= variance <= self.variance_max)

    def reset(self) -> None:
        """Early stopping 상태 초기화 (gradient history 포함)."""
        super().reset()
        self.grad_norm_history.clear()

    def get_state(self) -> dict[str, Any]:
        """현재 상태를 딕셔너리로 반환 (체크포인트 저장용)."""
        state = super().get_state()
        state["grad_norm_history"] = list(self.grad_norm_history)  # deque → list
        return state

    def load_state(self, state: dict[str, Any]) -> None:
        """체크포인트에서 상태 복원."""
        super().load_state(state)
        history = state.get("grad_norm_history", [])
        self.grad_norm_history = deque(history, maxlen=self.grad_norm_window_size)
