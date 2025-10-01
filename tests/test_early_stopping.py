"""
Unit tests for Early Stopping utilities (Phase 1).

Tests for ValueHeadEarlyStopping v2.0:
- Mode variations: "any", "all", "loss_only"
- Window-based gradient instability detection
- Variance out of range detection
- State save/load functionality
"""

from __future__ import annotations

import pytest

from src.utils.early_stopping import LossEarlyStopping, ValueHeadEarlyStopping


class TestLossEarlyStopping:
    """LossEarlyStopping 기본 기능 테스트 (변경 없음)."""

    def test_disabled_early_stopping(self):
        """비활성화 상태에서는 항상 False 반환."""
        es = LossEarlyStopping({"enabled": False})
        assert es.should_stop({"loss": 1.0}) is False
        assert es.should_stop({"loss": 0.1}) is False

    def test_loss_convergence(self):
        """Loss가 수렴하면 early stopping 트리거."""
        es = LossEarlyStopping(
            {"enabled": True, "patience": 3, "min_delta": 0.01, "monitor": "loss"}
        )

        # Loss가 개선되는 동안은 중단하지 않음
        assert es.should_stop({"loss": 1.0}) is False
        assert es.should_stop({"loss": 0.5}) is False
        assert es.should_stop({"loss": 0.3}) is False

        # Loss가 정체 (patience=3)
        assert es.should_stop({"loss": 0.3}) is False  # counter=1
        assert es.should_stop({"loss": 0.3}) is False  # counter=2
        assert es.should_stop({"loss": 0.3}) is True  # counter=3, 중단

        assert "Loss convergence" in es.stop_reason

    def test_state_save_and_load(self):
        """State 저장 및 복원."""
        es = LossEarlyStopping({"enabled": True, "patience": 5, "min_delta": 0.01})

        es.should_stop({"loss": 1.0})
        es.should_stop({"loss": 0.9})
        es.should_stop({"loss": 0.9})  # counter=1

        state = es.get_state()
        assert state["best_value"] == 0.9
        assert state["counter"] == 1

        # 새로운 인스턴스에 복원
        es2 = LossEarlyStopping({"enabled": True, "patience": 5, "min_delta": 0.01})
        es2.load_state(state)

        assert es2.best_value == 0.9
        assert es2.counter == 1


class TestValueHeadEarlyStopping:
    """ValueHeadEarlyStopping v2.0 기능 테스트."""

    # ========== Mode: "any" ==========

    def test_any_mode_loss_converged(self):
        """ANY 모드: Loss만 수렴해도 중단."""
        es = ValueHeadEarlyStopping(
            {
                "enabled": True,
                "mode": "any",
                "patience": 2,
                "min_delta": 0.01,
                "monitor": "value_loss",
            }
        )

        # Loss 개선
        assert (
            es.should_stop(
                {"value_loss": 1.0, "grad_norm": 10.0, "value_variance": 1.0}
            )
            is False
        )

        # Loss 정체 시작
        assert (
            es.should_stop(
                {"value_loss": 1.0, "grad_norm": 10.0, "value_variance": 1.0}
            )
            is False
        )  # counter=1

        # Loss 정체 지속 → 중단
        assert (
            es.should_stop(
                {"value_loss": 1.0, "grad_norm": 10.0, "value_variance": 1.0}
            )
            is True
        )  # counter=2, 중단

        assert "any mode" in es.stop_reason
        assert "loss converged" in es.stop_reason

    def test_any_mode_grad_unstable(self):
        """ANY 모드: Gradient 불안정만으로도 중단."""
        es = ValueHeadEarlyStopping(
            {
                "enabled": True,
                "mode": "any",
                "patience": 100,  # Loss는 수렴하지 않음
                "grad_norm_threshold": 50.0,
                "grad_norm_window_size": 5,
                "grad_norm_threshold_ratio": 0.6,  # 60% 이상
                "monitor": "value_loss",
            }
        )

        # Window 채우기 (5개 중 4개가 초과 = 80% > 60%)
        for i in range(5):
            grad_norm = 100.0 if i < 4 else 10.0  # 4개 초과, 1개 정상
            should_stop = es.should_stop(
                {
                    "value_loss": float(i),  # 계속 개선 (수렴 안 함)
                    "grad_norm": grad_norm,
                    "value_variance": 1.0,
                }
            )

            if i < 4:
                assert should_stop is False  # Window 차는 중
            else:
                assert should_stop is True  # 80% > 60%, 중단

        assert "any mode" in es.stop_reason
        assert "gradient unstable" in es.stop_reason

    def test_any_mode_variance_invalid(self):
        """ANY 모드: Variance 범위 이탈만으로도 중단."""
        es = ValueHeadEarlyStopping(
            {
                "enabled": True,
                "mode": "any",
                "patience": 100,  # Loss는 수렴하지 않음
                "grad_norm_threshold": 1000.0,  # Gradient는 안정
                "variance_min": 0.1,
                "variance_max": 5.0,
                "monitor": "value_loss",
            }
        )

        # Variance가 범위 내
        assert (
            es.should_stop(
                {
                    "value_loss": 1.0,
                    "grad_norm": 10.0,
                    "value_variance": 1.0,  # OK
                }
            )
            is False
        )

        # Variance가 너무 작음
        assert (
            es.should_stop(
                {
                    "value_loss": 0.9,  # 개선 (수렴 안 함)
                    "grad_norm": 10.0,
                    "value_variance": 0.05,  # < 0.1, invalid
                }
            )
            is True
        )

        assert "any mode" in es.stop_reason
        assert "variance out of range" in es.stop_reason

    # ========== Mode: "all" ==========

    def test_all_mode_requires_all_conditions(self):
        """ALL 모드: 모든 조건이 만족해야 중단."""
        es = ValueHeadEarlyStopping(
            {
                "enabled": True,
                "mode": "all",
                "patience": 2,
                "min_delta": 0.01,
                "grad_norm_threshold": 50.0,
                "grad_norm_window_size": 3,
                "grad_norm_threshold_ratio": 0.7,
                "variance_min": 0.1,
                "variance_max": 5.0,
                "monitor": "value_loss",
            }
        )

        # Loss 수렴 + Gradient 안정 + Variance OK
        # 먼저 Loss를 수렴시킴
        es.should_stop({"value_loss": 1.0, "grad_norm": 10.0, "value_variance": 1.0})
        es.should_stop(
            {"value_loss": 1.0, "grad_norm": 10.0, "value_variance": 1.0}
        )  # counter=1
        es.should_stop(
            {"value_loss": 1.0, "grad_norm": 10.0, "value_variance": 1.0}
        )  # counter=2

        # Window 채우기 (모두 안정)
        for _ in range(3):
            should_stop = es.should_stop(
                {
                    "value_loss": 1.0,
                    "grad_norm": 10.0,  # < 50, 안정
                    "value_variance": 1.0,  # 범위 내
                }
            )

        # 모든 조건 만족 → 중단
        assert should_stop is True
        assert "all mode" in es.stop_reason
        assert "all conditions met" in es.stop_reason

    def test_all_mode_fails_if_grad_unstable(self):
        """ALL 모드: Gradient 불안정하면 중단 안 함."""
        es = ValueHeadEarlyStopping(
            {
                "enabled": True,
                "mode": "all",
                "patience": 2,
                "grad_norm_threshold": 50.0,
                "grad_norm_window_size": 3,
                "grad_norm_threshold_ratio": 0.7,
                "monitor": "value_loss",
            }
        )

        # Loss 수렴
        es.should_stop({"value_loss": 1.0, "grad_norm": 100.0, "value_variance": 1.0})
        es.should_stop({"value_loss": 1.0, "grad_norm": 100.0, "value_variance": 1.0})
        should_stop = es.should_stop(
            {"value_loss": 1.0, "grad_norm": 100.0, "value_variance": 1.0}
        )

        # Loss는 수렴했지만 Gradient 불안정 → 중단 안 함
        assert should_stop is False

    # ========== Mode: "loss_only" ==========

    def test_loss_only_mode(self):
        """LOSS_ONLY 모드: Loss만 체크."""
        es = ValueHeadEarlyStopping(
            {
                "enabled": True,
                "mode": "loss_only",
                "patience": 2,
                "min_delta": 0.01,
                "monitor": "value_loss",
            }
        )

        # Gradient 불안정, Variance invalid이지만 무시
        es.should_stop(
            {
                "value_loss": 1.0,
                "grad_norm": 1000.0,  # 매우 불안정
                "value_variance": 100.0,  # 범위 초과
            }
        )
        es.should_stop(
            {"value_loss": 1.0, "grad_norm": 1000.0, "value_variance": 100.0}
        )
        should_stop = es.should_stop(
            {"value_loss": 1.0, "grad_norm": 1000.0, "value_variance": 100.0}
        )

        # Loss만 수렴했으면 중단
        assert should_stop is True
        assert "loss_only mode" in es.stop_reason

    # ========== Window-based Gradient Check ==========

    def test_window_based_gradient_check(self):
        """Window 기반 gradient 체크가 일시적 스파이크에 강인."""
        es = ValueHeadEarlyStopping(
            {
                "enabled": True,
                "mode": "any",
                "patience": 100,
                "grad_norm_threshold": 50.0,
                "grad_norm_window_size": 10,
                "grad_norm_threshold_ratio": 0.7,  # 70% 이상
                "monitor": "value_loss",
            }
        )

        # 일시적 스파이크 (10개 중 6개 초과 = 60% < 70%)
        grad_norms = [100, 100, 10, 100, 10, 100, 10, 100, 10, 100]
        # 초과: 6개, 정상: 4개 → 60% < 70% → 불안정 아님

        for i, grad_norm in enumerate(grad_norms):
            should_stop = es.should_stop(
                {
                    "value_loss": float(i),  # 계속 개선
                    "grad_norm": grad_norm,
                    "value_variance": 1.0,
                }
            )
            assert should_stop is False  # 70% 미만이므로 안정

        # 현재 window: [T, T, F, T, F, T, F, T, F, T] (6개 초과, 60%)
        # 이제 7개 초과 (70%)를 만들기 위해 교체 전략 사용
        # 연속 7개 초과 값 추가 → 최종적으로 7/10
        for _ in range(4):
            should_stop = es.should_stop(
                {
                    "value_loss": 10.0,
                    "grad_norm": 100.0,  # 초과
                    "value_variance": 1.0,
                }
            )

        # Window: [F, T, F, T, F, T, T, T, T, T] (7개 = 70%)
        assert should_stop is True
        assert "gradient unstable" in es.stop_reason

    # ========== State Management ==========

    def test_state_save_and_load_with_history(self):
        """Gradient history를 포함한 state 저장/복원."""
        es = ValueHeadEarlyStopping(
            {
                "enabled": True,
                "mode": "any",
                "patience": 5,
                "grad_norm_threshold": 50.0,
                "grad_norm_window_size": 5,
                "monitor": "value_loss",
            }
        )

        # Window 채우기
        for i in range(3):
            es.should_stop(
                {
                    "value_loss": 1.0,
                    "grad_norm": 100.0 if i % 2 == 0 else 10.0,
                    "value_variance": 1.0,
                }
            )

        # State 저장
        state = es.get_state()
        assert "grad_norm_history" in state
        assert len(state["grad_norm_history"]) == 3

        # 새 인스턴스에 복원
        es2 = ValueHeadEarlyStopping(
            {"enabled": True, "mode": "any", "grad_norm_window_size": 5}
        )
        es2.load_state(state)

        assert len(es2.grad_norm_history) == 3
        assert list(es2.grad_norm_history) == state["grad_norm_history"]

    def test_reset_clears_history(self):
        """Reset이 gradient history를 초기화."""
        es = ValueHeadEarlyStopping(
            {"enabled": True, "mode": "any", "grad_norm_window_size": 5}
        )

        # Window 채우기
        for _ in range(3):
            es.should_stop(
                {"value_loss": 1.0, "grad_norm": 100.0, "value_variance": 1.0}
            )

        assert len(es.grad_norm_history) == 3

        # Reset
        es.reset()
        assert len(es.grad_norm_history) == 0

    # ========== Edge Cases ==========

    def test_missing_optional_metrics(self):
        """Optional metrics 누락 시 해당 조건 무시."""
        es = ValueHeadEarlyStopping(
            {"enabled": True, "mode": "any", "patience": 2, "monitor": "value_loss"}
        )

        # grad_norm, value_variance 없음 → loss만 체크
        es.should_stop({"value_loss": 1.0})
        es.should_stop({"value_loss": 1.0})
        should_stop = es.should_stop({"value_loss": 1.0})

        assert should_stop is True
        assert "loss converged" in es.stop_reason

    def test_invalid_mode_raises_error(self):
        """잘못된 mode는 초기화 시 에러."""
        with pytest.raises(ValueError, match="Invalid mode"):
            ValueHeadEarlyStopping({"enabled": True, "mode": "invalid_mode"})

    def test_disabled_early_stopping(self):
        """비활성화 상태에서는 항상 False."""
        es = ValueHeadEarlyStopping({"enabled": False, "mode": "any"})

        # 모든 조건 만족해도 중단 안 함
        for _ in range(10):
            assert (
                es.should_stop(
                    {"value_loss": 1.0, "grad_norm": 1000.0, "value_variance": 1000.0}
                )
                is False
            )
