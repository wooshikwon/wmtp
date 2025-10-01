"""
Unit tests for Recipe schema (Phase 2).

Tests for the new schema structure:
- EarlyStoppingConfig
- PretrainConfig (replaces Stage1Config)
- Recipe.pretrain field
- Train.early_stopping field
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.settings.recipe_schema import (
    EarlyStoppingConfig,
    PretrainConfig,
    Train,
)


class TestEarlyStoppingConfig:
    """EarlyStoppingConfig validation tests."""

    def test_default_values(self):
        """Default values are correctly set."""
        es = EarlyStoppingConfig()
        assert es.enabled is False
        assert es.patience == 100
        assert es.min_delta == 1e-4
        assert es.monitor == "loss"
        assert es.mode == "any"
        assert es.grad_norm_threshold is None
        assert es.grad_norm_window_size == 10
        assert es.grad_norm_threshold_ratio == 0.7
        assert es.variance_min is None
        assert es.variance_max is None

    def test_mode_validation(self):
        """Mode must be one of: any, all, loss_only."""
        # Valid modes
        for mode in ["any", "all", "loss_only"]:
            es = EarlyStoppingConfig(mode=mode)
            assert es.mode == mode

        # Invalid mode
        with pytest.raises(ValidationError):
            EarlyStoppingConfig(mode="invalid")

    def test_patience_validation(self):
        """Patience must be >= 1."""
        EarlyStoppingConfig(patience=1)  # OK
        EarlyStoppingConfig(patience=100)  # OK

        with pytest.raises(ValidationError):
            EarlyStoppingConfig(patience=0)

    def test_grad_norm_threshold_ratio_validation(self):
        """Ratio must be in (0, 1]."""
        EarlyStoppingConfig(grad_norm_threshold_ratio=0.5)  # OK
        EarlyStoppingConfig(grad_norm_threshold_ratio=1.0)  # OK

        with pytest.raises(ValidationError):
            EarlyStoppingConfig(grad_norm_threshold_ratio=0)

        with pytest.raises(ValidationError):
            EarlyStoppingConfig(grad_norm_threshold_ratio=1.5)


class TestPretrainConfig:
    """PretrainConfig validation tests."""

    def test_default_values(self):
        """Default values are correctly set."""
        pc = PretrainConfig()
        assert pc.enabled is True
        assert pc.num_epochs == 3
        assert pc.max_steps == 2000
        assert pc.lr == 1e-4
        assert pc.save_value_head is True
        assert pc.early_stopping is None

    def test_with_early_stopping(self):
        """PretrainConfig can include early stopping."""
        pc = PretrainConfig(
            early_stopping=EarlyStoppingConfig(enabled=True, mode="any", patience=10)
        )
        assert pc.early_stopping.enabled is True
        assert pc.early_stopping.mode == "any"
        assert pc.early_stopping.patience == 10

    def test_num_epochs_validation(self):
        """num_epochs must be >= 1."""
        PretrainConfig(num_epochs=1)  # OK

        with pytest.raises(ValidationError):
            PretrainConfig(num_epochs=0)

    def test_lr_validation(self):
        """lr must be > 0."""
        PretrainConfig(lr=1e-5)  # OK

        with pytest.raises(ValidationError):
            PretrainConfig(lr=0)


class TestTrainSchema:
    """Train schema changes tests."""

    def test_train_has_early_stopping_field(self):
        """Train now has early_stopping field (not stage1)."""
        train = Train(
            algo="baseline-mtp",
            early_stopping=EarlyStoppingConfig(enabled=True, mode="loss_only"),
        )
        assert train.early_stopping.enabled is True
        assert train.early_stopping.mode == "loss_only"

    def test_train_early_stopping_is_optional(self):
        """early_stopping is optional."""
        train = Train(algo="baseline-mtp")
        assert train.early_stopping is None


class TestRecipeSchema:
    """Recipe schema integration tests."""

    def test_recipe_with_pretrain(self):
        """Recipe can include pretrain section."""
        from src.settings.loader import load_recipe

        recipe = load_recipe("tests/configs/recipe.critic_wmtp.yaml")

        assert recipe.pretrain is not None
        assert recipe.pretrain.enabled is True
        assert recipe.pretrain.num_epochs == 3
        assert recipe.pretrain.max_steps == 30
        assert recipe.pretrain.lr == 1e-4

    def test_pretrain_early_stopping(self):
        """Pretrain can have early stopping configured."""
        from src.settings.loader import load_recipe

        recipe = load_recipe("tests/configs/recipe.critic_wmtp.yaml")

        assert recipe.pretrain.early_stopping is not None
        assert recipe.pretrain.early_stopping.enabled is True
        assert recipe.pretrain.early_stopping.mode == "any"
        assert recipe.pretrain.early_stopping.patience == 10
        assert recipe.pretrain.early_stopping.grad_norm_threshold == 50.0
        assert recipe.pretrain.early_stopping.variance_min == 0.1
        assert recipe.pretrain.early_stopping.variance_max == 5.0

    def test_train_early_stopping(self):
        """Train can have early stopping configured."""
        from src.settings.loader import load_recipe

        recipe = load_recipe("tests/configs/recipe.critic_wmtp.yaml")

        assert recipe.train.early_stopping is not None
        assert recipe.train.early_stopping.enabled is False
        assert recipe.train.early_stopping.monitor == "wmtp_loss"

    def test_pretrain_is_optional_for_baseline(self):
        """Baseline-mtp doesn't need pretrain."""
        from src.settings.loader import load_recipe

        recipe = load_recipe("tests/configs/recipe.mtp_baseline.yaml")

        assert recipe.pretrain is None  # Baseline doesn't use pretraining

    def test_critic_wmtp_requires_pretrain(self):
        """critic-wmtp requires pretrain configuration."""
        from src.factory.component_factory import ComponentFactory
        from src.settings.loader import load_recipe

        recipe = load_recipe("tests/configs/recipe.critic_wmtp.yaml")

        # Should not raise
        pretrainer = ComponentFactory.create_pretrainer(recipe)
        assert pretrainer is not None


class TestBackwardIncompatibility:
    """Verify that old schema is incompatible (intentional)."""

    def test_stage1_field_removed_from_train(self):
        """train.stage1 field no longer exists."""
        from src.settings.loader import load_recipe

        recipe = load_recipe("tests/configs/recipe.critic_wmtp.yaml")

        assert not hasattr(recipe.train, "stage1")

    def test_pretrain_is_top_level(self):
        """pretrain is now top-level, not under train or critic."""
        from src.settings.loader import load_recipe

        recipe = load_recipe("tests/configs/recipe.critic_wmtp.yaml")

        assert hasattr(recipe, "pretrain")
        assert recipe.pretrain is not None
        assert not hasattr(recipe.critic, "pretrainer")
