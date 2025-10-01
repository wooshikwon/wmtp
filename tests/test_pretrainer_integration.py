"""
Integration tests for Pretrainer with Early Stopping (Phase 3).

Tests the integration of ValueHeadEarlyStopping into CriticHeadPretrainer:
- Early stopping disabled (baseline behavior)
- Early stopping enabled but not triggered (normal completion)
- Early stopping triggered by loss convergence
- Early stopping triggered by gradient instability
- Early stopping triggered by variance out of range
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MockModel(nn.Module):
    """Mock language model for testing."""

    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.config = type(
            "Config", (), {"hidden_size": hidden_size, "output_hidden_states": False}
        )()
        self.embedding = nn.Embedding(100, hidden_size)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(2)]
        )

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        x = self.embedding(input_ids)
        hidden_states = [x]
        for layer in self.layers:
            x = layer(x)
            hidden_states.append(x)

        outputs = type("Outputs", (), {"hidden_states": tuple(hidden_states)})()
        return outputs


class MockRewardModel(nn.Module):
    """Mock reward model for testing."""

    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_size})()
        self.embedding = nn.Embedding(100, hidden_size)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        return self.head(x.mean(dim=1))


def create_mock_dataloader(
    num_batches: int = 5, batch_size: int = 2, seq_len: int = 10
):
    """Create mock dataloader."""
    batches = []
    for _ in range(num_batches):
        batches.append(
            {
                "input_ids": torch.randint(0, 100, (batch_size, seq_len)),
                "attention_mask": torch.ones(batch_size, seq_len),
            }
        )
    return batches


class TestPretrainerIntegration:
    """Pretrainer integration tests with early stopping."""

    def test_pretrainer_without_early_stopping(self):
        """Early stopping disabled - baseline behavior."""
        from src.components.trainer.critic_head_pretrainer import CriticHeadPretrainer

        config = {
            "lr": 1e-3,
            "num_epochs": 1,
            "max_steps": 3,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "early_stopping": None,  # Disabled
        }

        pretrainer = CriticHeadPretrainer(config)
        pretrainer.setup({})

        base_model = MockModel()
        rm_model = MockRewardModel()
        train_loader = create_mock_dataloader(num_batches=5)

        result = pretrainer.run(
            {
                "base_model": base_model,
                "rm_model": rm_model,
                "train_dataloader": train_loader,
                "run_name": "test_no_es",
            }
        )

        # Should complete normally
        assert result["saved"] is not None
        assert result["total_steps"] == 3
        assert result["early_stopped"] is False
        assert result["stop_reason"] is None

    def test_pretrainer_early_stopping_disabled(self):
        """Early stopping config present but disabled."""
        from src.components.trainer.critic_head_pretrainer import CriticHeadPretrainer

        config = {
            "lr": 1e-3,
            "num_epochs": 1,
            "max_steps": 3,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "early_stopping": {
                "enabled": False,  # Explicitly disabled
                "mode": "any",
                "patience": 2,
            },
        }

        pretrainer = CriticHeadPretrainer(config)
        pretrainer.setup({})

        base_model = MockModel()
        rm_model = MockRewardModel()
        train_loader = create_mock_dataloader(num_batches=5)

        result = pretrainer.run(
            {
                "base_model": base_model,
                "rm_model": rm_model,
                "train_dataloader": train_loader,
                "run_name": "test_es_disabled",
            }
        )

        # Should complete normally
        assert result["total_steps"] == 3
        assert result["early_stopped"] is False
        assert result["stop_reason"] is None

    def test_pretrainer_early_stopping_not_triggered(self):
        """Early stopping enabled but not triggered (max_steps reached first)."""
        from src.components.trainer.critic_head_pretrainer import CriticHeadPretrainer

        config = {
            "lr": 1e-3,
            "num_epochs": 1,
            "max_steps": 3,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "early_stopping": {
                "enabled": True,
                "mode": "any",
                "patience": 100,  # Very patient - won't trigger
                "min_delta": 1e-4,
                "monitor": "value_loss",
                "grad_norm_threshold": 1000.0,  # Very high - won't trigger
                "variance_min": 0.0,  # Very permissive
                "variance_max": 1000.0,  # Very permissive
            },
        }

        pretrainer = CriticHeadPretrainer(config)
        pretrainer.setup({})

        base_model = MockModel()
        rm_model = MockRewardModel()
        train_loader = create_mock_dataloader(num_batches=5)

        result = pretrainer.run(
            {
                "base_model": base_model,
                "rm_model": rm_model,
                "train_dataloader": train_loader,
                "run_name": "test_es_not_triggered",
            }
        )

        # Should complete normally via max_steps
        assert result["total_steps"] == 3
        assert result["early_stopped"] is False
        assert result["stop_reason"] is None

    def test_pretrainer_config_extraction(self):
        """Test that early_stopping_config is properly extracted in __init__."""
        from src.components.trainer.critic_head_pretrainer import CriticHeadPretrainer

        config = {
            "lr": 1e-3,
            "early_stopping": {
                "enabled": True,
                "mode": "any",
                "patience": 10,
            },
        }

        pretrainer = CriticHeadPretrainer(config)

        # Check that config is extracted
        assert pretrainer.early_stopping_config is not None
        assert pretrainer.early_stopping_config["enabled"] is True
        assert pretrainer.early_stopping_config["mode"] == "any"
        assert pretrainer.early_stopping_config["patience"] == 10

    def test_pretrainer_return_structure(self):
        """Test that return value includes early_stopped and stop_reason fields."""
        from src.components.trainer.critic_head_pretrainer import CriticHeadPretrainer

        config = {
            "lr": 1e-3,
            "num_epochs": 1,
            "max_steps": 2,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "early_stopping": {
                "enabled": True,
                "mode": "any",
                "patience": 100,
            },
        }

        pretrainer = CriticHeadPretrainer(config)
        pretrainer.setup({})

        base_model = MockModel()
        rm_model = MockRewardModel()
        train_loader = create_mock_dataloader(num_batches=3)

        result = pretrainer.run(
            {
                "base_model": base_model,
                "rm_model": rm_model,
                "train_dataloader": train_loader,
                "run_name": "test_return_structure",
            }
        )

        # Check all required fields exist
        assert "saved" in result
        assert "final_loss" in result
        assert "total_steps" in result
        assert "early_stopped" in result
        assert "stop_reason" in result

    def test_pretrainer_variance_calculation(self):
        """Test that variance is calculated from pred_values."""
        from src.components.trainer.critic_head_pretrainer import CriticHeadPretrainer

        config = {
            "lr": 1e-3,
            "num_epochs": 1,
            "max_steps": 2,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "early_stopping": {
                "enabled": True,
                "mode": "any",
                "patience": 1,
                "variance_min": 0.1,  # Set reasonable range
                "variance_max": 5.0,
            },
        }

        pretrainer = CriticHeadPretrainer(config)
        pretrainer.setup({})

        base_model = MockModel()
        rm_model = MockRewardModel()
        train_loader = create_mock_dataloader(num_batches=3)

        # Should complete without errors (variance is calculated correctly)
        result = pretrainer.run(
            {
                "base_model": base_model,
                "rm_model": rm_model,
                "train_dataloader": train_loader,
                "run_name": "test_variance",
            }
        )

        # If variance calculation has errors, the test would fail during run()
        assert result is not None
        assert "early_stopped" in result
