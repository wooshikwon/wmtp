import types
from typing import Any

import numpy as np
import torch


def _dummy_model_forward(**batch: Any):
    B, S = batch["input_ids"].shape
    H = 4
    V = 8
    logits = torch.randn(B, S, H, V, dtype=torch.float32)
    return {"logits": logits}


class DummyModel(torch.nn.Module):
    def forward(self, **batch: Any):
        return _dummy_model_forward(**batch)


class DummyOptimizer:
    def __init__(self):
        self._last_lr = 1e-5
        self.grad_clip = 1.0

    def step(self):
        pass

    def zero_grad(self, set_to_none: bool = True):
        pass


def test_baseline_mtp_trainer_step():
    """Test BaselineMtpTrainer (Phase 2 refactored version)."""
    from src.components.trainer.baseline_mtp_trainer import BaselineMtpTrainer

    model = DummyModel()
    optimizer = DummyOptimizer()

    trainer = BaselineMtpTrainer(
        {
            "n_heads": 4,
            "horizon": 4,
            "loss_config": {"lambda": 0.3, "temperature": 0.7},
            "mixed_precision": "fp32",
        }
    )
    trainer.setup({"model": model, "optimizer": optimizer})

    batch = {
        "input_ids": torch.ones(2, 16, dtype=torch.long),
        "labels": torch.ones(2, 16, dtype=torch.long),
        "attention_mask": torch.ones(2, 16, dtype=torch.float32),
    }

    result = trainer.train_step(batch)
    assert "loss" in result
    assert result["loss"] > 0
    print("✓ BaselineMtpTrainer smoke test passed")


def test_critic_wmtp_trainer_step():
    """Test CriticWmtpTrainer with mock scorer."""
    from src.components.trainer.critic_wmtp_trainer import CriticWmtpTrainer

    model = DummyModel()
    optimizer = DummyOptimizer()

    # Mock scorer that returns weights
    class MockCriticScorer:
        def setup(self, ctx):
            pass

        def run(self, ctx):
            B = 2
            S = 16
            H = 4
            weights = torch.ones((B, S, H), dtype=torch.float32)
            return {"weights": weights}

    scorer = MockCriticScorer()
    
    trainer = CriticWmtpTrainer(
        {
            "n_heads": 4,
            "horizon": 4,
            "loss_config": {"lambda": 0.3, "temperature": 0.7},
            "mixed_precision": "fp32",
        }
    )
    trainer.setup({"model": model, "optimizer": optimizer})
    trainer.scorer = scorer

    # Mock hidden states for CriticWmtpTrainer
    class ModelWithHidden(torch.nn.Module):
        def forward(self, **batch):
            result = _dummy_model_forward(**batch)
            B, S = batch["input_ids"].shape
            result["hidden_states"] = torch.randn(B, S, 768)  # Mock hidden states
            return result

    trainer.model = ModelWithHidden()

    batch = {
        "input_ids": torch.ones(2, 16, dtype=torch.long),
        "labels": torch.ones(2, 16, dtype=torch.long),
        "attention_mask": torch.ones(2, 16, dtype=torch.float32),
    }

    result = trainer.train_step(batch)
    assert "loss" in result
    assert result["loss"] > 0
    print("✓ CriticWmtpTrainer smoke test passed")


def test_rho1_wmtp_trainer_step():
    """Test Rho1WmtpTrainer with reference model."""
    from src.components.trainer.rho1_wmtp_trainer import Rho1WmtpTrainer

    model = DummyModel()
    ref_model = DummyModel()
    optimizer = DummyOptimizer()

    trainer = Rho1WmtpTrainer(
        {
            "n_heads": 4,
            "horizon": 4,
            "loss_config": {"lambda": 0.3, "temperature": 0.7},
            "mixed_precision": "fp32",
        }
    )
    trainer.setup({"model": model, "optimizer": optimizer, "ref_model": ref_model})

    batch = {
        "input_ids": torch.ones(2, 16, dtype=torch.long),
        "labels": torch.ones(2, 16, dtype=torch.long),
        "attention_mask": torch.ones(2, 16, dtype=torch.float32),
    }

    result = trainer.train_step(batch)
    assert "loss" in result
    assert result["loss"] > 0
    print("✓ Rho1WmtpTrainer smoke test passed")


if __name__ == "__main__":
    print("Running Phase 2 Trainer Smoke Tests...")
    print("=" * 50)
    test_baseline_mtp_trainer_step()
    test_critic_wmtp_trainer_step()
    test_rho1_wmtp_trainer_step()
    print("=" * 50)
    print("✅ All trainer smoke tests passed!")