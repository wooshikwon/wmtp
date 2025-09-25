import numpy as np
import torch


class DummyModel(torch.nn.Module):
    def forward(self, **batch):
        B, S = batch["input_ids"].shape
        H, V = 1, 8
        logits = torch.randn(B, S, H, V, dtype=torch.float32)
        return {"logits": logits}


class DummyOptimizer:
    def __init__(self):
        self._last_lr = 1e-5
        self.grad_clip = 1.0

    def step(self):
        pass

    def zero_grad(self, set_to_none: bool = True):
        pass


def test_trainer_smoke_h1():
    """Test BaselineMtpTrainer with H=1 (single head)."""
    from src.components.trainer.baseline_mtp_trainer import BaselineMtpTrainer

    model = DummyModel()
    optimizer = DummyOptimizer()

    trainer = BaselineMtpTrainer(
        {
            "n_heads": 1,
            "horizon": 1,
            "loss_config": {"lambda": 0.3, "temperature": 0.7},
            "mixed_precision": "fp32",
        }
    )
    trainer.setup({"model": model, "optimizer": optimizer})

    batch = {
        "input_ids": torch.ones(2, 12, dtype=torch.long),
        "labels": torch.ones(2, 12, dtype=torch.long),
        "attention_mask": torch.ones(2, 12, dtype=torch.float32),
    }

    result = trainer.train_step(batch)
    assert "loss" in result, "Loss not in result"
    assert result["loss"] > 0, "Loss should be positive"
    print("✓ BaselineMtpTrainer H=1 smoke test passed")


if __name__ == "__main__":
    print("Running H=1 Trainer Smoke Test...")
    print("=" * 50)
    test_trainer_smoke_h1()
    print("=" * 50)
    print("✅ H=1 trainer smoke test passed!")