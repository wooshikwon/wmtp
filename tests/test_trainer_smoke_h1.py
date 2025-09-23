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
    from src.components.trainer.mtp_weighted_ce_trainer import MTPWeightedCETrainer

    model = DummyModel()
    optimizer = DummyOptimizer()

    class OnesScorer:
        def setup(self, ctx):
            pass

        def run(self, ctx):
            B, S = 2, 12
            return {"weights": np.ones((B, S), dtype=np.float32)}

    scorer = OnesScorer()
    trainer = MTPWeightedCETrainer(
        {
            "n_heads": 1,
            "horizon": 1,
            "loss_config": {"lambda": 0.3},
            "mixed_precision": "fp32",
        }
    )
    trainer.setup({"model": model, "optimizer": optimizer})
    trainer.scorer = scorer

    batch = {
        "input_ids": torch.randint(0, 8, (2, 12)),
        "labels": torch.randint(0, 8, (2, 12)),
    }

    out = trainer.train_step(batch)
    assert "loss" in out and out["loss"] > 0.0


