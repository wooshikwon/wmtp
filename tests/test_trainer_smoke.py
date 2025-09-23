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


def test_mtp_weighted_trainer_step():
    from src.components.trainer.mtp_weighted_ce_trainer import MTPWeightedCETrainer

    model = DummyModel()
    optimizer = DummyOptimizer()

    # Minimal scorer that returns ones
    class OnesScorer:
        def setup(self, ctx):
            pass

        def run(self, ctx):
            B = 2
            S = 16
            return {"weights": np.ones((B, S), dtype=np.float32)}

    scorer = OnesScorer()
    trainer = MTPWeightedCETrainer(
        {
            "n_heads": 4,
            "horizon": 4,
            "loss_config": {"lambda": 0.3},
            "mixed_precision": "fp32",
        }
    )

    trainer.setup({"model": model, "optimizer": optimizer})

    batch = {
        "input_ids": torch.randint(0, 8, (2, 16)),
        "labels": torch.randint(0, 8, (2, 16)),
    }

    # Inject scorer
    trainer.scorer = scorer

    out = trainer.train_step(batch)
    assert "loss" in out and isinstance(out["loss"], float)
    assert out["loss"] > 0.0
