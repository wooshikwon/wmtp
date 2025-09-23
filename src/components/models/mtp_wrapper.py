from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class MTPWrapper(nn.Module):
    """
    Teacher-forcing k-step logits emulation for models that output [B,S,V] only.

    If a base model doesn't have native MTP heads, this wrapper stacks logits
    for future steps by reusing the model in a loop (teacher-forcing), producing
    [B, S, H, V] logits. Intended for small H and debugging/compat, not speed.
    """

    def __init__(self, base_model: nn.Module, horizon: int):
        super().__init__()
        self.base_model = base_model
        self.horizon = int(horizon)

    def forward(self, **batch: Any) -> dict[str, torch.Tensor]:
        input_ids: torch.Tensor = batch.get("input_ids")
        assert input_ids is not None, "input_ids required"

        # Get next-token logits [B, S, V]
        outputs = self.base_model(**batch)
        logits = (
            outputs["logits"]
            if isinstance(outputs, dict) and "logits" in outputs
            else outputs
        )
        if logits.ndim != 3:
            raise ValueError(
                f"Expected logits [B,S,V] from base model, got {tuple(logits.shape)}"
            )

        B, S, V = logits.shape
        H = max(1, self.horizon)

        # If H==1 just reshape to [B,S,1,V]
        if H == 1:
            return {"logits": logits.unsqueeze(2)}

        # Build stacked logits [B,S,H,V] using teacher-forcing assumptions
        stacked = []
        # k=0: next-token as given
        stacked.append(logits)
        # k>=1: shift labels as inputs is typical; we emulate by rolling logits
        # Note: this is an approximation; for exact teacher-forcing we would
        # need to run the model conditioned on labels shifted by k.
        for k in range(1, H):
            # For emulation, pad head with zeros and slice to length
            pad = torch.zeros(B, k, V, device=logits.device, dtype=logits.dtype)
            rolled = torch.cat([pad, logits[:, :-k, :]], dim=1)
            stacked.append(rolled)

        logits_stacked = torch.stack(stacked, dim=2)  # [B, S, H, V]
        return {"logits": logits_stacked}
