from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_sequence_rewards(
    rm_model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    amp_dtype: torch.dtype | None = None,
) -> list[float]:
    """Reward computation utility colocated with model adapters.

    Falls back to negative mean next-token CE if explicit reward field is
    not provided by the RM.
    """
    autocast_kwargs = {}
    if amp_dtype is not None:
        autocast_kwargs = {
            "device_type": ("cuda" if torch.cuda.is_available() else "cpu"),
            "dtype": amp_dtype,
        }

    with (
        torch.autocast(**autocast_kwargs)
        if autocast_kwargs
        else torch.autocast(enabled=False, device_type="cpu")
    ):
        outputs = rm_model(input_ids=input_ids, attention_mask=attention_mask)

    if isinstance(outputs, dict):
        for key in ("reward", "rewards", "score", "scores", "value", "values"):
            if key in outputs:
                vals = outputs[key]
                if isinstance(vals, torch.Tensor):
                    vals = vals.detach().float().view(-1)
                    return vals.tolist()
                try:
                    return [float(v) for v in vals]
                except Exception:
                    pass

    logits = (
        outputs["logits"]
        if isinstance(outputs, dict) and "logits" in outputs
        else outputs
    )
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        return [0.0 for _ in range(int(input_ids.shape[0]))]

    B, S, V = logits.shape
    if S <= 1:
        return [0.0 for _ in range(B)]

    logits_shifted = logits[:, :-1, :].transpose(1, 2).contiguous()
    labels_shifted = input_ids[:, 1:].contiguous()

    ce = F.cross_entropy(logits_shifted, labels_shifted, reduction="none")

    if attention_mask is not None:
        mask = attention_mask[:, 1:].to(dtype=ce.dtype)
        token_counts = torch.clamp(mask.sum(dim=1), min=1.0)
        ce_mean = (ce * mask).sum(dim=1) / token_counts
    else:
        ce_mean = ce.mean(dim=1)

    rewards = (-ce_mean).detach().float().tolist()
    return rewards
