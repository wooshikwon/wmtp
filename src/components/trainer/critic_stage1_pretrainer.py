"""
Critic Stage1 pretrainer implemented as a trainer component.

Runs lightweight value-head regression using RM sequence rewards and
stores the trained head to cache for Stage2 usage.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.components.base import BaseComponent
from src.components.registry import trainer_registry
from src.components.reward.sequence_reward import compute_sequence_rewards
from src.components.scorer.critic_delta import CriticDeltaScorer


@trainer_registry.register(
    "critic-stage1-pretrainer-v1", category="trainer", version="1.0.0"
)
class CriticStage1Pretrainer(BaseComponent):
    """Stage1 trainer to fit a value head for critic-wmtp."""

    def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
        self.validate_initialized()

        base_model = ctx["base_model"]
        rm_model = ctx.get("rm_model")
        train_loader = ctx["train_dataloader"]
        cache_root = ctx["cache_root"]

        vh_path = cache_root / "value_head.pt"
        idx_path = cache_root / "index.jsonl"
        cache_root.mkdir(parents=True, exist_ok=True)

        # Skip if cached
        if vh_path.exists() and idx_path.exists():
            return {"skipped": True}

        hidden_size = getattr(
            getattr(base_model, "config", object()), "hidden_size", 4096
        )
        scorer = CriticDeltaScorer(
            {
                "target": self.config.get("target", "rm_sequence"),
                "token_spread": self.config.get("token_spread", "gae"),
                "delta_mode": self.config.get("delta_mode", "td"),
                "normalize": self.config.get("normalize", "zscore"),
                "temperature": self.config.get("temperature", 0.7),
            }
        )
        scorer.setup({"hidden_size": hidden_size})

        # Enable hidden states
        orig_flag = getattr(
            getattr(base_model, "config", object()), "output_hidden_states", False
        )
        if hasattr(base_model, "config"):
            base_model.config.output_hidden_states = True

        base_model.eval()
        vh = scorer.value_head
        assert vh is not None
        optimizer = torch.optim.AdamW(
            vh.parameters(), lr=float(self.config.get("lr", 1e-4))
        )
        loss_fn = torch.nn.MSELoss()

        with open(idx_path, "w") as fidx:
            for step, batch in enumerate(train_loader):
                input_ids = batch.get("input_ids")
                attention_mask = batch.get("attention_mask")
                if input_ids is None:
                    continue

                device = next(base_model.parameters()).device
                input_ids = input_ids.to(device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)

                with (
                    torch.no_grad(),
                    torch.autocast(
                        device_type=("cuda" if torch.cuda.is_available() else "cpu"),
                        dtype=torch.bfloat16,
                    ),
                ):
                    outputs = base_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
                    hidden_states = (
                        outputs.hidden_states[-1]
                        if hasattr(outputs, "hidden_states")
                        else outputs["hidden_states"][-1]
                    )

                rewards = compute_sequence_rewards(
                    rm_model, input_ids, attention_mask, amp_dtype=torch.bfloat16
                )

                # Build value targets with GAE over token-spread rewards
                seq_lengths = [
                    int(hidden_states.shape[1]) for _ in range(hidden_states.shape[0])
                ]
                value_targets = []
                for b in range(hidden_states.shape[0]):
                    r = float(rewards[b])
                    L = seq_lengths[b]
                    spread = scorer.spread_reward_to_tokens(r, L)
                    zeros = np.zeros(L, dtype=np.float32)
                    vt = scorer.compute_gae_returns(spread, zeros, next_value=0.0)
                    value_targets.append(vt)

                B, S, H = hidden_states.shape
                hs_flat = hidden_states.reshape(B * S, H)
                vt_flat = torch.tensor(
                    np.concatenate(value_targets, axis=0),
                    device=hs_flat.device,
                    dtype=hs_flat.dtype,
                ).view(B * S, 1)

                vh.train()
                optimizer.zero_grad(set_to_none=True)
                pred = vh(hs_flat)
                loss = loss_fn(pred, vt_flat)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(vh.parameters(), 1.0)
                optimizer.step()

                fidx.write(f"{step}\n")

        torch.save(scorer.value_head.state_dict(), vh_path)

        if hasattr(base_model, "config"):
            base_model.config.output_hidden_states = orig_flag

        return {"saved": str(vh_path)}
