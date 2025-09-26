import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class DistilGPT2MTP(nn.Module):
    """DistilGPT2 with Multi-Token Prediction heads."""

    def __init__(self, config=None):
        super().__init__()
        if config is None:
            self.base_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
            self.config = self.base_model.config
            self.n_future_tokens = 4
        else:
            self.config = config
            self.n_future_tokens = config.mtp_config.get("n_future_tokens", 4)
            # Build base model from config
            from transformers import GPT2LMHeadModel

            self.base_model = GPT2LMHeadModel(config)

        self.hidden_size = self.config.n_embd
        self.vocab_size = self.config.vocab_size

        # Create additional MTP heads
        self.extra_heads = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.vocab_size, bias=False)
                for _ in range(self.n_future_tokens - 1)
            ]
        )

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.base_model.transformer(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )

        hidden_states = outputs.last_hidden_state

        all_logits = []
        main_logits = self.base_model.lm_head(hidden_states)
        all_logits.append(main_logits)

        for head in self.extra_heads:
            logits = head(hidden_states)
            all_logits.append(logits)

        mtp_logits = torch.stack(all_logits, dim=2)

        return {"logits": mtp_logits}
