from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class DistilGPT2MTP(nn.Module):
    """DistilGPT2 with Multi-Token Prediction heads.

    Enhanced version with proper hidden_states support for Critic-WMTP training.
    Now with MPS optimization support.
    """

    def __init__(self, config=None, training_config: dict[str, Any] | None = None):
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

        # Store training config for MPS optimization decision
        self.training_config = training_config

        # Check if we should use MPS optimization
        self._use_mps_optimization = self._should_use_mps_path()

        # Create additional MTP heads
        self.extra_heads = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.vocab_size, bias=False)
                for _ in range(self.n_future_tokens - 1)
            ]
        )

    def _should_use_mps_path(self) -> bool:
        """Determine if MPS optimization should be used based on config."""
        if self.training_config is None:
            return False

        try:
            # Import here to avoid circular dependency
            from src.utils.mps_optimizer import MPSOptimizer

            return MPSOptimizer.should_use_mps_path(self.training_config)
        except ImportError:
            # If MPSOptimizer not available, fallback to default
            return False

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        output_hidden_states=None,
        output_attentions=None,
        return_dict=None,
        **kwargs,
    ):
        """
        Enhanced forward method with proper hidden_states support.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Labels for training (optional)
            output_hidden_states: Whether to return hidden states
            output_attentions: Whether to return attention weights
            return_dict: Whether to return ModelOutput object
            **kwargs: Additional arguments

        Returns:
            dict or CausalLMOutputWithCrossAttentions containing:
            - logits: MTP logits of shape [B, S, n_future_tokens, V]
            - hidden_states: Hidden states (if requested)
            - attentions: Attention weights (if requested)
            - last_hidden_state: Last layer hidden states (if requested)
        """
        # Handle default values for output flags
        if output_hidden_states is None:
            output_hidden_states = getattr(self.config, "output_hidden_states", False)
        if output_attentions is None:
            output_attentions = getattr(self.config, "output_attentions", False)
        if return_dict is None:
            return_dict = getattr(self.config, "use_return_dict", True)

        # Forward through base transformer with proper flags
        outputs = self.base_model.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,  # Always use return_dict for internal processing
            **kwargs,
        )

        # Extract hidden states
        last_hidden_state = outputs.last_hidden_state

        # Build MTP logits
        all_logits = []

        # Main head (reuse base model's lm_head)
        main_logits = self.base_model.lm_head(last_hidden_state)
        all_logits.append(main_logits)

        # Additional MTP heads
        for head in self.extra_heads:
            logits = head(last_hidden_state)
            all_logits.append(logits)

        # Stack to create MTP logits: [B, S, n_future_tokens, V]
        # Use MPS-optimized stacking if available
        if self._use_mps_optimization:
            try:
                from src.utils.mps_optimizer import MPSOptimizer

                mtp_logits = MPSOptimizer.optimize_4d_stack(
                    all_logits,
                    dim=2,
                    use_mps=True,  # We already know we want MPS optimization
                )
            except Exception:
                # Fallback to standard stacking if optimization fails
                mtp_logits = torch.stack(all_logits, dim=2)
        else:
            # Standard stacking for CUDA/CPU
            mtp_logits = torch.stack(all_logits, dim=2)

        # Prepare output
        if return_dict:
            # Return dict with all necessary fields
            result = {
                "logits": mtp_logits,
                "last_hidden_state": last_hidden_state,
            }
            if output_hidden_states:
                result["hidden_states"] = outputs.hidden_states
            if output_attentions:
                result["attentions"] = outputs.attentions

            # Return dict directly for trainer compatibility
            return result
        else:
            # Legacy tuple format
            result = (mtp_logits,)
            if output_hidden_states:
                result += (outputs.hidden_states,)
            if output_attentions:
                result += (outputs.attentions,)
            return result

    def generate(self, *args, **kwargs):
        """Delegate generation to base model for compatibility."""
        return self.base_model.generate(*args, **kwargs)

    def get_input_embeddings(self):
        """Get input embeddings for compatibility."""
        return self.base_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        """Set input embeddings for compatibility."""
        self.base_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        """Get output embeddings for compatibility."""
        return self.base_model.get_output_embeddings()

    def set_output_embeddings(self, value):
        """Set output embeddings for compatibility."""
        self.base_model.set_output_embeddings(value)
