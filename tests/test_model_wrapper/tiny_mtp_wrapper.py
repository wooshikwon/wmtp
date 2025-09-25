"""MTP Model Wrapper for testing WMTP pipeline with smaller models.

This wrapper takes a standard HuggingFace model (e.g., Sheared-LLaMA-2.7B)
and adds Multi-Token Prediction (MTP) heads to simulate the Facebook MTP model structure.

테스트 목적:
- 7B MTP 모델 대신 작은 2.7B 모델로 파이프라인 검증
- MacBook M3에서 실행 가능한 크기로 축소
- 동일한 MTP 구조 유지 (4개 헤드, horizon=4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig
from typing import Optional, Dict, Any, Tuple


class MTPModelWrapper(nn.Module):
    """Wraps a standard causal LM with Multi-Token Prediction heads.
    
    Architecture:
        Base Model (Sheared-LLaMA-2.7B)
                    |
            Hidden States
                    |
        ┌───────────┴───────────┐
        │                       │
    Main Head             Extra Heads
    (t+1 pred)         (t+2, t+3, t+4 pred)
        │                       │
        └───────────┬───────────┘
                    │
            MTP Logits [B, S, H, V]
    """
    
    def __init__(
        self,
        base_model_name_or_path: str = "princeton-nlp/Sheared-LLaMA-2.7B",
        n_future_tokens: int = 4,
        device: str = "auto"
    ):
        """Initialize MTP wrapper.
        
        Args:
            base_model_name_or_path: HuggingFace model name or local path
            n_future_tokens: Number of future tokens to predict (MTP heads)
            device: Device to use ("auto", "cuda", "mps", "cpu")
        """
        super().__init__()
        
        self.n_future_tokens = n_future_tokens
        
        # Robust cross-platform device resolution
        self.device = self._resolve_device_robust(device)
        print(f"[MTPWrapper] Using device: {self.device}")
        
        # Load base model with optimal settings
        print(f"[MTPWrapper] Loading base model: {base_model_name_or_path}")
        self.config = AutoConfig.from_pretrained(base_model_name_or_path)
        
        # Get optimal dtype for device
        optimal_dtype = self._get_optimal_dtype()
        print(f"[MTPWrapper] Using dtype: {optimal_dtype}")
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            torch_dtype=optimal_dtype,
            low_cpu_mem_usage=True,
            device_map={"": self.device} if self.device != "cpu" else None
        )
        
        # Extract hidden size from config
        self.hidden_size = self.config.hidden_size
        self.vocab_size = self.config.vocab_size
        
        # Create extra prediction heads for t+2, t+3, t+4
        # (t+1 uses the base model's original lm_head)
        self.extra_heads = nn.ModuleList()
        for i in range(1, n_future_tokens):
            # Each head is a simple linear projection
            head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
            # Initialize with small random weights
            nn.init.normal_(head.weight, mean=0.0, std=0.02)
            self.extra_heads.append(head)
            
        print(f"[MTPWrapper] Created {len(self.extra_heads)} extra MTP heads")
        
        # Move entire model to target device (ensures all components are on same device)
        self.to(self.device)
        print(f"[MTPWrapper] Moved all components to {self.device}")
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with MTP heads.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            **kwargs: Additional arguments for base model
            
        Returns:
            Dictionary containing:
                - logits: MTP logits [batch_size, seq_len, n_future_tokens, vocab_size]
                - hidden_states: Last hidden states (optional)
        """
        # Get base model outputs with hidden states
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )
        
        # Extract hidden states from the last layer
        # Shape: [batch_size, seq_len, hidden_size]
        hidden_states = base_outputs.hidden_states[-1]
        
        # Get logits from base model's lm_head (for t+1 prediction)
        base_logits = base_outputs.logits  # [B, S, V]
        
        # Compute logits for each MTP head
        all_logits = [base_logits]  # Start with t+1 predictions
        
        for head in self.extra_heads:
            # Apply head to hidden states
            head_logits = head(hidden_states)  # [B, S, V]
            all_logits.append(head_logits)
        
        # Stack all logits along a new dimension
        # Shape: [batch_size, seq_len, n_future_tokens, vocab_size]
        mtp_logits = torch.stack(all_logits, dim=2)
        
        return {
            "logits": mtp_logits,
            "hidden_states": hidden_states
        }
    
    def generate_mtp(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate tokens using MTP (for testing).
        
        This is a simplified generation method for testing purposes.
        In production, you'd use more sophisticated decoding strategies.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            use_cache: Whether to use KV cache (not implemented)
            
        Returns:
            generated_ids: Generated token IDs
            mtp_predictions: MTP predictions at each step
        """
        self.eval()
        
        generated = input_ids.clone()
        mtp_predictions = []
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get MTP predictions
                outputs = self.forward(generated)
                mtp_logits = outputs["logits"]
                
                # Use first head (t+1) for next token
                next_token_logits = mtp_logits[:, -1, 0, :]  # [B, V]
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Store MTP predictions for analysis
                mtp_predictions.append(mtp_logits[:, -1, :, :].cpu())
        
        return generated, torch.stack(mtp_predictions, dim=1)
    
    def prepare_for_training(self):
        """Prepare model for training (gradient checkpointing, etc.)"""
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            self.base_model.gradient_checkpointing_enable()
            print("[MTPWrapper] Enabled gradient checkpointing")
            
        # Freeze base model if needed (for testing with limited memory)
        # self.freeze_base_model()
        
    def freeze_base_model(self):
        """Freeze base model parameters (only train MTP heads)."""
        for param in self.base_model.parameters():
            param.requires_grad = False
        print("[MTPWrapper] Froze base model parameters")
        
    def unfreeze_base_model(self):
        """Unfreeze base model parameters."""
        for param in self.base_model.parameters():
            param.requires_grad = True
        print("[MTPWrapper] Unfroze base model parameters")
        
    def _resolve_device_robust(self, device: str) -> str:
        """Cross-platform device resolution with priority: CUDA > MPS > CPU."""
        if device != "auto":
            return device
            
        # Priority: CUDA > MPS > CPU
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps" 
        else:
            return "cpu"
            
    def _get_optimal_dtype(self) -> torch.dtype:
        """Get optimal dtype for current device."""
        if self.device == "cuda":
            return torch.bfloat16  # A100 supports bfloat16
        elif self.device == "mps":
            return torch.float32   # MPS works best with float32
        else:
            return torch.float32   # CPU fallback

    def get_memory_footprint(self):
        """Calculate approximate memory footprint."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Approximate memory in GB (4 bytes per parameter for float32)
        total_memory_gb = (total_params * 4) / (1024**3)
        trainable_memory_gb = (trainable_params * 4) / (1024**3)
        
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "total_memory_gb": total_memory_gb,
            "trainable_memory_gb": trainable_memory_gb
        }