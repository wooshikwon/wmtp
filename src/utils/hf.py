"""
WMTP HuggingFace 유틸리티: 모델 관리 및 호환성 보장

WMTP 연구 맥락:
WMTP는 다양한 모델(Base, RM, Ref)을 조합하여 사용하므로
모델 간 호환성과 효율적인 메모리 관리가 중요합니다.
특히 MTP 모델의 특수한 구조를 HuggingFace 생태계와 연결합니다.

핵심 기능:
- 토큰 임베딩 크기 조정: 모델-토크나이저 vocab 일치
- 모델 크기 측정: 메모리 요구사항 계산
- dtype 변환: bf16/fp16 혼합정밀도 지원
- 안전한 모델 로딩: 오류 처리 및 복구

WMTP 알고리즘과의 연결:
- Baseline: 표준 HF 모델 로딩
- Critic-WMTP: RM 모델과 Base 모델 호환성 보장
- Rho1-WMTP: Ref 모델과 Base 모델 토크나이저 통합

사용 예시:
    >>> from src.utils.hf import resize_token_embeddings, get_model_size
    >>>
    >>> # 토크나이저와 모델 vocab 일치시키기
    >>> resize_token_embeddings(model, len(tokenizer))
    >>>
    >>> # 모델 크기 확인
    >>> params, size_gb = get_model_size(model)
    >>> print(f"모델: {params:,} 파라미터, {size_gb:.2f}GB")

성능 최적화:
- 임베딩을 8의 배수로 패딩하여 GPU 효율 향상
- bf16 사용으로 메모리 50% 절감
- 토크나이저 캐싱으로 로딩 시간 단축

디버깅 팁:
- vocab 크기 불일치: resize_token_embeddings() 호출
- OOM 오류: get_model_size()로 메모리 요구사항 확인
- dtype 오류: get_dtype()으로 올바른 형식 변환

Note:
    모델/토크나이저 로딩은 src/components/loader/model/ 하위의
    전문 로더들이 담당합니다. 여기서는 유틸리티만 제공합니다.
"""

from typing import Any

import torch
from rich.console import Console
from transformers import PreTrainedModel

console = Console()


def resize_token_embeddings(
    model: PreTrainedModel,
    tokenizer_vocab_size: int,
    pad_to_multiple_of: int = 8,
) -> None:
    """
    Resize model token embeddings to match tokenizer vocabulary.

    Args:
        model: The model to resize
        tokenizer_vocab_size: Target vocabulary size
        pad_to_multiple_of: Pad to multiple of this for efficiency
    """
    model_vocab_size = model.get_input_embeddings().weight.shape[0]

    if model_vocab_size != tokenizer_vocab_size:
        # Calculate padded size
        if pad_to_multiple_of > 1:
            padded_size = (
                (tokenizer_vocab_size + pad_to_multiple_of - 1)
                // pad_to_multiple_of
                * pad_to_multiple_of
            )
        else:
            padded_size = tokenizer_vocab_size

        console.print(
            f"[yellow]Resizing embeddings from {model_vocab_size} to {padded_size}[/yellow]"
        )
        model.resize_token_embeddings(padded_size)


def get_model_size(model: PreTrainedModel) -> tuple[int, float]:
    """
    Get model parameter count and size in GB.

    Args:
        model: The model to measure

    Returns:
        Tuple of (parameter_count, size_in_gb)
    """
    param_count = sum(p.numel() for p in model.parameters())
    param_size_gb = param_count * 4 / (1024**3)  # Assuming fp32

    # Account for actual dtype
    if hasattr(model, "dtype"):
        if model.dtype == torch.float16 or model.dtype == torch.bfloat16:
            param_size_gb /= 2
        elif model.dtype == torch.int8:
            param_size_gb /= 4

    return param_count, param_size_gb


def get_dtype(dtype_str: str) -> torch.dtype:
    """
    Convert string dtype to torch dtype.

    Args:
        dtype_str: String representation of dtype

    Returns:
        Corresponding torch.dtype

    Raises:
        ValueError: If dtype string is not recognized
    """
    dtype_map = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "int8": torch.int8,
        "int4": torch.int8,  # Quantized to int8
        "auto": None,
    }

    if dtype_str not in dtype_map:
        raise ValueError(
            f"Unknown dtype: {dtype_str}. " f"Supported: {list(dtype_map.keys())}"
        )

    return dtype_map[dtype_str]


def safe_from_pretrained(
    model_class: type,
    model_id: str,
    **kwargs: Any,
) -> Any:
    """
    Safely load a model with proper error handling.

    Args:
        model_class: The model class to use
        model_id: Model identifier
        **kwargs: Additional arguments for from_pretrained

    Returns:
        Loaded model

    Raises:
        RuntimeError: If model cannot be loaded
    """
    try:
        model = model_class.from_pretrained(model_id, **kwargs)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_id}: {e}")


__all__ = [
    "resize_token_embeddings",
    "get_model_size",
    "get_dtype",
    "safe_from_pretrained",
]
