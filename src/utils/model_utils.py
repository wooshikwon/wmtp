"""모델 관련 공통 유틸리티"""

from typing import Any

import torch


def extract_hidden_states(outputs: Any) -> torch.Tensor:
    """모델 출력에서 hidden_states를 안전하게 추출

    다양한 모델 출력 형태를 지원:
    - dict: outputs["hidden_states"] 또는 outputs["last_hidden_state"]
    - object: outputs.hidden_states 또는 outputs.last_hidden_state
    - HuggingFace ModelOutput objects

    Args:
        outputs: 모델 출력 (dict, BaseModelOutput, CausalLMOutput 등)

    Returns:
        torch.Tensor: [B, S, D] 형태의 hidden states

    Raises:
        ValueError: hidden_states 추출 실패 시
    """
    hidden_states = None

    try:
        # Case 1: dict 형태에서 hidden_states 키 접근
        if isinstance(outputs, dict) and "hidden_states" in outputs:
            hs = outputs["hidden_states"]
            # list/tuple인 경우 마지막 레이어 선택
            hidden_states = hs[-1] if isinstance(hs, (list, tuple)) else hs

        # Case 2: dict 형태에서 last_hidden_state 키 접근
        elif isinstance(outputs, dict) and "last_hidden_state" in outputs:
            hidden_states = outputs["last_hidden_state"]

        # Case 3: object 형태에서 hidden_states 속성 접근
        elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            hs = outputs.hidden_states
            hidden_states = hs[-1] if isinstance(hs, (list, tuple)) else hs

        # Case 4: object 형태에서 last_hidden_state 속성 접근
        elif (
            hasattr(outputs, "last_hidden_state")
            and outputs.last_hidden_state is not None
        ):
            hidden_states = outputs.last_hidden_state

        # Case 5: HuggingFace ModelOutput - 추가 fallback
        elif hasattr(outputs, "__class__") and "Output" in outputs.__class__.__name__:
            # 가능한 속성들을 순서대로 시도
            for attr_name in [
                "last_hidden_state",
                "hidden_states",
                "encoder_last_hidden_state",
            ]:
                if hasattr(outputs, attr_name):
                    attr_value = getattr(outputs, attr_name)
                    if attr_value is not None:
                        if isinstance(attr_value, (list, tuple)):
                            hidden_states = attr_value[-1]
                        else:
                            hidden_states = attr_value
                        break

    except Exception:
        # 예상치 못한 오류는 조용히 넘어가고 아래에서 처리
        pass

    # 검증: hidden_states가 올바른 형태인지 확인
    if hidden_states is None:
        raise ValueError(
            f"Failed to extract hidden_states from model outputs. "
            f"Output type: {type(outputs)}, "
            f"Available keys/attributes: {_get_available_keys(outputs)}"
        )

    if not isinstance(hidden_states, torch.Tensor):
        raise ValueError(
            f"hidden_states must be torch.Tensor, got {type(hidden_states)}"
        )

    if hidden_states.ndim != 3:
        raise ValueError(
            f"Expected hidden_states shape [B, S, D], got {hidden_states.shape}"
        )

    return hidden_states


def ensure_output_hidden_states(model: torch.nn.Module) -> None:
    """모델이 hidden_states를 출력하도록 설정

    HuggingFace 모델의 config.output_hidden_states를 True로 설정합니다.
    설정이 불가능한 모델은 조용히 무시됩니다.

    Args:
        model: 설정할 모델
    """
    try:
        if hasattr(model, "config") and hasattr(model.config, "output_hidden_states"):
            model.config.output_hidden_states = True
    except Exception:
        # 설정 실패는 조용히 무시 (일부 custom 모델은 지원하지 않을 수 있음)
        pass


def get_model_output_info(outputs: Any) -> dict[str, Any]:
    """디버깅용: 모델 출력 구조 정보 반환

    Args:
        outputs: 모델 출력

    Returns:
        Dict containing output structure information
    """
    info = {
        "type": str(type(outputs)),
        "is_dict": isinstance(outputs, dict),
    }

    if isinstance(outputs, dict):
        info["dict_keys"] = list(outputs.keys())
        if "hidden_states" in outputs:
            hs = outputs["hidden_states"]
            info["hidden_states_type"] = str(type(hs))
            if isinstance(hs, (list, tuple)):
                info["hidden_states_length"] = len(hs)
                if len(hs) > 0:
                    info["hidden_states_last_shape"] = getattr(
                        hs[-1], "shape", "no shape"
                    )
            else:
                info["hidden_states_shape"] = getattr(hs, "shape", "no shape")
    else:
        # Object 형태의 속성들 조사
        attrs = [attr for attr in dir(outputs) if not attr.startswith("_")]
        info["attributes"] = attrs[:20]  # 처음 20개만

        if hasattr(outputs, "hidden_states"):
            hs = outputs.hidden_states
            info["hidden_states_type"] = str(type(hs))
            if isinstance(hs, (list, tuple)):
                info["hidden_states_length"] = len(hs)
                if len(hs) > 0:
                    info["hidden_states_last_shape"] = getattr(
                        hs[-1], "shape", "no shape"
                    )
            else:
                info["hidden_states_shape"] = getattr(hs, "shape", "no shape")

        if hasattr(outputs, "last_hidden_state"):
            lhs = outputs.last_hidden_state
            info["last_hidden_state_shape"] = getattr(lhs, "shape", "no shape")

    return info


def _get_available_keys(outputs: Any) -> str:
    """디버깅을 위한 사용 가능한 키/속성 목록 반환"""
    if isinstance(outputs, dict):
        return f"dict keys: {list(outputs.keys())}"
    else:
        attrs = [attr for attr in dir(outputs) if not attr.startswith("_")]
        return f"object attrs: {attrs[:10]}..."  # 처음 10개만
