"""Model loaders for WMTP framework."""

# Import all model loaders to ensure registry registration
from .checkpoint_loader import CheckpointLoader
from .hf_model_loader import HFModelLoader
from .mtp_native_cpu_loader import MTPNativeCPULoader
from .mtp_native_loader import MTPNativeLoader
from .sheared_llama_loader import ShearedLLaMALoader
from .starling_rm_loader import StarlingRMLoader

__all__ = [
    "HFModelLoader",
    "MTPNativeLoader",
    "MTPNativeCPULoader",
    "CheckpointLoader",
    "StarlingRMLoader",
    "ShearedLLaMALoader",
]
