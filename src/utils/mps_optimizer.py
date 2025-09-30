"""
MPS (Metal Performance Shaders) 백엔드 최적화 유틸리티
Apple Silicon GPU를 위한 4D 텐서 처리 최적화

WMTP 연구 맥락:
WMTP(Weighted Multi-Token Prediction)는 4D 텐서 [B, S, H, V]를 사용하여
여러 미래 토큰을 동시에 예측합니다. 그러나 MPS 백엔드는 4D→2D view() 연산에서
블로킹이 발생하는 문제가 있어, 이를 우회하는 최적화가 필요합니다.

핵심 기능:
- 설정 기반 MPS 경로 자동 판단
- 4D 텐서 스택 최적화 (연속 메모리 보장)
- Cross-Entropy 계산 최적화 (헤드별 3D 처리)

성능 최적화:
- MPS에서 4D→2D flatten 대신 헤드별 3D 처리로 블로킹 회피
- 연속 메모리 할당으로 MPS 커널 효율성 향상
- CPU 대비 2-3배 성능 향상 목표

사용 예시:
    >>> from src.utils.mps_optimizer import MPSOptimizer
    >>>
    >>> # 설정 기반 MPS 경로 판단
    >>> config = {"launcher": {"resources": {"gpu_type": "mps"}}}
    >>> use_mps = MPSOptimizer.should_use_mps_path(config)
    >>>
    >>> # 4D 텐서 스택 최적화
    >>> tensors = [torch.randn(2, 128, 50257) for _ in range(4)]
    >>> mtp_logits = MPSOptimizer.optimize_4d_stack(tensors, dim=2, use_mps=use_mps)
"""

from typing import Any, Callable

import torch


class MPSOptimizer:
    """MPS 특화 텐서 연산 최적화 클래스

    Apple Silicon GPU를 위한 WMTP 4D 텐서 처리 최적화를 제공합니다.
    설정 기반으로 자동으로 MPS 최적화 경로를 선택합니다.
    """

    @staticmethod
    def should_use_mps_path(config: dict[str, Any]) -> bool:
        """
        설정 기반 MPS 경로 사용 여부 결정

        판단 기준 (OR 조건):
        1. launcher.resources.gpu_type == "mps"
        2. devices.compute_backend == "mps"
        3. 실제 device가 mps

        Args:
            config: 전체 설정 딕셔너리

        Returns:
            bool: MPS 최적화 경로 사용 여부

        Example:
            >>> config = {"launcher": {"resources": {"gpu_type": "mps"}}}
            >>> MPSOptimizer.should_use_mps_path(config)
            True
        """
        # 1. launcher의 gpu_type 체크
        gpu_type = (
            config.get("launcher", {}).get("resources", {}).get("gpu_type", "")
        ).lower()

        # 2. devices의 compute_backend 체크
        compute_backend = (config.get("devices", {}).get("compute_backend", "")).lower()

        # 3. 실제 device 체크 (런타임)
        device_str = str(config.get("device", "")).lower()
        device_type = device_str.split(":")[0] if device_str else ""

        # 4. MPS 사용 가능 여부 확인
        mps_available = torch.backends.mps.is_available()

        # 최종 판단
        use_mps = (
            gpu_type == "mps" or compute_backend == "mps" or device_type == "mps"
        ) and mps_available

        if use_mps:
            print(
                f"🍎 MPS optimization enabled (gpu_type={gpu_type}, "
                f"backend={compute_backend}, device={device_type})"
            )

        return use_mps

    @staticmethod
    def optimize_4d_stack(
        tensors: list[torch.Tensor], dim: int, use_mps: bool
    ) -> torch.Tensor:
        """
        MPS 최적화된 4D 텐서 스택

        MPS 문제: torch.stack()이 불연속 메모리 생성으로 블로킹 발생
        해결책: 연속 메모리에 직접 할당 후 복사

        Args:
            tensors: 스택할 3D 텐서 리스트 [B, S, V]
            dim: 스택 차원 (2 for MTP)
            use_mps: MPS 최적화 사용 여부

        Returns:
            torch.Tensor: 4D 텐서 [B, S, H, V]

        Example:
            >>> tensors = [torch.randn(2, 128, 50257) for _ in range(4)]
            >>> result = MPSOptimizer.optimize_4d_stack(tensors, dim=2, use_mps=True)
            >>> result.shape
            torch.Size([2, 128, 4, 50257])
        """
        if use_mps and dim == 2:
            # MPS 최적화: 연속 메모리 할당 후 복사
            B, S, V = tensors[0].shape
            H = len(tensors)

            # 1. 연속 메모리 미리 할당
            result = torch.zeros(
                B, S, H, V, dtype=tensors[0].dtype, device=tensors[0].device
            )

            # 2. 각 헤드 직접 복사 (메모리 연속성 보장)
            for i, tensor in enumerate(tensors):
                result[:, :, i, :] = tensor

            # 3. 명시적 연속성 보장
            return result.contiguous()
        else:
            # 기존 CUDA/CPU 방식
            return torch.stack(tensors, dim=dim)

    @staticmethod
    def compute_ce_per_head_mps(
        logits: torch.Tensor, target_labels: torch.Tensor, ignore_index: int = -100
    ) -> torch.Tensor:
        """
        MPS 최적화 Cross-Entropy 계산

        MPS 문제: 4D→2D view() 연산에서 무한 블로킹
        해결책: 헤드별 3D 처리 (view 없이)

        Args:
            logits: 4D 텐서 [B, S, H, V]
            target_labels: 3D 텐서 [B, S, H]
            ignore_index: 무시할 라벨 인덱스

        Returns:
            torch.Tensor: CE per head [B, S, H]

        Example:
            >>> logits = torch.randn(2, 128, 4, 50257)
            >>> labels = torch.randint(0, 50257, (2, 128, 4))
            >>> ce = MPSOptimizer.compute_ce_per_head_mps(logits, labels)
            >>> ce.shape
            torch.Size([2, 128, 4])
        """
        B, S, H, V = logits.shape
        ce_list = []

        # 헤드별 3D 처리 (MPS 친화적)
        for h in range(H):
            # 3D 슬라이스 추출 (view 없음!)
            logits_h = logits[:, :, h, :].contiguous()  # [B, S, V]
            labels_h = target_labels[:, :, h].contiguous()  # [B, S]

            # 3D Cross-Entropy (MPS가 잘 처리)
            # transpose로 [B, V, S] 형태로 변경
            ce_h = torch.nn.functional.cross_entropy(
                logits_h.transpose(1, 2),  # [B, V, S]
                labels_h,  # [B, S]
                ignore_index=ignore_index,
                reduction="none",
            )  # [B, S]

            ce_list.append(ce_h)

        # 헤드 차원으로 스택
        ce_per_head = torch.stack(ce_list, dim=2)  # [B, S, H]
        return ce_per_head.contiguous()

    @staticmethod
    def get_device_info() -> dict[str, Any]:
        """
        현재 사용 가능한 디바이스 정보 반환

        Returns:
            dict: 디바이스 정보 딕셔너리
                - mps_available: MPS 사용 가능 여부
                - cuda_available: CUDA 사용 가능 여부
                - current_device: 현재 추천 디바이스
                - device_name: 디바이스 이름

        Example:
            >>> info = MPSOptimizer.get_device_info()
            >>> print(info['current_device'])
            mps
        """
        info = {
            "mps_available": torch.backends.mps.is_available(),
            "cuda_available": torch.cuda.is_available(),
            "current_device": "cpu",
            "device_name": "CPU",
        }

        if info["mps_available"]:
            info["current_device"] = "mps"
            info["device_name"] = "Apple Silicon GPU"
        elif info["cuda_available"]:
            info["current_device"] = "cuda"
            info["device_name"] = (
                torch.cuda.get_device_name(0)
                if torch.cuda.device_count() > 0
                else "CUDA"
            )

        return info

    @staticmethod
    def validate_tensor_compatibility(tensor: torch.Tensor, device_type: str) -> bool:
        """
        텐서가 특정 디바이스와 호환되는지 확인

        Args:
            tensor: 확인할 텐서
            device_type: 대상 디바이스 타입 ("mps", "cuda", "cpu")

        Returns:
            bool: 호환성 여부

        Example:
            >>> tensor = torch.randn(2, 3, device="cpu")
            >>> MPSOptimizer.validate_tensor_compatibility(tensor, "mps")
            True
        """
        # 텐서 디바이스 확인
        tensor_device = str(tensor.device).split(":")[0]

        # 디바이스 호환성 체크
        if tensor_device != device_type and device_type != "cpu":
            print(f"⚠️ Tensor device ({tensor_device}) != target device ({device_type})")
            return False

        # MPS 특별 처리
        if device_type == "mps":
            # MPS는 특정 dtype만 지원
            supported_dtypes = [torch.float32, torch.float16]
            if tensor.dtype not in supported_dtypes:
                print(f"⚠️ MPS does not support {tensor.dtype}. Converting to float32.")
                return False

            # 텐서 크기 제한 확인 (MPS는 매우 큰 텐서에서 문제 발생 가능)
            if tensor.numel() > 1e9:  # 1B elements
                print(f"⚠️ Tensor too large for MPS: {tensor.numel()} elements")
                return False

        return True

    @staticmethod
    def benchmark_operation(
        operation_func: Callable[..., Any], *args, num_iterations: int = 10, warmup: int = 3
    ) -> float:
        """
        특정 연산의 성능 벤치마크

        Args:
            operation_func: 벤치마크할 함수
            *args: 함수 인자들
            num_iterations: 측정 반복 횟수
            warmup: 워밍업 반복 횟수

        Returns:
            float: 평균 실행 시간 (초)

        Example:
            >>> def test_op(x): return x * 2
            >>> tensor = torch.randn(1000, 1000, device="mps")
            >>> time = MPSOptimizer.benchmark_operation(test_op, tensor)
            >>> print(f"Average time: {time:.4f}s")
        """
        import time

        # Warmup
        for _ in range(warmup):
            operation_func(*args)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elif torch.backends.mps.is_available():
                torch.mps.synchronize()

        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            operation_func(*args)

            # 동기화 (GPU 연산 완료 대기)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elif torch.backends.mps.is_available():
                torch.mps.synchronize()

            end = time.perf_counter()
            times.append(end - start)

        return sum(times) / len(times)
