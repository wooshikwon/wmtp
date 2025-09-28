"""
MPS Optimizer 유닛 테스트

MPS 최적화 유틸리티의 각 기능을 테스트합니다.
MPS가 없는 환경에서도 테스트가 가능하도록 설계되었습니다.
"""

from unittest.mock import patch

import pytest
import torch

from src.utils.mps_optimizer import MPSOptimizer


class TestMPSOptimizer:
    """MPSOptimizer 클래스 테스트"""

    def test_should_use_mps_path_with_gpu_type(self):
        """launcher.resources.gpu_type이 "mps"일 때 MPS 경로 선택"""
        config = {"launcher": {"resources": {"gpu_type": "mps"}}}

        with patch("torch.backends.mps.is_available", return_value=True):
            assert MPSOptimizer.should_use_mps_path(config) == True

        with patch("torch.backends.mps.is_available", return_value=False):
            assert MPSOptimizer.should_use_mps_path(config) == False

    def test_should_use_mps_path_with_compute_backend(self):
        """devices.compute_backend가 "mps"일 때 MPS 경로 선택"""
        config = {"devices": {"compute_backend": "mps"}}

        with patch("torch.backends.mps.is_available", return_value=True):
            assert MPSOptimizer.should_use_mps_path(config) == True

    def test_should_use_mps_path_with_device(self):
        """device가 "mps:0"일 때 MPS 경로 선택"""
        config = {"device": "mps:0"}

        with patch("torch.backends.mps.is_available", return_value=True):
            assert MPSOptimizer.should_use_mps_path(config) == True

    def test_should_not_use_mps_path_for_cuda(self):
        """CUDA 설정일 때 MPS 경로 선택하지 않음"""
        config = {
            "launcher": {"resources": {"gpu_type": "A100"}},
            "devices": {"compute_backend": "cuda"},
        }

        with patch("torch.backends.mps.is_available", return_value=True):
            assert MPSOptimizer.should_use_mps_path(config) == False

    def test_optimize_4d_stack_mps_path(self):
        """MPS 경로에서 4D 텐서 스택 최적화"""
        B, S, V, H = 2, 128, 50257, 4
        tensors = [torch.randn(B, S, V) for _ in range(H)]

        # MPS 최적화 경로
        result = MPSOptimizer.optimize_4d_stack(tensors, dim=2, use_mps=True)

        assert result.shape == (B, S, H, V)
        assert result.is_contiguous()

        # 값 검증
        for i in range(H):
            assert torch.allclose(result[:, :, i, :], tensors[i])

    def test_optimize_4d_stack_cuda_path(self):
        """CUDA 경로에서 일반 torch.stack 사용"""
        B, S, V, H = 2, 128, 50257, 4
        tensors = [torch.randn(B, S, V) for _ in range(H)]

        # CUDA 경로 (일반 stack)
        result = MPSOptimizer.optimize_4d_stack(tensors, dim=2, use_mps=False)
        expected = torch.stack(tensors, dim=2)

        assert result.shape == (B, S, H, V)
        assert torch.allclose(result, expected)

    def test_compute_ce_per_head_mps(self):
        """MPS 최적화 Cross-Entropy 계산"""
        B, S, H, V = 2, 10, 4, 100  # 작은 크기로 테스트
        logits = torch.randn(B, S, H, V)
        target_labels = torch.randint(0, V, (B, S, H))

        # MPS CE 계산
        ce_mps = MPSOptimizer.compute_ce_per_head_mps(logits, target_labels)

        assert ce_mps.shape == (B, S, H)
        assert ce_mps.is_contiguous()

        # 값 검증: 각 헤드별로 CE 계산 결과와 비교
        for h in range(H):
            logits_h = logits[:, :, h, :]
            labels_h = target_labels[:, :, h]
            expected_ce_h = torch.nn.functional.cross_entropy(
                logits_h.transpose(1, 2), labels_h, reduction="none"
            )
            assert torch.allclose(ce_mps[:, :, h], expected_ce_h, rtol=1e-5)

    def test_compute_ce_with_ignore_index(self):
        """ignore_index가 있는 경우 CE 계산"""
        B, S, H, V = 2, 10, 4, 100
        ignore_index = -100

        logits = torch.randn(B, S, H, V)
        target_labels = torch.randint(0, V, (B, S, H))

        # 일부 라벨을 ignore_index로 설정
        target_labels[0, 0, 0] = ignore_index
        target_labels[1, 5, 2] = ignore_index

        ce = MPSOptimizer.compute_ce_per_head_mps(
            logits, target_labels, ignore_index=ignore_index
        )

        assert ce.shape == (B, S, H)
        # ignore_index 위치의 CE는 0이어야 함
        assert ce[0, 0, 0] == 0
        assert ce[1, 5, 2] == 0

    def test_get_device_info(self):
        """디바이스 정보 반환 테스트"""
        info = MPSOptimizer.get_device_info()

        assert "mps_available" in info
        assert "cuda_available" in info
        assert "current_device" in info
        assert "device_name" in info

        # 현재 환경에 따라 적절한 디바이스가 선택되었는지 확인
        if info["mps_available"]:
            assert info["current_device"] == "mps"
            assert info["device_name"] == "Apple Silicon GPU"
        elif info["cuda_available"]:
            assert info["current_device"] == "cuda"
        else:
            assert info["current_device"] == "cpu"
            assert info["device_name"] == "CPU"

    def test_validate_tensor_compatibility(self):
        """텐서 호환성 검증 테스트"""
        # float32 텐서 (MPS 호환)
        tensor_f32 = torch.randn(10, 10, dtype=torch.float32)
        assert MPSOptimizer.validate_tensor_compatibility(tensor_f32, "mps") == True

        # float64 텐서 (MPS 비호환)
        tensor_f64 = torch.randn(10, 10, dtype=torch.float64)
        assert MPSOptimizer.validate_tensor_compatibility(tensor_f64, "mps") == False

        # 매우 큰 텐서 (MPS 제한)
        large_tensor = torch.randn(1000, 1000, 1001)  # > 1B elements
        assert MPSOptimizer.validate_tensor_compatibility(large_tensor, "mps") == False

        # CUDA/CPU는 모든 dtype 지원
        assert MPSOptimizer.validate_tensor_compatibility(tensor_f64, "cuda") == True
        assert MPSOptimizer.validate_tensor_compatibility(tensor_f64, "cpu") == True

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS not available"
    )
    def test_mps_integration(self):
        """실제 MPS 디바이스에서 통합 테스트 (MPS가 있을 때만)"""
        device = torch.device("mps")
        B, S, V, H = 2, 128, 1000, 4  # 적당한 크기

        # MPS에서 텐서 생성
        tensors = [torch.randn(B, S, V, device=device) for _ in range(H)]

        # 4D 스택
        mtp_logits = MPSOptimizer.optimize_4d_stack(tensors, dim=2, use_mps=True)
        assert mtp_logits.device.type == "mps"
        assert mtp_logits.shape == (B, S, H, V)

        # CE 계산
        target_labels = torch.randint(0, V, (B, S, H), device=device)
        ce = MPSOptimizer.compute_ce_per_head_mps(mtp_logits, target_labels)
        assert ce.device.type == "mps"
        assert ce.shape == (B, S, H)

    def test_benchmark_operation(self):
        """벤치마크 함수 테스트"""

        def simple_op(x):
            return x * 2

        tensor = torch.randn(100, 100)
        avg_time = MPSOptimizer.benchmark_operation(
            simple_op, tensor, num_iterations=5, warmup=1
        )

        assert isinstance(avg_time, float)
        assert avg_time > 0

    def test_edge_cases(self):
        """엣지 케이스 테스트"""
        # 빈 설정
        assert MPSOptimizer.should_use_mps_path({}) == False

        # 대소문자 구분 없음
        config = {
            "launcher": {
                "resources": {
                    "gpu_type": "MPS"  # 대문자
                }
            }
        }
        with patch("torch.backends.mps.is_available", return_value=True):
            assert MPSOptimizer.should_use_mps_path(config) == True

        # 단일 텐서 스택
        single_tensor = torch.randn(2, 3, 4)
        result = MPSOptimizer.optimize_4d_stack([single_tensor], dim=2, use_mps=True)
        assert result.shape == (2, 3, 1, 4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
