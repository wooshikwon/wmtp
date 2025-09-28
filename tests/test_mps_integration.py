"""
MPS 통합 테스트 스크립트
실제 MPS 디바이스에서 4D 텐서 블로킹 문제를 재현하고 최적화를 검증합니다.
"""

import time
from pathlib import Path

import torch
import yaml

from src.utils.mps_optimizer import MPSOptimizer


def test_mps_blocking_issue():
    """MPS에서 4D 텐서 블로킹 문제 재현"""
    print("\n" + "=" * 60)
    print("MPS 4D Tensor Blocking Test")
    print("=" * 60)

    # 디바이스 정보 출력
    device_info = MPSOptimizer.get_device_info()
    print(f"Device Info: {device_info}")

    if not device_info["mps_available"]:
        print("⚠️ MPS not available. Skipping test.")
        return

    device = torch.device("mps")
    print(f"✅ Using device: {device}")

    # WMTP와 동일한 크기
    B, S, H, V = 2, 128, 4, 50257
    print(f"\n📊 Tensor dimensions: B={B}, S={S}, H={H}, V={V}")

    # 1. 4D 텐서 생성 (modeling.py 시뮬레이션)
    print("\n1️⃣ Creating 4D tensor (torch.stack)...")
    tensors = [torch.randn(B, S, V, device=device) for _ in range(H)]

    try:
        start = time.time()
        # 기존 방식 (블로킹 가능)
        mtp_logits = torch.stack(tensors, dim=2)  # [B, S, H, V]
        torch.mps.synchronize()  # MPS 동기화
        elapsed = time.time() - start
        print(f"   ✅ Stack completed: {elapsed:.4f}s")
        print(f"   Shape: {mtp_logits.shape}, Contiguous: {mtp_logits.is_contiguous()}")
    except Exception as e:
        print(f"   ❌ Stack failed: {e}")
        return

    # 2. 4D→2D flatten (base_wmtp_trainer.py 시뮬레이션)
    print("\n2️⃣ Flattening to 2D (view)...")
    try:
        start = time.time()
        flat = mtp_logits.view(B * S * H, V)  # ⚠️ 여기서 블로킹 가능
        torch.mps.synchronize()
        elapsed = time.time() - start
        print(f"   ✅ Flatten completed: {elapsed:.4f}s")
        print(f"   Shape: {flat.shape}")
    except Exception as e:
        print(f"   ❌ Flatten failed or timed out: {e}")

    # 3. Cross-Entropy 계산
    print("\n3️⃣ Computing Cross-Entropy...")
    try:
        labels = torch.randint(0, V, (B * S * H,), device=device)
        start = time.time()
        loss = torch.nn.functional.cross_entropy(flat, labels)
        torch.mps.synchronize()
        elapsed = time.time() - start
        print(f"   ✅ CE completed: {elapsed:.4f}s")
        print(f"   Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"   ❌ CE failed: {e}")


def test_mps_optimized_path():
    """MPS 최적화 경로 테스트"""
    print("\n" + "=" * 60)
    print("MPS Optimized Path Test")
    print("=" * 60)

    device_info = MPSOptimizer.get_device_info()
    if not device_info["mps_available"]:
        print("⚠️ MPS not available. Skipping test.")
        return

    device = torch.device("mps")
    B, S, H, V = 2, 128, 4, 50257
    print(f"📊 Tensor dimensions: B={B}, S={S}, H={H}, V={V}")

    # 설정 시뮬레이션
    config = {
        "launcher": {"resources": {"gpu_type": "mps"}},
        "devices": {"compute_backend": "mps"},
        "device": device,
    }

    use_mps = MPSOptimizer.should_use_mps_path(config)
    print(f"\n🔍 MPS optimization enabled: {use_mps}")

    # 1. 최적화된 4D 스택
    print("\n1️⃣ Optimized 4D stack...")
    tensors = [torch.randn(B, S, V, device=device) for _ in range(H)]

    start = time.time()
    mtp_logits = MPSOptimizer.optimize_4d_stack(tensors, dim=2, use_mps=True)
    torch.mps.synchronize()
    elapsed = time.time() - start
    print(f"   ✅ Optimized stack: {elapsed:.4f}s")
    print(f"   Shape: {mtp_logits.shape}, Contiguous: {mtp_logits.is_contiguous()}")

    # 2. 최적화된 CE 계산 (헤드별 3D)
    print("\n2️⃣ Optimized CE computation (per-head 3D)...")
    target_labels = torch.randint(0, V, (B, S, H), device=device)

    start = time.time()
    ce_per_head = MPSOptimizer.compute_ce_per_head_mps(mtp_logits, target_labels)
    torch.mps.synchronize()
    elapsed = time.time() - start
    print(f"   ✅ CE per head: {elapsed:.4f}s")
    print(f"   Shape: {ce_per_head.shape}, Mean CE: {ce_per_head.mean().item():.4f}")

    # 3. 성능 비교
    print("\n3️⃣ Performance comparison...")

    # 기존 방식 시간 측정
    def traditional_ce():
        logits_flat = mtp_logits.contiguous().view(B * S * H, V)
        labels_flat = target_labels.view(B * S * H)
        ce = torch.nn.functional.cross_entropy(
            logits_flat, labels_flat, reduction="none"
        )
        return ce.view(B, S, H)

    # MPS 최적화 방식 시간 측정
    def optimized_ce():
        return MPSOptimizer.compute_ce_per_head_mps(mtp_logits, target_labels)

    # 벤치마크
    traditional_time = MPSOptimizer.benchmark_operation(
        traditional_ce, num_iterations=10, warmup=3
    )
    optimized_time = MPSOptimizer.benchmark_operation(
        optimized_ce, num_iterations=10, warmup=3
    )

    speedup = traditional_time / optimized_time if optimized_time > 0 else 0
    print("\n📈 Performance Results:")
    print(f"   Traditional (4D→2D): {traditional_time:.6f}s")
    print(f"   Optimized (3D per-head): {optimized_time:.6f}s")
    print(f"   Speedup: {speedup:.2f}x")


def test_config_integration():
    """실제 config.yaml과의 통합 테스트"""
    print("\n" + "=" * 60)
    print("Config Integration Test")
    print("=" * 60)

    config_path = Path("tests/configs/config.local_test.yaml")
    if not config_path.exists():
        print(f"⚠️ Config file not found: {config_path}")
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"📄 Loaded config: {config_path}")

    # GPU 타입 확인
    gpu_type = config.get("launcher", {}).get("resources", {}).get("gpu_type", "")
    compute_backend = config.get("devices", {}).get("compute_backend", "")

    print(f"   GPU Type: {gpu_type}")
    print(f"   Compute Backend: {compute_backend}")

    # MPS 경로 판단
    use_mps = MPSOptimizer.should_use_mps_path(config)
    print(f"\n🔍 Should use MPS path: {use_mps}")

    if gpu_type == "mps" and not use_mps:
        print("⚠️ Config specifies MPS but MPS path not enabled!")
        print("   Check if MPS is available on this system.")


def main():
    """메인 테스트 실행"""
    print("\n" + "🚀 Starting MPS Integration Tests " + "🚀")

    # 1. 블로킹 문제 재현
    test_mps_blocking_issue()

    # 2. 최적화 경로 테스트
    test_mps_optimized_path()

    # 3. 설정 통합 테스트
    test_config_integration()

    print("\n" + "✅ All tests completed! " + "✅")


if __name__ == "__main__":
    main()
