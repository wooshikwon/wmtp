"""
MPS 대규모 텐서 테스트
더 큰 크기의 텐서로 MPS 블로킹 문제를 재현합니다.
"""

import time

import torch

from src.utils.mps_optimizer import MPSOptimizer


def test_large_scale_mps():
    """대규모 텐서에서 MPS 성능 테스트"""

    if not torch.backends.mps.is_available():
        print("MPS not available")
        return

    device = torch.device("mps")
    print(f"🍎 Testing on {device}")

    # 다양한 크기로 테스트
    test_configs = [
        # (B, S, H, V) - 점진적으로 크기 증가
        (2, 128, 4, 50257),  # 기본
        (4, 256, 4, 50257),  # 2x batch/seq
        (8, 512, 4, 50257),  # 4x batch/seq
        (16, 1024, 4, 50257),  # 8x batch/seq (매우 큼)
    ]

    results = []

    for B, S, H, V in test_configs:
        print(f"\n{'='*60}")
        print(f"Testing size: B={B}, S={S}, H={H}, V={V}")
        print(f"Total elements: {B*S*H*V:,}")

        try:
            # 텐서 생성
            tensors = [torch.randn(B, S, V, device=device) for _ in range(H)]

            # 1. 기존 방식 (stack + view)
            print("\n📊 Traditional approach (stack + view):")
            start = time.time()
            mtp_logits = torch.stack(tensors, dim=2)
            torch.mps.synchronize()
            stack_time = time.time() - start
            print(f"   Stack: {stack_time:.4f}s")

            start = time.time()
            flat = mtp_logits.view(B * S * H, V)
            torch.mps.synchronize()
            view_time = time.time() - start
            print(f"   View: {view_time:.4f}s")

            labels_flat = torch.randint(0, V, (B * S * H,), device=device)
            start = time.time()
            ce_flat = torch.nn.functional.cross_entropy(
                flat, labels_flat, reduction="none"
            )
            ce = ce_flat.view(B, S, H)
            torch.mps.synchronize()
            ce_time = time.time() - start
            print(f"   CE: {ce_time:.4f}s")

            traditional_total = stack_time + view_time + ce_time
            print(f"   Total: {traditional_total:.4f}s")

            # 2. MPS 최적화 방식
            print("\n🚀 Optimized approach (direct allocation + per-head):")
            start = time.time()
            mtp_logits_opt = MPSOptimizer.optimize_4d_stack(tensors, 2, True)
            torch.mps.synchronize()
            opt_stack_time = time.time() - start
            print(f"   Optimized stack: {opt_stack_time:.4f}s")

            labels_3d = torch.randint(0, V, (B, S, H), device=device)
            start = time.time()
            ce_opt = MPSOptimizer.compute_ce_per_head_mps(mtp_logits_opt, labels_3d)
            torch.mps.synchronize()
            opt_ce_time = time.time() - start
            print(f"   Optimized CE: {opt_ce_time:.4f}s")

            optimized_total = opt_stack_time + opt_ce_time
            print(f"   Total: {optimized_total:.4f}s")

            # 성능 비교
            speedup = traditional_total / optimized_total if optimized_total > 0 else 0
            print(f"\n📈 Speedup: {speedup:.2f}x")

            results.append(
                {
                    "size": (B, S, H, V),
                    "traditional": traditional_total,
                    "optimized": optimized_total,
                    "speedup": speedup,
                }
            )

        except RuntimeError as e:
            print(f"\n❌ Error: {e}")
            if "out of memory" in str(e).lower():
                print("   MPS ran out of memory. Stopping tests.")
                break

    # 결과 요약
    print(f"\n{'='*60}")
    print("📊 SUMMARY")
    print(f"{'='*60}")

    for r in results:
        B, S, H, V = r["size"]
        print(
            f"Size ({B}x{S}x{H}x{V}): "
            f"Traditional={r['traditional']:.4f}s, "
            f"Optimized={r['optimized']:.4f}s, "
            f"Speedup={r['speedup']:.2f}x"
        )

    # 평균 성능 향상
    if results:
        avg_speedup = sum(r["speedup"] for r in results) / len(results)
        print(f"\n🎯 Average speedup: {avg_speedup:.2f}x")

        if avg_speedup > 1.0:
            print("✅ MPS optimization is beneficial!")
        else:
            print("⚠️ MPS optimization may not be needed for current MPS version")


if __name__ == "__main__":
    test_large_scale_mps()
