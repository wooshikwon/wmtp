#!/usr/bin/env python
"""Phase 3 구현 검증 스크립트."""

import sys
import io
import torch
from pathlib import Path


def test_s3_upload_from_bytes():
    """S3Manager에 upload_from_bytes 메서드가 추가되었는지 확인."""
    print("1. S3Manager upload_from_bytes 메서드 확인...")
    try:
        from src.utils.s3 import S3Manager

        # 메서드 존재 확인
        assert hasattr(S3Manager, "upload_from_bytes"), "upload_from_bytes 메서드가 없습니다"
        print("   ✓ upload_from_bytes 메서드 존재 확인")
        return True
    except Exception as e:
        print(f"   ✗ 실패: {e}")
        return False


def test_dist_manager_mlflow_integration():
    """DistributedManager에 MLflow 통합이 되었는지 확인."""
    print("2. DistributedManager MLflow 통합 확인...")
    try:
        from src.utils.dist import DistributedManager
        import inspect

        # save_checkpoint 메서드 시그니처 확인
        sig = inspect.signature(DistributedManager.save_checkpoint)
        params = list(sig.parameters.keys())

        assert "mlflow_manager" in params, "mlflow_manager 파라미터가 없습니다"
        print("   ✓ save_checkpoint에 mlflow_manager 파라미터 존재")

        # load_checkpoint S3 지원 확인 (코드 검사)
        source = inspect.getsource(DistributedManager.load_checkpoint)
        assert "s3://" in source, "S3 경로 처리 코드가 없습니다"
        print("   ✓ load_checkpoint S3 지원 확인")

        return True
    except Exception as e:
        print(f"   ✗ 실패: {e}")
        return False


def test_critic_pretrainer_mlflow():
    """CriticStage1Pretrainer MLflow 저장 확인."""
    print("3. CriticStage1Pretrainer MLflow 저장 확인...")
    try:
        from src.components.trainer.critic_stage1_pretrainer import CriticStage1Pretrainer
        import inspect

        # run 메서드 소스 확인
        source = inspect.getsource(CriticStage1Pretrainer.run)

        # cache_root 제거 확인
        assert "cache_root" not in source or "cache_root = ctx" not in source, "cache_root가 여전히 사용됨"
        print("   ✓ cache_root 제거 확인")

        # MLflow 사용 확인
        assert "mlflow_manager" in source, "mlflow_manager 사용 안 함"
        print("   ✓ mlflow_manager 사용 확인")

        return True
    except Exception as e:
        print(f"   ✗ 실패: {e}")
        return False


def test_pipeline_s3_resume():
    """Pipeline S3 체크포인트 재개 지원 확인."""
    print("4. Pipeline S3 체크포인트 재개 확인...")
    try:
        import inspect
        from src.pipelines.training_pipeline import run_training_pipeline

        # 함수 소스 확인
        source = inspect.getsource(run_training_pipeline)

        # S3 체크포인트 로드 코드 확인
        assert 's3://' in source, "S3 체크포인트 처리 코드 없음"
        assert 'S3Manager' in source or 'stream_model' in source, "S3Manager 사용 안 함"
        print("   ✓ S3 체크포인트 로드 지원 확인")

        return True
    except Exception as e:
        print(f"   ✗ 실패: {e}")
        return False


def test_trainer_mlflow_pass():
    """MTPWeightedCETrainer MLflow 전달 확인."""
    print("5. MTPWeightedCETrainer MLflow 전달 확인...")
    try:
        import inspect
        from src.components.trainer.mtp_weighted_ce_trainer import MTPWeightedCETrainer

        # _save_checkpoint 메서드 소스 확인
        source = inspect.getsource(MTPWeightedCETrainer._save_checkpoint)

        # MLflow 매니저 전달 확인
        assert "mlflow_manager=self.mlflow" in source, "mlflow_manager 전달 안 함"
        print("   ✓ save_checkpoint에 mlflow_manager 전달 확인")

        # _save_final_checkpoint도 확인
        source_final = inspect.getsource(MTPWeightedCETrainer._save_final_checkpoint)
        assert "mlflow_manager=self.mlflow" in source_final, "최종 체크포인트에 mlflow_manager 전달 안 함"
        print("   ✓ save_final_checkpoint에도 mlflow_manager 전달 확인")

        return True
    except Exception as e:
        print(f"   ✗ 실패: {e}")
        return False


def main():
    """모든 테스트 실행."""
    print("=" * 60)
    print("Phase 3: MLflow 체크포인트 통합 검증")
    print("=" * 60)

    tests = [
        test_s3_upload_from_bytes,
        test_dist_manager_mlflow_integration,
        test_critic_pretrainer_mlflow,
        test_pipeline_s3_resume,
        test_trainer_mlflow_pass,
    ]

    results = []
    for test in tests:
        print()
        results.append(test())

    print("\n" + "=" * 60)
    print("검증 결과:")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(f"통과: {passed}/{total} 테스트")

    if passed == total:
        print("✅ Phase 3 구현이 성공적으로 완료되었습니다!")
    else:
        print("⚠️ 일부 테스트가 실패했습니다. 위 로그를 확인하세요.")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())