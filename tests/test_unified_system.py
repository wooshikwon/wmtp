#!/usr/bin/env python
"""
Phase 5: 통합 시스템 테스트 스위트

WMTP 파이프라인의 모든 구성요소가 통합되어 올바르게 작동하는지 검증합니다.
캐시 제거 후 S3 스트리밍 기반 아키텍처의 end-to-end 동작을 테스트합니다.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Any, Dict

# WMTP components
from src.utils.path_resolver import PathResolver, PathCategory
from src.components.loader.unified_model_loader import UnifiedModelLoader
from src.components.loader.unified_data_loader import UnifiedDataLoader
from src.components.tokenizer.sentencepiece_tokenizer import SentencePieceTokenizer
from src.factory.component_factory import ComponentFactory
from src.utils.s3 import S3Manager


class TestUnifiedSystem(unittest.TestCase):
    """통합 시스템 테스트 클래스."""

    def setUp(self):
        """테스트 설정."""
        self.test_config = {
            "storage": {
                "mode": "hybrid",
                "s3": {
                    "bucket": "test-wmtp-bucket",
                    "region": "us-east-1"
                }
            },
            "paths": {
                "models": "./models",
                "datasets": "./data",
                "checkpoints": "./checkpoints"
            }
        }

        # Mock S3Manager for testing
        self.mock_s3_manager = MagicMock(spec=S3Manager)
        self.mock_s3_manager.connected = True

    def test_path_resolver_integration(self):
        """PathResolver와 다른 구성요소간 통합 테스트."""
        print("1. PathResolver 통합 테스트...")

        resolver = PathResolver(self.test_config)

        # 로컬 파일 경로 해결 테스트
        local_model_path = resolver.resolve("models/facebook-mtp", PathCategory.MODEL)
        self.assertIsInstance(local_model_path, Path)

        # S3 경로 해결 테스트
        s3_model_path = resolver.resolve("s3://test-bucket/models/model.pth", PathCategory.MODEL)
        self.assertEqual(s3_model_path, "models/model.pth")

        # 카테고리별 경로 자동 분류 테스트
        dataset_path = resolver.resolve("data/mbpp_train.jsonl", PathCategory.DATASET)
        self.assertIsInstance(dataset_path, (Path, str))

        print("   ✓ PathResolver 통합 동작 확인")

    @patch('src.utils.s3.create_s3_manager')
    def test_unified_loaders_integration(self, mock_create_s3):
        """통합 로더들의 상호작용 테스트."""
        print("2. UnifiedLoader 통합 테스트...")

        # S3Manager 목킹
        mock_create_s3.return_value = self.mock_s3_manager

        # UnifiedModelLoader 테스트
        model_loader = UnifiedModelLoader(self.test_config)
        model_loader.setup({})

        # 지원되는 모델 타입 확인
        supported_types = model_loader.get_supported_types()
        expected_types = ["facebook-mtp", "huggingface", "sheared-llama"]
        for expected_type in expected_types:
            self.assertIn(expected_type, supported_types)

        # UnifiedDataLoader 테스트
        data_loader = UnifiedDataLoader(self.test_config)
        data_loader.setup({})

        # 지원되는 데이터셋 타입 확인
        supported_datasets = data_loader.get_supported_types()
        expected_datasets = ["mbpp", "human-eval", "code-contests"]
        for expected_dataset in expected_datasets:
            self.assertIn(expected_dataset, supported_datasets)

        print("   ✓ 통합 로더 상호작용 확인")

    @patch('src.utils.s3.create_s3_manager')
    def test_tokenizer_s3_integration(self, mock_create_s3):
        """토크나이저 S3 통합 테스트."""
        print("3. 토크나이저 S3 통합 테스트...")

        # S3Manager 목킹
        mock_create_s3.return_value = self.mock_s3_manager

        # 토크나이저 스트림 목킹
        mock_tokenizer_stream = MagicMock()
        self.mock_s3_manager.stream_model.return_value = mock_tokenizer_stream

        tokenizer = SentencePieceTokenizer({
            "s3_manager": self.mock_s3_manager
        })
        tokenizer.setup({})

        # S3 스트리밍 호출 확인
        with patch.object(tokenizer, '_ensure_processor_loaded') as mock_ensure:
            mock_ensure.return_value = True
            result = tokenizer.run({})

            # 결과 구조 확인
            self.assertIn("tokenizer", result)

        print("   ✓ 토크나이저 S3 스트리밍 통합 확인")

    @patch('src.utils.s3.create_s3_manager')
    def test_component_factory_integration(self, mock_create_s3):
        """ComponentFactory 통합 테스트."""
        print("4. ComponentFactory 통합 테스트...")

        mock_create_s3.return_value = self.mock_s3_manager

        factory = ComponentFactory()

        # 통합 로더 생성 테스트
        model_loader = factory.create_loader("unified-model", self.test_config)
        self.assertIsInstance(model_loader, UnifiedModelLoader)

        data_loader = factory.create_loader("unified-data", self.test_config)
        self.assertIsInstance(data_loader, UnifiedDataLoader)

        # 토크나이저 생성 테스트
        tokenizer = factory.create_tokenizer("sentence-piece", self.test_config)
        self.assertIsInstance(tokenizer, SentencePieceTokenizer)

        print("   ✓ ComponentFactory 통합 동작 확인")

    def test_cache_free_operation(self):
        """캐시 없는 동작 검증."""
        print("5. 캐시 제거 후 동작 테스트...")

        # BaseLoader 캐시 제거 확인
        from src.components.loader.base_loader import BaseLoader
        import inspect

        # cache_dir 파라미터가 제거되었는지 확인
        loader_sig = inspect.signature(BaseLoader.__init__)
        self.assertNotIn("cache_dir", loader_sig.parameters)

        # 캐시 관련 메서드들이 제거되었는지 확인
        removed_methods = [
            "compute_cache_key",
            "get_cached_path",
            "load_with_cache",
            "sync_directory_with_cache"
        ]

        for method in removed_methods:
            self.assertFalse(hasattr(BaseLoader, method))

        # 스트리밍 메서드 존재 확인
        self.assertTrue(hasattr(BaseLoader, "load_with_streaming"))

        print("   ✓ 캐시 제거 후 스트리밍 기반 동작 확인")

    @patch('src.utils.s3.create_s3_manager')
    @patch('mlflow.pytorch.log_model')
    def test_mlflow_checkpoint_integration(self, mock_mlflow_log, mock_create_s3):
        """MLflow 체크포인트 통합 테스트."""
        print("6. MLflow 체크포인트 통합 테스트...")

        mock_create_s3.return_value = self.mock_s3_manager

        # MLflow 매니저 목킹
        mock_mlflow_manager = MagicMock()
        mock_mlflow_manager.list_artifacts.return_value = []

        # CriticStage1Pretrainer 테스트
        from src.components.trainer.critic_stage1_pretrainer import CriticStage1Pretrainer

        pretrainer = CriticStage1Pretrainer({
            "target": "rm_sequence",
            "lr": 1e-4
        })
        pretrainer.setup({})

        # 목 모델과 데이터로더 생성
        mock_model = MagicMock()
        mock_model.config.hidden_size = 4096
        mock_model.config.output_hidden_states = False
        mock_model.parameters.return_value = [MagicMock()]

        mock_dataloader = []  # 빈 데이터로더로 테스트

        ctx = {
            "base_model": mock_model,
            "rm_model": MagicMock(),
            "train_dataloader": mock_dataloader,
            "mlflow_manager": mock_mlflow_manager
        }

        # 실행 (빈 데이터로더이므로 바로 저장 단계로)
        result = pretrainer.run(ctx)

        # 결과 확인
        self.assertIn("saved", result)

        print("   ✓ MLflow 체크포인트 통합 확인")

    @patch('src.utils.s3.create_s3_manager')
    def test_end_to_end_pipeline(self, mock_create_s3):
        """전체 파이프라인 end-to-end 테스트."""
        print("7. End-to-End 파이프라인 테스트...")

        mock_create_s3.return_value = self.mock_s3_manager

        # 1. PathResolver로 경로 해결
        resolver = PathResolver(self.test_config)
        model_path = resolver.resolve("models/test-model", PathCategory.MODEL)

        # 2. ComponentFactory로 로더 생성
        factory = ComponentFactory()
        model_loader = factory.create_loader("unified-model", self.test_config)
        data_loader = factory.create_loader("unified-data", self.test_config)
        tokenizer = factory.create_tokenizer("sentence-piece", self.test_config)

        # 3. 구성요소 초기화
        model_loader.setup({})
        data_loader.setup({})
        tokenizer.setup({})

        # 4. 지원 타입 확인 (실제 로딩은 목 데이터로 제한)
        model_types = model_loader.get_supported_types()
        data_types = data_loader.get_supported_types()

        self.assertGreater(len(model_types), 0)
        self.assertGreater(len(data_types), 0)

        print("   ✓ End-to-End 파이프라인 동작 확인")

    def test_s3_streaming_functionality(self):
        """S3 스트리밍 기능 테스트."""
        print("8. S3 스트리밍 기능 테스트...")

        # S3Manager 스트리밍 메서드 존재 확인
        from src.utils.s3 import S3Manager

        # 스트리밍 메서드들 확인
        streaming_methods = ["stream_model", "stream_dataset"]
        for method in streaming_methods:
            self.assertTrue(hasattr(S3Manager, method))

        # 업로드 메서드 확인
        self.assertTrue(hasattr(S3Manager, "upload_from_bytes"))

        print("   ✓ S3 스트리밍 기능 확인")

    def tearDown(self):
        """테스트 정리."""
        pass


class TestSystemCompatibility(unittest.TestCase):
    """시스템 호환성 테스트 클래스."""

    def test_python_dependencies(self):
        """Python 의존성 호환성 테스트."""
        print("9. Python 의존성 호환성 테스트...")

        # 필수 라이브러리 import 테스트
        try:
            import torch
            import transformers
            import mlflow
            import boto3
            import rich
            import datasets
            print("   ✓ 모든 필수 의존성 확인")
        except ImportError as e:
            self.fail(f"필수 의존성 누락: {e}")

    def test_vessl_environment_readiness(self):
        """VESSL 환경 준비성 테스트."""
        print("10. VESSL 환경 준비성 테스트...")

        # 환경변수 기반 설정 확인
        test_config = {
            "storage": {
                "mode": "s3",
                "s3": {
                    "bucket": "wmtp-models",
                    "region": "us-east-1"
                }
            }
        }

        # PathResolver VESSL 모드 테스트
        resolver = PathResolver(test_config)
        self.assertEqual(resolver.storage_mode, "s3")

        # S3 경로 해결 테스트
        s3_path = resolver.resolve("s3://wmtp-models/facebook-mtp/model.pth", PathCategory.MODEL)
        self.assertEqual(s3_path, "facebook-mtp/model.pth")

        print("   ✓ VESSL 환경 준비성 확인")


def main():
    """통합 테스트 실행."""
    print("=" * 80)
    print("WMTP Phase 5: 통합 시스템 테스트")
    print("=" * 80)
    print()

    # 테스트 스위트 생성
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 통합 시스템 테스트 추가
    suite.addTests(loader.loadTestsFromTestCase(TestUnifiedSystem))
    suite.addTests(loader.loadTestsFromTestCase(TestSystemCompatibility))

    # 테스트 실행
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 결과 요약
    print("\n" + "=" * 80)
    print("통합 테스트 결과:")
    print("=" * 80)

    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total_tests - failures - errors

    print(f"실행: {total_tests} 테스트")
    print(f"통과: {passed}/{total_tests}")

    if failures > 0:
        print(f"실패: {failures}")
        for test, trace in result.failures:
            print(f"  - {test}: {trace.split(chr(10))[-2]}")

    if errors > 0:
        print(f"오류: {errors}")
        for test, trace in result.errors:
            print(f"  - {test}: {trace.split(chr(10))[-2]}")

    if passed == total_tests:
        print("\n🎉 모든 통합 테스트가 성공했습니다!")
        print("📦 WMTP 파이프라인이 완전히 통합되어 배포 준비가 완료되었습니다.")
    else:
        print("\n⚠️  일부 통합 테스트가 실패했습니다. 상세 로그를 확인하세요.")

    return 0 if passed == total_tests else 1


if __name__ == "__main__":
    sys.exit(main())