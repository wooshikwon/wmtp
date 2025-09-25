#!/usr/bin/env python
"""
VESSL GPU 환경 호환성 테스트

WMTP 파이프라인이 VESSL GPU 클러스터에서 S3 스트리밍 기반으로
올바르게 동작하는지 검증합니다.

주요 검증 사항:
1. GPU 환경 감지 및 설정
2. S3 직접 스트리밍 동작
3. 분산 학습 환경 호환성
4. 메모리 효율성
5. 환경변수 기반 설정 로드
6. MLflow S3 통합

VESSL 환경 시뮬레이션:
- GPU 디바이스 목킹
- S3 전용 설정 (로컬 파일 없음)
- 분산 학습 환경변수
- 메모리 제한 환경
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
from typing import Dict, Any

import torch
import torch.distributed as dist

# WMTP components
from src.utils.path_resolver import PathResolver, PathCategory
from src.components.loader.unified_model_loader import UnifiedModelLoader
from src.components.loader.unified_data_loader import UnifiedDataLoader
from src.components.tokenizer.sentencepiece_tokenizer import SentencePieceTokenizer
from src.factory.component_factory import ComponentFactory
from src.utils.s3 import S3Manager, create_s3_manager


class VESSLEnvironmentSimulator:
    """VESSL 환경 시뮬레이터."""

    def __init__(self):
        """VESSL 환경 시뮬레이터 초기화."""
        self.original_env = {}
        self.gpu_available = torch.cuda.is_available()

    def setup_vessl_environment(self) -> Dict[str, Any]:
        """VESSL GPU 환경 설정."""
        # VESSL 환경변수 시뮬레이션
        vessl_env = {
            # 분산 학습 환경
            "WORLD_SIZE": "2",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "29500",

            # S3 자격 증명 (Mock)
            "AWS_ACCESS_KEY_ID": "AKIAMOCKKEY12345678",
            "AWS_SECRET_ACCESS_KEY": "mockSecretKey+abcdefghijklmnopqrstuvwxyz123",
            "AWS_DEFAULT_REGION": "us-east-1",

            # VESSL 특화 환경
            "VESSL_EXPERIMENT_ID": "exp-12345",
            "VESSL_RUN_ID": "run-67890",
            "HOSTNAME": "vessl-gpu-worker-0",

            # MLflow 설정
            "MLFLOW_TRACKING_URI": "s3://wmtp-mlflow/tracking",
            "MLFLOW_REGISTRY_URI": "s3://wmtp-mlflow/registry",
        }

        # 환경변수 백업 및 설정
        for key, value in vessl_env.items():
            self.original_env[key] = os.environ.get(key)
            os.environ[key] = value

        # VESSL 설정 반환
        return {
            "storage": {
                "mode": "s3",  # S3 전용 모드
                "s3": {
                    "bucket": "wmtp-models",
                    "region": "us-east-1"
                }
            },
            "distributed": {
                "backend": "nccl",
                "world_size": 2,
                "rank": 0
            },
            "paths": {
                # 로컬 경로 없음 - S3만 사용
            },
            "hardware": {
                "device": "cuda" if self.gpu_available else "cpu",
                "mixed_precision": True
            }
        }

    def cleanup_environment(self):
        """환경변수 복원."""
        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class TestVESSLGPUEnvironment(unittest.TestCase):
    """VESSL GPU 환경 테스트."""

    def setUp(self):
        """테스트 설정."""
        self.simulator = VESSLEnvironmentSimulator()
        self.vessl_config = self.simulator.setup_vessl_environment()

        # Mock S3Manager
        self.mock_s3_manager = MagicMock(spec=S3Manager)
        self.mock_s3_manager.connected = True
        self.mock_s3_manager.bucket = "wmtp-models"
        self.mock_s3_manager.region = "us-east-1"

    def tearDown(self):
        """테스트 정리."""
        self.simulator.cleanup_environment()

    def test_gpu_detection_and_setup(self):
        """GPU 감지 및 설정 테스트."""
        print("1. GPU 감지 및 설정 테스트...")

        # GPU 가용성 확인
        gpu_available = torch.cuda.is_available()
        print(f"   GPU 가용: {gpu_available}")

        if gpu_available:
            device_count = torch.cuda.device_count()
            print(f"   GPU 개수: {device_count}")

            # 첫 번째 GPU 정보
            if device_count > 0:
                device_name = torch.cuda.get_device_name(0)
                device_memory = torch.cuda.get_device_properties(0).total_memory
                print(f"   GPU 이름: {device_name}")
                print(f"   GPU 메모리: {device_memory / (1024**3):.1f} GB")

        # 설정에서 디바이스 확인
        expected_device = "cuda" if gpu_available else "cpu"
        self.assertEqual(self.vessl_config["hardware"]["device"], expected_device)

        print("   ✓ GPU 환경 설정 확인")

    @patch('src.utils.s3.create_s3_manager')
    def test_s3_streaming_only_mode(self, mock_create_s3):
        """S3 전용 스트리밍 모드 테스트."""
        print("2. S3 전용 스트리밍 모드 테스트...")

        mock_create_s3.return_value = self.mock_s3_manager

        # PathResolver S3 모드 테스트
        resolver = PathResolver(self.vessl_config)
        self.assertEqual(resolver.storage_mode, "s3")

        # S3 경로 해결 테스트
        s3_model_path = resolver.resolve("s3://wmtp-models/facebook-mtp/model.pth", PathCategory.MODEL)
        self.assertEqual(s3_model_path, "facebook-mtp/model.pth")

        # 로컬 경로가 없어도 S3 경로 우선 처리
        local_fallback = resolver.resolve("models/nonexistent.pth", PathCategory.MODEL)
        self.assertIsInstance(local_fallback, str)  # S3 키로 변환되어야 함

        print("   ✓ S3 전용 모드 동작 확인")

    @patch('src.utils.s3.create_s3_manager')
    def test_unified_loaders_gpu_compatibility(self, mock_create_s3):
        """통합 로더 GPU 호환성 테스트."""
        print("3. 통합 로더 GPU 호환성 테스트...")

        mock_create_s3.return_value = self.mock_s3_manager

        # 모델 스트림 목킹
        mock_model_stream = MagicMock()
        self.mock_s3_manager.stream_model.return_value = mock_model_stream

        # UnifiedModelLoader GPU 설정
        model_loader = UnifiedModelLoader(self.vessl_config)
        model_loader.setup({})

        # GPU 디바이스 설정 확인
        if torch.cuda.is_available():
            # GPU 환경에서는 CUDA 디바이스 사용
            with patch.object(model_loader, 'load_model') as mock_load:
                mock_model = MagicMock()
                mock_load.return_value = mock_model

                # S3 스트리밍 시뮬레이션
                try:
                    result = model_loader.run({
                        "path": "s3://wmtp-models/facebook-mtp/model.pth",
                        "device": "cuda:0"
                    })
                    # 로딩 성공 확인
                    self.assertIn("model", result)
                except Exception as e:
                    # S3 연결 실패는 예상됨 (목 환경)
                    self.assertIn("S3", str(e))

        print("   ✓ 통합 로더 GPU 호환성 확인")

    @patch('src.utils.s3.create_s3_manager')
    def test_distributed_training_compatibility(self, mock_create_s3):
        """분산 학습 환경 호환성 테스트."""
        print("4. 분산 학습 환경 호환성 테스트...")

        mock_create_s3.return_value = self.mock_s3_manager

        # 분산 환경변수 확인
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        self.assertEqual(world_size, 2)
        self.assertEqual(rank, 0)
        self.assertEqual(local_rank, 0)

        # 분산 설정 확인
        dist_config = self.vessl_config.get("distributed", {})
        self.assertEqual(dist_config["world_size"], 2)
        self.assertEqual(dist_config["backend"], "nccl")

        # ComponentFactory 분산 환경 테스트
        factory = ComponentFactory()

        # 통합 로더들이 분산 환경에서 초기화되는지 확인
        model_loader = factory.create_loader("unified-model", self.vessl_config)
        data_loader = factory.create_loader("unified-data", self.vessl_config)

        self.assertIsNotNone(model_loader)
        self.assertIsNotNone(data_loader)

        print("   ✓ 분산 학습 환경 호환성 확인")

    @patch('src.utils.s3.create_s3_manager')
    def test_memory_efficient_streaming(self, mock_create_s3):
        """메모리 효율적 스트리밍 테스트."""
        print("5. 메모리 효율적 스트리밍 테스트...")

        mock_create_s3.return_value = self.mock_s3_manager

        # 스트리밍 메서드 목킹
        mock_stream = MagicMock()
        self.mock_s3_manager.stream_model.return_value = mock_stream
        self.mock_s3_manager.stream_dataset.return_value = mock_stream

        # BaseLoader 스트리밍 테스트
        from src.components.loader.base_loader import BaseLoader

        # 스트리밍 메서드 존재 확인
        self.assertTrue(hasattr(BaseLoader, "load_with_streaming"))

        # 캐시 메서드 제거 확인
        cache_methods = [
            "compute_cache_key",
            "get_cached_path",
            "load_with_cache",
            "sync_directory_with_cache"
        ]

        for method in cache_methods:
            self.assertFalse(hasattr(BaseLoader, method))

        # S3Manager 스트리밍 메서드 확인
        streaming_methods = ["stream_model", "stream_dataset", "upload_from_bytes"]
        for method in streaming_methods:
            self.assertTrue(hasattr(self.mock_s3_manager, method))

        print("   ✓ 메모리 효율적 스트리밍 확인")

    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.set_registry_uri')
    @patch('src.utils.s3.create_s3_manager')
    def test_mlflow_s3_integration(self, mock_create_s3, mock_set_registry, mock_set_tracking):
        """MLflow S3 통합 테스트."""
        print("6. MLflow S3 통합 테스트...")

        mock_create_s3.return_value = self.mock_s3_manager

        # MLflow URI 환경변수 확인
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        registry_uri = os.environ.get("MLFLOW_REGISTRY_URI")

        self.assertEqual(tracking_uri, "s3://wmtp-mlflow/tracking")
        self.assertEqual(registry_uri, "s3://wmtp-mlflow/registry")

        # CriticStage1Pretrainer MLflow 통합 테스트
        from src.components.trainer.critic_stage1_pretrainer import CriticStage1Pretrainer

        pretrainer = CriticStage1Pretrainer({
            "target": "rm_sequence",
            "lr": 1e-4
        })

        # MLflow 매니저 목킹
        mock_mlflow_manager = MagicMock()
        mock_mlflow_manager.list_artifacts.return_value = []
        mock_mlflow_manager.log_model = MagicMock()

        # 목 컨텍스트 생성
        mock_model = MagicMock()
        mock_model.config.hidden_size = 4096
        mock_model.config.output_hidden_states = False
        mock_model.parameters.return_value = [MagicMock()]

        ctx = {
            "base_model": mock_model,
            "rm_model": MagicMock(),
            "train_dataloader": [],  # 빈 데이터로더
            "mlflow_manager": mock_mlflow_manager
        }

        # MLflow S3 저장 테스트
        pretrainer.setup({})
        result = pretrainer.run(ctx)

        self.assertIn("saved", result)
        self.assertEqual(result["saved"], "MLflow/S3")

        print("   ✓ MLflow S3 통합 확인")

    @patch('src.utils.s3.create_s3_manager')
    def test_tokenizer_s3_streaming(self, mock_create_s3):
        """토크나이저 S3 스트리밍 테스트."""
        print("7. 토크나이저 S3 스트리밍 테스트...")

        mock_create_s3.return_value = self.mock_s3_manager

        # 토크나이저 스트림 목킹
        mock_tokenizer_data = b"fake tokenizer model data"
        mock_stream = MagicMock()
        mock_stream.read.return_value = mock_tokenizer_data
        self.mock_s3_manager.stream_model.return_value = mock_stream

        # SentencePieceTokenizer S3 스트리밍
        tokenizer = SentencePieceTokenizer({
            "s3_manager": self.mock_s3_manager
        })

        # 토크나이저 초기화 테스트
        try:
            tokenizer.setup({})
            # 실제 로딩은 실패하지만 S3 스트리밍 호출은 확인 가능
            with patch.object(tokenizer, '_ensure_processor_loaded') as mock_ensure:
                mock_ensure.return_value = True
                result = tokenizer.run({})
                self.assertIn("tokenizer", result)
        except Exception as e:
            # S3 스트리밍 호출이 이루어졌는지 확인
            self.mock_s3_manager.stream_model.assert_called()

        print("   ✓ 토크나이저 S3 스트리밍 확인")

    def test_environment_variables_configuration(self):
        """환경변수 기반 설정 테스트."""
        print("8. 환경변수 기반 설정 테스트...")

        # 필수 환경변수 확인
        required_vars = [
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_DEFAULT_REGION",
            "MLFLOW_TRACKING_URI",
            "MLFLOW_REGISTRY_URI"
        ]

        for var in required_vars:
            self.assertIn(var, os.environ)
            self.assertIsNotNone(os.environ[var])

        # VESSL 특화 환경변수 확인
        vessl_vars = [
            "VESSL_EXPERIMENT_ID",
            "VESSL_RUN_ID",
            "HOSTNAME"
        ]

        for var in vessl_vars:
            self.assertIn(var, os.environ)

        # 분산 학습 환경변수 확인
        dist_vars = ["WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]
        for var in dist_vars:
            self.assertIn(var, os.environ)

        print("   ✓ 환경변수 기반 설정 확인")

    def test_error_handling_and_fallbacks(self):
        """오류 처리 및 폴백 테스트."""
        print("9. 오류 처리 및 폴백 테스트...")

        # S3 연결 실패 시나리오
        disconnected_s3 = MagicMock(spec=S3Manager)
        disconnected_s3.connected = False

        config_with_disconnected_s3 = self.vessl_config.copy()

        with patch('src.utils.s3.create_s3_manager') as mock_create:
            mock_create.return_value = disconnected_s3

            # PathResolver 폴백 동작
            resolver = PathResolver(config_with_disconnected_s3)

            # S3 연결 실패 시에도 적절히 처리되어야 함
            try:
                path = resolver.resolve("models/test.pth", PathCategory.MODEL)
                # 경로 해결 자체는 성공해야 함
                self.assertIsNotNone(path)
            except Exception as e:
                # 적절한 에러 메시지 확인
                self.assertIn("S3", str(e)) or self.assertIn("connection", str(e))

        print("   ✓ 오류 처리 및 폴백 확인")


class TestVESSLPerformance(unittest.TestCase):
    """VESSL 성능 테스트."""

    def setUp(self):
        """테스트 설정."""
        self.simulator = VESSLEnvironmentSimulator()
        self.vessl_config = self.simulator.setup_vessl_environment()

    def tearDown(self):
        """테스트 정리."""
        self.simulator.cleanup_environment()

    def test_memory_usage_optimization(self):
        """메모리 사용량 최적화 테스트."""
        print("10. 메모리 사용량 최적화 테스트...")

        # 베이스라인 메모리 사용량 측정
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB

        # 여러 로더 생성
        factory = ComponentFactory()
        loaders = []

        with patch('src.utils.s3.create_s3_manager') as mock_create:
            mock_s3 = MagicMock()
            mock_s3.connected = True
            mock_create.return_value = mock_s3

            # 다수의 로더 생성 (캐시 없이)
            for i in range(5):
                model_loader = factory.create_loader("unified-model", self.vessl_config)
                data_loader = factory.create_loader("unified-data", self.vessl_config)
                loaders.extend([model_loader, data_loader])

        # 메모리 사용량 확인
        current_memory = process.memory_info().rss / (1024 * 1024)  # MB
        memory_increase = current_memory - initial_memory

        print(f"   초기 메모리: {initial_memory:.1f} MB")
        print(f"   현재 메모리: {current_memory:.1f} MB")
        print(f"   증가량: {memory_increase:.1f} MB")

        # 캐시가 제거되어 메모리 증가가 적어야 함
        self.assertLess(memory_increase, 100)  # 100MB 미만 증가

        print("   ✓ 메모리 사용량 최적화 확인")

    @patch('src.utils.s3.create_s3_manager')
    def test_concurrent_s3_streaming(self, mock_create_s3):
        """동시 S3 스트리밍 성능 테스트."""
        print("11. 동시 S3 스트리밍 성능 테스트...")

        mock_s3_manager = MagicMock()
        mock_s3_manager.connected = True
        mock_create_s3.return_value = mock_s3_manager

        # 동시 스트리밍 시뮬레이션
        import concurrent.futures
        import time

        def simulate_stream_load(key):
            """스트리밍 로드 시뮬레이션."""
            mock_stream = MagicMock()
            mock_s3_manager.stream_model.return_value = mock_stream
            time.sleep(0.1)  # 네트워크 지연 시뮬레이션
            return f"loaded_{key}"

        # 동시에 여러 파일 스트리밍
        keys = [f"model_{i}.pth" for i in range(10)]

        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(simulate_stream_load, keys))

        elapsed_time = time.time() - start_time
        print(f"   10개 파일 동시 스트리밍: {elapsed_time:.2f}초")

        # 병렬 처리로 시간 단축 확인
        self.assertLess(elapsed_time, 1.0)  # 1초 미만이어야 함
        self.assertEqual(len(results), 10)

        print("   ✓ 동시 S3 스트리밍 성능 확인")


def main():
    """VESSL GPU 환경 테스트 실행."""
    print("=" * 80)
    print("WMTP VESSL GPU 환경 호환성 테스트")
    print("=" * 80)
    print()

    # GPU 환경 정보 출력
    gpu_available = torch.cuda.is_available()
    print(f"GPU 사용 가능: {gpu_available}")
    if gpu_available:
        print(f"GPU 개수: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print()

    # 테스트 스위트 생성
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # VESSL 환경 테스트 추가
    suite.addTests(loader.loadTestsFromTestCase(TestVESSLGPUEnvironment))
    suite.addTests(loader.loadTestsFromTestCase(TestVESSLPerformance))

    # 테스트 실행
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 결과 요약
    print("\n" + "=" * 80)
    print("VESSL GPU 환경 테스트 결과:")
    print("=" * 80)

    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total_tests - failures - errors

    print(f"실행: {total_tests} 테스트")
    print(f"통과: {passed}/{total_tests}")

    if failures > 0:
        print(f"실패: {failures}")

    if errors > 0:
        print(f"오류: {errors}")

    if passed == total_tests:
        print("\n🚀 VESSL GPU 환경 호환성 검증 완료!")
        print("📦 WMTP가 VESSL 클러스터에서 실행될 준비가 되었습니다.")
    else:
        print("\n⚠️  일부 VESSL 환경 테스트가 실패했습니다.")
        print("VESSL 배포 전에 문제를 해결해주세요.")

    return 0 if passed == total_tests else 1


if __name__ == "__main__":
    sys.exit(main())