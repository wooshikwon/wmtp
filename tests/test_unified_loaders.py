"""
UnifiedLoader 테스트 코드

Phase 2 통합 로더의 기능을 검증합니다.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.components.loader.unified_data_loader import UnifiedDataLoader
from src.components.loader.unified_model_loader import UnifiedModelLoader
from src.utils.path_resolver import PathResolver


def test_unified_model_loader_init():
    """UnifiedModelLoader 초기화 테스트"""
    print("\n[UnifiedModelLoader 초기화 테스트]")

    config = {
        "storage": {
            "mode": "auto",
            "s3": {"bucket": "wmtp", "region": "ap-northeast-2"},
        },
        "devices": {"compute_backend": "auto", "mixed_precision": "bf16"},
    }

    loader = UnifiedModelLoader(config)

    assert loader.path_resolver is not None
    assert isinstance(loader.path_resolver, PathResolver)
    assert loader.mixed_precision == "bf16"
    print("✓ UnifiedModelLoader 초기화 성공")


def test_model_type_detection():
    """모델 타입 자동 감지 테스트"""
    print("\n[모델 타입 감지 테스트]")

    config = {"storage": {"mode": "local"}}
    loader = UnifiedModelLoader(config)

    test_cases = [
        ("models/consolidated.pth", "mtp_native"),
        ("models/7b_mtp_model.bin", "mtp_native"),
        ("checkpoints/epoch_5.pt", "checkpoint"),
        ("models/sheared_llama_2.7b", "sheared_llama"),
        ("models/starling-rm-7b", "starling_rm"),
        ("models/codellama-7b", "huggingface"),
    ]

    for path, expected_type in test_cases:
        detected = loader._detect_model_type(path)
        assert (
            detected == expected_type
        ), f"경로 {path}: 기대 {expected_type}, 실제 {detected}"
        print(f"✓ {path} → {detected}")


def test_unified_data_loader_init():
    """UnifiedDataLoader 초기화 테스트"""
    print("\n[UnifiedDataLoader 초기화 테스트]")

    config = {
        "storage": {
            "mode": "auto",
            "s3": {"bucket": "wmtp", "region": "ap-northeast-2"},
        },
        "split": "train",
        "max_samples": 100,
        "seed": 42,
    }

    loader = UnifiedDataLoader(config)

    assert loader.path_resolver is not None
    assert loader.split == "train"
    assert loader.max_samples == 100
    assert loader.seed == 42
    print("✓ UnifiedDataLoader 초기화 성공")


def test_dataset_type_detection():
    """데이터셋 타입 자동 감지 테스트"""
    print("\n[데이터셋 타입 감지 테스트]")

    config = {"storage": {"mode": "local"}}
    loader = UnifiedDataLoader(config)

    test_cases = [
        ("dataset/mbpp/train.jsonl", "mbpp"),
        ("s3://wmtp/datasets/mbpp", "mbpp"),
        ("dataset/codecontests_v2", "codecontests"),
        ("benchmarks/contest/test.json", "codecontests"),
        ("dataset/humaneval/test.jsonl", "humaneval"),
        ("custom_data.jsonl", "custom"),
    ]

    for path, expected_type in test_cases:
        detected = loader._detect_dataset_type(path)
        assert (
            detected == expected_type
        ), f"경로 {path}: 기대 {expected_type}, 실제 {detected}"
        print(f"✓ {path} → {detected}")


def test_model_loader_with_mock_s3():
    """S3 모의 객체를 사용한 모델 로더 테스트"""
    print("\n[모델 로더 S3 스트리밍 테스트 (모의)]")

    config = {
        "storage": {"mode": "s3", "s3": {"bucket": "wmtp", "region": "ap-northeast-2"}},
        "devices": {"compute_backend": "cpu"},
    }

    with patch(
        "src.components.loader.unified_model_loader.create_s3_manager"
    ) as mock_s3:
        # S3Manager 모의 객체 설정
        mock_manager = MagicMock()
        mock_manager.stream_model = MagicMock(return_value=MagicMock())
        mock_s3.return_value = mock_manager

        loader = UnifiedModelLoader(config)
        loader.s3_manager = mock_manager

        # S3 경로로 모델 로드 시도
        inputs = {"model_path": "s3://wmtp/models/7b_mtp.pth"}

        # _load_mtp_native 메서드를 모의 처리
        with patch.object(loader, "_load_mtp_native", return_value={"model": "test"}):
            result = loader.run(inputs)

        assert result["model_type"] == "mtp_native"
        assert result["path"] == "s3://wmtp/models/7b_mtp.pth"
        print("✓ S3 모델 스트리밍 로드 시뮬레이션 성공")


def test_data_loader_json_processing():
    """JSON/JSONL 파일 처리 테스트"""
    print("\n[데이터 로더 JSON 처리 테스트]")

    config = {"storage": {"mode": "local"}}
    loader = UnifiedDataLoader(config)

    # 임시 JSONL 파일 생성
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        tmp.write('{"text": "Problem 1", "code": "def solution(): return 1"}\n')
        tmp.write('{"text": "Problem 2", "code": "def solution(): return 2"}\n')
        tmp_path = tmp.name

    try:
        # JSONL 파일 로드
        dataset = loader._load_json_file(Path(tmp_path))

        assert len(dataset) == 2
        assert dataset[0]["text"] == "Problem 1"
        assert dataset[1]["code"] == "def solution(): return 2"
        print(f"✓ JSONL 파일 로드 성공: {len(dataset)}개 항목")

    finally:
        Path(tmp_path).unlink()


def test_dataset_preprocessing():
    """데이터셋 전처리 테스트"""
    print("\n[데이터셋 전처리 테스트]")

    config = {"storage": {"mode": "local"}}
    loader = UnifiedDataLoader(config)

    # MBPP 형식 데이터
    from datasets import Dataset

    mbpp_data = Dataset.from_list(
        [
            {
                "text": "Write a function to add two numbers",
                "code": "def add(a, b): return a + b",
            },
            {
                "text": "Write a function to multiply",
                "code": "def mul(a, b): return a * b",
            },
        ]
    )

    # MBPP 전처리
    processed = loader._preprocess_mbpp(mbpp_data)

    assert "prompt" in processed[0]
    assert "completion" in processed[0]
    assert "Write a function" in processed[0]["prompt"]
    print("✓ MBPP 데이터셋 전처리 성공")


def test_component_factory_integration():
    """ComponentFactory와의 통합 테스트"""
    print("\n[ComponentFactory 통합 테스트]")

    from src.factory.component_factory import ComponentFactory
    from src.settings.config_schema import Config

    # 테스트 설정
    config_dict = {
        "project": "test",
        "seed": 42,
        "storage": {"mode": "local"},
        "paths": {
            "models": {"base": "models/7b_mtp", "rm": "models/rm", "ref": "models/ref"},
            "datasets": {"mbpp": "dataset/mbpp", "contest": "dataset/contest"},
        },
        "mlflow": {
            "experiment": "test",
            "tracking_uri": "./mlruns",
            "registry_uri": "./mlruns",
        },
        "launcher": {"target": "local"},
        "devices": {"compute_backend": "cpu"},
    }

    config = Config(**config_dict)

    # 모델 로더 생성
    model_loader = ComponentFactory.create_model_loader(config)
    assert model_loader is not None
    assert hasattr(model_loader, "run")
    print("✓ UnifiedModelLoader 생성 성공")

    # 데이터 로더 생성
    data_loader = ComponentFactory.create_data_loader("mbpp", config)
    assert data_loader is not None
    assert hasattr(data_loader, "run")
    print("✓ UnifiedDataLoader 생성 성공")


def main():
    """모든 테스트 실행"""
    print("=" * 60)
    print("UnifiedLoader 테스트 시작")
    print("=" * 60)

    test_functions = [
        test_unified_model_loader_init,
        test_model_type_detection,
        test_unified_data_loader_init,
        test_dataset_type_detection,
        test_model_loader_with_mock_s3,
        test_data_loader_json_processing,
        test_dataset_preprocessing,
        test_component_factory_integration,
    ]

    failed = 0
    for test_func in test_functions:
        try:
            test_func()
        except Exception as e:
            print(f"\n✗ {test_func.__name__} 실패: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    if failed == 0:
        print("✅ 모든 UnifiedLoader 테스트 성공!")
    else:
        print(f"⚠️ {failed}개 테스트 실패")
    print("=" * 60)


if __name__ == "__main__":
    main()
