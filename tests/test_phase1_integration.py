"""
Phase 1 통합 테스트

통합 경로 시스템의 전체 흐름을 검증합니다.
- PathResolver 동작
- S3Manager 스트리밍 메서드
- Config 스키마 변경
"""

import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.settings.config_schema import Config
from src.utils.path_resolver import PathResolver
from src.utils.s3 import S3Manager, create_s3_manager


def test_config_schema():
    """Config 스키마 변경 테스트"""
    print("\n[Config 스키마 테스트]")

    # 통합 설정 테스트
    config_dict = {
        "project": "mtp_ft",
        "seed": 42,
        "storage": {
            "mode": "auto",
            "s3": {"bucket": "wmtp", "region": "ap-northeast-2", "prefix": ""},
        },
        "paths": {
            "models": {
                "base": "s3://wmtp/models/7b_1t_4",
                "rm": "models/Llama_3_8B_RM",
                "ref": "s3://wmtp/models/Sheared-LLaMA-2.7B",
            },
            "datasets": {
                "mbpp": "dataset/mbpp",
                "contest": "s3://wmtp/datasets/contest",
            },
            # cache 필드 없음
        },
        "mlflow": {
            "experiment": "mtp/wmtp",
            "tracking_uri": "./mlflow_runs",
            "registry_uri": "./mlflow_runs",
        },
        "launcher": {"target": "local", "resources": {"gpus": 4, "gpu_type": "A100"}},
        "devices": {"compute_backend": "auto", "mixed_precision": "bf16"},
    }

    # Config 인스턴스 생성
    try:
        config = Config(**config_dict)
        print("✓ Config 생성 성공")
        print(f"  - Storage mode: {config.storage.mode}")
        print(f"  - Base model: {config.paths.models.base}")
        print(f"  - MBPP dataset: {config.paths.datasets.mbpp}")

        # cache 필드가 없는지 확인
        assert not hasattr(config.paths, "cache"), "cache 필드가 제거되어야 함"
        print("✓ cache 필드 제거 확인")

    except Exception as e:
        print(f"✗ Config 생성 실패: {e}")
        raise


def test_path_resolver_with_config():
    """Config 경로를 PathResolver로 처리하는 테스트"""
    print("\n[PathResolver + Config 통합 테스트]")

    resolver = PathResolver()

    # Config에서 사용하는 경로들 테스트
    test_paths = {
        "s3://wmtp/models/7b_1t_4": ("s3", "s3://wmtp/models/7b_1t_4"),
        "models/Llama_3_8B_RM": ("local", None),  # 절대 경로로 변환됨
        "dataset/mbpp": ("local", None),
        "s3://wmtp/datasets/contest": ("s3", "s3://wmtp/datasets/contest"),
    }

    for path, (expected_type, expected_resolved) in test_paths.items():
        path_type, resolved = resolver.resolve(path)
        assert path_type == expected_type, f"경로 타입 불일치: {path}"

        if expected_resolved:
            assert resolved == expected_resolved, f"해석된 경로 불일치: {path}"

        print(f"✓ {path_type:5} 경로 처리: {path[:30]}...")


def test_s3_manager_creation():
    """S3Manager 생성 테스트"""
    print("\n[S3Manager 생성 테스트]")

    # auto 모드 config
    config_auto = {
        "storage": {
            "mode": "auto",
            "s3": {"bucket": "wmtp", "region": "ap-northeast-2"},
        }
    }

    s3_manager = create_s3_manager(config_auto)
    assert s3_manager is not None, "auto 모드에서 S3Manager가 생성되어야 함"
    assert s3_manager.bucket == "wmtp"
    print("✓ auto 모드에서 S3Manager 생성 성공")

    # local 모드 config
    config_local = {"storage": {"mode": "local"}}

    s3_manager = create_s3_manager(config_local)
    assert s3_manager is None, "local 모드에서 S3Manager는 None이어야 함"
    print("✓ local 모드에서 S3Manager None 반환 확인")


def test_s3_streaming_methods():
    """S3Manager 스트리밍 메서드 테스트 (모의)"""
    print("\n[S3Manager 스트리밍 메서드 테스트]")

    # S3Manager가 stream_model과 stream_dataset 메서드를 가지고 있는지 확인
    s3_manager = S3Manager(
        bucket="test-bucket", region="ap-northeast-2", cache_dir="/tmp/.test_cache"
    )

    # 메서드 존재 확인
    assert hasattr(s3_manager, "stream_model"), "stream_model 메서드가 없음"
    assert hasattr(s3_manager, "stream_dataset"), "stream_dataset 메서드가 없음"
    assert hasattr(s3_manager, "download_to_bytes"), "download_to_bytes 메서드가 없음"

    print("✓ stream_model 메서드 확인")
    print("✓ stream_dataset 메서드 확인")
    print("✓ download_to_bytes 메서드 확인")

    # 실제 S3 연결 없이도 메서드 시그니처 확인
    import inspect

    # stream_model 시그니처 확인
    sig = inspect.signature(s3_manager.stream_model)
    assert "s3_key" in sig.parameters, "stream_model은 s3_key 파라미터를 받아야 함"
    print("✓ stream_model 시그니처 확인")

    # stream_dataset 시그니처 확인
    sig = inspect.signature(s3_manager.stream_dataset)
    assert "s3_key" in sig.parameters, "stream_dataset은 s3_key 파라미터를 받아야 함"
    print("✓ stream_dataset 시그니처 확인")


def test_integration_flow():
    """전체 통합 플로우 테스트"""
    print("\n[전체 통합 플로우 테스트]")

    # 1. Config 로드 시뮬레이션
    config_dict = {
        "storage": {
            "mode": "auto",
            "s3": {"bucket": "wmtp", "region": "ap-northeast-2"},
        },
        "paths": {
            "models": {"base": "s3://wmtp/models/7b_mtp"},
            "datasets": {"mbpp": "dataset/mbpp"},
        },
    }

    # 2. PathResolver 생성
    resolver = PathResolver()

    # 3. 모델 경로 해석
    model_path = config_dict["paths"]["models"]["base"]
    path_type, resolved = resolver.resolve(model_path)

    if path_type == "s3":
        print(f"✓ S3 모델 경로 감지: {resolved}")
        # S3Manager로 스트리밍 (실제 연결은 하지 않음)
        bucket, key = resolver.extract_bucket_and_key(resolved)
        print(f"  → S3에서 스트리밍: bucket={bucket}, key={key}")
    else:
        print(f"✓ 로컬 모델 경로 감지: {resolved}")
        print(f"  → 로컬 파일 직접 로드: {resolved}")

    # 4. 데이터셋 경로 해석
    dataset_path = config_dict["paths"]["datasets"]["mbpp"]
    path_type, resolved = resolver.resolve(dataset_path)

    if path_type == "s3":
        print(f"✓ S3 데이터셋 경로 감지: {resolved}")
    else:
        print(f"✓ 로컬 데이터셋 경로 감지: {resolved}")
        print(f"  → 로컬 파일 직접 로드: {resolved}")


def main():
    """모든 통합 테스트 실행"""
    print("=" * 60)
    print("Phase 1 통합 테스트 시작")
    print("=" * 60)

    test_functions = [
        test_config_schema,
        test_path_resolver_with_config,
        test_s3_manager_creation,
        test_s3_streaming_methods,
        test_integration_flow,
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
        print("✅ 모든 Phase 1 통합 테스트 성공!")
    else:
        print(f"⚠️ {failed}개 테스트 실패")
    print("=" * 60)


if __name__ == "__main__":
    main()
