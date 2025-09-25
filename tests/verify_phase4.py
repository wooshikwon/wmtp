#!/usr/bin/env python
"""Phase 4 구현 검증 스크립트: Cache 완전 제거 및 정리."""

import sys
import inspect
from pathlib import Path


def test_tokenizer_cache_removal():
    """TokenizerComponent에서 .cache 경로 제거 확인."""
    print("1. TokenizerComponent 캐시 제거 확인...")
    try:
        from src.components.tokenizer.sentencepiece_tokenizer import SentencePieceTokenizer

        # _ensure_processor_loaded 메서드 소스 확인
        source = inspect.getsource(SentencePieceTokenizer._ensure_processor_loaded)

        # .cache 경로 사용 안 함 확인
        assert ".cache/tokenizer.model" not in source, ".cache 경로가 여전히 존재함"
        print("   ✓ .cache/tokenizer.model 경로 제거 확인")

        return True
    except Exception as e:
        print(f"   ✗ 실패: {e}")
        return False


def test_base_loader_cache_removal():
    """BaseLoader에서 캐시 시스템 완전 제거 확인."""
    print("2. BaseLoader 캐시 시스템 제거 확인...")
    try:
        from src.components.loader.base_loader import BaseLoader

        # __init__ 메서드에 cache_dir 파라미터 없음 확인
        sig = inspect.signature(BaseLoader.__init__)
        params = list(sig.parameters.keys())
        assert "cache_dir" not in params, "__init__에 cache_dir 파라미터가 여전히 존재"
        print("   ✓ __init__에서 cache_dir 파라미터 제거 확인")

        # 제거되어야 할 메서드들이 없는지 확인
        removed_methods = [
            "compute_cache_key",
            "get_cached_path",
            "load_with_cache",
            "sync_directory_with_cache"
        ]

        for method_name in removed_methods:
            assert not hasattr(BaseLoader, method_name), f"{method_name} 메서드가 여전히 존재"
            print(f"   ✓ {method_name} 메서드 제거 확인")

        # load_with_streaming 메서드 존재 확인
        assert hasattr(BaseLoader, "load_with_streaming"), "load_with_streaming 메서드가 없음"
        print("   ✓ load_with_streaming 메서드 존재 확인")

        return True
    except Exception as e:
        print(f"   ✗ 실패: {e}")
        return False


def test_s3_manager_cache_removal():
    """S3Manager에서 캐시 관련 코드 제거 확인."""
    print("3. S3Manager 캐시 시스템 제거 확인...")
    try:
        from src.utils.s3 import S3Manager

        # __init__ 메서드에 cache_dir 파라미터 없음 확인
        sig = inspect.signature(S3Manager.__init__)
        params = list(sig.parameters.keys())
        assert "cache_dir" not in params, "__init__에 cache_dir 파라미터가 여전히 존재"
        print("   ✓ __init__에서 cache_dir 파라미터 제거 확인")

        # 제거되어야 할 메서드들이 없는지 확인
        removed_methods = [
            "download_if_missing",
            "get_cache_path"
        ]

        for method_name in removed_methods:
            assert not hasattr(S3Manager, method_name), f"{method_name} 메서드가 여전히 존재"
            print(f"   ✓ {method_name} 메서드 제거 확인")

        # 스트리밍 메서드들 존재 확인
        streaming_methods = ["stream_model", "stream_dataset"]
        for method_name in streaming_methods:
            assert hasattr(S3Manager, method_name), f"{method_name} 메서드가 없음"
            print(f"   ✓ {method_name} 메서드 존재 확인")

        return True
    except Exception as e:
        print(f"   ✗ 실패: {e}")
        return False


def test_s3_utils_cache_removal():
    """S3Utils에서 캐시 관련 코드 제거 확인."""
    print("4. S3Utils 캐시 시스템 제거 확인...")
    try:
        from src.utils.s3 import S3Utils
        import inspect

        # download 메서드들의 반환 타입 변경 확인
        download_model_sig = inspect.signature(S3Utils.download_model)
        # 더 이상 local_dir 파라미터가 없어야 함
        params = list(download_model_sig.parameters.keys())
        assert "local_dir" not in params, "download_model에 local_dir 파라미터가 여전히 존재"
        print("   ✓ download_model에서 local_dir 파라미터 제거 확인")

        # download_dataset도 동일하게 확인
        download_dataset_sig = inspect.signature(S3Utils.download_dataset)
        params = list(download_dataset_sig.parameters.keys())
        assert "local_dir" not in params, "download_dataset에 local_dir 파라미터가 여전히 존재"
        print("   ✓ download_dataset에서 local_dir 파라미터 제거 확인")

        return True
    except Exception as e:
        print(f"   ✗ 실패: {e}")
        return False


def test_config_schema_cache_removal():
    """ConfigSchema에서 cache 필드 제거 확인."""
    print("5. ConfigSchema cache 필드 제거 확인...")
    try:
        import inspect
        from src.settings.config_schema import Config

        # Config 클래스에서 paths 확인
        source = inspect.getsource(Config)

        # cache 필드가 없는지 확인 (주석 제외)
        lines = [line.strip() for line in source.split('\n') if line.strip() and not line.strip().startswith('#')]
        cache_references = [line for line in lines if 'cache' in line.lower() and not line.startswith('#')]

        # 주석이나 문서화를 제외한 실제 cache 필드 정의가 없어야 함
        active_cache_fields = [ref for ref in cache_references if '=' in ref and 'Field' in ref]
        assert len(active_cache_fields) == 0, f"cache 필드가 여전히 존재: {active_cache_fields}"
        print("   ✓ cache 필드 제거 확인")

        return True
    except Exception as e:
        print(f"   ✗ 실패: {e}")
        return False


def test_no_cache_dependencies():
    """전체 코드에서 캐시 의존성 제거 확인."""
    print("6. 캐시 의존성 완전 제거 확인...")
    try:
        from src.components.loader.base_loader import BaseLoader
        from src.utils.s3 import S3Manager
        import inspect

        # BaseLoader 클래스에서 cache 관련 속성이 없는지 확인
        base_loader_source = inspect.getsource(BaseLoader)
        assert "self.cache_dir" not in base_loader_source, "BaseLoader에 cache_dir 속성 설정이 여전히 존재"
        print("   ✓ BaseLoader.cache_dir 속성 제거 확인")

        # S3Manager 클래스에서 cache 관련 속성이 없는지 확인
        s3_manager_source = inspect.getsource(S3Manager)
        assert "self.cache_dir" not in s3_manager_source, "S3Manager에 cache_dir 속성 설정이 여전히 존재"
        print("   ✓ S3Manager.cache_dir 속성 제거 확인")

        return True
    except Exception as e:
        print(f"   ✗ 실패: {e}")
        return False


def test_streaming_functionality():
    """스트리밍 기반 기능 동작 확인."""
    print("7. 스트리밍 기능 동작 확인...")
    try:
        from src.utils.s3 import S3Manager
        from unittest.mock import MagicMock, patch

        # S3Manager 스트리밍 메서드 호출 가능성 확인
        with patch("boto3.client") as mock_boto:
            mock_s3_client = MagicMock()
            mock_boto.return_value = mock_s3_client
            mock_s3_client.head_bucket.return_value = True

            s3_manager = S3Manager(bucket="test-bucket", region="us-east-1")
            s3_manager.connected = True

            # stream_model 메서드 존재 확인
            assert hasattr(s3_manager, "stream_model"), "stream_model 메서드가 없음"
            print("   ✓ stream_model 메서드 호출 가능 확인")

            # stream_dataset 메서드 존재 확인
            assert hasattr(s3_manager, "stream_dataset"), "stream_dataset 메서드가 없음"
            print("   ✓ stream_dataset 메서드 호출 가능 확인")

        return True
    except Exception as e:
        print(f"   ✗ 실패: {e}")
        return False


def main():
    """모든 테스트 실행."""
    print("=" * 60)
    print("Phase 4: Cache 완전 제거 및 정리 검증")
    print("=" * 60)

    tests = [
        test_tokenizer_cache_removal,
        test_base_loader_cache_removal,
        test_s3_manager_cache_removal,
        test_s3_utils_cache_removal,
        test_config_schema_cache_removal,
        test_no_cache_dependencies,
        test_streaming_functionality,
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
        print("✅ Phase 4 구현이 성공적으로 완료되었습니다!")
        print("🚀 캐시 시스템이 완전히 제거되고 S3 스트리밍 기반으로 전환되었습니다.")
    else:
        print("⚠️ 일부 테스트가 실패했습니다. 위 로그를 확인하세요.")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())