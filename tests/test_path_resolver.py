"""
PathResolver 테스트 코드

Phase 1 통합 경로 시스템의 핵심 기능을 검증합니다.
"""

import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.path_resolver import PathResolver, create_path_resolver


def test_path_resolver_local_paths():
    """로컬 경로 해석 테스트"""
    resolver = PathResolver()

    # 상대 경로 테스트
    path_type, resolved = resolver.resolve("./models/test.pth")
    assert path_type == "local"
    assert Path(resolved).is_absolute()
    print(f"✓ 상대 경로 해석: {resolved}")

    # 절대 경로 테스트
    abs_path = "/tmp/models/test.pth"
    path_type, resolved = resolver.resolve(abs_path)
    assert path_type == "local"
    assert resolved == abs_path
    print(f"✓ 절대 경로 해석: {resolved}")


def test_path_resolver_s3_paths():
    """S3 경로 해석 테스트"""
    resolver = PathResolver()

    # S3 URI 테스트
    s3_uri = "s3://wmtp/models/7b_mtp.pth"
    path_type, resolved = resolver.resolve(s3_uri)
    assert path_type == "s3"
    assert resolved == s3_uri
    print(f"✓ S3 URI 해석: {resolved}")

    # S3 버킷과 키 추출 테스트
    bucket, key = resolver.extract_bucket_and_key(s3_uri)
    assert bucket == "wmtp"
    assert key == "models/7b_mtp.pth"
    print(f"✓ S3 버킷/키 추출: bucket={bucket}, key={key}")


def test_path_category_detection():
    """경로 카테고리 감지 테스트"""
    resolver = PathResolver()

    # 모델 경로 테스트
    model_paths = [
        "model.pth",
        "checkpoint.pt",
        "weights.safetensors",
        "s3://wmtp/models/7b.bin",
    ]

    for path in model_paths:
        assert resolver.is_model_path(path), f"모델 경로로 인식 실패: {path}"
        assert resolver.get_path_category(path) == "model"
    print(f"✓ 모델 경로 감지: {len(model_paths)}개 테스트 성공")

    # 데이터셋 경로 테스트
    dataset_paths = [
        "dataset/mbpp/test.json",
        "data/train.jsonl",
        "benchmark/eval.csv",
        "s3://wmtp/datasets/codecontests.parquet",
    ]

    for path in dataset_paths:
        assert resolver.is_dataset_path(path), f"데이터셋 경로로 인식 실패: {path}"
        assert resolver.get_path_category(path) == "dataset"
    print(f"✓ 데이터셋 경로 감지: {len(dataset_paths)}개 테스트 성공")


def test_path_normalization():
    """경로 정규화 테스트"""
    resolver = PathResolver()

    # 상대 경로를 S3 URI로 변환
    normalized = resolver.normalize_s3_path("models/7b.pth")
    assert normalized == "s3://wmtp/models/7b.pth"
    print(f"✓ 상대 경로 S3 정규화: {normalized}")

    # 이미 S3 URI인 경우
    s3_uri = "s3://custom-bucket/path/file.pth"
    normalized = resolver.normalize_s3_path(s3_uri)
    assert normalized == s3_uri
    print(f"✓ S3 URI 유지: {normalized}")


def test_error_handling():
    """에러 처리 테스트"""
    resolver = PathResolver()

    # 빈 경로
    try:
        resolver.resolve("")
        assert False, "빈 경로에서 에러가 발생해야 함"
    except ValueError as e:
        print(f"✓ 빈 경로 에러 처리: {e}")

    # 잘못된 S3 URI
    try:
        resolver.extract_bucket_and_key("not-s3-uri")
        assert False, "잘못된 S3 URI에서 에러가 발생해야 함"
    except ValueError as e:
        print(f"✓ 잘못된 S3 URI 에러 처리: {e}")


def test_create_helper():
    """헬퍼 함수 테스트"""
    resolver = create_path_resolver()
    assert isinstance(resolver, PathResolver)
    print("✓ PathResolver 생성 헬퍼 함수 동작")


def main():
    """모든 테스트 실행"""
    print("=" * 50)
    print("PathResolver 테스트 시작")
    print("=" * 50)

    test_functions = [
        test_path_resolver_local_paths,
        test_path_resolver_s3_paths,
        test_path_category_detection,
        test_path_normalization,
        test_error_handling,
        test_create_helper,
    ]

    for test_func in test_functions:
        print(f"\n[{test_func.__name__}]")
        try:
            test_func()
        except Exception as e:
            print(f"✗ 테스트 실패: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 50)
    print("모든 PathResolver 테스트 완료!")
    print("=" * 50)


if __name__ == "__main__":
    main()
