"""
DataLoader V2: 순차적이고 직관적인 데이터 로더

4단계 순차 프로세스:
1. 데이터셋 소스 확인
2. 메타데이터 로드 및 가용성 확인
3. 데이터 다운로드/로드
4. 포맷 정규화 및 전처리
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset, load_from_disk
from src.components.loader.base_loader import DatasetLoader
from src.components.registry import loader_registry
from src.utils import get_console_output
from src.utils.path_resolver import PathResolver
from src.utils.s3 import create_s3_manager


@loader_registry.register(
    "unified-data-loader",
    version="3.0.0",
    description="Sequential and intuitive data loader",
)
class DataLoader(DatasetLoader):
    """4단계 프로세스로 데이터셋을 로드합니다.
    각 단계는 독립적이며 순차적으로 실행됩니다.
    """

    def __init__(self, config: dict[str, Any]):
        """초기화: 필수 설정만 저장"""
        super().__init__(config)

        # 경로와 타입 설정
        self.dataset_path = config.get("dataset_path")
        self.dataset_type = config.get("dataset_type")
        self.split = config.get("split", "train")

        # 유틸리티
        self.path_resolver = PathResolver()
        self.s3_manager = create_s3_manager(config)

        # 옵션
        self.max_samples = config.get("max_samples")
        self.seed = config.get("seed", 42)

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """메인 실행 메서드"""
        # Factory에서 이미 설정된 경로 사용 (오버라이드 가능)
        dataset_path = inputs.get("dataset_path", self.dataset_path)
        if not dataset_path:
            raise ValueError("dataset_path is required")

        # 순차적 4단계 실행
        dataset = self.load_dataset_sequential(dataset_path)

        return {
            "dataset": dataset,
            "dataset_type": self.dataset_type,
            "split": self.split,
            "size": len(dataset),
            "path": dataset_path,
            "loader": self.__class__.__name__,
        }

    def load(self, path: str, **kwargs) -> Any:
        """BaseLoader abstract method 구현"""
        return self.load_dataset_sequential(path)

    def preprocess(self, data: Any, **kwargs) -> Any:
        """BaseLoader abstract method 구현"""
        # 이미 step4에서 전처리를 수행하므로 그대로 반환
        return data

    def load_dataset_sequential(self, dataset_path: str) -> Dataset:
        """
        데이터셋 로딩의 전체 흐름을 관리하는 메인 메서드
        4단계를 순차적으로 실행
        """
        # Step 1: 데이터셋 소스 확인
        dataset_type, path_type, resolved_path = self.step1_identify_source(
            dataset_path
        )

        # Step 2: 메타데이터 확인
        metadata = self.step2_check_metadata(resolved_path, dataset_type, path_type)

        # Step 3: 데이터 로드
        raw_dataset = self.step3_load_data(resolved_path, path_type)

        # Step 4: 포맷 정규화 및 전처리
        dataset = self.step4_normalize_and_preprocess(
            raw_dataset, dataset_type, metadata
        )

        return dataset

    # ============= STEP 1: 소스 확인 =============
    def step1_identify_source(self, dataset_path: str) -> tuple[str, str, str]:
        """Step 1: 데이터셋 소스와 경로 타입 확인"""
        console_out = get_console_output()

        # 경로 타입 해석 (s3 또는 local)
        path_type, resolved_path = self.path_resolver.resolve(dataset_path)

        # 데이터셋 타입 결정 (Factory에서 전달 또는 자동 감지)
        dataset_type = self.dataset_type or self._detect_dataset_type(dataset_path)

        console_out.detail(f"데이터셋 소스: {dataset_type} ({path_type})")
        return dataset_type, path_type, resolved_path

    # ============= STEP 2: 메타데이터 확인 =============
    def step2_check_metadata(
        self, resolved_path: str, dataset_type: str, path_type: str
    ) -> dict:
        """Step 2: 데이터셋 메타데이터 및 가용성 확인"""
        console_out = get_console_output()

        metadata = {}

        # MBPP나 CodeContests 같은 표준 데이터셋은 메타데이터 정의
        if dataset_type == "mbpp":
            metadata = {
                "expected_fields": ["task_id", "text", "code", "test_list"],
                "format": "jsonl",
                "split_available": ["train", "validation", "test"],
            }
        elif dataset_type == "codecontests":
            metadata = {
                "expected_fields": ["name", "description", "solutions"],
                "format": "parquet",
                "split_available": ["train", "valid", "test"],
            }
        elif dataset_type == "humaneval":
            metadata = {
                "expected_fields": ["task_id", "prompt", "canonical_solution", "test"],
                "format": "jsonl",
                "split_available": ["test"],
            }
        else:
            # Custom 데이터셋은 메타데이터 파일 확인
            metadata = self._load_custom_metadata(resolved_path, path_type)

        console_out.detail(f"데이터 포맷: {metadata.get('format', 'unknown')}")
        return metadata

    # ============= STEP 3: 데이터 로드 =============
    def step3_load_data(self, resolved_path: str, path_type: str) -> Dataset:
        """Step 3: 실제 데이터 로드"""
        console_out = get_console_output()

        if path_type == "s3":
            dataset = self._load_from_s3(resolved_path)
        else:
            dataset = self._load_from_local(resolved_path)

        # 샘플 수 제한
        if self.max_samples and len(dataset) > self.max_samples:
            dataset = dataset.select(range(self.max_samples))
            console_out.detail(f"샘플 제한: {self.max_samples}개")

        return dataset

    # ============= STEP 4: 정규화 및 전처리 =============
    def step4_normalize_and_preprocess(
        self,
        raw_dataset: Dataset,
        dataset_type: str,
        metadata: dict,
    ) -> Dataset:
        """Step 4: 데이터 포맷 정규화 및 전처리"""
        console_out = get_console_output()

        # 데이터셋별 정규화
        if dataset_type == "mbpp":
            dataset = self._normalize_mbpp(raw_dataset)
        elif dataset_type == "codecontests":
            dataset = self._normalize_codecontests(raw_dataset)
        elif dataset_type == "humaneval":
            dataset = self._normalize_humaneval(raw_dataset)
        else:
            # Custom은 그대로 사용
            dataset = raw_dataset

        # 공통 필드 확인 및 추가
        dataset = self._ensure_common_fields(dataset)

        console_out.detail(f"정규화 완료: {len(dataset)} samples")
        return dataset

    # ============= 데이터셋별 정규화 메서드 =============
    def _normalize_mbpp(self, dataset: Dataset) -> Dataset:
        """MBPP 데이터셋 정규화"""

        def normalize_example(example):
            return {
                "instruction": example.get("text", ""),
                "input": "",  # MBPP는 별도 입력 없음
                "output": example.get("code", ""),
                "task_id": example.get("task_id", ""),
                "test_cases": example.get("test_list", []),
            }

        return dataset.map(normalize_example)

    def _normalize_codecontests(self, dataset: Dataset) -> Dataset:
        """CodeContests 데이터셋 정규화"""

        def normalize_example(example):
            # solutions가 리스트인 경우 첫 번째 선택
            solutions = example.get("solutions", {})
            if isinstance(solutions, dict):
                python_sols = solutions.get("python", [])
                code = python_sols[0] if python_sols else ""
            else:
                code = ""

            return {
                "instruction": example.get("description", ""),
                "input": example.get("public_tests", {}).get("input", [""])[0],
                "output": code,
                "task_id": example.get("name", ""),
                "test_cases": example.get("public_tests", {}),
            }

        return dataset.map(normalize_example)

    def _normalize_humaneval(self, dataset: Dataset) -> Dataset:
        """HumanEval 데이터셋 정규화"""

        def normalize_example(example):
            return {
                "instruction": example.get("prompt", ""),
                "input": "",  # HumanEval은 별도 입력 없음
                "output": example.get("canonical_solution", ""),
                "task_id": example.get("task_id", ""),
                "test_cases": example.get("test", ""),
            }

        return dataset.map(normalize_example)

    def _ensure_common_fields(self, dataset: Dataset) -> Dataset:
        """모든 데이터셋에 공통 필드 보장"""
        required_fields = ["instruction", "input", "output"]

        for field in required_fields:
            if field not in dataset.column_names:
                dataset = dataset.add_column(field, [""] * len(dataset))

        return dataset

    # ============= 유틸리티 메서드 =============
    def _detect_dataset_type(self, path: str) -> str:
        """경로에서 데이터셋 타입 자동 감지"""
        path_lower = path.lower()

        if "mbpp" in path_lower:
            return "mbpp"
        elif "codecontests" in path_lower or "contest" in path_lower:
            return "codecontests"
        elif "humaneval" in path_lower:
            return "humaneval"
        else:
            return "custom"

    def _load_from_local(self, path: str) -> Dataset:
        """로컬에서 데이터셋 로드"""
        path_obj = Path(path)

        # HuggingFace datasets 형식
        if path_obj.is_dir() and (path_obj / "dataset_info.json").exists():
            return load_from_disk(str(path_obj))[self.split]

        # Parquet 파일
        if path_obj.suffix == ".parquet":
            return load_dataset("parquet", data_files=str(path_obj))[self.split]

        # JSONL/JSON 파일
        if path_obj.suffix in [".jsonl", ".json"]:
            return load_dataset("json", data_files=str(path_obj))["train"]

        # 디렉토리인 경우 split별 파일 찾기
        if path_obj.is_dir():
            split_file = path_obj / f"{self.split}.jsonl"
            if split_file.exists():
                return load_dataset("json", data_files=str(split_file))["train"]

        raise ValueError(f"Cannot load dataset from {path}")

    def _load_from_s3(self, s3_path: str) -> Dataset:
        """S3에서 데이터셋 로드"""
        if not self.s3_manager:
            raise RuntimeError("S3 manager not available")

        # S3에서 스트리밍으로 로드
        bucket, key = self.path_resolver.extract_bucket_and_key(s3_path)

        # 파일 형식에 따라 로드
        if key.endswith(".parquet"):
            # Parquet는 직접 스트리밍 가능
            return load_dataset(
                "parquet",
                data_files=f"s3://{bucket}/{key}",
                split=self.split,
            )
        else:
            # JSONL/JSON은 다운로드 후 로드
            stream = self.s3_manager.stream_dataset(key)
            data = json.loads(stream.read())
            return Dataset.from_list(data)

    def _load_custom_metadata(self, path: str, path_type: str) -> dict:
        """Custom 데이터셋의 메타데이터 로드"""
        metadata = {
            "format": "custom",
            "expected_fields": [],
            "split_available": ["train"],
        }

        # metadata.json 파일이 있으면 로드
        if path_type == "local":
            metadata_file = Path(path) / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata.update(json.load(f))

        return metadata
