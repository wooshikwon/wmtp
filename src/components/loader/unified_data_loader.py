"""
UnifiedDataLoader: 모든 데이터셋을 처리하는 통합 로더

Phase 2 리팩토링의 데이터셋 통합 구현체로, 기존 4개 데이터셋 로더를 하나로 통합합니다.
PathResolver를 활용하여 로컬/S3 경로를 자동 판별하고,
데이터셋 타입을 자동 감지하여 적절한 로드 방식을 적용합니다.

통합 대상:
- mbpp_loader.py
- codecontests_loader.py
- humaneval_loader.py
- custom_loader.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset, load_from_disk

from src.components.loader.base_loader import DatasetLoader
from src.components.registry import loader_registry
from src.utils.path_resolver import PathResolver
from src.utils.s3 import create_s3_manager


@loader_registry.register(
    "unified-data-loader",
    version="2.0.0",
    description="Unified dataset loader for all dataset types",
)
class UnifiedDataLoader(DatasetLoader):
    """모든 데이터셋을 처리하는 통합 로더

    WMTP Phase 2 리팩토링:
    - 경로 자동 판별: PathResolver로 로컬/S3 자동 구분
    - 데이터셋 타입 자동 감지: 경로/파일명 패턴으로 타입 추론
    - 스트리밍 지원: S3에서 직접 스트리밍으로 데이터 로드
    - 통합 인터페이스: 모든 데이터셋에 동일한 API 제공

    지원하는 데이터셋:
    1. MBPP: Google Python 프로그래밍 문제
    2. CodeContests: DeepMind 알고리즘 경진 문제
    3. HumanEval: OpenAI 함수 구현 평가
    4. Custom: 사용자 정의 JSONL/JSON 데이터셋
    """

    def __init__(self, config: dict[str, Any]):
        """통합 데이터 로더 초기화

        Args:
            config: 환경 설정 딕셔너리
                - storage: 스토리지 설정
                - paths: 경로 설정
        """
        super().__init__(config)

        # PathResolver와 S3Manager 초기화
        self.path_resolver = PathResolver()
        self.s3_manager = create_s3_manager(config)

        # 기본 설정
        self.split = config.get("split", "train")
        self.max_samples = config.get("max_samples")
        self.seed = config.get("seed", 42)
        self.dataset_path = config.get("dataset_path")  # Factory에서 전달된 경로

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """통합 데이터셋 로드 실행

        Args:
            inputs: 입력 딕셔너리
                - dataset_path: 데이터셋 경로 (로컬 또는 S3 URI)
                - dataset_type: (선택) 명시적 데이터셋 타입
                - split: (선택) 데이터 분할 (train/valid/test)
                - max_samples: (선택) 최대 샘플 수

        Returns:
            로드된 데이터셋과 메타데이터를 포함하는 딕셔너리
        """
        dataset_path = inputs.get("dataset_path", self.dataset_path)
        if not dataset_path:
            raise ValueError("dataset_path is required")

        # 경로 해석
        path_type, resolved = self.path_resolver.resolve(dataset_path)

        # 데이터셋 타입 감지
        dataset_type = inputs.get("dataset_type") or self._detect_dataset_type(
            dataset_path
        )

        # 분할 설정
        split = inputs.get("split", self.split)
        max_samples = inputs.get("max_samples", self.max_samples)

        # 경로 타입에 따른 로드
        if path_type == "s3":
            dataset = self._load_from_s3(resolved, dataset_type, split)
        else:
            dataset = self._load_from_local(resolved, dataset_type, split)

        # 샘플 수 제한
        if max_samples and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))

        # 데이터셋별 전처리
        dataset = self._preprocess_dataset(dataset, dataset_type)

        return {
            "dataset": dataset,
            "dataset_type": dataset_type,
            "split": split,
            "size": len(dataset),
            "path": dataset_path,
            "loader": self.__class__.__name__,
        }

    def _detect_dataset_type(self, path: str) -> str:
        """경로에서 데이터셋 타입 자동 감지

        Args:
            path: 데이터셋 경로

        Returns:
            감지된 데이터셋 타입
        """
        path_lower = path.lower()

        if "mbpp" in path_lower:
            return "mbpp"
        elif "codecontests" in path_lower or "contest" in path_lower:
            return "codecontests"
        elif "humaneval" in path_lower:
            return "humaneval"
        else:
            return "custom"

    def _load_from_s3(self, s3_path: str, dataset_type: str, split: str) -> Dataset:
        """S3에서 데이터셋 스트리밍 로드

        Args:
            s3_path: S3 URI
            dataset_type: 데이터셋 타입
            split: 데이터 분할

        Returns:
            로드된 데이터셋
        """
        if not self.s3_manager:
            raise RuntimeError("S3 manager not available")

        # S3 경로에서 버킷과 키 추출
        bucket, key = self.path_resolver.extract_bucket_and_key(s3_path)

        # 분할별 파일 경로 구성
        if dataset_type in ["mbpp", "codecontests", "humaneval"]:
            # 표준 데이터셋은 split별 파일 구조
            split_key = f"{key}/{split}.jsonl"
        else:
            # Custom 데이터셋은 단일 파일
            split_key = key

        # S3에서 스트리밍으로 데이터 로드
        data_list = []
        for item in self.s3_manager.stream_dataset(split_key):
            data_list.append(item)

        # HuggingFace Dataset으로 변환
        return Dataset.from_list(data_list)

    def _load_from_local(
        self, local_path: str, dataset_type: str, split: str
    ) -> Dataset:
        """로컬 파일에서 데이터셋 로드

        Args:
            local_path: 로컬 경로
            dataset_type: 데이터셋 타입
            split: 데이터 분할

        Returns:
            로드된 데이터셋
        """
        path = Path(local_path)

        # HuggingFace 데이터셋 디렉토리 구조 확인
        if path.is_dir() and (path / "dataset_info.json").exists():
            # HuggingFace 형식으로 로드
            dataset = load_from_disk(str(path))
            if split in dataset:
                return dataset[split]
            return dataset

        # JSONL/JSON 파일 로드
        if path.is_file():
            return self._load_json_file(path)

        # 디렉토리에서 split별 파일 찾기
        split_file = path / f"{split}.jsonl"
        if not split_file.exists():
            split_file = path / f"{split}.json"

        if split_file.exists():
            return self._load_json_file(split_file)

        # HuggingFace Hub에서 시도
        if dataset_type in ["mbpp", "humaneval"]:
            hub_id = self._get_huggingface_id(dataset_type)
            if hub_id:
                dataset = load_dataset(hub_id, split=split)
                return dataset

        raise FileNotFoundError(f"Dataset not found at {local_path}")

    def _load_json_file(self, file_path: Path) -> Dataset:
        """JSON/JSONL 파일 로드

        Args:
            file_path: 파일 경로

        Returns:
            로드된 데이터셋
        """
        data_list = []

        if file_path.suffix == ".jsonl":
            # JSONL 형식
            with open(file_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data_list.append(json.loads(line))
        else:
            # JSON 형식
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    data_list = data
                else:
                    data_list = [data]

        return Dataset.from_list(data_list)

    def _get_huggingface_id(self, dataset_type: str) -> str | None:
        """데이터셋 타입에 따른 HuggingFace ID 반환

        Args:
            dataset_type: 데이터셋 타입

        Returns:
            HuggingFace 데이터셋 ID 또는 None
        """
        hub_ids = {
            "mbpp": "google-research-datasets/mbpp",
            "humaneval": "openai_humaneval",
            "codecontests": "deepmind/code_contests",
        }
        return hub_ids.get(dataset_type)

    def _preprocess_dataset(self, dataset: Dataset, dataset_type: str) -> Dataset:
        """데이터셋별 전처리

        Args:
            dataset: 원본 데이터셋
            dataset_type: 데이터셋 타입

        Returns:
            전처리된 데이터셋
        """
        if dataset_type == "mbpp":
            return self._preprocess_mbpp(dataset)
        elif dataset_type == "codecontests":
            return self._preprocess_codecontests(dataset)
        elif dataset_type == "humaneval":
            return self._preprocess_humaneval(dataset)
        else:
            return dataset  # Custom은 전처리 없음

    def _preprocess_mbpp(self, dataset: Dataset) -> Dataset:
        """MBPP 데이터셋 전처리

        Args:
            dataset: 원본 MBPP 데이터셋

        Returns:
            전처리된 데이터셋
        """

        # MBPP 특화 전처리
        def process_example(example):
            # 문제와 해답을 자연어 프롬프트로 결합
            if "text" in example and "code" in example:
                example["prompt"] = f"# {example['text']}\n\ndef solution():\n"
                example["completion"] = example["code"]
            return example

        return dataset.map(process_example)

    def _preprocess_codecontests(self, dataset: Dataset) -> Dataset:
        """CodeContests 데이터셋 전처리

        Args:
            dataset: 원본 CodeContests 데이터셋

        Returns:
            전처리된 데이터셋
        """

        # CodeContests 특화 전처리
        def process_example(example):
            if "description" in example:
                example["prompt"] = f"# Problem: {example['description']}\n\n"
                if "python_solutions" in example:
                    # Python 솔루션만 추출
                    solutions = example.get("python_solutions", [])
                    if solutions:
                        example["completion"] = solutions[0]
            return example

        return dataset.map(process_example)

    def _preprocess_humaneval(self, dataset: Dataset) -> Dataset:
        """HumanEval 데이터셋 전처리

        Args:
            dataset: 원본 HumanEval 데이터셋

        Returns:
            전처리된 데이터셋
        """

        # HumanEval 특화 전처리
        def process_example(example):
            if "prompt" in example and "canonical_solution" in example:
                example["completion"] = example["canonical_solution"]
            return example

        return dataset.map(process_example)

    def preprocess(self, data: Any, **kwargs) -> Any:
        """DatasetLoader 인터페이스 구현

        Args:
            data: 원본 데이터
            **kwargs: 전처리 파라미터

        Returns:
            전처리된 데이터
        """
        dataset_type = kwargs.get("dataset_type", "custom")
        return self._preprocess_dataset(data, dataset_type)

    def load(self, path: str, **kwargs) -> Any:
        """BaseLoader의 추상 메서드 구현

        Args:
            path: 데이터셋 경로
            **kwargs: 추가 파라미터

        Returns:
            로드된 데이터셋
        """
        inputs = {"dataset_path": path, **kwargs}
        result = self.run(inputs)
        return result["dataset"]
