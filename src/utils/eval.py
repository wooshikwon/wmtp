"""
WMTP 평가 프로토콜: 코드 생성 벤치마크 평가 기반 시스템

WMTP 연구 맥락:
WMTP의 핵심 가설인 "토큰 가중치가 성능을 향상시킨다"를 검증하기 위한
평가 시스템입니다. Chen et al. (2021)의 pass@k 메트릭을 구현하여
코드 생성 능력을 정량적으로 측정합니다.

핵심 기능:
- pass@k 계산: unbiased estimator로 정확한 성공률 측정
- 코드 추출: 모델 출력에서 실행 가능한 코드 파싱
- 배치 생성: 효율적인 GPU 활용으로 빠른 평가
- 메트릭 계산: exact match, BLEU 등 다양한 지표

WMTP 알고리즘과의 연결:
- Baseline MTP: 균등 가중치의 기준 pass@k 제공
- Critic-WMTP: Value 기반 가중치로 개선된 코드 품질 측정
- Rho1-WMTP: CE 기반 가중치로 syntax/logic 정확도 향상 확인

사용 예시:
    >>> protocol = EvaluationProtocol(sampling_config={"temperature": 0.8})
    >>> metrics = protocol.evaluate(
    >>>     model=model,
    >>>     tokenizer=tokenizer,
    >>>     dataset=mbpp_dataset,
    >>>     batch_size=16
    >>> )
    >>> print(f"pass@1: {metrics['pass@1']:.2%}")

성능 최적화:
- 배치 추론으로 GPU 효율 극대화
- KV 캐시 활용으로 생성 속도 향상
- 병렬 코드 실행으로 평가 시간 단축

디버깅 팁:
- 낮은 pass@k: temperature 조정 (0.2 → 0.8)
- 코드 파싱 실패: extract_code() 패턴 확인
- OOM 오류: batch_size 감소
"""

import re
from typing import Any

import torch
from rich.console import Console
from rich.table import Table

from ..components.base import Component

console = Console()


class EvaluationProtocol(Component):
    """
    코드 생성 벤치마크용 기본 평가 프로토콜.

    WMTP 연구 맥락:
    토큰 가중치의 효과를 정량화하기 위한 표준화된 평가 방법을 제공합니다.
    Meta MTP 논문의 평가 프로토콜을 준수하여 공정한 비교가 가능합니다.

    구현 내용:
    - 공통 평가 로직과 메트릭 계산
    - pass@k unbiased estimator (Chen et al., 2021)
    - 코드 추출 및 정제 파이프라인
    - 결과 시각화 및 보고서 생성

    WMTP 특화:
    - 알고리즘별 샘플링 설정 최적화
    - 토큰 가중치 효과의 정량적 측정
    - syntax vs semantic 정확도 분리 평가
    """

    def __init__(
        self,
        sampling_config: dict[str, Any] | None = None,
        device: torch.device | None = None,
    ):
        """
        평가 프로토콜 초기화.

        WMTP 맥락:
        각 알고리즘에 최적화된 샘플링 설정을 사용합니다.
        Rho1은 낮은 temperature, Critic은 높은 temperature가 효과적입니다.

        매개변수:
            sampling_config: 샘플링 설정
                - temperature: 0.2~0.8 (기본 0.2)
                - top_p: 0.9~1.0 (기본 0.95)
                - max_length: 최대 토큰 수 (기본 2048)
                - n: 샘플 수 (pass@k용, 기본 1)
            device: 평가 디바이스 (None시 자동 감지)

        예시:
            >>> protocol = EvaluationProtocol(
            >>>     sampling_config={"temperature": 0.8, "top_p": 0.95},
            >>>     device=torch.device("cuda:0")
            >>> )

        디버깅 팁:
            - temperature 높으면 다양성 증가, 정확도 감소
            - top_p 낮으면 안전한 토큰만 선택
            - max_length 부족시 불완전한 코드 생성
        """
        self.sampling_config = sampling_config or {
            "temperature": 0.2,
            "top_p": 0.95,
            "max_length": 2048,
            "n": 1,
        }
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.initialized = False

    def setup(self, config: dict[str, Any]) -> None:
        """Setup evaluator from configuration."""
        # Update sampling config if provided
        if "sampling" in config:
            self.sampling_config.update(config["sampling"])

        # Update device if provided
        if "device" in config:
            self.device = torch.device(config["device"])

        self.initialized = True

    def evaluate(
        self,
        model: Any,
        tokenizer: Any,
        dataset: Any,
        batch_size: int = 8,
        num_samples: int | None = None,
    ) -> dict[str, float]:
        """
        Evaluate model on dataset.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            dataset: Evaluation dataset
            batch_size: Batch size
            num_samples: Optional number of samples to evaluate

        Returns:
            Dictionary of evaluation metrics
        """
        raise NotImplementedError("Subclasses must implement evaluate()")

    def generate_completion(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_new_tokens: int = 512,
    ) -> str:
        """
        Generate code completion for a prompt.

        Args:
            model: Language model
            tokenizer: Tokenizer
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate

        Returns:
            Generated completion
        """
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.sampling_config["temperature"],
                top_p=self.sampling_config["top_p"],
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        completion = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the prompt from completion
        if completion.startswith(prompt):
            completion = completion[len(prompt) :]

        return completion.strip()

    def extract_code(self, completion: str) -> str:
        """
        Extract code from model completion.

        Args:
            completion: Model output

        Returns:
            Extracted code
        """
        # Try to extract code block
        code_pattern = r"```python\n(.*?)```"
        match = re.search(code_pattern, completion, re.DOTALL)

        if match:
            return match.group(1).strip()

        # Try to extract function definition
        func_pattern = r"def\s+\w+.*?(?=\n\n|\Z)"
        match = re.search(func_pattern, completion, re.DOTALL)

        if match:
            return match.group(0).strip()

        # Return full completion as fallback
        return completion.strip()

    def display_results(
        self, metrics: dict[str, float], title: str = "Evaluation Results"
    ) -> None:
        """Display evaluation results."""
        table = Table(title=title)
        table.add_column("Metric", style="cyan")
        table.add_column("Score", style="green")

        for metric, score in metrics.items():
            table.add_row(metric, f"{score:.2%}")

        console.print(table)


# Export main classes and functions
__all__ = [
    "EvaluationProtocol",
]
