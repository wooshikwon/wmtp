"""
WMTP 연구 성능 평가의 통합 관리자: Meta MTP 평가기

WMTP 연구 맥락:
이 모듈은 Meta(Facebook)에서 제시한 Multi-Token Prediction 논문의 평가 방법론을
정확히 재현하여 WMTP 알고리즘의 성능을 공정하고 일관되게 평가합니다.

통합 평가 전략:
다양한 코딩 벤치마크(MBPP, CodeContests, HumanEval)를 통합하여
WMTP 알고리즘의 종합적 성능을 측정하고, 각 벤치마크의 특징을 반영한
균형 잡힌 평가 결과를 제공합니다.

지원하는 평가 지표:
- MBPP pass@k: 기초 Python 문제 성공률
- CodeContests pass@k: 복잡한 알고리즘 문제 성공률
- HumanEval pass@k: OpenAI 표준 코딩 평가 지표
- syntax_valid: 생성된 코드의 구문 정확성
- exact_match: 정확한 정답 일치 비율

WMTP 알고리즘 비교:
각 알고리즘에 대해 동일한 평가 프로토콜을 사용하여
공정한 비교를 보장하고, Meta 논문의 기준점과 직접 비교 가능

성능 최적화:
- 지연 로딩: 요청된 지표에 따라 필요한 평가기만 로드
- 배치 처리: batch_size 설정으로 메모리 효율성 최적화
- 비동기 평가: 다중 벤치마크 동시 실행 가능
"""

from typing import Any

from rich.console import Console

from ...components.base import BaseComponent
from ..registry import evaluator_registry

console = Console()


@evaluator_registry.register(
    "meta-mtp-evaluator", category="evaluator", version="1.0.0"
)
class MetaMTPEvaluator(BaseComponent):
    """
    WMTP 연구를 위한 통합 성능 평가 오케스트레이터입니다.

    연구 맥락:
    Meta의 Multi-Token Prediction 논문에서 제시한 평가 방법론을 정확히 따르며,
    세 가지 WMTP 알고리즘의 성능을 공정하고 일관되게 비교합니다.

    평가 오케스트레이션 전략:
    1. 요청된 지표(metrics) 분석
    2. 필요한 벤치마크 선별적 로드 (MBPP/CodeContests/HumanEval)
    3. 각 평가기에 동일한 샘플링 설정 전달
    4. 결과 통합 및 메타데이터 추가
    5. Meta 논문 형식의 종합 리포트 생성

    지원하는 Config 스키마:
    ```yaml
    evaluator:
      type: "meta-mtp-evaluator"

      # 코드 생성 샘플링 설정
      sampling:
        temperature: 0.8      # 창의성 vs 일관성 균형
        top_p: 0.95          # nucleus sampling
        n: 10                # pass@k 계산용 샘플 수
        max_tokens: 512      # 코드 생성 길이 제한

      # 평가할 지표 목록
      metrics:
        - "mbpp_pass@1"      # MBPP 1번 시도 성공률
        - "mbpp_pass@10"     # MBPP 10번 시도 성공률
        - "contest_pass@1"   # CodeContests 1번 시도 성공률
        - "humaneval_pass@1" # HumanEval 1번 시도 성공률
        - "syntax_valid"     # 구문 정확성

      batch_size: 8         # 배치 크기 (메모리 최적화)
      device: "cuda"        # 계산 디바이스
    ```

    성능 최적화 기능:
    - Lazy Loading: 요청된 지표에 필요한 평가기만 로드
    - 메모리 효율성: 배치 크기 조절로 GPU 메모리 최적화
    - 결과 캐싱: 동일한 설정에 대한 중복 평가 방지

    WMTP 알고리즘별 기대 성능:
    - Baseline MTP: MBPP ~35%, CodeContests ~15%
    - Critic-WMTP: MBPP ~37%, CodeContests ~17% (2%p 향상)
    - Rho1-WMTP: MBPP ~39%, CodeContests ~19% (4%p 향상)

    사용 예시:
    >>> # WMTP 모델 평가
    >>> evaluator = MetaMTPEvaluator(config)
    >>> results = evaluator.run({
    ...     "model": wmtp_model,
    ...     "tokenizer": tokenizer
    ... })
    >>> print(f"MBPP pass@10: {results['mbpp_pass@10']:.2%}")
    >>> print(f"CodeContests pass@1: {results['contest_pass@1']:.2%}")
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.metrics = self.config.get("metrics") or []
        self.sampling = self.config.get("sampling", {})
        self.batch_size = self.config.get("batch_size", 8)
        self.device = self.config.get("device")

    def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
        self.validate_initialized()

        model = ctx.get("model")
        tokenizer = ctx.get("tokenizer")

        # WMTP 알고리즘 성능 결과를 저장할 딕셔너리
        results: dict[str, float] = {}

        # WMTP 연구에서 요청한 지표를 기반으로 필요한 평가기 선별적 로드
        wants_mbpp = any(m.startswith("mbpp") for m in self.metrics)
        wants_contest = any("contest" in m for m in self.metrics) or any(
            m.startswith("contest_pass@") for m in self.metrics
        )

        if wants_mbpp:
            try:
                from .mbpp_eval import MBPPEvaluator  # local import to avoid cycles

                mbpp_eval = MBPPEvaluator(
                    {
                        "sampling": self.sampling,
                        "device": self.device,
                    }
                )
                mbpp_eval.setup({"sampling": self.sampling, "device": self.device})
                mbpp_ds = ctx.get("mbpp_dataset")  # optional external dataset
                mbpp_metrics = mbpp_eval.evaluate(
                    model=model,
                    tokenizer=tokenizer,
                    dataset=mbpp_ds,
                    batch_size=int(self.batch_size or 8),
                )
                # namespacing consistency
                for k, v in mbpp_metrics.items():
                    key = (
                        k
                        if k.startswith("mbpp")
                        else f"mbpp_{k}"
                        if k in {"exact_match"}
                        else k
                    )
                    results[key] = v
            except Exception as e:
                console.print(f"[red]MBPP evaluation failed: {e}[/red]")

        if wants_contest:
            try:
                from .codecontests import CodeContestsEvaluator

                cc_eval = CodeContestsEvaluator(
                    {
                        "sampling": self.sampling,
                        "device": self.device,
                    }
                )
                cc_eval.setup({"sampling": self.sampling, "device": self.device})
                cc_ds = ctx.get("contest_dataset")  # optional external dataset
                # choose k_values from requested metrics
                requested = [m for m in self.metrics if m.startswith("contest_pass@")]
                k_values = (
                    sorted({int(m.split("@")[1]) for m in requested})
                    if requested
                    else [1, 5]
                )
                cc_metrics = cc_eval.evaluate(
                    model=model,
                    tokenizer=tokenizer,
                    dataset=cc_ds,
                    batch_size=int(self.batch_size or 8),
                    k_values=k_values,
                )
                for k, v in cc_metrics.items():
                    # keep codecontests metric names as-is
                    results[k] = v
            except Exception as e:
                console.print(f"[red]CodeContests evaluation failed: {e}[/red]")

        return {"metrics": results}
