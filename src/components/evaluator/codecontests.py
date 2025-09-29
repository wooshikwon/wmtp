"""
WMTP CodeContests 평가자: 경쟁 프로그래밍 벤치마크 평가 시스템

WMTP 연구 맥락:
CodeContests는 더 복잡한 알고리즘 문제를 포함하는 고난도 벤치마크입니다.
MBPP보다 긴 코드 생성과 복잡한 로직 구현이 필요하여,
토큰 가중치가 장문 코드 생성에 미치는 영향을 측정합니다.

핵심 기능:
- 경쟁 프로그래밍 스타일 평가
- pass@k 메트릭 (k=1,5,10)
- 입출력 기반 정확성 검증
- 다중 테스트 케이스 실행

WMTP 알고리즘과의 연결:
- Baseline MTP: 복잡한 알고리즘에서 기준 성능
- Critic-WMTP: 중요 로직 부분에 가중치 부여
- Rho1-WMTP: 참조 모델로 더 나은 알고리즘 선택

성능 기대치:
- Baseline: pass@1 ~15%, pass@10 ~30%
- Critic: pass@1 ~18%, pass@10 ~35%
- Rho1: pass@1 ~20%, pass@10 ~38%
"""

from collections import defaultdict
from typing import Any

from rich.console import Console
from tqdm import tqdm

from ...utils.evaluation_protocol import EvaluationProtocol
from ..registry import evaluator_registry

console = Console()


@evaluator_registry.register("codecontests", category="evaluator", version="1.0.0")
class CodeContestsEvaluator(EvaluationProtocol):
    """
    CodeContests 벤치마크 평가자.

    WMTP 연구 맥락:
    경쟁 프로그래밍 수준의 복잡한 알고리즘 문제를 평가합니다.
    MBPP보다 10배 긴 코드 생성이 필요하며, 효율적인 알고리즘과
    엣지 케이스 처리가 중요합니다.

    특징:
    - 평균 100줄 이상의 코드 생성
    - 시간/공간 복잡도 고려 필요
    - 다양한 입력 형식 처리
    - 정확한 출력 형식 준수

    WMTP 최적화:
    - Critic: 알고리즘 핵심 로직에 높은 가중치
    - Rho1: 참조 모델의 알고리즘 패턴 학습
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        CodeContests 평가자 초기화.

        매개변수:
            config: 평가 설정
                - sampling: 높은 temperature(0.8) 권장
                - max_new_tokens: 512 이상 필요
                - timeout: 테스트 실행 제한 시간
        """
        super().__init__(
            sampling_config=config.get("sampling", None) if config else None,
            device=config.get("device", None) if config else None,
        )
        self.config = config or {}
        self.dataset_name = "codecontests"

    def evaluate(
        self,
        model: Any,
        tokenizer: Any,
        dataset: Any | None = None,
        batch_size: int = 8,  # noqa: ARG002
        num_samples: int | None = None,
        k_values: list[int] | None = None,
    ) -> dict[str, float]:
        """
        CodeContests 데이터셋에서 모델 평가.

        WMTP 맥락:
        복잡한 알고리즘 문제에서 토큰 가중치의 효과를 측정합니다.
        특히 반복문, 조건문, 재귀 등 제어 구조의 정확성이
        전체 알고리즘 성능에 미치는 영향을 분석합니다.

        매개변수:
            model: 평가할 모델
            tokenizer: 토크나이저
            dataset: CodeContests 데이터셋
            batch_size: 배치 크기
            num_samples: 평가할 문제 수
            k_values: pass@k의 k 값 리스트 (기본 [1, 5])

        반환값:
            pass@k 메트릭 딕셔너리

        주의사항:
            - 각 문제당 k개의 솔루션 생성 (메모리 주의)
            - 테스트 실행시 타임아웃 설정 필요
        """
        if dataset is None:
            dataset = self.load_codecontests_dataset()

        if k_values is None:
            k_values = [1, 5]

        metrics = defaultdict(list)

        # Sample subset if specified
        if num_samples:
            dataset = dataset[:num_samples]

        console.print(
            f"[cyan]Evaluating on {len(dataset)} CodeContests problems...[/cyan]"
        )

        for problem in tqdm(dataset, desc="Generating solutions"):
            # Generate multiple solutions for pass@k
            solutions = []
            for _ in range(max(k_values)):
                prompt = self.format_codecontests_prompt(problem)
                completion = self.generate_completion(
                    model, tokenizer, prompt, max_new_tokens=512
                )
                code = self.extract_code(completion)
                solutions.append(code)

            # Evaluate solutions
            results = self.evaluate_solutions(solutions, problem["test_cases"])

            # Compute pass@k for this problem
            for k in k_values:
                passed = any(results[:k])
                metrics[f"pass@{k}"].append(passed)

        # Average metrics
        final_metrics = {}
        for metric, values in metrics.items():
            final_metrics[metric] = sum(values) / len(values)

        self.display_results(final_metrics, title="CodeContests Evaluation Results")
        return final_metrics

    def load_codecontests_dataset(self) -> list[dict[str, Any]]:
        """
        CodeContests 데이터셋 로드.

        참고:
        실제 구현시 HuggingFace datasets 라이브러리나
        공식 CodeContests 데이터를 로드합니다.
        """
        # 데모용 모의 구현
        # 실제로는 CodeContests 데이터셋 로드
        return [
            {
                "description": "Given two integers, return their sum.",
                "test_cases": [
                    {"input": "1 2", "output": "3"},
                    {"input": "-1 1", "output": "0"},
                ],
            },
            {
                "description": "Find the maximum element in an array.",
                "test_cases": [
                    {"input": "5\n1 3 5 2 4", "output": "5"},
                    {"input": "3\n-1 -5 -3", "output": "-1"},
                ],
            },
        ]

    def format_codecontests_prompt(self, problem: dict[str, Any]) -> str:
        """CodeContests 문제를 프롬프트로 변환."""
        prompt = f"""Problem: {problem["description"]}

Write a complete Python solution:

```python
"""
        return prompt

    def evaluate_solutions(
        self,
        solutions: list[str],
        test_cases: list[dict[str, str]],
    ) -> list[bool]:
        """
        여러 솔루션을 테스트 케이스로 평가.

        WMTP 맥락:
        각 솔루션의 함수적 정확성을 검증합니다.
        다양한 입력에 대한 올바른 출력 생성 여부를 확인합니다.

        매개변수:
            solutions: 생성된 솔루션 리스트
            test_cases: 입출력 테스트 케이스

        반환값:
            각 솔루션의 통과/실패 리스트
        """
        results = []

        for solution in solutions:
            passed = self.run_test_cases(solution, test_cases)
            results.append(passed)

        return results

    def run_test_cases(
        self,
        code: str,
        test_cases: list[dict[str, str]],
    ) -> bool:
        """
        코드를 테스트 케이스로 실행.

        WMTP 맥락:
        경쟁 프로그래밍 방식으로 표준 입출력을 통해 평가합니다.
        시간 제한과 메모리 제한을 고려한 실행이 필요합니다.

        매개변수:
            code: 솔루션 코드
            test_cases: 입출력 테스트 케이스

        반환값:
            모든 테스트 통과시 True

        안전성:
            - stdin/stdout 리다이렉션으로 격리 실행
            - 타임아웃 처리 필요 (향후 구현)
        """
        try:
            for test in test_cases:
                # Create execution namespace
                namespace = {}

                # Mock stdin for input
                import io
                import sys

                old_stdin = sys.stdin
                sys.stdin = io.StringIO(test["input"])

                # Capture stdout
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()

                try:
                    # Execute the code
                    exec(code, namespace)

                    # Get output
                    output = sys.stdout.getvalue().strip()

                    # Check if output matches expected
                    if output != test["output"]:
                        return False

                finally:
                    # Restore stdin/stdout
                    sys.stdin = old_stdin
                    sys.stdout = old_stdout

            return True

        except Exception:
            return False

    def compute_pass_at_k(
        self,
        n: int,
        c: int,
        k: int,
    ) -> float:
        """
        pass@k 메트릭 계산.

        WMTP 맥락:
        Chen et al. (2021)의 unbiased estimator를 사용하여
        정확한 pass@k를 계산합니다.

        매개변수:
            n: 생성된 전체 샘플 수
            c: 정답 샘플 수
            k: pass@k의 k 값

        반환값:
            pass@k 점수 (0.0 ~ 1.0)

        수식:
            pass@k = 1 - C(n-c, k) / C(n, k)
            여기서 C는 조합(combination)
        """
        if n - c < k:
            return 1.0

        return 1.0 - float(self._comb(n - c, k)) / float(self._comb(n, k))

    def _comb(self, n: int, k: int) -> int:
        """이항 계수 계산: n개에서 k개를 선택하는 경우의 수."""
        if k > n:
            return 0
        if k == 0 or k == n:
            return 1

        result = 1
        for i in range(min(k, n - k)):
            result = result * (n - i) // (i + 1)
        return result
