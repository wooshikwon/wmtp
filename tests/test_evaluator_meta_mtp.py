"""
Tests for Meta MTP evaluator orchestrator.
"""

from src.components.evaluator.meta_mtp import MetaMTPEvaluator


class TestMetaMTPEvaluator:
    def test_registration_and_basic_run_mbpp_only(self):
        evaluator = MetaMTPEvaluator(
            {
                "metrics": ["mbpp_exact"],
                "sampling": {"temperature": 0.2, "top_p": 0.95, "n": 1},
                "batch_size": 2,
            }
        )
        evaluator.setup({"sampling": {"temperature": 0.2}})

        # Use dummy model/tokenizer placeholders; component handles errors gracefully
        result = evaluator.run({"model": object(), "tokenizer": object()})

        assert isinstance(result, dict)
        assert "metrics" in result

    def test_registration_and_basic_run_contest_only(self):
        evaluator = MetaMTPEvaluator(
            {
                "metrics": ["contest_pass@1", "contest_pass@5"],
                "sampling": {"temperature": 0.2, "top_p": 0.95, "n": 1},
                "batch_size": 2,
            }
        )
        evaluator.setup({"sampling": {"temperature": 0.2}})

        result = evaluator.run({"model": object(), "tokenizer": object()})
        assert isinstance(result, dict)
        assert "metrics" in result

    def test_registration_and_basic_run_both(self):
        evaluator = MetaMTPEvaluator(
            {
                "metrics": ["mbpp_exact", "contest_pass@1"],
                "sampling": {"temperature": 0.2, "top_p": 0.95, "n": 1},
                "batch_size": 2,
            }
        )
        evaluator.setup({"sampling": {"temperature": 0.2}})

        result = evaluator.run({"model": object(), "tokenizer": object()})
        assert isinstance(result, dict)
        assert "metrics" in result
