"""
Phase 2 평가기 통합 테스트

이 테스트는 Phase 2에서 구현한 세 가지 평가기가
올바르게 작동하는지 검증합니다:
1. SelfSpeculativeEvaluator
2. PerplexityMeasurer
3. MetricsVisualizer

각 평가기는 Registry에 등록되고, ComponentFactory를 통해
생성되어야 합니다.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from src.components.evaluator.self_speculative import SelfSpeculativeEvaluator
from src.components.evaluator.perplexity_measurer import PerplexityMeasurer
from src.components.evaluator.metrics_visualizer import MetricsVisualizer
from src.components.registry import evaluator_registry
from src.factory.component_factory import ComponentFactory
from src.settings import Config, Recipe


class MockMTPModel(nn.Module):
    """Mock MTP model for testing."""

    def __init__(self, vocab_size: int = 1000, hidden_size: int = 768, n_heads: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_heads = n_heads

        # 간단한 임베딩과 출력 레이어
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.output_heads = nn.ModuleList([
            nn.Linear(hidden_size, vocab_size) for _ in range(n_heads)
        ])

    def forward(self, input_ids, attention_mask=None):
        # 임베딩
        hidden = self.embedding(input_ids)  # [B, S, H]

        # 각 헤드별 출력
        logits = []
        for head in self.output_heads:
            logits.append(head(hidden))  # [B, S, V]

        # Stack to [B, S, H, V]
        logits = torch.stack(logits, dim=2)

        # Mock outputs object
        outputs = MagicMock()
        outputs.logits = logits
        outputs.hidden_states = [hidden]  # For critic scorer

        return outputs


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self):
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.vocab_size = 1000

    def encode(self, text, return_tensors=None):
        # 간단한 토큰화 시뮬레이션
        tokens = [1] + [i % 100 + 10 for i in range(len(text) // 2)] + [2]
        if return_tensors == "pt":
            return torch.tensor([tokens])
        return tokens

    def decode(self, token_ids, skip_special_tokens=False):
        return "decoded text"

    def __call__(self, text, max_length=512, truncation=True, padding="max_length", return_tensors=None):
        # Handle batch input
        if isinstance(text, list):
            all_tokens = []
            all_attention_mask = []
            for t in text:
                tokens = self.encode(t)

                # Padding/truncation
                if len(tokens) < max_length:
                    tokens = tokens + [self.pad_token_id] * (max_length - len(tokens))
                else:
                    tokens = tokens[:max_length]

                all_tokens.append(tokens)
                all_attention_mask.append([1 if t != self.pad_token_id else 0 for t in tokens])

            if return_tensors == "pt":
                return {
                    "input_ids": torch.tensor(all_tokens),
                    "attention_mask": torch.tensor(all_attention_mask)
                }
            return {"input_ids": all_tokens}
        else:
            tokens = self.encode(text)

            # Padding/truncation
            if len(tokens) < max_length:
                tokens = tokens + [self.pad_token_id] * (max_length - len(tokens))
            else:
                tokens = tokens[:max_length]

            if return_tensors == "pt":
                return {
                    "input_ids": torch.tensor([tokens]),
                    "attention_mask": torch.tensor([[1 if t != self.pad_token_id else 0 for t in tokens]])
                }
            return {"input_ids": tokens}


@pytest.fixture
def mock_model():
    """Create a mock MTP model."""
    return MockMTPModel()


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    return MockTokenizer()


@pytest.fixture
def test_config():
    """Create test configuration."""
    config = MagicMock()
    config.seed = 42
    config.devices.mixed_precision = "bf16"
    return config


@pytest.fixture
def test_recipe():
    """Create test recipe."""
    recipe = MagicMock()
    recipe.eval.protocol = "meta-mtp"
    recipe.eval.sampling.temperature = 0.8
    recipe.eval.sampling.top_p = 0.95
    recipe.eval.sampling.model_dump.return_value = {
        "temperature": 0.8,
        "top_p": 0.95
    }
    recipe.data.eval.batch_size = 4
    return recipe


class TestSelfSpeculativeEvaluator:
    """Test SelfSpeculativeEvaluator functionality."""

    def test_registration(self):
        """Test that evaluator is registered."""
        assert "self-speculative" in evaluator_registry.list_keys("evaluator")

    def test_initialization(self):
        """Test evaluator initialization."""
        config = {
            "num_sequences": 10,
            "max_tokens": 128,
            "temperature": 0.8,
            "device": "cpu"
        }
        evaluator = SelfSpeculativeEvaluator(config)
        assert evaluator.num_sequences == 10
        assert evaluator.max_tokens == 128
        assert evaluator.temperature == 0.8

    def test_speculative_decode_step(self, mock_model, mock_tokenizer):
        """Test single speculative decoding step."""
        evaluator = SelfSpeculativeEvaluator({"device": "cpu"})
        evaluator.setup({})

        # 입력 준비
        input_ids = torch.randint(0, 100, (1, 10))

        # Speculative decode step
        accepted_tokens, num_accepted, head_probs = evaluator._speculative_decode_step(
            mock_model, input_ids, draft_length=4
        )

        assert accepted_tokens is not None
        assert isinstance(num_accepted, int)
        assert 0 <= num_accepted <= 4
        assert len(head_probs) == 4

    def test_run_evaluation(self, mock_model, mock_tokenizer):
        """Test full evaluation run."""
        evaluator = SelfSpeculativeEvaluator({
            "num_sequences": 2,
            "max_tokens": 32,
            "measure_speedup": False,
            "device": "cpu"
        })
        evaluator.setup({})

        # Context 준비
        ctx = {
            "model": mock_model,
            "tokenizer": mock_tokenizer,
            "prompts": ["def test():", "class Example:"]
        }

        # 평가 실행
        results = evaluator.run(ctx)

        assert "metrics" in results
        metrics = results["metrics"]
        assert "overall_acceptance_rate" in metrics
        assert "average_accepted_length" in metrics
        assert "position_acceptance_rates" in metrics
        assert len(metrics["position_acceptance_rates"]) == 4


class TestPerplexityMeasurer:
    """Test PerplexityMeasurer functionality."""

    def test_registration(self):
        """Test that evaluator is registered."""
        assert "perplexity-measurer" in evaluator_registry.list_keys("evaluator")

    def test_initialization(self):
        """Test evaluator initialization."""
        config = {
            "batch_size": 4,
            "max_length": 512,
            "analyze_token_types": True,
            "device": "cpu"
        }
        measurer = PerplexityMeasurer(config)
        assert measurer.batch_size == 4
        assert measurer.max_length == 512
        assert measurer.analyze_token_types is True

    def test_compute_perplexity_batch(self, mock_model):
        """Test perplexity computation for a batch."""
        measurer = PerplexityMeasurer({"device": "cpu"})
        measurer.setup({})

        # 입력 준비
        input_ids = torch.randint(0, 100, (2, 20))

        # Perplexity 계산
        ppl, ce_losses = measurer._compute_perplexity_batch(
            mock_model, input_ids
        )

        assert isinstance(ppl, float)
        assert ppl > 0
        assert ce_losses.shape == (2, 19)  # seq_len - 1

    def test_classify_token_type(self, mock_tokenizer):
        """Test token type classification."""
        measurer = PerplexityMeasurer()
        measurer.setup({})

        # 특수 토큰
        assert measurer._classify_token_type(0, mock_tokenizer) == "special"
        assert measurer._classify_token_type(1, mock_tokenizer) == "special"
        assert measurer._classify_token_type(2, mock_tokenizer) == "special"

        # 일반 토큰 (mock에서는 text로 분류됨)
        assert measurer._classify_token_type(10, mock_tokenizer) in ["code", "text"]

    def test_run_evaluation(self, mock_model, mock_tokenizer):
        """Test full evaluation run."""
        measurer = PerplexityMeasurer({
            "batch_size": 2,
            "max_length": 64,
            "analyze_token_types": False,
            "compute_head_perplexity": False,
            "device": "cpu"
        })
        measurer.setup({})

        # Context 준비
        ctx = {
            "model": mock_model,
            "tokenizer": mock_tokenizer,
            "eval_dataset": None  # Will use default test data
        }

        # 평가 실행
        results = measurer.run(ctx)

        assert "metrics" in results
        metrics = results["metrics"]
        assert "overall_perplexity" in metrics
        assert "overall_ce" in metrics
        assert metrics["overall_perplexity"] > 0


class TestMetricsVisualizer:
    """Test MetricsVisualizer functionality."""

    def test_registration(self):
        """Test that evaluator is registered."""
        assert "metrics-visualizer" in evaluator_registry.list_keys("evaluator")

    def test_initialization(self):
        """Test evaluator initialization with temp directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "output_dir": tmpdir,
                "save_formats": ["png"],
                "use_plotly": False,
                "upload_to_mlflow": False
            }
            visualizer = MetricsVisualizer(config)
            assert visualizer.output_dir == Path(tmpdir)
            assert visualizer.save_formats == ["png"]

    def test_create_inference_speed_chart(self):
        """Test inference speed chart creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = MetricsVisualizer({
                "output_dir": tmpdir,
                "use_plotly": False
            })
            visualizer.setup({})

            metrics = {
                "batch_sizes": [1, 4, 8],
                "mtp_speeds": [100, 350, 650],
                "ntp_speeds": [40, 140, 260]
            }

            chart_path = visualizer.create_inference_speed_chart(metrics)
            assert chart_path.exists()
            assert chart_path.suffix == ".png"

    def test_create_perplexity_heatmap(self):
        """Test perplexity heatmap creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = MetricsVisualizer({
                "output_dir": tmpdir,
                "use_plotly": False
            })
            visualizer.setup({})

            metrics = {
                "perplexity_matrix": [
                    [15.2, 16.8, 18.5, 22.3],
                    [16.1, 17.2, 19.1, 23.8],
                    [17.5, 18.9, 20.7, 25.2],
                    [19.2, 21.3, 23.5, 28.1]
                ]
            }

            chart_path = visualizer.create_perplexity_heatmap(metrics)
            assert chart_path.exists()
            assert chart_path.suffix == ".png"

    def test_run_visualization(self):
        """Test full visualization run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = MetricsVisualizer({
                "output_dir": tmpdir,
                "use_plotly": False,
                "upload_to_mlflow": False
            })
            visualizer.setup({})

            # Context 준비
            ctx = {
                "metrics": {
                    "batch_sizes": [1, 4, 8],
                    "mtp_speeds": [100, 350, 650],
                    "ntp_speeds": [40, 140, 260],
                    "position_acceptance_rates": [0.85, 0.72, 0.58, 0.41],
                    "pass_at_1": [35.2, 37.1, 39.5],
                    "perplexity": [22.5, 20.8, 19.2]
                },
                "chart_types": ["inference_speed", "acceptance", "comparison"]
            }

            # 시각화 실행
            results = visualizer.run(ctx)

            assert "charts" in results
            assert len(results["charts"]) >= 3
            assert all(chart.exists() for chart in results["charts"])


class TestComponentFactoryIntegration:
    """Test ComponentFactory integration with Phase 2 evaluators."""

    def test_create_evaluator_by_type_self_speculative(self, test_recipe, test_config):
        """Test creating self-speculative evaluator via factory."""
        evaluator = ComponentFactory.create_evaluator_by_type(
            "self-speculative", test_recipe, test_config
        )
        assert isinstance(evaluator, SelfSpeculativeEvaluator)

    def test_create_evaluator_by_type_perplexity(self, test_recipe, test_config):
        """Test creating perplexity measurer via factory."""
        evaluator = ComponentFactory.create_evaluator_by_type(
            "perplexity-measurer", test_recipe, test_config
        )
        assert isinstance(evaluator, PerplexityMeasurer)

    def test_create_evaluator_by_type_visualizer(self, test_recipe, test_config):
        """Test creating metrics visualizer via factory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # config에 output_dir 설정 추가
            test_config.output_dir = tmpdir

            evaluator = ComponentFactory.create_evaluator_by_type(
                "metrics-visualizer", test_recipe, test_config
            )
            assert isinstance(evaluator, MetricsVisualizer)


class TestEndToEndIntegration:
    """End-to-end integration tests for Phase 2."""

    @pytest.mark.slow
    def test_complete_evaluation_pipeline(self, mock_model, mock_tokenizer, test_recipe, test_config):
        """Test complete evaluation pipeline with all Phase 2 components."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Self-speculative 평가
            spec_evaluator = SelfSpeculativeEvaluator({
                "num_sequences": 2,
                "max_tokens": 32,
                "measure_speedup": False,
                "device": "cpu"
            })
            spec_evaluator.setup({})
            spec_results = spec_evaluator.run({
                "model": mock_model,
                "tokenizer": mock_tokenizer,
                "prompts": ["def test():", "class Example:"]
            })

            # 2. Perplexity 측정
            ppl_measurer = PerplexityMeasurer({
                "batch_size": 2,
                "max_length": 64,
                "device": "cpu"
            })
            ppl_measurer.setup({})
            ppl_results = ppl_measurer.run({
                "model": mock_model,
                "tokenizer": mock_tokenizer
            })

            # 3. 결과 시각화
            all_metrics = {
                **spec_results["metrics"],
                **ppl_results["metrics"]
            }

            visualizer = MetricsVisualizer({
                "output_dir": tmpdir,
                "use_plotly": False,
                "upload_to_mlflow": False
            })
            visualizer.setup({})
            vis_results = visualizer.run({
                "metrics": all_metrics,
                "chart_types": ["all"]
            })

            # 검증
            assert spec_results["metrics"]["overall_acceptance_rate"] >= 0
            assert ppl_results["metrics"]["overall_perplexity"] > 0
            assert len(vis_results["charts"]) >= 2

            # 메타데이터 파일 확인
            metadata_path = Path(tmpdir) / "visualization_metadata.json"
            assert metadata_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])