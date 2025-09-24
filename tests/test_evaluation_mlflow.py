"""
Test script for verifying Phase 9 Stage 3 MLflow integration.

This test verifies that the evaluation pipeline correctly:
1. Logs metrics to MLflow
2. Saves prediction samples as artifacts
3. Saves weight statistics
4. Generates evaluation reports
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_evaluation_pipeline_mlflow_integration():
    """Test that evaluation pipeline properly integrates with MLflow."""

    # Import after all patches are in place
    from src.pipelines.evaluation_pipeline import EvaluationPipeline

    # Create mock config and recipe
    mock_config = MagicMock()
    mock_config.seed = 42
    mock_config.model_dump.return_value = {"seed": 42}

    mock_recipe = MagicMock()
    mock_recipe.run.name = "test_run"
    mock_recipe.train.algo = "critic-wmtp"
    mock_recipe.model.base_id = "test-model"
    mock_recipe.model.mtp.n_heads = 4
    mock_recipe.model.mtp.horizon = 4
    mock_recipe.eval.protocol = "meta-mtp"
    mock_recipe.eval.sampling.temperature = 0.7
    mock_recipe.eval.sampling.top_p = 0.95
    mock_recipe.eval.sampling.n = 1
    mock_recipe.eval.sampling.model_dump.return_value = {
        "temperature": 0.7,
        "top_p": 0.95,
        "n": 1,
    }
    mock_recipe.data.eval.sources = ["mbpp"]
    mock_recipe.data.eval.batch_size = 8
    mock_recipe.data.eval.max_length = 2048
    mock_recipe.loss.temperature = 0.7

    # Create pipeline
    pipeline = EvaluationPipeline(mock_config, mock_recipe)

    # Mock MLflow manager
    mock_mlflow = MagicMock()
    pipeline.mlflow = mock_mlflow

    # Create test results
    test_results = {
        "metrics": {
            "mbpp_exact_match": 0.65,
            "mbpp_pass@1": 0.70,
            "contest_pass@1": 0.45,
            "contest_pass@5": 0.62,
        },
        "predictions": ["def foo(): return 1", "def bar(): return 2"],
        "references": ["def foo(): return 1", "def bar(): return 2"],
        "weight_statistics": {
            "mean_weight": 1.0,
            "std_weight": 0.15,
            "min_weight": 0.05,
            "max_weight": 2.5,
        },
    }

    # Create a temporary checkpoint file
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        checkpoint_path = Path(tmp.name)

    try:
        # Test _log_results with all options enabled
        pipeline._log_results(
            results=test_results,
            save_predictions=True,
            save_report=True,
            checkpoint=checkpoint_path,
        )

        # Verify metrics were logged
        assert mock_mlflow.log_metric.called
        metric_calls = {
            call[0][0]: call[0][1] for call in mock_mlflow.log_metric.call_args_list
        }

        # Check that all metrics were logged
        assert "mbpp_exact_match" in metric_calls
        assert "mbpp_pass@1" in metric_calls
        assert "contest_pass@1" in metric_calls
        assert "contest_pass@5" in metric_calls
        assert "avg_performance" in metric_calls

        # Check metric values
        assert metric_calls["mbpp_exact_match"] == 0.65
        assert metric_calls["mbpp_pass@1"] == 0.70

        # Verify parameters were logged
        assert mock_mlflow.log_param.called
        param_calls = {
            call[0][0]: call[0][1] for call in mock_mlflow.log_param.call_args_list
        }

        # Check key parameters
        assert "eval_protocol" in param_calls
        assert "algorithm" in param_calls
        assert "model_id" in param_calls
        assert "mtp_heads" in param_calls

        # Verify artifacts were logged
        assert mock_mlflow.log_artifact.called
        artifact_calls = [
            (call[0][0], call[0][1] if len(call[0]) > 1 else None)
            for call in mock_mlflow.log_artifact.call_args_list
        ]

        # Check that different artifact types were saved
        artifact_paths = [call[1] for call in artifact_calls if call[1]]
        assert "evaluation" in artifact_paths
        assert "predictions" in artifact_paths
        assert "weights" in artifact_paths
        assert "reports" in artifact_paths

    finally:
        # Clean up
        checkpoint_path.unlink(missing_ok=True)


def test_evaluation_report_generation():
    """Test that evaluation reports are generated correctly."""

    from datetime import datetime

    from src.pipelines.evaluation_pipeline import EvaluationPipeline

    # Create mock config and recipe
    mock_config = MagicMock()
    mock_config.seed = 42

    mock_recipe = MagicMock()
    mock_recipe.train.algo = "rho1-wmtp"
    mock_recipe.model.base_id = "test-model"
    mock_recipe.model.mtp.n_heads = 4
    mock_recipe.model.mtp.horizon = 4
    mock_recipe.eval.protocol = "meta-mtp"
    mock_recipe.eval.sampling.temperature = 0.7
    mock_recipe.eval.sampling.top_p = 0.95
    mock_recipe.loss.temperature = 0.7
    mock_recipe.rho1 = {"percentile_top_p": 0.2}

    pipeline = EvaluationPipeline(mock_config, mock_recipe)
    pipeline.mlflow = MagicMock()

    # Test report generation
    test_results = {"weight_statistics": {"mean_weight": 1.0}}
    test_metrics = {
        "mbpp_exact_match": 0.72,
        "contest_pass@1": 0.48,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint = Path("/tmp/test_checkpoint.pt")

    # Generate report
    pipeline._generate_evaluation_report(
        results=test_results,
        metrics=test_metrics,
        checkpoint=checkpoint,
        timestamp=timestamp,
    )

    # Verify report was saved as artifact
    assert pipeline.mlflow.log_artifact.called

    # Check that the report path includes "reports" artifact path
    calls = pipeline.mlflow.log_artifact.call_args_list
    report_saved = any("reports" in str(call) for call in calls)
    assert report_saved


def test_weight_statistics_saving():
    """Test that weight statistics are properly saved."""

    import numpy as np

    from src.pipelines.evaluation_pipeline import EvaluationPipeline

    mock_config = MagicMock()
    mock_config.seed = 42

    mock_recipe = MagicMock()
    mock_recipe.train.algo = "critic-wmtp"
    mock_recipe.loss.temperature = 0.7

    pipeline = EvaluationPipeline(mock_config, mock_recipe)
    pipeline.mlflow = MagicMock()

    # Create test weight data
    test_weights = np.random.normal(1.0, 0.2, size=100).tolist()
    test_results = {
        "weights": test_weights,
        "weight_statistics": {
            "mean_weight": 1.01,
            "std_weight": 0.19,
        },
    }

    timestamp = "20240101_120000"

    # Save weight statistics
    pipeline._save_weight_statistics(test_results, timestamp)

    # Verify artifact was saved
    assert pipeline.mlflow.log_artifact.called

    # Check that weights path was used
    calls = pipeline.mlflow.log_artifact.call_args_list
    weights_saved = any("weights" in str(call) for call in calls)
    assert weights_saved


def test_prediction_samples_saving():
    """Test that prediction samples are properly saved."""

    from src.pipelines.evaluation_pipeline import EvaluationPipeline

    mock_config = MagicMock()
    mock_config.seed = 42

    mock_recipe = MagicMock()

    pipeline = EvaluationPipeline(mock_config, mock_recipe)
    pipeline.mlflow = MagicMock()

    # Create test predictions
    test_results = {
        "predictions": [
            "def add(a, b): return a + b",
            "def multiply(a, b): return a * b",
            "def subtract(a, b): return a - b",
        ],
        "references": [
            "def add(x, y): return x + y",
            "def multiply(x, y): return x * y",
            "def subtract(x, y): return x - y",
        ],
    }

    timestamp = "20240101_120000"

    # Save prediction samples
    pipeline._save_prediction_samples(test_results, timestamp)

    # Verify artifact was saved
    assert pipeline.mlflow.log_artifact.called

    # Check that predictions path was used
    calls = pipeline.mlflow.log_artifact.call_args_list
    predictions_saved = any("predictions" in str(call) for call in calls)
    assert predictions_saved


if __name__ == "__main__":
    # Run tests
    print("Testing MLflow integration...")
    test_evaluation_pipeline_mlflow_integration()
    print("✓ MLflow integration test passed")

    print("Testing report generation...")
    test_evaluation_report_generation()
    print("✓ Report generation test passed")

    print("Testing weight statistics saving...")
    test_weight_statistics_saving()
    print("✓ Weight statistics test passed")

    print("Testing prediction samples saving...")
    test_prediction_samples_saving()
    print("✓ Prediction samples test passed")

    print("\nAll tests passed!")
