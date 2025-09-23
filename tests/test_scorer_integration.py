"""
Integration tests for scorer components.

Validates that:
1. Recipe YAML examples are valid
2. Scorers are properly registered
3. ComponentFactory can create scorers from recipes
4. Scorers produce expected outputs
"""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

# Import scorers to ensure they are registered
import src.components.scorer  # noqa: F401
from src.components.registry import scorer_registry
from src.factory.component_factory import ComponentFactory
from src.settings.recipe_schema import Recipe


class TestRecipeCompatibility:
    """Test recipe YAML compatibility with schemas."""

    def test_critic_recipe_valid(self):
        """Test that critic recipe example is valid."""
        recipe_path = Path("configs/recipe.example.yaml")

        if recipe_path.exists():
            with open(recipe_path) as f:
                recipe_data = yaml.safe_load(f)

            # Validate with Pydantic schema
            recipe = Recipe(**recipe_data)

            # Check critical fields
            assert recipe.train.algo == "critic-wmtp"
            assert recipe.critic.target == "rm_sequence"
            assert recipe.critic.token_spread == "gae"
            assert recipe.loss.lambda_weight == 0.3
            assert recipe.loss.temperature == 0.7

    def test_rho1_recipe_valid(self):
        """Test that rho1 recipe example is valid."""
        recipe_path = Path("configs/recipe.rho1.example.yaml")

        if recipe_path.exists():
            with open(recipe_path) as f:
                recipe_data = yaml.safe_load(f)

            # Validate with Pydantic schema
            recipe = Recipe(**recipe_data)

            # Check critical fields
            assert recipe.train.algo == "rho1-wmtp"
            assert recipe.rho1.score == "abs_excess_ce"
            assert recipe.rho1.percentile_top_p == 0.15
            assert recipe.loss.lambda_weight == 0.5


class TestScorerRegistry:
    """Test scorer registration and retrieval."""

    def test_scorers_registered(self):
        """Check that expected scorers are registered."""
        assert scorer_registry.exists("critic-delta-v1")
        assert scorer_registry.exists("rho1-excess-v1")

    def test_scorer_metadata(self):
        """Verify scorer metadata."""
        critic_meta = scorer_registry.get_metadata("critic-delta-v1")
        assert critic_meta["category"] == "scorer"
        assert critic_meta["version"] == "1.0.0"

        rho1_meta = scorer_registry.get_metadata("rho1-excess-v1")
        assert rho1_meta["category"] == "scorer"
        assert rho1_meta["version"] == "1.0.0"


class TestComponentFactoryScorers:
    """Test ComponentFactory scorer creation."""

    def test_create_critic_scorer(self):
        """Test creating critic scorer from recipe."""
        # Create mock recipe
        recipe_data = {
            "run": {"name": "test", "tags": []},
            "model": {
                "base_id": "facebook/multi-token-prediction",
                "rm_id": "test-rm",
                "ref_id": "test-ref",
                "mtp": {"n_heads": 4, "horizon": 4},
            },
            "train": {"algo": "critic-wmtp", "full_finetune": True},
            "optim": {"optimizer": "adamw"},
            "data": {
                "train": {"sources": ["mbpp"]},
                "eval": {"sources": ["mbpp"]},
            },
            "batching": {},
            "loss": {
                "lambda": 0.3,
                "temperature": 0.7,
                "epsilon": 0.05,
                "max_weight": 3.0,
            },
            "critic": {
                "target": "rm_sequence",
                "token_spread": "gae",
                "delta_mode": "td",
                "normalize": "zscore",
            },
            "rho1": {},
            "eval": {},
        }

        recipe = Recipe(**recipe_data)

        # Create scorer
        scorer = ComponentFactory.create_scorer(recipe)

        # Verify scorer configuration
        assert scorer is not None
        if hasattr(scorer, "target"):
            assert scorer.target == "rm_sequence"
        if hasattr(scorer, "token_spread"):
            assert scorer.token_spread == "gae"
        if hasattr(scorer, "temperature"):
            assert scorer.temperature == 0.7

    def test_create_rho1_scorer(self):
        """Test creating Rho-1 scorer from recipe."""
        # Create mock recipe
        recipe_data = {
            "run": {"name": "test", "tags": []},
            "model": {
                "base_id": "facebook/multi-token-prediction",
                "rm_id": "test-rm",
                "ref_id": "test-ref",
                "mtp": {"n_heads": 4, "horizon": 4},
            },
            "train": {"algo": "rho1-wmtp", "full_finetune": True},
            "optim": {"optimizer": "adamw"},
            "data": {
                "train": {"sources": ["mbpp"]},
                "eval": {"sources": ["mbpp"]},
            },
            "batching": {},
            "loss": {
                "lambda": 0.5,
                "temperature": 0.5,
                "epsilon": 0.1,
                "max_weight": 2.5,
            },
            "critic": {
                "target": "rm_sequence",
                "token_spread": "uniform",
                "delta_mode": "td",
                "normalize": "zscore",
            },
            "rho1": {
                "score": "abs_excess_ce",
                "percentile_top_p": 0.15,
                "refresh_per_epoch": True,
            },
            "eval": {},
        }

        recipe = Recipe(**recipe_data)

        # Create scorer
        scorer = ComponentFactory.create_scorer(recipe)

        # Verify scorer configuration
        assert scorer is not None
        if hasattr(scorer, "score_method"):
            assert scorer.score_method == "abs_excess_ce"
        if hasattr(scorer, "percentile_top_p"):
            assert scorer.percentile_top_p == 0.15
        if hasattr(scorer, "temperature"):
            assert scorer.temperature == 0.5


class TestScorerFunctionality:
    """Test scorer actual functionality."""

    def test_critic_scorer_output(self):
        """Test critic scorer produces valid outputs."""
        from src.components.scorer.critic_delta import CriticDeltaScorer

        config = {
            "target": "rm_sequence",
            "token_spread": "gae",
            "delta_mode": "td",
            "normalize": "zscore",
            "temperature": 0.7,
        }

        scorer = CriticDeltaScorer(config)
        scorer.setup({})

        # Run scorer
        ctx = {"seq_lengths": [100]}  # Changed to seq_lengths (plural) with list
        result = scorer.run(ctx)

        # Validate output structure
        assert "weights" in result
        assert "deltas" in result
        assert "statistics" in result

        # Validate weights
        weights_tensor = result["weights"]
        assert isinstance(weights_tensor, torch.Tensor)
        weights = weights_tensor.detach().cpu().numpy()
        assert weights.shape == (1, 100)  # Expecting batch dimension
        weights = weights[0]  # Get first batch

        # Check mean is approximately 1.0
        assert 0.95 <= np.mean(weights) <= 1.05

        # Check all weights are positive
        assert np.all(weights > 0)

        # Check weights are within expected range
        assert np.all(weights >= 0.05)  # epsilon
        assert np.all(weights <= 3.0)  # max_weight

    def test_rho1_scorer_output(self):
        """Test Rho-1 scorer produces valid outputs."""
        from src.components.scorer.rho1_excess import Rho1ExcessScorer

        config = {
            "score": "abs_excess_ce",
            "percentile_top_p": 0.2,
            "refresh_per_epoch": False,
            "temperature": 0.7,
        }

        scorer = Rho1ExcessScorer(config)
        scorer.setup({})

        # Run scorer
        ctx = {"seq_lengths": [100]}  # Changed to seq_lengths (plural) with list
        result = scorer.run(ctx)

        # Validate output structure
        assert "weights" in result
        assert "scores" in result
        assert "statistics" in result

        # Validate weights
        weights_tensor = result["weights"]
        assert isinstance(weights_tensor, torch.Tensor)
        weights = weights_tensor.detach().cpu().numpy()
        assert weights.shape == (1, 100)  # Expecting batch dimension
        weights = weights[0]  # Get first batch

        # Check mean is approximately 1.0
        assert 0.95 <= np.mean(weights) <= 1.05

        # Check all weights are positive
        assert np.all(weights > 0)

        # Check statistics
        stats = result["statistics"]
        assert "mean_weight" in stats
        assert "std_weight" in stats
        assert "top_p_threshold" in stats


class TestScorerCompatibility:
    """Test complete scorer compatibility chain."""

    def test_end_to_end_critic_flow(self):
        """Test complete flow from recipe to scorer execution."""
        # Load recipe
        recipe_data = {
            "run": {"name": "test", "tags": ["test"]},
            "model": {
                "base_id": "facebook/multi-token-prediction",
                "rm_id": "test-rm",
                "ref_id": "test-ref",
                "mtp": {"n_heads": 4, "horizon": 4},
            },
            "train": {
                "algo": "critic-wmtp",
                "full_finetune": True,
                "lora": {"enabled": False},
            },
            "optim": {
                "optimizer": "adamw",
                "lr": 1.2e-5,
                "weight_decay": 0.1,
                "betas": [0.9, 0.95],
                "grad_clip": 1.0,
                "scheduler": "cosine",
                "warmup_ratio": 0.03,
            },
            "data": {
                "train": {"sources": ["mbpp"], "max_length": 2048},
                "eval": {"sources": ["mbpp"], "max_length": 2048},
            },
            "batching": {
                "global_batch_tokens": 4000000,
                "micro_batch_size": 1,
                "grad_accum_steps": 64,
            },
            "loss": {
                "weight_norm": "mean1.0_clip",
                "lambda": 0.3,
                "temperature": 0.7,
                "epsilon": 0.05,
                "max_weight": 3.0,
            },
            "critic": {
                "target": "rm_sequence",
                "token_spread": "gae",
                "delta_mode": "td",
                "normalize": "zscore",
            },
            "rho1": {
                "score": "abs_excess_ce",
                "percentile_top_p": 0.2,
                "refresh_per_epoch": False,
            },
            "eval": {
                "protocol": "meta-mtp",
                "sampling": {"temperature": 0.2, "top_p": 0.95, "n": 1},
                "metrics": ["mbpp_exact"],
            },
        }

        # Validate recipe
        recipe = Recipe(**recipe_data)
        assert recipe.train.algo == "critic-wmtp"

        # Create scorer through factory
        scorer = ComponentFactory.create_scorer(recipe)
        assert scorer is not None

        # Initialize and run scorer
        scorer.setup({})
        result = scorer.run({"seq_lengths": [50], "horizon": 4})

        # Validate results - expecting head-level weights [B,S,H] for critic
        assert "weights" in result
        weights_tensor = result["weights"]
        assert isinstance(weights_tensor, torch.Tensor)
        weights = weights_tensor.detach().cpu().numpy()
        assert weights.shape == (1, 50, 4)  # [batch, seq_len, horizon]

        # Verify head-level weights satisfy constraints
        overall_mean = np.mean(weights)
        assert 0.8 <= overall_mean <= 1.2  # Relaxed range for head-level weights

    def test_end_to_end_rho1_flow(self):
        """Test complete flow from recipe to scorer execution for Rho-1."""
        # Similar to above but with rho1-wmtp algorithm
        recipe_data = {
            "run": {"name": "test", "tags": ["test"]},
            "model": {
                "base_id": "facebook/multi-token-prediction",
                "rm_id": "test-rm",
                "ref_id": "test-ref",
                "mtp": {"n_heads": 4, "horizon": 4},
            },
            "train": {
                "algo": "rho1-wmtp",
                "full_finetune": False,
                "lora": {
                    "enabled": True,
                    "r": 32,
                    "alpha": 64,
                    "dropout": 0.1,
                    "target_modules": ["q_proj", "v_proj"],
                },
            },
            "optim": {
                "optimizer": "adamw",
                "lr": 2e-5,
                "weight_decay": 0.01,
                "betas": [0.9, 0.999],
                "grad_clip": 1.0,
                "scheduler": "linear",
                "warmup_ratio": 0.05,
            },
            "data": {
                "train": {"sources": ["contest"], "max_length": 1024},
                "eval": {"sources": ["contest"], "max_length": 1024},
            },
            "batching": {
                "global_batch_tokens": 2000000,
                "micro_batch_size": 2,
                "grad_accum_steps": 32,
            },
            "loss": {
                "weight_norm": "mean1.0_clip",
                "lambda": 0.5,
                "temperature": 0.5,
                "epsilon": 0.1,
                "max_weight": 2.5,
            },
            "critic": {
                "target": "rm_sequence",
                "token_spread": "uniform",
                "delta_mode": "td",
                "normalize": "zscore",
            },
            "rho1": {
                "score": "abs_excess_ce",
                "percentile_top_p": 0.15,
                "refresh_per_epoch": True,
            },
            "eval": {
                "protocol": "meta-mtp",
                "sampling": {"temperature": 0.1, "top_p": 0.9, "n": 5},
                "metrics": ["contest_pass@1", "contest_pass@5"],
            },
        }

        # Validate recipe
        recipe = Recipe(**recipe_data)
        assert recipe.train.algo == "rho1-wmtp"

        # Create scorer through factory
        scorer = ComponentFactory.create_scorer(recipe)
        assert scorer is not None

        # Initialize and run scorer
        scorer.setup({})
        result = scorer.run({"seq_lengths": [50], "horizon": 4})

        # Validate results - expecting head-level weights [B,S,H] for rho1
        assert "weights" in result
        weights_tensor = result["weights"]
        assert isinstance(weights_tensor, torch.Tensor)
        weights = weights_tensor.detach().cpu().numpy()
        assert weights.shape == (1, 50, 4)  # [batch, seq_len, horizon]

        # Verify head-level weights satisfy constraints
        overall_mean = np.mean(weights)
        assert 0.8 <= overall_mean <= 1.2  # Relaxed range for head-level weights

        # Verify distance-aware decay for rho1 scorer
        head_means = [np.mean(weights[:, :, k]) for k in range(4)]
        for k in range(3):
            assert head_means[k] >= head_means[k+1] * 0.5, f"Rho1 Head {k} mean should be reasonably higher than Head {k+1} due to distance decay"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
