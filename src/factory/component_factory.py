"""
Component factory for building and assembling components from configuration.

This module provides factory functions that take validated Pydantic configurations
and create appropriate component instances using the registry pattern.
"""

from typing import Any

from src.components.base import (
    Evaluator,
    Loader,
    Optimizer,
    Scorer,
    Trainer,
)
from src.components.registry import (
    evaluator_registry,
    loader_registry,
    optimizer_registry,
    scorer_registry,
    trainer_registry,
)
from src.settings import Config, Recipe


class ComponentFactory:
    """
    Factory for creating components from configuration.

    Maps configuration values to registry keys and creates
    appropriate component instances.
    """

    # Mapping from algorithm to scorer registry key
    ALGO_TO_SCORER = {
        "critic-wmtp": "critic-delta-v1",
        "rho1-wmtp": "rho1-excess-v1",
    }

    # Mapping from optimizer name to registry key
    OPTIMIZER_MAP = {
        "adamw": "adamw-bf16-fused",
        "lion": "lion-optimizer",
        "sgd": "sgd-optimizer",
    }

    # Mapping from data source to loader registry key
    LOADER_MAP = {
        "mbpp": "dataset-mbpp-loader",
        "contest": "dataset-contest-loader",
        "model": "hf-local-s3-loader",
    }

    # Trainer selection based on configuration
    TRAINER_MAP = {
        "mtp-weighted": "mtp-weighted-ce-trainer",
        "mtp-standard": "mtp-standard-trainer",
    }

    # Evaluator selection based on protocol
    EVALUATOR_MAP = {
        "meta-mtp": "meta-mtp-evaluator",
        "mbpp": "mbpp-v1",
        "codecontests": "codecontests-v1",
    }

    @classmethod
    def create_scorer(cls, recipe: Recipe) -> Scorer:
        """
        Create scorer based on training algorithm.

        Args:
            recipe: Recipe configuration

        Returns:
            Scorer instance
        """
        algo = recipe.train.algo
        scorer_key = cls.ALGO_TO_SCORER.get(algo)

        if not scorer_key:
            raise ValueError(
                f"No scorer mapping found for algorithm '{algo}'. "
                f"Available algorithms: {list(cls.ALGO_TO_SCORER.keys())}"
            )

        # Prepare scorer configuration based on algorithm
        if algo == "critic-wmtp":
            scorer_config = {
                "target": recipe.critic.target,
                "token_spread": recipe.critic.token_spread,
                "delta_mode": recipe.critic.delta_mode,
                "normalize": recipe.critic.normalize,
                "temperature": recipe.loss.temperature,
            }
        elif algo == "rho1-wmtp":
            scorer_config = {
                "score": recipe.rho1.score,
                "percentile_top_p": recipe.rho1.percentile_top_p,
                "refresh_per_epoch": recipe.rho1.refresh_per_epoch,
                "temperature": recipe.loss.temperature,
            }
        else:
            scorer_config = {}

        # Create scorer from registry
        return scorer_registry.create(scorer_key, scorer_config)

    @classmethod
    def create_trainer(
        cls,
        recipe: Recipe,
        config: Config,
        scorer: Scorer | None = None,
    ) -> Trainer:
        """
        Create trainer based on configuration.

        Args:
            recipe: Recipe configuration
            config: Environment configuration
            scorer: Optional scorer instance

        Returns:
            Trainer instance
        """
        # Determine trainer type
        trainer_key = cls.TRAINER_MAP.get("mtp-weighted")

        trainer_config = {
            "n_heads": recipe.model.mtp.n_heads,
            "horizon": recipe.model.mtp.horizon,
            "loss_config": {
                "weight_norm": recipe.loss.weight_norm,
                "lambda": recipe.loss.lambda_weight,
                "temperature": recipe.loss.temperature,
                "epsilon": recipe.loss.epsilon,
                "max_weight": recipe.loss.max_weight,
            },
            "full_finetune": recipe.train.full_finetune,
            "lora_config": recipe.train.lora.model_dump()
            if recipe.train.lora.enabled
            else None,
            "mixed_precision": config.devices.mixed_precision,
            "fsdp_config": config.devices.fsdp.model_dump()
            if config.devices.fsdp.enabled
            else None,
            "scorer": scorer,
        }

        return trainer_registry.create(trainer_key, trainer_config)

    @classmethod
    def create_optimizer(cls, recipe: Recipe, model_params: Any) -> Optimizer:
        """
        Create optimizer based on configuration.

        Args:
            recipe: Recipe configuration
            model_params: Model parameters to optimize

        Returns:
            Optimizer instance
        """
        optimizer_key = cls.OPTIMIZER_MAP.get(recipe.optim.optimizer)

        if not optimizer_key:
            raise ValueError(
                f"No optimizer mapping found for '{recipe.optim.optimizer}'. "
                f"Available optimizers: {list(cls.OPTIMIZER_MAP.keys())}"
            )

        optimizer_config = {
            "params": model_params,
            "lr": recipe.optim.lr,
            "weight_decay": recipe.optim.weight_decay,
            "betas": recipe.optim.betas,
            "grad_clip": recipe.optim.grad_clip,
            "scheduler": recipe.optim.scheduler,
            "warmup_ratio": recipe.optim.warmup_ratio,
        }

        return optimizer_registry.create(optimizer_key, optimizer_config)

    @classmethod
    def create_data_loader(cls, source: str, config: Config) -> Loader:
        """
        Create data loader for a specific source.

        Args:
            source: Data source name
            config: Environment configuration

        Returns:
            Loader instance
        """
        loader_key = cls.LOADER_MAP.get(source)

        if not loader_key:
            raise ValueError(
                f"No loader mapping found for source '{source}'. "
                f"Available sources: {list(cls.LOADER_MAP.keys())}"
            )

        loader_config = {
            "cache_dir": str(config.paths.cache),
            # Nested storage config only (flattened keys removed)
            "storage": {
                "mode": config.storage.mode,
                "s3": (config.storage.s3.model_dump() if config.storage.s3 else None),
            },
            # Ensure S3 utils share the same cache root
            "paths": {
                "cache": str(config.paths.cache),
            },
        }

        # Add source-specific configuration
        if source == "mbpp":
            loader_config["local_path"] = str(config.paths.datasets.mbpp_local)
        elif source == "contest":
            loader_config["local_path"] = str(config.paths.datasets.contest_local)
        elif source == "model":
            loader_config["model_paths"] = {
                "base": str(config.paths.models.base_local),
                "rm": str(config.paths.models.rm_local),
                "ref": str(config.paths.models.ref_local),
            }

        # Note: s3_config(flat) is no longer supported. Use nested 'storage' only.

        return loader_registry.create(loader_key, loader_config)

    @classmethod
    def create_model_loader(cls, config: Config) -> Loader:
        """
        Create model loader (separated from generic data loader for clarity).

        Returns:
            Loader instance configured for model loading
        """
        loader_key = cls.LOADER_MAP.get("model")
        if not loader_key:
            raise ValueError("No loader mapping found for 'model'.")

        loader_config = {
            "cache_dir": str(config.paths.cache),
            "storage": {
                "mode": config.storage.mode,
                "s3": (config.storage.s3.model_dump() if config.storage.s3 else None),
            },
            "paths": {
                "cache": str(config.paths.cache),
            },
            "model_paths": {
                "base": str(config.paths.models.base_local),
                "rm": str(config.paths.models.rm_local),
                "ref": str(config.paths.models.ref_local),
            },
        }

        return loader_registry.create(loader_key, loader_config)

    @classmethod
    def create_evaluator(cls, recipe: Recipe, config: Config) -> Evaluator:
        """
        Create evaluator based on protocol.

        Args:
            recipe: Recipe configuration
            config: Environment configuration

        Returns:
            Evaluator instance
        """
        protocol = recipe.eval.protocol
        evaluator_key = cls.EVALUATOR_MAP.get(protocol)

        if not evaluator_key:
            raise ValueError(
                f"No evaluator mapping found for protocol '{protocol}'. "
                f"Available protocols: {list(cls.EVALUATOR_MAP.keys())}"
            )

        evaluator_config = {
            "sampling": recipe.eval.sampling.model_dump(),
            "metrics": recipe.eval.metrics,
            "batch_size": recipe.data.eval.batch_size,
        }

        return evaluator_registry.create(evaluator_key, evaluator_config)

    # build_pipeline_components removed: use create_* methods directly in pipelines
