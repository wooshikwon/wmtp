"""
Training pipeline orchestrator for WMTP framework.

Assembles components from settings, initializes MLflow/dist/seed, loads
models and datasets, builds dataloaders, and runs the trainer.
"""

from __future__ import annotations

from typing import Any, Iterable

import math
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import default_data_collator

from src.components.base import Component
from src.factory.component_factory import ComponentFactory
from src.settings import Config, Recipe
from src.utils import (
    FSDPConfig,
    MLflowManager,
    create_mlflow_manager,
    get_dist_manager,
    set_seed,
)


@dataclass
class PipelineOutputs:
    trainer_metrics: dict[str, Any]


class TrainingPipeline:
    """End-to-end training orchestrator."""

    def __init__(self, config: Config, recipe: Recipe):
        self.config = config
        self.recipe = recipe
        self.components: dict[str, Component] = {}
        self.mlflow: MLflowManager | None = None

    def _init_env(self, run_name: str | None, tags: list[str] | None) -> None:
        set_seed(self.config.seed)

        # MLflow
        self.mlflow = create_mlflow_manager(self.config.model_dump())
        tag_map = {str(i): t for i, t in enumerate(tags or [])}
        self.mlflow.start_run(run_name=run_name or self.recipe.run.name, tags=tag_map)

        # Log config/recipe basics
        from src.utils.mlflow import auto_log_config

        auto_log_config(self.config.model_dump(), self.recipe.model_dump(by_alias=True))

    def _shutdown_env(self) -> None:
        if self.mlflow:
            self.mlflow.end_run("FINISHED")

    def _build_components(self) -> None:
        self.components = ComponentFactory.build_pipeline_components(
            self.recipe, self.config
        )

    def _load_models(self) -> dict[str, Any]:
        # Use model loader
        model_loader = self.components["model_loader"]
        model_loader.setup({})
        models = model_loader.run({"load_all_models": True})["models"]
        return models

    def _prepare_datasets(self) -> dict[str, Any]:
        datasets: dict[str, Any] = {}

        # Train sources
        for source in self.recipe.data.train.sources:
            loader = self.components[f"train_loader_{source}"]
            loader.setup({})
            ds = loader.run(
                {
                    "split": "train",
                    "max_length": self.recipe.data.train.max_length,
                    "add_solution": True,
                }
            )
            datasets[f"train_{source}"] = ds["dataset"]

        # Eval sources
        for source in self.recipe.data.eval.sources:
            loader = self.components[f"eval_loader_{source}"]
            loader.setup({})
            ds = loader.run(
                {
                    "split": "val",
                    "max_length": self.recipe.data.eval.max_length,
                    "add_solution": False,
                }
            )
            datasets[f"eval_{source}"] = ds["dataset"]

        return datasets

    def _tokenize_function(self, tokenizer, example: dict[str, Any]) -> dict[str, Any]:
        text = example.get("full_text") or example.get("prompt") or ""
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=self.recipe.data.train.max_length,
            padding=False,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    def _build_dataloader(self, dataset, tokenizer, split: str) -> DataLoader:
        # Tokenize with HF datasets caching enabled by default
        tokenized = dataset.map(
            lambda ex: self._tokenize_function(tokenizer, ex),
            remove_columns=dataset.column_names,
            desc=f"Tokenizing {split}",
            load_from_cache_file=True,
        )

        # Distributed sampler for training
        sampler = None
        try:
            import torch.distributed as dist

            if split == "train" and dist.is_available() and dist.is_initialized():
                sampler = DistributedSampler(tokenized, shuffle=True)
        except Exception:
            sampler = None

        # DataLoader
        return DataLoader(
            tokenized,
            batch_size=self.recipe.data.train.batch_size or 1,
            shuffle=(sampler is None and split == "train"),
            sampler=sampler,
            collate_fn=default_data_collator,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )

    def run(
        self,
        run_name: str | None = None,
        tags: list[str] | None = None,
        resume: bool = False,
        dry_run: bool = False,
        max_steps: int | None = None,
        verbose: bool = False,
    ) -> PipelineOutputs:
        self._init_env(run_name, tags)
        try:
            # Components
            self._build_components()

            # Models
            models = self._load_models()
            base = models["base"]["model"]
            tokenizer = models["base"]["tokenizer"]

            # Optional: ensure logits shape compliance; if model outputs [B,S,V], fallback to H=1
            horizon = self.recipe.model.mtp.horizon
            if hasattr(base, "config") and getattr(base.config, "mtp_heads", None) is None:
                if horizon > 1:
                    # Log a warning and fallback to H=1 behavior (trainer will validate)
                    from rich.console import Console

                    Console().print(
                        f"[yellow]Model does not declare MTP heads. Falling back to H=1 (requested H={horizon}).[/yellow]"
                    )
                    self.recipe.model.mtp.horizon = 1  # soft fallback in runtime object

            # Optimizer using factory (after model is ready)
            optimizer = ComponentFactory.create_optimizer(self.recipe, base.parameters())
            num_steps = max_steps or 0
            optimizer.setup({"num_training_steps": num_steps})

            # Datasets and dataloaders
            datasets = self._prepare_datasets()
            train_key = f"train_{self.recipe.data.train.sources[0]}"
            train_ds = datasets[train_key]["train"] if hasattr(datasets[train_key], "keys") else datasets[train_key]
            train_dl = self._build_dataloader(train_ds, tokenizer, "train")

            # Trainer
            trainer = self.components["trainer"]
            trainer.setup(
                {
                    "model": base,
                    "optimizer": optimizer,
                    "mlflow_manager": self.mlflow,
                }
            )

            if dry_run:
                return PipelineOutputs(trainer_metrics={"dry_run": True})

            metrics = trainer.run({"train_dataloader": train_dl, "max_steps": max_steps})
            return PipelineOutputs(trainer_metrics=metrics)
        finally:
            self._shutdown_env()


def get_pipeline(algo: str) -> type[TrainingPipeline]:
    # For now a single training pipeline supports both algos
    return TrainingPipeline


