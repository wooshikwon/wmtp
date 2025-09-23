"""
Minimal training pipeline (assembly-only).

Builds components via factory, loads models/datasets, optionally runs
critic Stage1 pretrainer, then runs the main trainer. Branching logic is
kept minimal and delegated to factory/registered modules as much as possible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import default_data_collator

from src.factory.component_factory import ComponentFactory
from src.settings import Config, Recipe
from src.utils import create_mlflow_manager, set_seed


@dataclass
class RunOutputs:
    trainer_metrics: dict[str, Any]


def run_training_pipeline(
    config: Config,
    recipe: Recipe,
    run_name: str | None = None,
    tags: list[str] | None = None,
    dry_run: bool = False,
    max_steps: int | None = None,
) -> RunOutputs:
    set_seed(config.seed)

    mlflow = create_mlflow_manager(config.model_dump())
    tag_map = {str(i): t for i, t in enumerate(tags or [])}
    mlflow.start_run(run_name=run_name or recipe.run.name, tags=tag_map)

    # Create components via factory (no bulk builder)
    model_loader = ComponentFactory.create_model_loader(config)
    model_loader.setup({})
    models = model_loader.run({"load_all_models": True})["models"]
    base = models["base"]["model"]
    tokenizer = models["base"]["tokenizer"]
    ref_model = (
        models.get("ref", {}).get("model")
        if isinstance(models.get("ref"), dict)
        else None
    )
    rm_model = (
        models.get("rm", {}).get("model")
        if isinstance(models.get("rm"), dict)
        else None
    )

    optimizer = ComponentFactory.create_optimizer(recipe, base.parameters())
    optimizer.setup({"num_training_steps": max_steps or 0})

    # Optional: wrap base with MTPWrapper when mtp_heads is missing but H>1 requested
    try:
        horizon = recipe.model.mtp.horizon
        if (
            hasattr(base, "config")
            and getattr(base.config, "mtp_heads", None) is None
            and horizon > 1
        ):
            from rich.console import Console

            from src.components.models.mtp_wrapper import MTPWrapper

            Console().print(
                f"[yellow]Model lacks native MTP heads. Applying MTPWrapper for H={horizon} (teacher-forcing emulation).[/yellow]"
            )
            base = MTPWrapper(base, horizon=horizon)
    except Exception:
        pass

    # Build datasets via loaders
    train_source = recipe.data.train.sources[0]
    train_loader_comp = ComponentFactory.create_data_loader(train_source, config)
    train_loader_comp.setup({})
    train_ds = train_loader_comp.run(
        {
            "split": "train",
            "max_length": recipe.data.train.max_length,
            "add_solution": True,
        }
    )["dataset"]

    # Tokenize
    def _tokenize_function(example: dict[str, Any]) -> dict[str, Any]:
        text = example.get("full_text") or example.get("prompt") or ""
        tok = tokenizer(
            text,
            truncation=True,
            max_length=recipe.data.train.max_length,
            padding=False,
        )
        tok["labels"] = tok["input_ids"].copy()
        return tok

    tokenized = train_ds.map(
        _tokenize_function,
        remove_columns=train_ds.column_names,
        desc="Tokenizing train",
        load_from_cache_file=True,
    )

    # Sampler
    sampler = None
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            sampler = DistributedSampler(tokenized, shuffle=True)
    except Exception:
        sampler = None

    train_dl = DataLoader(
        tokenized,
        batch_size=recipe.data.train.batch_size or 1,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=default_data_collator,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    # Optional Stage1 (critic-wmtp)
    if recipe.train.algo == "critic-wmtp" and rm_model is not None and not dry_run:
        from pathlib import Path

        from src.components.registry import trainer_registry

        pre_cfg = {
            "target": getattr(recipe.critic, "target", "rm_sequence")
            if hasattr(recipe, "critic")
            else "rm_sequence",
            "token_spread": getattr(recipe.critic, "token_spread", "gae")
            if hasattr(recipe, "critic")
            else "gae",
            "delta_mode": getattr(recipe.critic, "delta_mode", "td")
            if hasattr(recipe, "critic")
            else "td",
            "normalize": getattr(recipe.critic, "normalize", "zscore")
            if hasattr(recipe, "critic")
            else "zscore",
            "temperature": recipe.loss.temperature,
            "lr": 1e-4,
        }
        pretrainer = trainer_registry.create("critic-stage1-pretrainer-v1", pre_cfg)
        cache_root = (
            Path(config.paths.cache) / "critic" / (recipe.run.name or "default")
        )
        pretrainer.setup({})
        pretrainer.run(
            {
                "base_model": base,
                "rm_model": rm_model,
                "train_dataloader": train_dl,
                "cache_root": cache_root,
            }
        )

    # Trainer
    scorer = ComponentFactory.create_scorer(recipe)
    # Provide value_head checkpoint path to scorer (if exists)
    try:
        from pathlib import Path

        vh_path = (
            Path(config.paths.cache)
            / "critic"
            / (recipe.run.name or "default")
            / "value_head.pt"
        )
        if vh_path.exists():
            scorer.setup({"value_head_path": vh_path})
        else:
            scorer.setup({})
    except Exception:
        scorer.setup({})
    trainer = ComponentFactory.create_trainer(recipe, config, scorer)
    trainer.setup(
        {
            "model": base,
            "optimizer": optimizer,
            "mlflow_manager": mlflow,
            "ref_model": ref_model,
            "base_tokenizer": tokenizer,
            "rm_model": rm_model,
        }
    )

    if dry_run:
        mlflow.end_run("FINISHED")
        return RunOutputs(trainer_metrics={"dry_run": True})

    metrics = trainer.run({"train_dataloader": train_dl, "max_steps": max_steps})
    mlflow.end_run("FINISHED")
    return RunOutputs(trainer_metrics=metrics)
