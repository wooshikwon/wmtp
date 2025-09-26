#!/usr/bin/env python3
"""
WMTP Local Model Standardization Only
로컬 테스트 모델들만 표준화하는 안전한 버전
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()


def standardize_local_model(
    source_path: Path, target_path: Path, wmtp_type: str, model_info: dict[str, Any]
):
    """로컬 모델을 표준 형식으로 변환"""
    console.print(f"[blue]Standardizing: {source_path} → {target_path}[/blue]")

    if not source_path.exists():
        console.print(f"[red]❌ Source path does not exist: {source_path}[/red]")
        return

    target_path.mkdir(parents=True, exist_ok=True)

    # 1. 기존 HuggingFace 파일들 복사
    required_files = [
        "config.json",
        "model.safetensors",
        "pytorch_model.bin",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "vocab.json",
        "merges.txt",
    ]

    for file_name in required_files:
        source_file = source_path / file_name
        if source_file.exists():
            shutil.copy2(source_file, target_path / file_name)
            console.print(f"  ✅ Copied {file_name}")

    # 2. WMTP 메타데이터 생성
    metadata = {
        "wmtp_type": wmtp_type,
        "training_algorithm": model_info.get("training_algorithm", "unknown"),
        "horizon": model_info.get("horizon", 4),
        "n_heads": model_info.get("n_heads", 4),
        "base_architecture": model_info.get("architecture", "unknown"),
        "model_size": model_info.get("size", "unknown"),
        "storage_version": "2.0",
        "created_by": "wmtp_local_standardizer",
        "standardization_date": datetime.now().isoformat(),
        "original_format": model_info.get("original_format", "unknown"),
        "source_path": str(source_path),
    }

    with open(target_path / "wmtp_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    console.print("  ✅ Generated wmtp_metadata.json")

    # 3. MLflow 호환성 파일들
    mlflow_dir = target_path / "mlflow_model"
    mlflow_dir.mkdir(exist_ok=True)

    # requirements.txt
    with open(mlflow_dir / "requirements.txt", "w") as f:
        f.write("torch>=2.0.0\ntransformers>=4.40.0\nsafetensors>=0.4.0\n")

    # conda.yaml
    conda_env = {
        "channels": ["conda-forge", "pytorch"],
        "dependencies": ["python=3.12", "pytorch>=2.0.0"],
        "name": f"wmtp_{wmtp_type}_env",
    }

    import yaml

    with open(mlflow_dir / "conda.yaml", "w") as f:
        yaml.dump(conda_env, f)

    console.print("  ✅ Generated MLflow files")
    console.print(f"[green]✅ Completed: {target_path}[/green]\n")


def main():
    console.print("[bold blue]WMTP Local Model Standardization[/bold blue]")

    # 로컬 테스트 모델들 표준화 설정
    local_models = [
        {
            "source": Path("tests/tiny_models/distilgpt2-mtp"),
            "target": Path("tests/tiny_models_v2/distilgpt2-mtp"),
            "wmtp_type": "mtp_huggingface",
            "info": {
                "architecture": "gpt2",
                "size": "120m",
                "training_algorithm": "mtp",
                "horizon": 4,
                "n_heads": 4,
                "original_format": "huggingface_safetensors",
            },
        },
        {
            "source": Path("tests/tiny_models/tiny-reward-model"),
            "target": Path("tests/tiny_models_v2/tiny-reward-model"),
            "wmtp_type": "reward_model",
            "info": {
                "architecture": "gpt2",
                "size": "120m",
                "training_algorithm": "reward_modeling",
                "original_format": "huggingface_safetensors",
            },
        },
        {
            "source": Path("tests/tiny_models/distilgpt2"),
            "target": Path("tests/tiny_models_v2/distilgpt2"),
            "wmtp_type": "base_model",
            "info": {
                "architecture": "gpt2",
                "size": "120m",
                "training_algorithm": "base_training",
                "original_format": "huggingface_safetensors",
            },
        },
    ]

    # 표준화 실행
    for model in local_models:
        try:
            standardize_local_model(
                model["source"], model["target"], model["wmtp_type"], model["info"]
            )
        except Exception as e:
            console.print(f"[red]❌ Failed to standardize {model['source']}: {e}[/red]")

    console.print("[bold green]🎉 Local model standardization completed![/bold green]")


if __name__ == "__main__":
    main()
