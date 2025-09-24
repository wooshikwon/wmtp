#!/usr/bin/env python3
"""
컴포넌트 로딩 테스트 스크립트
세 알고리즘의 개별 컴포넌트들이 정상적으로 로딩되는지 확인
"""

import os
from pathlib import Path

# MLflow 환경변수 설정
os.environ["MLFLOW_TRACKING_URI"] = "./mlflow_runs"
os.environ["MLFLOW_REGISTRY_URI"] = "./mlflow_runs"

from rich.console import Console
from src.settings import load_config, load_recipe

console = Console()

def test_config_loading():
    """설정 파일 로딩 테스트"""
    console.print("[bold blue]1. 설정 파일 로딩 테스트[/bold blue]")

    try:
        config = load_config("configs/config.experiment.yaml")
        console.print(f"✅ 환경 설정 로딩 성공: {config.project}")
        console.print(f"   - 디바이스: {config.devices.compute_backend}")
        console.print(f"   - MLflow URI: {config.mlflow.tracking_uri}")
        return config
    except Exception as e:
        console.print(f"❌ 환경 설정 로딩 실패: {e}")
        return None

def test_recipe_loading():
    """레시피 파일들 로딩 테스트"""
    console.print("\n[bold blue]2. 레시피 파일 로딩 테스트[/bold blue]")

    recipes = {
        "MTP Baseline": "configs/recipe.baseline_quick.yaml",
        "Critic WMTP": "configs/recipe.critic_quick.yaml",
        "Rho-1 WMTP": "configs/recipe.rho1_quick.yaml"
    }

    loaded_recipes = {}
    for name, path in recipes.items():
        try:
            if Path(path).exists():
                recipe = load_recipe(path)
                console.print(f"✅ {name}: {recipe.train.algo}")
                loaded_recipes[name] = recipe
            else:
                console.print(f"❌ {name}: 파일 없음 ({path})")
        except Exception as e:
            console.print(f"❌ {name}: 로딩 실패 - {e}")

    return loaded_recipes

def test_component_creation():
    """컴포넌트 생성 테스트"""
    console.print("\n[bold blue]3. 컴포넌트 생성 테스트[/bold blue]")

    try:
        config = load_config("configs/config.experiment.yaml")
        from src.factory.component_factory import ComponentFactory

        factory = ComponentFactory()

        # MTP Native Loader 테스트
        console.print("🔧 MTP Native Loader 생성 테스트...")
        recipe = load_recipe("configs/recipe.baseline_quick.yaml")
        mtp_loader = factory.create_model_loader(config, recipe)
        console.print(f"✅ MTP Native Loader 생성 성공: {type(mtp_loader).__name__}")

        return True

    except Exception as e:
        console.print(f"❌ 컴포넌트 생성 실패: {e}")
        return False

def test_model_loading():
    """실제 모델 로딩 테스트"""
    console.print("\n[bold blue]4. 모델 로딩 테스트[/bold blue]")

    try:
        config = load_config("configs/config.experiment.yaml")
        from src.factory.component_factory import ComponentFactory

        factory = ComponentFactory()
        recipe = load_recipe("configs/recipe.baseline_quick.yaml")
        mtp_loader = factory.create_model_loader(config, recipe)

        model_path = "models/7b_1t_4"
        console.print(f"🔧 Facebook MTP 모델 로딩 시작: {model_path}")

        model = mtp_loader.load_model(model_path)
        console.print(f"✅ 모델 로딩 성공: {type(model).__name__}")
        console.print(f"   - 디바이스: {next(model.parameters()).device}")

        return True

    except Exception as e:
        console.print(f"❌ 모델 로딩 실패: {e}")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return False

def main():
    console.print("[bold green]🧪 WMTP 컴포넌트 로딩 진단 테스트[/bold green]")
    console.print("각 단계별로 컴포넌트들이 정상 동작하는지 확인합니다.\n")

    # 단계별 테스트
    results = []

    config = test_config_loading()
    results.append(config is not None)

    recipes = test_recipe_loading()
    results.append(len(recipes) > 0)

    component_ok = test_component_creation()
    results.append(component_ok)

    model_ok = test_model_loading()
    results.append(model_ok)

    # 결과 요약
    console.print(f"\n[bold blue]📊 테스트 결과 요약[/bold blue]")
    console.print(f"성공한 테스트: {sum(results)}/4")

    if all(results):
        console.print("[green]🎉 모든 컴포넌트가 정상 동작합니다![/green]")
        console.print("[dim]실험 실패 원인은 훈련 루프나 MLflow 로깅에 있을 수 있습니다.[/dim]")
    else:
        console.print("[red]⚠️ 일부 컴포넌트에 문제가 있습니다.[/red]")
        console.print("[dim]위의 오류 메시지를 확인하여 문제를 해결하세요.[/dim]")

if __name__ == "__main__":
    main()