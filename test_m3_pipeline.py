#!/usr/bin/env python3
"""Test WMTP Pipeline on MacBook M3.

This script tests the WMTP training pipeline using:
- Sheared-LLaMA-2.7B wrapped with MTP heads (instead of 7B MTP model)
- Tiny test dataset (5 samples)
- MPS (Metal Performance Shaders) for M3 GPU acceleration
- Minimal training steps (10 steps)

Usage:
    python test_m3_pipeline.py
    
    # Use even smaller model (distilgpt2, 82M params):
    python test_m3_pipeline.py --tiny
    
    # Dry run (no actual training):
    python test_m3_pipeline.py --dry-run
"""

import sys
import os
from pathlib import Path
import argparse
import torch
import warnings
from rich.console import Console
from rich.panel import Panel
from rich.progress import track

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

console = Console()

def check_environment():
    """Check if the environment is suitable for testing."""
    issues = []
    
    # Check PyTorch installation
    try:
        import torch
        console.print(f"✓ PyTorch version: {torch.__version__}")
    except ImportError:
        issues.append("PyTorch is not installed")
    
    # Check for MPS availability (M3 GPU)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        console.print("✓ MPS (Metal Performance Shaders) is available")
        console.print(f"  MPS device: {torch.backends.mps.is_built()}")
    else:
        console.print("⚠ MPS is not available, will use CPU")
        issues.append("MPS not available (expected on M3)")
    
    # Check memory
    try:
        import psutil
        total_memory = psutil.virtual_memory().total / (1024**3)
        available_memory = psutil.virtual_memory().available / (1024**3)
        console.print(f"✓ Total memory: {total_memory:.1f} GB")
        console.print(f"  Available memory: {available_memory:.1f} GB")
        
        if available_memory < 10:
            issues.append(f"Low available memory: {available_memory:.1f} GB")
    except ImportError:
        console.print("⚠ psutil not installed, cannot check memory")
    
    # Check transformers
    try:
        import transformers
        console.print(f"✓ Transformers version: {transformers.__version__}")
    except ImportError:
        issues.append("Transformers is not installed")
    
    return issues

def modify_config_for_tiny_model(config, recipe):
    """Modify configuration to use tiny model (distilgpt2)."""
    console.print("\n[yellow]Using tiny model (distilgpt2) for ultra-light testing[/yellow]")
    
    # Change to tiny model
    config.paths.models.base = "distilgpt2"
    config.paths.models.ref = "distilgpt2"
    
    # Reduce batch size and sequence length even more
    recipe.data.train.max_length = 64
    recipe.data.train.batch_size = 1
    recipe.data.eval.max_length = 64
    recipe.data.eval.batch_size = 1
    
    # Reduce training steps
    recipe.train.max_steps = 5
    
    console.print("  Model: distilgpt2 (82M parameters)")
    console.print("  Max sequence length: 64")
    console.print("  Training steps: 5")

def create_test_data_loader():
    """Create a simple test data loader that doesn't require HuggingFace datasets."""
    from src.components.base import BaseComponent
    from src.components.registry import loader_registry
    
    @loader_registry.register("test-data-loader", category="loader", version="1.0.0")
    class TestDataLoader(BaseComponent):
        """Simple test data loader."""
        
        def run(self, inputs):
            # Create minimal test dataset
            test_samples = [
                {"text": "def hello(): return 'world'", "labels": "def hello(): return 'world'"},
                {"text": "x = 1 + 2", "labels": "x = 1 + 2"},
                {"text": "print('test')", "labels": "print('test')"},
            ]
            
            # Return as simple list (will be tokenized later)
            return {"dataset": test_samples}
    
    return TestDataLoader

def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test WMTP Pipeline on M3")
    parser.add_argument("--tiny", action="store_true", help="Use tiny model (distilgpt2)")
    parser.add_argument("--dry-run", action="store_true", help="Dry run without training")
    parser.add_argument("--config", default="configs/config.m3_test.yaml", help="Config file")
    parser.add_argument("--recipe", default="configs/recipe.m3_test.yaml", help="Recipe file")
    args = parser.parse_args()
    
    console.print(Panel.fit(
        "[bold cyan]WMTP Pipeline Test on MacBook M3[/bold cyan]\n"
        "Testing with small model and dataset",
        title="🧪 Test Mode"
    ))
    
    # Check environment
    console.print("\n[bold]Checking environment...[/bold]")
    issues = check_environment()
    
    if issues:
        console.print("\n[red]Environment issues found:[/red]")
        for issue in issues:
            console.print(f"  • {issue}")
        
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            console.print("Exiting...")
            return
    
    # Load configurations
    console.print("\n[bold]Loading configurations...[/bold]")
    
    try:
        # Import here to avoid early import errors
        from src.settings import Config, Recipe
        import yaml

        # Load YAML files and validate with Pydantic models
        with open(args.config, 'r') as f:
            config_data = yaml.safe_load(f)
        config = Config.model_validate(config_data)

        with open(args.recipe, 'r') as f:
            recipe_data = yaml.safe_load(f)
        recipe = Recipe.model_validate(recipe_data)

        console.print(f"✓ Loaded config: {args.config}")
        console.print(f"✓ Loaded recipe: {args.recipe}")
    except Exception as e:
        console.print(f"[red]Failed to load configurations: {e}[/red]")
        import traceback
        traceback.print_exc()
        return
    
    # Modify for tiny model if requested
    if args.tiny:
        modify_config_for_tiny_model(config, recipe)
    
    # Override model loader to use test MTP wrapper
    console.print("\n[bold]Configuring test model loader...[/bold]")
    
    # Register test loaders if not already registered
    try:
        from src.components.loader.test_mtp_loader import TestMTPLoader, TinyMTPLoader
        console.print("✓ Test model loaders registered")
    except Exception as e:
        console.print(f"[yellow]Warning: Could not import test loaders: {e}[/yellow]")
    
    # Register test data loader
    create_test_data_loader()
    console.print("✓ Test data loader registered")
    
    # Set random seed
    from src.utils import set_seed
    set_seed(config.seed)
    console.print(f"✓ Random seed set to {config.seed}")
    
    # Prepare for training
    console.print("\n[bold]Pipeline Configuration:[/bold]")
    console.print(f"  Algorithm: {recipe.train.algo}")
    console.print(f"  Model: {config.paths.models.base}")
    console.print(f"  Device: {config.devices.compute_backend}")
    console.print(f"  Batch size: {recipe.data.train.batch_size}")
    console.print(f"  Max steps: {recipe.train.max_steps}")
    console.print(f"  Dry run: {args.dry_run}")
    
    # Confirm before starting
    if not args.dry_run:
        console.print("\n[yellow]Starting actual training (auto-confirmed for testing)...[/yellow]")
    
    # Run pipeline
    console.print("\n[bold green]Starting WMTP pipeline test...[/bold green]")
    
    try:
        # Now we can import normally since __init__.py is fixed
        from src.pipelines import run_training_pipeline

        console.print("✓ Training pipeline imported successfully")
        
        # Modify factory to use test loader
        from src.factory.component_factory import ComponentFactory
        
        # Monkey-patch the create_model_loader to use our test loader
        original_create_model_loader = ComponentFactory.create_model_loader
        
        def test_create_model_loader(config, recipe=None):
            """Override to use test MTP loader."""
            from src.components.loader.test_mtp_loader import TestMTPLoader, TinyMTPLoader
            
            if args.tiny:
                return TinyMTPLoader(config.model_dump())
            else:
                return TestMTPLoader(config.model_dump())
        
        ComponentFactory.create_model_loader = staticmethod(test_create_model_loader)
        
        # Run the training pipeline
        outputs = run_training_pipeline(
            config=config,
            recipe=recipe,
            dry_run=args.dry_run
        )
        
        # Restore original loader
        ComponentFactory.create_model_loader = original_create_model_loader
        
        console.print("\n[bold green]✓ Pipeline test completed successfully![/bold green]")
        
        # Print results
        if outputs and outputs.trainer_metrics:
            console.print("\n[bold]Training Metrics:[/bold]")
            for key, value in outputs.trainer_metrics.items():
                if isinstance(value, float):
                    console.print(f"  {key}: {value:.4f}")
                else:
                    console.print(f"  {key}: {value}")
    
    except Exception as e:
        console.print(f"\n[red]Pipeline test failed: {e}[/red]")
        console.print("\n[yellow]This is expected for initial testing.[/yellow]")
        console.print("Common issues:")
        console.print("  1. Model loading issues - check paths")
        console.print("  2. Memory issues - use --tiny flag")
        console.print("  3. MPS issues - may need to use CPU")
        
        import traceback
        console.print("\n[dim]Full traceback:[/dim]")
        traceback.print_exc()
        return
    
    console.print("\n[bold cyan]Test complete! The WMTP pipeline is working.[/bold cyan]")

if __name__ == "__main__":
    # Suppress some warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
    
    main()