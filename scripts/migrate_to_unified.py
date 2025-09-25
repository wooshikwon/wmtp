#!/usr/bin/env python
"""
WMTP 통합 시스템 마이그레이션 스크립트

Phase 4 완료 후 사용자가 기존 캐시 기반 시스템에서
새로운 S3 스트리밍 기반 통합 시스템으로 전환할 수 있도록 지원합니다.

주요 기능:
1. 기존 캐시 디렉토리 정리
2. 설정 파일 업데이트 가이드
3. S3 설정 검증
4. 기존 체크포인트 마이그레이션
5. 환경 호환성 검증

사용법:
    python scripts/migrate_to_unified.py --config configs/config.yaml [옵션]

옵션:
    --config: 설정 파일 경로 (필수)
    --cleanup: 기존 캐시 디렉토리 자동 정리
    --verify-only: 마이그레이션 검증만 수행
    --backup: 기존 설정 백업 생성
    --force: 확인 없이 실행
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

import yaml
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# WMTP components
try:
    from src.utils.s3 import create_s3_manager, S3Manager
    from src.utils.path_resolver import PathResolver, PathCategory
    from src.components.loader.unified_model_loader import UnifiedModelLoader
    from src.components.loader.unified_data_loader import UnifiedDataLoader
    from src.factory.component_factory import ComponentFactory
except ImportError as e:
    print(f"❌ WMTP 모듈을 import할 수 없습니다: {e}")
    print("현재 디렉토리에서 실행하거나 PYTHONPATH를 설정해주세요.")
    sys.exit(1)

console = Console()


class WMTPMigrationTool:
    """WMTP 마이그레이션 도구."""

    def __init__(self, config_path: Path):
        """
        마이그레이션 도구 초기화.

        Args:
            config_path: 설정 파일 경로
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.backup_dir = Path("./migration_backup")

        # 마이그레이션 상태
        self.migration_results = {
            "cache_cleanup": False,
            "config_update": False,
            "s3_verification": False,
            "checkpoint_migration": False,
            "system_verification": False
        }

    def _load_config(self) -> Dict[str, Any]:
        """설정 파일 로드."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            console.print(f"[red]❌ 설정 파일 로드 실패: {e}[/red]")
            sys.exit(1)

    def create_backup(self) -> bool:
        """기존 설정 백업 생성."""
        try:
            self.backup_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_config = self.backup_dir / f"config_backup_{timestamp}.yaml"

            shutil.copy2(self.config_path, backup_config)
            console.print(f"[green]✓ 설정 백업 생성: {backup_config}[/green]")
            return True
        except Exception as e:
            console.print(f"[red]❌ 백업 생성 실패: {e}[/red]")
            return False

    def cleanup_cache_directories(self, force: bool = False) -> bool:
        """기존 캐시 디렉토리 정리."""
        console.print("\n[bold]1. 캐시 디렉토리 정리[/bold]")

        # 정리할 캐시 디렉토리들
        cache_dirs = [
            Path(".cache"),
            Path("./cache"),
            Path("./checkpoints/.cache"),
            Path("./models/.cache"),
            Path("./data/.cache")
        ]

        # 설정에서 캐시 경로 찾기
        if "cache" in self.config:
            cache_dirs.append(Path(self.config["cache"]))

        paths_config = self.config.get("paths", {})
        if "cache" in paths_config:
            cache_dirs.append(Path(paths_config["cache"]))

        cleaned_dirs = []
        total_size = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("캐시 디렉토리 검사 중...", total=None)

            for cache_dir in cache_dirs:
                if cache_dir.exists():
                    # 디렉토리 크기 계산
                    size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
                    total_size += size
                    cleaned_dirs.append((cache_dir, size))

            progress.update(task, description=f"{len(cleaned_dirs)}개 캐시 디렉토리 발견")

        if not cleaned_dirs:
            console.print("[green]✓ 정리할 캐시 디렉토리가 없습니다.[/green]")
            return True

        # 정리할 디렉토리 목록 표시
        table = Table(title="정리할 캐시 디렉토리")
        table.add_column("경로", style="cyan")
        table.add_column("크기", justify="right", style="magenta")

        for cache_dir, size in cleaned_dirs:
            size_mb = size / (1024 * 1024)
            table.add_row(str(cache_dir), f"{size_mb:.1f} MB")

        table.add_row("[bold]합계[/bold]", f"[bold]{total_size / (1024 * 1024):.1f} MB[/bold]")
        console.print(table)

        # 사용자 확인
        if not force:
            confirm = console.input(f"\n{len(cleaned_dirs)}개 디렉토리 ({total_size / (1024 * 1024):.1f} MB)를 삭제하시겠습니까? [y/N]: ")
            if confirm.lower() != 'y':
                console.print("[yellow]캐시 정리를 건너뜁니다.[/yellow]")
                return False

        # 디렉토리 삭제
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("캐시 디렉토리 삭제 중...", total=len(cleaned_dirs))

                for i, (cache_dir, _) in enumerate(cleaned_dirs):
                    shutil.rmtree(cache_dir)
                    progress.update(task, advance=1, description=f"삭제됨: {cache_dir.name}")

            console.print(f"[green]✓ {len(cleaned_dirs)}개 캐시 디렉토리 정리 완료[/green]")
            return True

        except Exception as e:
            console.print(f"[red]❌ 캐시 정리 실패: {e}[/red]")
            return False

    def update_config_format(self) -> bool:
        """설정 파일 형식 업데이트 가이드."""
        console.print("\n[bold]2. 설정 파일 업데이트 가이드[/bold]")

        # 구 설정 형식 확인
        old_format_found = []

        # cache 섹션 확인
        if "cache" in self.config:
            old_format_found.append("cache 섹션")

        paths = self.config.get("paths", {})
        if "cache" in paths:
            old_format_found.append("paths.cache")

        # 개별 로더 설정에서 cache_dir 확인
        loaders = self.config.get("loaders", {})
        for loader_name, loader_config in loaders.items():
            if isinstance(loader_config, dict) and "cache_dir" in loader_config:
                old_format_found.append(f"loaders.{loader_name}.cache_dir")

        if not old_format_found:
            console.print("[green]✓ 설정 파일이 이미 새로운 형식입니다.[/green]")
            return True

        # 업데이트 가이드 표시
        panel_content = """
[bold red]발견된 구 설정 형식:[/bold red]
"""
        for item in old_format_found:
            panel_content += f"• {item}\n"

        panel_content += """
[bold green]권장 새 설정 형식:[/bold green]

[cyan]# 캐시 설정 제거[/cyan]
# cache: ./cache  # 삭제
# paths:
#   cache: ./cache  # 삭제

[cyan]# S3 스트리밍 설정 추가[/cyan]
storage:
  mode: "hybrid"  # local, s3, hybrid
  s3:
    bucket: "your-wmtp-bucket"
    region: "us-east-1"

[cyan]# 경로 설정 단순화[/cyan]
paths:
  models: "./models"
  datasets: "./data"
  checkpoints: "./checkpoints"

[cyan]# 로더 설정에서 cache_dir 제거[/cyan]
loaders:
  unified-model:
    type: "unified-model"
    # cache_dir: ./cache  # 삭제
  unified-data:
    type: "unified-data"
    # cache_dir: ./cache  # 삭제
"""

        console.print(Panel(panel_content, title="설정 업데이트 가이드", border_style="blue"))

        console.print("[yellow]⚠️  수동으로 설정 파일을 업데이트해주세요.[/yellow]")
        return False

    def verify_s3_configuration(self) -> bool:
        """S3 설정 검증."""
        console.print("\n[bold]3. S3 설정 검증[/bold]")

        # S3 설정 확인
        storage_config = self.config.get("storage", {})
        if not storage_config:
            console.print("[red]❌ storage 설정이 없습니다.[/red]")
            return False

        s3_config = storage_config.get("s3", {})
        if not s3_config:
            console.print("[yellow]⚠️  S3 설정이 없습니다. 로컬 모드로 동작합니다.[/yellow]")
            return True

        # S3Manager 생성 및 연결 테스트
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("S3 연결 테스트 중...", total=None)

                s3_manager = create_s3_manager(self.config)
                if s3_manager and s3_manager.connected:
                    console.print("[green]✓ S3 연결 성공[/green]")

                    # 버킷 접근 테스트
                    progress.update(task, description="S3 버킷 접근 테스트 중...")
                    bucket = s3_config.get("bucket")
                    if bucket:
                        # 간단한 리스트 테스트
                        try:
                            s3_manager.s3_client.head_bucket(Bucket=bucket)
                            console.print(f"[green]✓ 버킷 접근 가능: {bucket}[/green]")
                            return True
                        except Exception as e:
                            console.print(f"[red]❌ 버킷 접근 실패: {e}[/red]")
                            return False
                else:
                    console.print("[red]❌ S3 연결 실패[/red]")
                    return False

        except Exception as e:
            console.print(f"[red]❌ S3 검증 실패: {e}[/red]")
            return False

    def migrate_checkpoints(self) -> bool:
        """기존 체크포인트 마이그레이션."""
        console.print("\n[bold]4. 체크포인트 마이그레이션[/bold]")

        # 체크포인트 디렉토리 확인
        checkpoint_dirs = [
            Path("./checkpoints"),
            Path("./cache/checkpoints"),
            Path("./.cache/checkpoints")
        ]

        # 설정에서 체크포인트 경로 찾기
        paths = self.config.get("paths", {})
        if "checkpoints" in paths:
            checkpoint_dirs.append(Path(paths["checkpoints"]))

        found_checkpoints = []

        for checkpoint_dir in checkpoint_dirs:
            if checkpoint_dir.exists():
                # .pth, .pt, .safetensors 파일 찾기
                for ext in ['*.pth', '*.pt', '*.safetensors']:
                    found_checkpoints.extend(checkpoint_dir.rglob(ext))

        if not found_checkpoints:
            console.print("[green]✓ 마이그레이션할 체크포인트가 없습니다.[/green]")
            return True

        # 체크포인트 목록 표시
        table = Table(title="발견된 체크포인트")
        table.add_column("파일", style="cyan")
        table.add_column("크기", justify="right", style="magenta")
        table.add_column("위치", style="yellow")

        for checkpoint in found_checkpoints:
            size_mb = checkpoint.stat().st_size / (1024 * 1024)
            table.add_row(checkpoint.name, f"{size_mb:.1f} MB", str(checkpoint.parent))

        console.print(table)

        # 체크포인트는 자동 마이그레이션하지 않고 가이드만 제공
        guide_content = """
[bold]체크포인트 마이그레이션 가이드:[/bold]

1. [cyan]로컬 사용[/cyan]: 체크포인트를 ./checkpoints/ 디렉토리로 이동
2. [cyan]S3 업로드[/cyan]: MLflow 또는 직접 S3 업로드 사용

[bold]MLflow를 사용한 업로드 예시:[/bold]
```python
import mlflow.pytorch
mlflow.pytorch.log_model(model, "models/my-model")
```

[bold]직접 S3 업로드 예시:[/bold]
```python
from src.utils.s3 import create_s3_manager
s3_manager = create_s3_manager(config)
with open("checkpoint.pth", "rb") as f:
    s3_manager.upload_from_bytes(
        f.read(),
        "checkpoints/my-checkpoint.pth"
    )
```
"""

        console.print(Panel(guide_content, title="체크포인트 마이그레이션", border_style="blue"))
        console.print("[yellow]⚠️  체크포인트는 수동으로 마이그레이션해주세요.[/yellow]")
        return False

    def verify_system_integration(self) -> bool:
        """시스템 통합 검증."""
        console.print("\n[bold]5. 시스템 통합 검증[/bold]")

        verification_results = {}

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                # PathResolver 테스트
                task = progress.add_task("PathResolver 검증 중...", total=5)

                try:
                    resolver = PathResolver(self.config)
                    verification_results["PathResolver"] = "✓"
                    progress.update(task, advance=1)
                except Exception as e:
                    verification_results["PathResolver"] = f"❌ {str(e)[:50]}"

                # ComponentFactory 테스트
                progress.update(task, description="ComponentFactory 검증 중...")
                try:
                    factory = ComponentFactory()
                    verification_results["ComponentFactory"] = "✓"
                    progress.update(task, advance=1)
                except Exception as e:
                    verification_results["ComponentFactory"] = f"❌ {str(e)[:50]}"

                # UnifiedLoaders 테스트
                progress.update(task, description="UnifiedLoaders 검증 중...")
                try:
                    model_loader = UnifiedModelLoader(self.config)
                    data_loader = UnifiedDataLoader(self.config)
                    verification_results["UnifiedLoaders"] = "✓"
                    progress.update(task, advance=1)
                except Exception as e:
                    verification_results["UnifiedLoaders"] = f"❌ {str(e)[:50]}"

                # S3Manager 테스트
                progress.update(task, description="S3Manager 검증 중...")
                try:
                    s3_manager = create_s3_manager(self.config)
                    verification_results["S3Manager"] = "✓"
                    progress.update(task, advance=1)
                except Exception as e:
                    verification_results["S3Manager"] = f"❌ {str(e)[:50]}"

                # 캐시 제거 확인
                progress.update(task, description="캐시 제거 검증 중...")
                try:
                    from src.components.loader.base_loader import BaseLoader
                    import inspect

                    # cache_dir 파라미터 제거 확인
                    sig = inspect.signature(BaseLoader.__init__)
                    if "cache_dir" not in sig.parameters:
                        verification_results["Cache Removal"] = "✓"
                    else:
                        verification_results["Cache Removal"] = "❌ cache_dir still exists"
                    progress.update(task, advance=1)
                except Exception as e:
                    verification_results["Cache Removal"] = f"❌ {str(e)[:50]}"

        except Exception as e:
            console.print(f"[red]❌ 검증 실패: {e}[/red]")
            return False

        # 검증 결과 표시
        table = Table(title="시스템 통합 검증 결과")
        table.add_column("컴포넌트", style="cyan")
        table.add_column("상태", justify="center")

        all_passed = True
        for component, result in verification_results.items():
            style = "green" if result == "✓" else "red"
            table.add_row(component, f"[{style}]{result}[/{style}]")
            if result != "✓":
                all_passed = False

        console.print(table)

        if all_passed:
            console.print("[green]✓ 모든 시스템 통합 검증 통과[/green]")
        else:
            console.print("[red]❌ 일부 시스템 검증 실패[/red]")

        return all_passed

    def run_migration(self, cleanup: bool = False, verify_only: bool = False,
                     backup: bool = True, force: bool = False) -> bool:
        """마이그레이션 실행."""
        console.print(Panel.fit(
            "[bold blue]WMTP 통합 시스템 마이그레이션[/bold blue]\n"
            f"설정 파일: {self.config_path}",
            border_style="blue"
        ))

        # 백업 생성
        if backup and not verify_only:
            if not self.create_backup():
                if not force:
                    return False

        migration_steps = []

        # 검증만 수행하는 경우
        if verify_only:
            migration_steps = [
                ("S3 설정 검증", self.verify_s3_configuration),
                ("시스템 통합 검증", self.verify_system_integration)
            ]
        else:
            # 전체 마이그레이션 수행
            if cleanup:
                migration_steps.append(("캐시 정리", lambda: self.cleanup_cache_directories(force)))

            migration_steps.extend([
                ("설정 업데이트", self.update_config_format),
                ("S3 설정 검증", self.verify_s3_configuration),
                ("체크포인트 마이그레이션", self.migrate_checkpoints),
                ("시스템 통합 검증", self.verify_system_integration)
            ])

        # 마이그레이션 단계 실행
        total_success = 0
        for step_name, step_func in migration_steps:
            try:
                success = step_func()
                if success:
                    total_success += 1
                    self.migration_results[step_name.lower().replace(" ", "_")] = True
            except Exception as e:
                console.print(f"[red]❌ {step_name} 실행 중 오류: {e}[/red]")

        # 최종 결과
        console.print("\n" + "=" * 80)
        if total_success == len(migration_steps):
            console.print("[green]🎉 마이그레이션이 성공적으로 완료되었습니다![/green]")
            console.print("[green]📦 WMTP 통합 시스템을 사용할 준비가 되었습니다.[/green]")
            return True
        else:
            console.print(f"[yellow]⚠️  마이그레이션이 부분적으로 완료되었습니다. ({total_success}/{len(migration_steps)})[/yellow]")
            console.print("[yellow]위의 가이드를 참고하여 수동으로 완료해주세요.[/yellow]")
            return False


def main():
    """메인 함수."""
    parser = argparse.ArgumentParser(
        description="WMTP 통합 시스템 마이그레이션 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python scripts/migrate_to_unified.py --config configs/config.yaml
  python scripts/migrate_to_unified.py --config configs/config.yaml --cleanup --backup
  python scripts/migrate_to_unified.py --config configs/config.yaml --verify-only
        """
    )

    parser.add_argument("--config", type=Path, required=True,
                       help="설정 파일 경로")
    parser.add_argument("--cleanup", action="store_true",
                       help="기존 캐시 디렉토리 자동 정리")
    parser.add_argument("--verify-only", action="store_true",
                       help="마이그레이션 검증만 수행")
    parser.add_argument("--backup", action="store_true", default=True,
                       help="기존 설정 백업 생성 (기본값)")
    parser.add_argument("--force", action="store_true",
                       help="확인 없이 실행")

    args = parser.parse_args()

    # 설정 파일 존재 확인
    if not args.config.exists():
        console.print(f"[red]❌ 설정 파일을 찾을 수 없습니다: {args.config}[/red]")
        return 1

    # 마이그레이션 도구 생성 및 실행
    try:
        migration_tool = WMTPMigrationTool(args.config)
        success = migration_tool.run_migration(
            cleanup=args.cleanup,
            verify_only=args.verify_only,
            backup=args.backup,
            force=args.force
        )
        return 0 if success else 1

    except KeyboardInterrupt:
        console.print("\n[yellow]마이그레이션이 중단되었습니다.[/yellow]")
        return 1
    except Exception as e:
        console.print(f"[red]❌ 마이그레이션 실행 중 오류: {e}[/red]")
        return 1


if __name__ == "__main__":
    sys.exit(main())