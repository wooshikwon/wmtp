"""
WMTP 설정 로더: YAML 파일 검증 및 로딩 시스템

WMTP 연구 맥락:
WMTP 실험은 config.yaml(환경)과 recipe.yaml(알고리즘) 두 설정 파일로 제어됩니다.
이 모듈은 두 파일을 로드하고, Pydantic 스키마로 검증하며,
환경 변수 치환과 상세한 오류 메시지를 제공합니다.

핵심 기능:
- YAML 파일 로드 및 파싱
- 환경 변수 치환 (${VAR_NAME} 패턴 지원)
- Pydantic 스키마 검증 (타입 체크, 필수 필드 확인)
- 친화적 오류 메시지 출력 (Rich 라이브러리 활용)
- 알고리즘별 설정 일관성 검증

WMTP 알고리즘과의 연결:
- Baseline: 최소 설정으로 실행 가능
- Critic-WMTP: rm_id, critic 설정 검증 강화
- Rho1-WMTP: ref_id, rho1 설정 검증 강화

사용 예시:
    >>> from src.settings.loader import load_config, load_recipe
    >>>
    >>> # 설정 파일 로드
    >>> config = load_config("configs/config.yaml")
    >>> recipe = load_recipe("configs/recipe_rho1.yaml")
    >>>
    >>> # 설정 활용
    >>> if config.storage.mode == "s3":
    >>>     print(f"S3 버킷: {config.storage.s3.bucket}")
    >>> print(f"알고리즘: {recipe.train.algo}")

성능 최적화:
- 설정 파일은 한 번만 로드되어 메모리에 캐시
- 환경 변수는 런타임에 동적 치환
- 검증 실패 시 조기 종료로 불필요한 연산 방지

디버깅 팁:
- ValidationError: 필수 필드 누락 또는 타입 불일치
- ConfigurationError: 파일 없음 또는 YAML 문법 오류
- 환경 변수 오류: ${VAR_NAME:-default} 형식으로 기본값 제공
- 상세 오류: Rich 패널에서 정확한 위치와 원인 확인
"""

import os
import re
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from .config_schema import Config, S3AuthConfig
from .recipe_schema import Recipe

console = Console()
T = TypeVar("T", bound=BaseModel)


class ConfigurationError(Exception):
    """설정 오류 전용 예외 클래스.

    WMTP 설정 파일 로딩/검증 중 발생하는 모든 오류를 처리합니다.
    상세한 오류 메시지로 디버깅을 용이하게 합니다.
    """

    pass


def substitute_env_vars(data: Any) -> Any:
    """
    환경 변수 치환: 설정 데이터에서 환경 변수를 실제 값으로 교체

    WMTP 연구 맥락:
    클러스터 환경(VESSL)과 로컬 개발을 동일한 설정 파일로 지원하기 위해
    환경 변수를 활용합니다. S3 자격증명, MLflow URI 등이 대표적입니다.

    지원 패턴:
    - ${VAR_NAME}: 환경 변수 필수 (없으면 오류)
    - ${VAR_NAME:-default}: 기본값 제공 (없어도 동작)

    구체적 동작:
    1. 데이터 구조를 재귀적으로 탐색
    2. 문자열에서 ${} 패턴 검색
    3. 환경 변수 값으로 치환
    4. 없으면 기본값 사용 또는 오류 발생

    매개변수:
        data: 설정 데이터 (dict, list, str 등 모든 타입)
            - dict: 모든 값에 대해 재귀 적용
            - list: 모든 요소에 대해 재귀 적용
            - str: 환경 변수 패턴 치환

    반환값:
        Any: 환경 변수가 치환된 데이터

    예시:
        >>> # 설정에서
        >>> mlflow:
        >>>   tracking_uri: ${MLFLOW_TRACKING_URI:-./mlruns}
        >>>
        >>> # 환경 변수 설정 후
        >>> os.environ['MLFLOW_TRACKING_URI'] = 's3://bucket/mlflow'
        >>> # 결과: tracking_uri = 's3://bucket/mlflow'

    WMTP 활용 예:
        - ${AWS_PROFILE}: S3 접근용 AWS 프로필
        - ${MLFLOW_TRACKING_URI}: 실험 추적 서버
        - ${CUDA_VISIBLE_DEVICES}: GPU 선택

    디버깅 팁:
        - ConfigurationError: 필수 환경 변수 미설정
        - export VAR_NAME=value 또는 .env 파일 사용
        - 기본값 제공으로 오류 방지 권장
    """
    if isinstance(data, dict):
        return {key: substitute_env_vars(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [substitute_env_vars(item) for item in data]
    elif isinstance(data, str):
        # Pattern for ${VAR_NAME} or ${VAR_NAME:-default}
        pattern = r"\$\{([^}]+)\}"

        def replace_env_var(match):
            var_expr = match.group(1)

            # Check for default value pattern: VAR_NAME:-default
            if ":-" in var_expr:
                var_name, default_value = var_expr.split(":-", 1)
                return os.getenv(var_name.strip(), default_value.strip())
            else:
                # Simple variable substitution
                var_name = var_expr.strip()
                env_value = os.getenv(var_name)
                if env_value is None:
                    raise ConfigurationError(
                        f"Environment variable '{var_name}' not found and no default provided"
                    )
                return env_value

        return re.sub(pattern, replace_env_var, data)
    else:
        return data


def load_yaml(path: str | Path) -> dict[str, Any]:
    """
    Load a YAML file and return as dictionary with environment variable substitution.

    Args:
        path: Path to YAML file

    Returns:
        Dictionary containing YAML contents with environment variables substituted

    Raises:
        ConfigurationError: If file cannot be loaded or parsed
    """
    path = Path(path)

    if not path.exists():
        raise ConfigurationError(f"Configuration file not found: {path}")

    if not path.is_file():
        raise ConfigurationError(f"Path is not a file: {path}")

    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if data is None:
            raise ConfigurationError(f"Empty configuration file: {path}")

        # Substitute environment variables
        data = substitute_env_vars(data)

        return data

    except yaml.YAMLError as e:
        raise ConfigurationError(f"Failed to parse YAML file {path}: {e}")
    except Exception as e:
        raise ConfigurationError(f"Failed to load configuration from {path}: {e}")


def validate_schema(data: dict[str, Any], schema: type[T], name: str = "config") -> T:
    """
    Validate data against a Pydantic schema with detailed error reporting.

    Args:
        data: Dictionary to validate
        schema: Pydantic model class
        name: Configuration name for error messages

    Returns:
        Validated Pydantic model instance

    Raises:
        ConfigurationError: If validation fails
    """
    try:
        return schema(**data)
    except ValidationError as e:
        # Format detailed error message
        error_messages = []
        for error in e.errors():
            loc = " → ".join(str(x) for x in error["loc"])
            msg = error["msg"]
            error_messages.append(f"  • {loc}: {msg}")

        error_detail = "\n".join(error_messages)
        raise ConfigurationError(
            f"Validation failed for {name}:\n{error_detail}\n\n"
            f"Please check your configuration against the schema requirements."
        )
    except Exception as e:
        raise ConfigurationError(f"Unexpected error validating {name}: {e}")


def migrate_storage_config(data: dict[str, Any]) -> dict[str, Any]:
    """
    Phase 2 하위 호환성: 기존 storage 설정을 새 형식으로 마이그레이션

    기존 형식:
        storage:
          mode: s3
          s3:
            bucket: wmtp
            prefix: checkpoints
        paths:
          models:
            base: models/7b_1t_4

    새 형식:
        s3_auth:
          default_bucket: wmtp
          region: ap-northeast-2
        paths:
          models:
            base: s3://wmtp/checkpoints/models/7b_1t_4

    Args:
        data: 원본 config 딕셔너리

    Returns:
        마이그레이션된 config 딕셔너리
    """
    # 이미 새 형식이면 그대로 반환
    if 'storage' not in data:
        return data

    storage = data.get('storage', {})
    storage_mode = storage.get('mode', 'local')

    # storage를 s3_auth로 변환
    if storage_mode in ['s3', 'auto'] and 's3' in storage:
        s3_config = storage['s3']
        data['s3_auth'] = {
            'default_bucket': s3_config.get('bucket'),
            'region': s3_config.get('region', 'ap-northeast-2')
        }

        # S3 모드일 때 경로에 프로토콜 추가
        if storage_mode == 's3':
            bucket = s3_config.get('bucket', '')
            prefix = s3_config.get('prefix', '')

            # paths 섹션의 모든 경로에 s3:// 프리픽스 추가
            if 'paths' in data:
                for category, paths in data['paths'].items():
                    if isinstance(paths, dict):
                        for key, path in paths.items():
                            if path and not path.startswith(('s3://', 'file://', '/')):
                                # 상대 경로를 S3 URI로 변환
                                if prefix:
                                    data['paths'][category][key] = f"s3://{bucket}/{prefix}/{path}"
                                else:
                                    data['paths'][category][key] = f"s3://{bucket}/{path}"

    # auto 모드 처리 - 경로는 그대로 유지 (PathResolver가 처리)
    elif storage_mode == 'auto':
        # s3_auth는 위에서 이미 설정됨
        # 경로는 수정하지 않음 (사용자가 이미 프로토콜 포함 가능)
        pass

    # local 모드는 특별한 처리 불필요

    # storage 섹션을 유지 (deprecated 필드로 남김)
    # 완전히 제거하면 validation error가 발생할 수 있음

    # Print migration warning
    console.print("[yellow]⚠️  Legacy 'storage' configuration detected. "
                 "Migrating to new protocol-based format.[/yellow]")
    console.print("[yellow]    Please update your config to use s3_auth "
                 "and protocol prefixes in paths.[/yellow]")

    return data


def load_config(path: str | Path, verbose: bool = False) -> Config:
    """
    Load and validate environment configuration.

    Args:
        path: Path to config.yaml file
        verbose: Whether to print loaded configuration

    Returns:
        Validated Config object

    Raises:
        ConfigurationError: If loading or validation fails
    """
    path = Path(path)

    if verbose:
        console.print(f"[cyan]Loading configuration from {path}...[/cyan]")

    # Load YAML
    data = load_yaml(path)

    # Phase 2: 하위 호환성을 위한 마이그레이션
    data = migrate_storage_config(data)

    # Validate against schema
    config = validate_schema(data, Config, "config.yaml")

    if verbose:
        # Pretty print configuration
        yaml_str = yaml.dump(data, default_flow_style=False)
        syntax = Syntax(yaml_str, "yaml", theme="monokai", line_numbers=False)
        panel = Panel(syntax, title="[green]Configuration Loaded[/green]", expand=False)
        console.print(panel)

    return config


def load_recipe(path: str | Path, verbose: bool = False) -> Recipe:
    """
    Load and validate recipe configuration.

    Args:
        path: Path to recipe.yaml file
        verbose: Whether to print loaded configuration

    Returns:
        Validated Recipe object

    Raises:
        ConfigurationError: If loading or validation fails
    """
    path = Path(path)

    if verbose:
        console.print(f"[cyan]Loading recipe from {path}...[/cyan]")

    # Load YAML
    data = load_yaml(path)

    # Validate against schema
    recipe = validate_schema(data, Recipe, "recipe.yaml")

    if verbose:
        # Pretty print key settings
        console.print("[green]Recipe loaded:[/green]")
        console.print(f"  • Run: {recipe.run.name}")
        console.print(f"  • Algorithm: {recipe.train.algo}")
        console.print(f"  • Learning Rate: {recipe.optim.lr}")
        # Phase 3: recipe.model 제거됨, batching 필드도 없음
        console.print(f"  • Batch Size: {recipe.data.train.batch_size}")

    return recipe


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merge two configuration dictionaries.

    Args:
        base: Base configuration
        override: Override configuration

    Returns:
        Merged configuration with overrides applied
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def save_config(config: Config | Recipe, path: str | Path) -> None:
    """
    Save a configuration object to YAML file.

    Args:
        config: Configuration object to save
        path: Path to save to
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = config.model_dump(exclude_none=True, exclude_unset=False)

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    console.print(f"[green]Configuration saved to {path}[/green]")


# Export main functions
__all__ = [
    "load_config",
    "load_recipe",
    "load_yaml",
    "validate_schema",
    "merge_configs",
    "save_config",
    "migrate_storage_config",
    "ConfigurationError",
]
