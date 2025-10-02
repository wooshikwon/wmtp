"""통합 콘솔 출력 유틸리티.

모든 WMTP 컴포넌트의 콘솔 출력을 일관되게 관리하여
구조화된 로그와 명확한 Phase 구분을 제공합니다.

핵심 기능:
- Phase context manager: 자동 시작/종료 출력
- 계층적 indentation: Pipeline > Phase > Task > Detail
- 통합 Progress Bar: rich 기반 일관된 스타일
- 전역 인스턴스: 모든 컴포넌트가 동일한 출력 스타일 공유

사용 예시:
    >>> from src.utils.console_output import get_console_output
    >>> console_out = get_console_output()
    >>>
    >>> console_out.pipeline_start("WMTP 훈련 프레임워크")
    >>> with console_out.phase("모델 로딩"):
    ...     console_out.task("Base 모델 로딩")
    ...     console_out.detail("메타데이터 확인")
    >>> console_out.pipeline_end()
"""

from __future__ import annotations

from contextlib import contextmanager
from enum import Enum

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)


class LogLevel(Enum):
    """콘솔 출력 레벨 정의.

    각 레벨은 특정 아이콘과 indentation을 가집니다.
    """

    PIPELINE = 0  # 🎯, 🏁 - Pipeline 전체
    PHASE = 1  # 📦, ✅ - Phase 단위
    TASK = 2  # → - Sub-task
    DETAIL = 3  # • - Detail
    INFO = 4  # ℹ️ - 정보
    WARNING = 5  # ⚠️ - 경고
    ERROR = 6  # ❌ - 에러


class ConsoleOutput:
    """통합 콘솔 출력 유틸리티.

    모든 WMTP 컴포넌트의 콘솔 출력을 일관되게 관리합니다.

    Attributes:
        console: Rich Console 인스턴스
        _current_phase: 현재 실행 중인 Phase 이름
        _phase_count: Phase 카운터 (Phase 1, Phase 2, ...)
    """

    def __init__(self, console: Console | None = None):
        """ConsoleOutput 초기화.

        Args:
            console: Rich Console 인스턴스 (None이면 새로 생성)
        """
        self.console = console or Console()
        self._current_phase: str | None = None
        self._phase_count: int = 0

    # ============= Pipeline Level =============

    def pipeline_start(self, description: str = "WMTP 훈련 파이프라인") -> None:
        """Pipeline 시작 선언.

        Args:
            description: Pipeline 설명
        """
        self.console.print(f"[bold cyan]🎯 {description} 시작[/bold cyan]")

    def pipeline_end(self, description: str = "파이프라인") -> None:
        """Pipeline 완료 선언.

        Args:
            description: Pipeline 설명
        """
        self.console.print(f"[bold green]🏁 {description} 완료[/bold green]")

    # ============= Phase Level =============

    @contextmanager
    def phase(self, name: str):
        """Phase context manager - 자동으로 시작/종료 출력.

        Phase는 자동으로 번호가 매겨지며 (Phase 1, Phase 2, ...)
        context 진입/종료 시 시작/완료 메시지를 자동 출력합니다.

        Args:
            name: Phase 이름

        Yields:
            None

        Example:
            >>> with console_out.phase("모델 & 토크나이저 준비"):
            ...     console_out.task("Base 모델 로딩")
            ...     console_out.task("토크나이저 생성")
            📦 Phase 1: 모델 & 토크나이저 준비
              → Base 모델 로딩
              → 토크나이저 생성
            ✅ Phase 1 완료
        """
        self._phase_count += 1
        phase_num = self._phase_count
        self._current_phase = name

        self.console.print(f"[bold blue]📦 Phase {phase_num}: {name}[/bold blue]")

        try:
            yield
        finally:
            self.console.print(f"[green]✅ Phase {phase_num} 완료[/green]")
            self._current_phase = None

    # ============= Task Level =============

    def task(self, description: str) -> None:
        """Sub-task 출력 (Level 2 - indentation 2 spaces).

        Args:
            description: Task 설명
        """
        self.console.print(f"  → {description}")

    def detail(self, description: str) -> None:
        """Detail 출력 (Level 3 - indentation 4 spaces).

        Args:
            description: Detail 설명
        """
        self.console.print(f"    • {description}")

    # ============= Message Level =============

    def info(self, message: str) -> None:
        """정보 메시지 출력.

        Args:
            message: 정보 메시지
        """
        self.console.print(f"[cyan]ℹ️  {message}[/cyan]")

    def warning(self, message: str) -> None:
        """경고 메시지 출력.

        Args:
            message: 경고 메시지
        """
        self.console.print(f"[yellow]⚠️  {message}[/yellow]")

    def error(self, message: str) -> None:
        """에러 메시지 출력.

        Args:
            message: 에러 메시지
        """
        self.console.print(f"[red]❌ {message}[/red]")

    # ============= Progress Bar =============

    def create_progress(self) -> Progress:
        """표준 Progress Bar 생성.

        모든 WMTP 컴포넌트에서 일관된 스타일의 progress bar를 사용하도록
        통일된 Progress 인스턴스를 생성합니다.

        Returns:
            rich.progress.Progress 인스턴스

        Example:
            >>> with console_out.create_progress() as progress:
            ...     task = progress.add_task("데이터셋 토크나이징", total=100)
            ...     for i in range(100):
            ...         progress.update(task, advance=1)
        """
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )


# ============= Global Instance =============

_global_console_output: ConsoleOutput | None = None


def get_console_output() -> ConsoleOutput:
    """전역 ConsoleOutput 인스턴스 가져오기.

    모든 컴포넌트가 동일한 ConsoleOutput 인스턴스를 사용하여
    일관된 출력 스타일과 Phase 카운팅을 공유합니다.

    Returns:
        전역 ConsoleOutput 인스턴스

    Example:
        >>> from src.utils.console_output import get_console_output
        >>> console_out = get_console_output()
        >>> console_out.info("설정 로드 완료")
    """
    global _global_console_output
    if _global_console_output is None:
        _global_console_output = ConsoleOutput()
    return _global_console_output
