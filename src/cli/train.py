"""WMTP(Weighted Multi-Token Prediction) 훈련 프레임워크의 메인 진입점.

연구 철학: "Not All Tokens Are What You Need"
========================================

우리의 WMTP 접근법은 기존 MTP의 균등한 토큰 가중치 문제를 해결합니다.
모든 토큰이 동일한 중요도를 갖지 않는다는 통찰을 바탕으로,
토큰별 중요도를 동적으로 계산하여 학습 효율성을 극대화합니다.

지원 알고리즘:
  1. mtp-baseline: 기본 MTP 접근법 (균등 가중치)
  2. critic-wmtp: Value Function 기반 동적 가중치
     - Stage 1: 시퀀스 레벨 보상으로 Value Head 훈련
     - Stage 2: δ_t = V_t - V_{t-1} 차분값으로 토큰 중요도 계산
  3. rho1-wmtp: Reference Model 차이 기반 가중치
     - |CE^ref_t - CE^base_t| 값으로 토큰의 학습 난이도 측정

이 CLI는 config.yaml(환경설정)과 recipe.yaml(훈련설정)을 받아
완전한 분산 훈련 파이프라인을 실행합니다.
"""

import sys  # 시스템 종료 코드 관리
from pathlib import Path  # 파일 경로 처리를 위한 모던 Python 패스 객체

import typer  # 현대적인 CLI 인터페이스 라이브러리
from rich.console import Console  # 컬러풀한 터미널 출력
from rich.traceback import install  # 아름다운 에러 메시지 표시

# Rich 트레이스백 활성화 - 에러 발생시 읽기 쉬운 형태로 표시
install(show_locals=False)
console = Console()  # 전역 콘솔 객체 - 모든 출력에 사용

# Typer CLI 애플리케이션 정의
# WMTP 훈련의 모든 명령어를 관리하는 메인 앱
app = typer.Typer(
    name="wmtp-train",
    help="WMTP 훈련 CLI - 가중치 기반 Multi-Token Prediction 모델 훈련",
    pretty_exceptions_show_locals=False,  # 보안을 위해 로컬 변수 숨김
)


@app.command()  # Typer 데코레이터 - 이 함수를 CLI 명령으로 등록
def train(
    # 필수 매개변수: 환경 설정 파일 (GPU, 메모리, 저장소 등)
    config: Path = typer.Option(
        ...,  # 필수 입력 표시
        "--config",
        "-c",
        help="환경 설정 YAML 파일 경로 (GPU, 분산훈련, S3 설정)",
        exists=True,  # 파일이 존재해야 함
        dir_okay=False,  # 디렉토리는 허용하지 않음
        readable=True,  # 읽기 권한이 있어야 함
    ),
    # 필수 매개변수: 훈련 레시피 (알고리즘, 하이퍼파라미터)
    recipe: Path = typer.Option(
        ...,
        "--recipe",
        "-r",
        help="훈련 레시피 YAML 파일 경로 (알고리즘, 모델, 데이터셋 설정)",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    # 선택적: MLflow 실험 추적을 위한 실행 이름
    run_name: str | None = typer.Option(
        None,
        "--run-name",
        help="MLflow 추적용 실행 이름 (선택사항)",
    ),
    # 선택적: 중단된 훈련 재개를 위한 체크포인트
    resume: Path | None = typer.Option(
        None,
        "--resume",
        help="훈련 재개를 위한 체크포인트 파일 경로",
        exists=True,
        dir_okay=False,
    ),
    # 선택적: 실험 분류를 위한 태그들
    tags: str | None = typer.Option(
        None,
        "--tags",
        help="MLflow용 쉼표 구분 태그 (예: 'exp1,critic,mbpp')",
    ),
    # 선택적: 설정 검증만 수행 (실제 훈련 X)
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="실제 훈련 없이 설정 검증만 수행",
    ),
    # 선택적: 상세 로그 출력
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="상세 로그 출력 활성화",
    ),
):
    """WMTP 기반 모델 훈련을 실행하는 메인 함수.

    연구 목표:
        "Not All Tokens Are What You Need" 철학을 구현하여
        토큰별 중요도를 동적으로 계산하고 가중치를 적용하는
        새로운 Multi-Token Prediction 훈련 방식입니다.

    파이프라인 실행 단계:
        1. 설정 파일 로드 및 검증 (config.yaml + recipe.yaml)
        2. 분산 훈련 환경 초기화 (FSDP, Mixed Precision)
        3. 모델 및 데이터셋 로드 (Facebook MTP + 선택된 데이터셋)
        4. 알고리즘별 컴포넌트 생성:
           - Baseline: 균등 가중치 MTP
           - Critic: Value Function 기반 동적 가중치
           - Rho1: Reference Model 차이 기반 가중치
        5. MLflow 추적과 함께 훈련 루프 실행
        6. 체크포인트 및 아티팩트 저장 (로컬 + S3)

    Args:
        config: 환경 설정 파일 (GPU, 분산훈련, 저장소 설정)
        recipe: 훈련 레시피 파일 (알고리즘, 모델, 데이터 설정)
        run_name: MLflow 실험 추적용 이름
        resume: 중단된 훈련 재개용 체크포인트 경로
        tags: MLflow 분류용 태그 리스트
        dry_run: True시 설정 검증만 수행
        verbose: 상세 로그 출력 여부
    """
    # WMTP 프레임워크 시작 메시지
    console.print("[bold blue]WMTP 훈련 프레임워크 시작[/bold blue]")
    console.print(f"환경 설정: {config}")
    console.print(f"훈련 레시피: {recipe}")

    # 검증 모드인지 확인 (실제 훈련 없이 설정만 체크)
    if dry_run:
        console.print("[yellow]DRY RUN 모드 - 설정 검증만 수행합니다[/yellow]")

    # 태그 문자열을 리스트로 파싱 (MLflow 분류용)
    tag_list = []
    if tags:
        tag_list = [t.strip() for t in tags.split(",")]  # 쉼표로 분리하고 공백 제거
        console.print(f"실험 태그: {tag_list}")

    # 체크포인트에서 재개하는지 확인
    if resume:
        console.print(f"[green]체크포인트에서 훈련 재개: {resume}[/green]")

    try:
        # 훈련 파이프라인과 설정 로더 임포트
        from src.pipelines import run_training  # 통합 훈련 파이프라인
        from src.settings import load_config, load_recipe  # Pydantic 기반 설정 로더

        # YAML 파일들을 Pydantic 모델로 로드 및 검증
        cfg = load_config(config, verbose=verbose)  # 환경 설정 (GPU, 저장소 등)
        rcp = load_recipe(
            recipe, verbose=verbose
        )  # 훈련 레시피 (알고리즘, 하이퍼파라미터)

        # 선택된 WMTP 알고리즘 표시
        console.print(f"[cyan]선택된 알고리즘: {rcp.train.algo}[/cyan]")

        # 알고리즘별 간단한 설명 출력
        algo_descriptions = {
            "mtp-baseline": "기본 MTP (균등 가중치)",
            "critic-wmtp": "Critic 기반 동적 가중치 (Value Function)",
            "rho1-wmtp": "Rho-1 기반 동적 가중치 (Reference Model 차이)",
        }
        if rcp.train.algo in algo_descriptions:
            console.print(f"[dim]{algo_descriptions[rcp.train.algo]}[/dim]")

        # run_name과 tags를 recipe에 반영
        if run_name:
            rcp.run.name = run_name
        if tag_list:
            rcp.run.tags = tag_list

        # 통합 훈련 파이프라인 실행
        # 모든 WMTP 알고리즘이 동일한 파이프라인을 사용하되 다른 컴포넌트 조합
        outputs = run_training(
            cfg,  # 환경 설정
            rcp,  # 훈련 레시피
            dry_run=dry_run,  # 검증 모드 여부
            resume_checkpoint=resume,  # 체크포인트에서 재개
        )

        # 실행 결과 출력
        if dry_run:
            console.print("[green]✅ 설정 검증 완료! 모든 설정이 올바릅니다.[/green]")
        else:
            console.print(
                f"[green]🎉 훈련 완료! 최종 메트릭: {outputs.trainer_metrics}[/green]"
            )

    # 예외 처리: 다양한 오류 상황에 대한 사용자 친화적 메시지
    except FileNotFoundError as e:
        console.print(f"[red]❌ 파일을 찾을 수 없습니다: {e}[/red]")
        console.print("[dim]config 또는 recipe 파일 경로를 확인해주세요.[/dim]")
        sys.exit(1)
    except ValueError as e:
        console.print(f"[red]❌ 설정 오류: {e}[/red]")
        console.print("[dim]YAML 파일의 설정값들을 확인해주세요.[/dim]")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️ 사용자에 의해 훈련이 중단되었습니다[/yellow]")
        console.print(
            "[dim]체크포인트가 저장되었다면 --resume으로 재개할 수 있습니다.[/dim]"
        )
        sys.exit(130)  # SIGINT 표준 종료 코드
    except Exception as e:
        console.print(f"[red]❌ 예상치 못한 오류: {e}[/red]")
        console.print(
            "[dim]이 오류가 계속 발생하면 GitHub Issues에 보고해주세요.[/dim]"
        )
        if verbose:
            console.print_exception()  # 상세 모드시 전체 스택 트레이스 출력
        sys.exit(1)


# Python 모듈이 직접 실행될 때만 CLI 앱 시작
# 다른 곳에서 import할 때는 실행되지 않음
if __name__ == "__main__":
    app()  # Typer CLI 애플리케이션 실행
