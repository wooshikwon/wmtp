"""
언어모델 기본 성능 지표 측정: Perplexity & Cross-Entropy

WMTP 연구 맥락:
MTP vs NTP의 언어모델링 성능을 정량적으로 비교하기 위한
기본 지표인 Perplexity와 Cross-Entropy를 측정합니다.
특히 위치별, 토큰 타입별 세밀한 분석을 제공합니다.

Meta 논문 관련:
- Perplexity는 언어모델의 불확실성을 측정하는 표준 지표
- MTP는 미래 정보를 활용하여 더 낮은 perplexity 달성 가능
- 위치별 CE loss 분석으로 long-range dependency 성능 확인

측정 지표:
1. 전체 Perplexity
2. 위치별 Cross-Entropy
3. 토큰 타입별 Perplexity (code, text, special)
4. 헤드별 Perplexity (MTP 전용)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from torch.utils.data import DataLoader

from ...components.base import BaseComponent
from ..registry import evaluator_registry

console = Console()


@evaluator_registry.register(
    "perplexity-measurer", category="evaluator", version="1.0.0"
)
class PerplexityMeasurer(BaseComponent):
    """
    언어모델 Perplexity 및 Cross-Entropy 측정기.

    표준 언어모델 성능 지표를 측정하고, MTP 모델의 경우
    헤드별 성능까지 세밀하게 분석합니다.

    주요 메트릭:
    - Overall Perplexity
    - Position-wise Cross-Entropy
    - Token-type specific Perplexity
    - Per-head Perplexity (MTP only)

    설정 예시:
    ```yaml
    perplexity-measurer:
      batch_size: 8
      max_length: 2048
      position_buckets: [[0, 128], [128, 512], [512, 1024], [1024, 2048]]
      analyze_token_types: true
      compute_head_perplexity: true
    ```
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.batch_size = self.config.get("batch_size", 8)
        self.max_length = self.config.get("max_length", 2048)
        self.position_buckets = self.config.get(
            "position_buckets", [[0, 128], [128, 512], [512, 1024], [1024, 2048]]
        )
        self.analyze_token_types = self.config.get("analyze_token_types", True)
        self.compute_head_perplexity = self.config.get("compute_head_perplexity", True)
        self.device = self.config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )

    def _classify_token_type(self, token_id: int, tokenizer: Any) -> str:
        """
        토큰을 타입별로 분류 (code, text, special).

        Args:
            token_id: 토큰 ID
            tokenizer: 토크나이저

        Returns:
            토큰 타입
        """
        # 토큰을 디코드
        token_str = tokenizer.decode([token_id], skip_special_tokens=False)

        # Special 토큰 확인
        if token_id in [
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
        ]:
            return "special"

        # 코드 관련 토큰 패턴
        code_patterns = [
            "def",
            "class",
            "import",
            "if",
            "else",
            "for",
            "while",
            "return",
            "(",
            ")",
            "[",
            "]",
            "{",
            "}",
            ":",
            ";",
            "=",
            "+",
            "-",
            "*",
            "/",
            "self",
            "None",
            "True",
            "False",
            "__",
            "->",
            "==",
            "!=",
            "<=",
            ">=",
        ]

        # 코드 토큰 확인
        if any(pattern in token_str for pattern in code_patterns):
            return "code"

        # 나머지는 텍스트
        return "text"

    def _compute_perplexity_batch(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[float, torch.Tensor]:
        """
        배치에 대한 perplexity 계산.

        Args:
            model: 언어모델
            input_ids: 입력 토큰 [batch, seq_len]
            attention_mask: 어텐션 마스크

        Returns:
            perplexity: 배치 perplexity
            ce_losses: 토큰별 CE 손실 [batch, seq_len]
        """
        batch_size, seq_len = input_ids.shape

        with torch.no_grad():
            # 모델 forward
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # 로짓 추출
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            # MTP 모델 처리
            if logits.ndim == 4:  # [B, S, H, V]
                # 첫 번째 헤드 사용 (next token prediction)
                logits = logits[:, :, 0, :]  # [B, S, V]

            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            # Cross-entropy 계산
            ce_losses = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            ).view(batch_size, seq_len - 1)

            # 마스킹 처리
            if attention_mask is not None:
                shift_mask = attention_mask[:, 1:].contiguous()
                ce_losses = ce_losses * shift_mask
                valid_tokens = shift_mask.sum()
            else:
                valid_tokens = ce_losses.numel()

            # 평균 CE 및 Perplexity
            avg_ce = ce_losses.sum() / valid_tokens if valid_tokens > 0 else 0
            perplexity = math.exp(avg_ce.item()) if avg_ce < 50 else float("inf")

            return perplexity, ce_losses

    def _compute_head_perplexities(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """
        MTP 모델의 헤드별 perplexity 계산.

        Args:
            model: MTP 모델
            input_ids: 입력 토큰
            attention_mask: 어텐션 마스크

        Returns:
            헤드별 perplexity
        """
        head_perplexities = {}

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            if logits.ndim != 4:  # Not MTP
                return {}

            batch_size, seq_len, num_heads, vocab_size = logits.shape

            for h in range(num_heads):
                # 각 헤드는 h+1 위치 예측
                shift = h + 1

                if shift >= seq_len:
                    continue

                # 해당 헤드의 로짓
                head_logits = logits[:, :-shift, h, :]  # [B, S-shift, V]
                head_labels = input_ids[:, shift:]  # [B, S-shift]

                # 길이 맞추기
                min_len = min(head_logits.shape[1], head_labels.shape[1])
                head_logits = head_logits[:, :min_len, :]
                head_labels = head_labels[:, :min_len]

                # CE 계산
                ce_loss = F.cross_entropy(
                    head_logits.reshape(-1, vocab_size),
                    head_labels.reshape(-1),
                    reduction="mean",
                )

                # Perplexity
                perplexity = math.exp(ce_loss.item()) if ce_loss < 50 else float("inf")
                head_perplexities[f"head_{h + 1}"] = perplexity

        return head_perplexities

    def _analyze_position_wise_ce(
        self, ce_losses: list[torch.Tensor]
    ) -> dict[str, float]:
        """
        위치별 Cross-Entropy 분석.

        Args:
            ce_losses: 배치별 CE 손실 리스트

        Returns:
            위치 구간별 평균 CE
        """
        position_ce = {}

        # 모든 CE를 연결
        all_ce = torch.cat(
            [ce.cpu() for ce in ce_losses], dim=0
        )  # [total_samples, seq_len-1]

        for start, end in self.position_buckets:
            # 해당 위치 구간의 CE 추출
            if end > all_ce.shape[1]:
                end = all_ce.shape[1]

            if start < end:
                bucket_ce = all_ce[:, start:end]
                # 0이 아닌 값들의 평균 (padding 제외)
                valid_ce = bucket_ce[bucket_ce > 0]
                if valid_ce.numel() > 0:
                    avg_ce = valid_ce.mean().item()
                    position_ce[f"pos_{start}_{end}"] = avg_ce
                    position_ce[f"ppl_{start}_{end}"] = (
                        math.exp(avg_ce) if avg_ce < 50 else float("inf")
                    )

        return position_ce

    def _analyze_token_type_perplexity(
        self, model: torch.nn.Module, dataloader: DataLoader, tokenizer: Any
    ) -> dict[str, float]:
        """
        토큰 타입별 perplexity 분석.

        Args:
            model: 언어모델
            dataloader: 데이터 로더
            tokenizer: 토크나이저

        Returns:
            타입별 perplexity
        """
        type_ce = {"code": [], "text": [], "special": []}

        for batch in dataloader:
            input_ids = batch.get("input_ids", batch).to(self.device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs

                if logits.ndim == 4:  # MTP
                    logits = logits[:, :, 0, :]

                # CE 계산
                for i in range(input_ids.shape[0]):
                    for j in range(input_ids.shape[1] - 1):
                        token_id = input_ids[i, j + 1].item()
                        token_type = self._classify_token_type(token_id, tokenizer)

                        # 해당 토큰의 CE
                        ce = F.cross_entropy(
                            logits[i, j].unsqueeze(0),
                            input_ids[i, j + 1].unsqueeze(0),
                            reduction="none",
                        )
                        type_ce[token_type].append(ce.item())

            # 메모리 절약을 위해 일부만 처리
            if len(type_ce["code"]) > 10000:
                break

        # 타입별 평균 perplexity 계산
        type_perplexity = {}
        for token_type, ce_list in type_ce.items():
            if ce_list:
                avg_ce = np.mean(ce_list)
                type_perplexity[f"{token_type}_ce"] = avg_ce
                type_perplexity[f"{token_type}_ppl"] = (
                    math.exp(avg_ce) if avg_ce < 50 else float("inf")
                )

        return type_perplexity

    def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """
        Perplexity 측정 실행.

        Args:
            ctx: 평가 컨텍스트
                - model: 평가할 모델
                - tokenizer: 토크나이저
                - eval_dataset: 평가 데이터셋 (선택)

        Returns:
            Perplexity 메트릭
        """
        self.validate_initialized()

        model = ctx.get("model")
        tokenizer = ctx.get("tokenizer")
        eval_dataset = ctx.get("eval_dataset")

        if not model or not tokenizer:
            raise ValueError(
                "Model and tokenizer are required for perplexity measurement"
            )

        # 모델 설정
        model.eval()
        model = model.to(self.device)

        # 평가 데이터 준비
        if eval_dataset is None:
            # 기본 테스트 데이터 생성
            test_texts = [
                "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr",
                "The quick brown fox jumps over the lazy dog.",
                "import numpy as np\nimport torch\nfrom transformers import AutoModel",
                "Machine learning is a subset of artificial intelligence.",
                "class NeuralNetwork(nn.Module):\n    def __init__(self):\n        super().__init__()",
            ] * 20  # 반복하여 충분한 데이터 생성

            # 토큰화
            eval_dataset = []
            for text in test_texts:
                tokens = tokenizer(
                    text,
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                # Squeeze batch dimension if present (from return_tensors="pt")
                if isinstance(tokens, dict):
                    tokens = {
                        k: v.squeeze(0) if v.dim() > 1 else v for k, v in tokens.items()
                    }
                eval_dataset.append(tokens)

        # DataLoader 생성
        if not isinstance(eval_dataset, DataLoader):
            dataloader = DataLoader(
                eval_dataset, batch_size=self.batch_size, shuffle=False
            )
        else:
            dataloader = eval_dataset

        console.print("[bold cyan]Starting Perplexity Measurement[/bold cyan]")

        # 전체 perplexity 계산
        total_perplexity = 0
        all_ce_losses = []
        num_batches = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task(
                "[cyan]Computing perplexity...", total=len(dataloader)
            )

            for batch in dataloader:
                # 배치 데이터 준비
                if isinstance(batch, dict):
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch.get("attention_mask", None)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.device)
                else:
                    input_ids = batch.to(self.device)
                    attention_mask = None

                # Perplexity 계산
                batch_ppl, ce_losses = self._compute_perplexity_batch(
                    model, input_ids, attention_mask
                )

                total_perplexity += batch_ppl
                all_ce_losses.append(ce_losses)
                num_batches += 1

                progress.update(task, advance=1)

        # 평균 perplexity
        avg_perplexity = (
            total_perplexity / num_batches if num_batches > 0 else float("inf")
        )

        # 위치별 CE 분석
        console.print("\n[yellow]Analyzing position-wise cross-entropy...[/yellow]")
        position_metrics = self._analyze_position_wise_ce(all_ce_losses)

        # 헤드별 perplexity (MTP 모델인 경우)
        head_metrics = {}
        if self.compute_head_perplexity:
            console.print("[yellow]Computing per-head perplexity...[/yellow]")
            # 샘플 배치로 헤드별 perplexity 계산
            sample_batch = next(iter(dataloader))
            if isinstance(sample_batch, dict):
                sample_ids = sample_batch["input_ids"].to(self.device)
                sample_mask = sample_batch.get("attention_mask", None)
                if sample_mask is not None:
                    sample_mask = sample_mask.to(self.device)
            else:
                sample_ids = sample_batch.to(self.device)
                sample_mask = None

            head_metrics = self._compute_head_perplexities(
                model, sample_ids, sample_mask
            )

        # 토큰 타입별 perplexity
        type_metrics = {}
        if self.analyze_token_types:
            console.print("[yellow]Analyzing token-type perplexity...[/yellow]")
            type_metrics = self._analyze_token_type_perplexity(
                model, dataloader, tokenizer
            )

        # 결과 출력
        self._print_results(
            avg_perplexity, position_metrics, head_metrics, type_metrics
        )

        # 메트릭 통합
        metrics = {
            "overall_perplexity": avg_perplexity,
            "overall_ce": math.log(avg_perplexity)
            if avg_perplexity < float("inf")
            else 50.0,
            **position_metrics,
            **head_metrics,
            **type_metrics,
            "config": {
                "batch_size": self.batch_size,
                "max_length": self.max_length,
                "num_samples": num_batches * self.batch_size,
            },
        }

        return {"metrics": metrics}

    def _print_results(
        self,
        overall_ppl: float,
        position_metrics: dict,
        head_metrics: dict,
        type_metrics: dict,
    ):
        """결과를 테이블 형식으로 출력."""
        # 전체 결과 테이블
        table = Table(title="Perplexity Measurement Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        # Overall metrics
        table.add_row("Overall Perplexity", f"{overall_ppl:.2f}")
        table.add_row(
            "Overall Cross-Entropy",
            f"{math.log(overall_ppl):.4f}" if overall_ppl < float("inf") else "inf",
        )

        # Position-wise metrics
        for key, value in position_metrics.items():
            if "ppl" in key:
                table.add_row(f"Position {key}", f"{value:.2f}")

        # Head-specific metrics (MTP)
        for key, value in head_metrics.items():
            table.add_row(f"Perplexity {key}", f"{value:.2f}")

        # Token type metrics
        for key, value in type_metrics.items():
            if "ppl" in key:
                table.add_row(f"Type {key}", f"{value:.2f}")

        console.print(table)
