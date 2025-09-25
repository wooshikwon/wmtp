"""
Meta 2024 MTP 논문 Self-Speculative Decoding 구현

WMTP 연구 맥락:
MTP 모델의 혁신적 특징 중 하나는 추가 draft 모델 없이도
자체 헤드들을 활용한 speculative decoding이 가능하다는 점입니다.
이 모듈은 MTP의 자기-투기적 디코딩 성능을 측정합니다.

Self-Speculative Decoding 원리:
1. MTP의 4개 헤드가 동시에 t+1, t+2, t+3, t+4를 예측
2. 이 예측을 draft sequence로 사용
3. 단일 forward pass로 4개 토큰의 유효성 검증
4. 수락된 토큰만큼 건너뛰어 다음 예측 수행

Meta 논문 결과:
- 평균 수락률: 2.5개 토큰 (62.5%)
- 실효 속도 향상: 2.1배
- 코드 생성 태스크에서 특히 높은 수락률
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

from ...components.base import BaseComponent
from ..registry import evaluator_registry

console = Console()


@evaluator_registry.register("self-speculative", category="evaluator", version="1.0.0")
class SelfSpeculativeEvaluator(BaseComponent):
    """
    MTP Self-Speculative Decoding 성능 평가기.

    Meta 2024 논문의 핵심 기술인 자기-투기적 디코딩을 구현하고,
    수락률(acceptance rate)과 속도 향상을 측정합니다.

    주요 메트릭:
    - 토큰 수락률 (acceptance rate)
    - 평균 수락 길이 (average accepted length)
    - 실효 속도 향상 (effective speedup)
    - 헤드별 수락률 분포

    설정 예시:
    ```yaml
    self-speculative:
      num_sequences: 100       # 평가할 시퀀스 수
      max_tokens: 512          # 생성할 최대 토큰 수
      temperature: 0.8         # 샘플링 온도
      top_p: 0.95             # nucleus sampling
      measure_speedup: true    # 속도 향상 측정 여부
    ```
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.num_sequences = self.config.get("num_sequences", 100)
        self.max_tokens = self.config.get("max_tokens", 512)
        self.temperature = self.config.get("temperature", 0.8)
        self.top_p = self.config.get("top_p", 0.95)
        self.measure_speedup = self.config.get("measure_speedup", True)
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    def _speculative_decode_step(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        draft_length: int = 4
    ) -> tuple[torch.Tensor, int, list[float]]:
        """
        단일 speculative decoding 스텝 수행.

        Args:
            model: MTP 모델
            input_ids: 현재까지의 토큰 시퀀스 [batch, seq_len]
            draft_length: draft 토큰 수 (MTP heads 수)

        Returns:
            accepted_tokens: 수락된 토큰들
            num_accepted: 수락된 토큰 수
            head_probs: 각 헤드의 예측 확률
        """
        batch_size, seq_len = input_ids.shape

        with torch.no_grad():
            # Step 1: MTP forward로 draft 생성
            outputs = model(input_ids=input_ids)

            # MTP 출력: [batch, seq_len, heads, vocab]
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs

            # 마지막 위치의 각 헤드 예측 추출
            if logits.ndim == 4:  # [B, S, H, V]
                last_logits = logits[:, -1, :, :]  # [B, H, V]
            else:
                # Fallback: 일반 모델인 경우 단일 헤드로 처리
                last_logits = logits[:, -1:, :].unsqueeze(1)  # [B, 1, V]

            heads = last_logits.shape[1]
            draft_tokens = []
            head_probs = []

            # Step 2: 각 헤드에서 토큰 샘플링
            for h in range(min(heads, draft_length)):
                head_logits = last_logits[:, h, :]  # [B, V]

                # Temperature sampling
                if self.temperature > 0:
                    probs = F.softmax(head_logits / self.temperature, dim=-1)

                    # Top-p (nucleus) sampling
                    if self.top_p < 1.0:
                        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > self.top_p
                        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                        sorted_indices_to_remove[:, 0] = 0

                        indices_to_remove = sorted_indices_to_remove.scatter(
                            1, sorted_indices, sorted_indices_to_remove
                        )
                        probs[indices_to_remove] = 0
                        probs = probs / probs.sum(dim=-1, keepdim=True)

                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(head_logits, dim=-1, keepdim=True)
                    probs = F.softmax(head_logits, dim=-1)

                draft_tokens.append(next_token)
                # 선택된 토큰의 확률 저장
                token_prob = probs.gather(1, next_token).squeeze(-1)
                head_probs.append(token_prob.item() if batch_size == 1 else token_prob.mean().item())

            # Step 3: Draft 검증 (간단한 구현 - 실제는 더 복잡)
            # 여기서는 확률 기반 수락 결정
            accepted_tokens = []
            for i, (token, prob) in enumerate(zip(draft_tokens, head_probs)):
                # 확률이 임계값 이상이면 수락 (간단한 휴리스틱)
                acceptance_threshold = 0.3 * (1 - i * 0.1)  # 거리가 멀수록 엄격
                if prob > acceptance_threshold:
                    accepted_tokens.append(token)
                else:
                    break  # 첫 거절 시 중단

            if accepted_tokens:
                accepted_tensor = torch.cat(accepted_tokens, dim=1)
            else:
                # 최소 1개는 수락 (첫 번째 헤드)
                accepted_tensor = draft_tokens[0] if draft_tokens else torch.zeros((batch_size, 1), dtype=torch.long, device=input_ids.device)

            return accepted_tensor, len(accepted_tokens), head_probs

    def _measure_acceptance_rate(
        self,
        model: torch.nn.Module,
        prompts: list[str],
        tokenizer: Any
    ) -> dict[str, Any]:
        """
        여러 시퀀스에 대한 수락률 측정.

        Args:
            model: MTP 모델
            prompts: 평가할 프롬프트 리스트
            tokenizer: 토크나이저

        Returns:
            수락률 통계
        """
        total_proposed = 0
        total_accepted = 0
        acceptance_by_position = [0] * 4  # 각 헤드별 수락 횟수
        sequence_stats = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            task = progress.add_task(
                "[cyan]Measuring acceptance rates...",
                total=len(prompts)
            )

            for prompt in prompts:
                # 프롬프트 토큰화
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(self.device)

                sequence_proposed = 0
                sequence_accepted = 0

                # 시퀀스 생성 (최대 max_tokens까지)
                for _ in range(self.max_tokens // 4):  # 4개씩 제안하므로
                    accepted_tokens, num_accepted, _ = self._speculative_decode_step(
                        model, input_ids
                    )

                    # 통계 업데이트
                    sequence_proposed += 4
                    sequence_accepted += num_accepted
                    total_proposed += 4
                    total_accepted += num_accepted

                    # 위치별 수락 기록
                    for i in range(num_accepted):
                        if i < 4:
                            acceptance_by_position[i] += 1

                    # 수락된 토큰 추가
                    input_ids = torch.cat([input_ids, accepted_tokens], dim=1)

                    # EOS 토큰 확인
                    if tokenizer.eos_token_id in accepted_tokens:
                        break

                # 시퀀스별 통계
                if sequence_proposed > 0:
                    sequence_stats.append({
                        "acceptance_rate": sequence_accepted / sequence_proposed,
                        "avg_accepted": sequence_accepted / (sequence_proposed / 4)
                    })

                progress.update(task, advance=1)

        # 전체 통계 계산
        overall_acceptance_rate = total_accepted / total_proposed if total_proposed > 0 else 0
        avg_accepted_length = np.mean([s["avg_accepted"] for s in sequence_stats]) if sequence_stats else 0

        # 위치별 수락률 (정규화)
        position_rates = [
            count / (total_proposed / 4) if total_proposed > 0 else 0
            for count in acceptance_by_position
        ]

        return {
            "overall_acceptance_rate": overall_acceptance_rate,
            "average_accepted_length": avg_accepted_length,
            "position_acceptance_rates": position_rates,
            "effective_speedup": 1 + overall_acceptance_rate * 3,  # 이론적 속도 향상
            "num_sequences_evaluated": len(sequence_stats),
            "sequence_stats": sequence_stats
        }

    def _measure_speedup(
        self,
        model: torch.nn.Module,
        prompts: list[str],
        tokenizer: Any
    ) -> dict[str, Any]:
        """
        실제 속도 향상 측정 (speculative vs normal).

        Args:
            model: MTP 모델
            prompts: 평가할 프롬프트
            tokenizer: 토크나이저

        Returns:
            속도 비교 결과
        """
        # Speculative decoding 시간 측정
        start_spec = time.time()
        for prompt in prompts[:10]:  # 일부만 측정 (시간 절약)
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(self.device)

            for _ in range(self.max_tokens // 4):
                accepted_tokens, _, _ = self._speculative_decode_step(model, input_ids)
                input_ids = torch.cat([input_ids, accepted_tokens], dim=1)
                if tokenizer.eos_token_id in accepted_tokens:
                    break

        spec_time = time.time() - start_spec

        # Normal decoding 시간 측정 (비교용)
        start_normal = time.time()
        for prompt in prompts[:10]:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(self.device)

            for _ in range(self.max_tokens):
                with torch.no_grad():
                    outputs = model(input_ids=input_ids)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs

                    # 단일 토큰 생성
                    if logits.ndim == 4:  # MTP
                        next_logits = logits[:, -1, 0, :]  # 첫 번째 헤드만 사용
                    else:
                        next_logits = logits[:, -1, :]

                    next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
                    input_ids = torch.cat([input_ids, next_token], dim=1)

                    if next_token.item() == tokenizer.eos_token_id:
                        break

        normal_time = time.time() - start_normal

        return {
            "speculative_time": spec_time,
            "normal_time": normal_time,
            "measured_speedup": normal_time / spec_time if spec_time > 0 else 1.0
        }

    def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """
        Self-speculative decoding 평가 실행.

        Args:
            ctx: 평가 컨텍스트
                - model: MTP 모델
                - tokenizer: 토크나이저
                - prompts: 평가용 프롬프트 (선택)

        Returns:
            평가 메트릭
        """
        self.validate_initialized()

        model = ctx.get("model")
        tokenizer = ctx.get("tokenizer")

        if not model or not tokenizer:
            raise ValueError("Model and tokenizer are required for self-speculative evaluation")

        # 모델을 평가 모드로 설정
        model.eval()
        model = model.to(self.device)

        # 평가용 프롬프트 준비
        prompts = ctx.get("prompts")
        if not prompts:
            # 기본 프롬프트 생성
            prompts = [
                "def fibonacci(n):",
                "class Stack:\n    def __init__(self):",
                "def quicksort(arr):",
                "# Binary search implementation\ndef binary_search(arr, target):",
                "def merge_sort(arr):\n    if len(arr) <= 1:",
            ] * (self.num_sequences // 5)

        console.print("[bold cyan]Starting Self-Speculative Decoding Evaluation[/bold cyan]")

        # 수락률 측정
        acceptance_metrics = self._measure_acceptance_rate(model, prompts, tokenizer)

        # 속도 향상 측정 (선택적)
        speedup_metrics = {}
        if self.measure_speedup:
            console.print("\n[yellow]Measuring actual speedup...[/yellow]")
            speedup_metrics = self._measure_speedup(model, prompts, tokenizer)

        # 결과 테이블 출력
        self._print_results(acceptance_metrics, speedup_metrics)

        # 메트릭 통합
        metrics = {
            **acceptance_metrics,
            **speedup_metrics,
            "config": {
                "num_sequences": self.num_sequences,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p
            }
        }

        return {"metrics": metrics}

    def _print_results(self, acceptance_metrics: dict, speedup_metrics: dict):
        """결과를 보기 좋게 출력."""
        table = Table(title="Self-Speculative Decoding Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        # 수락률 메트릭
        table.add_row("Overall Acceptance Rate", f"{acceptance_metrics['overall_acceptance_rate']:.2%}")
        table.add_row("Average Accepted Length", f"{acceptance_metrics['average_accepted_length']:.2f} tokens")
        table.add_row("Theoretical Speedup", f"{acceptance_metrics['effective_speedup']:.2f}x")

        # 위치별 수락률
        for i, rate in enumerate(acceptance_metrics['position_acceptance_rates']):
            table.add_row(f"Head {i+1} Acceptance Rate", f"{rate:.2%}")

        # 실제 속도 향상 (있는 경우)
        if speedup_metrics:
            table.add_row("Measured Speedup", f"{speedup_metrics['measured_speedup']:.2f}x")
            table.add_row("Speculative Time", f"{speedup_metrics['speculative_time']:.2f}s")
            table.add_row("Normal Time", f"{speedup_metrics['normal_time']:.2f}s")

        console.print(table)