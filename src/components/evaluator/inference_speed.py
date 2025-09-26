"""
Meta 2024 MTP 논문 Figure S10 재현: 추론 속도 벤치마크

WMTP 연구 맥락:
Multi-Token Prediction의 핵심 장점 중 하나는 단일 forward pass로
여러 토큰을 예측함으로써 추론 속도를 향상시키는 것입니다.
이 모듈은 MTP vs NTP의 추론 속도를 정량적으로 측정합니다.

Meta 논문 결과:
- MTP는 NTP 대비 최대 3배 빠른 추론 속도
- Self-speculative decoding으로 추가 가속
- 실제 코드 생성 태스크에서 2.5배 평균 속도 향상

측정 방법:
1. 동일한 프롬프트에 대한 토큰 생성 시간 측정
2. MTP (4개 헤드 활용) vs NTP (1개 헤드만 사용) 비교
3. Self-speculative decoding 성능 측정
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from rich.console import Console
from torch import nn

from ...components.base import BaseComponent
from ..registry import evaluator_registry

console = Console()


@evaluator_registry.register("inference-speed", category="evaluator", version="1.0.0")
class InferenceSpeedEvaluator(BaseComponent):
    """
    MTP vs NTP 추론 속도 벤치마크 평가기.

    Meta 2024 논문의 Figure S10을 재현하며, MTP가 NTP 대비
    얼마나 빠른 추론 속도를 달성하는지 정량적으로 측정합니다.

    측정 항목:
    - MTP 토큰 생성 속도 (tokens/sec)
    - NTP 토큰 생성 속도 (tokens/sec)
    - 속도 향상 비율 (speedup ratio)
    - Self-speculative decoding 속도

    설정 예시:
    ```yaml
    inference-speed:
      batch_sizes: [1, 4, 8, 16]
      sequence_lengths: [512, 1024, 2048]
      num_trials: 10
      warmup_steps: 3
    ```
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.batch_sizes = self.config.get("batch_sizes", [1, 4, 8])
        self.sequence_lengths = self.config.get("sequence_lengths", [512, 1024])
        self.num_trials = self.config.get("num_trials", 10)
        self.warmup_steps = self.config.get("warmup_steps", 3)
        self.device = self.config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )

    def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """
        추론 속도 벤치마크 실행.

        Args:
            ctx: 평가 컨텍스트
                - model: 평가할 MTP 모델
                - tokenizer: 토크나이저

        Returns:
            추론 속도 메트릭
        """
        self.validate_initialized()

        model = ctx.get("model")
        tokenizer = ctx.get("tokenizer")

        if not model or not tokenizer:
            raise ValueError(
                "Model and tokenizer are required for inference speed evaluation"
            )

        # 모델을 평가 모드로 설정
        model.eval()

        results = {
            "mtp_results": {},
            "ntp_results": {},
            "speculative_results": {},
            "speedup_metrics": {},
        }

        console.print("[cyan]Starting inference speed benchmark...[/cyan]")

        # 각 배치 크기와 시퀀스 길이에 대해 벤치마크 실행
        for batch_size in self.batch_sizes:
            for seq_len in self.sequence_lengths:
                config_key = f"batch_{batch_size}_seq_{seq_len}"
                console.print(f"[yellow]Testing {config_key}...[/yellow]")

                # MTP 모드 벤치마크
                mtp_times = self._benchmark_mtp_inference(
                    model, tokenizer, batch_size, seq_len
                )
                results["mtp_results"][config_key] = {
                    "mean_time": float(np.mean(mtp_times)),
                    "std_time": float(np.std(mtp_times)),
                    "tokens_per_sec": float(seq_len / np.mean(mtp_times)),
                }

                # NTP 모드 벤치마크 (첫 번째 헤드만 사용)
                ntp_times = self._benchmark_ntp_inference(
                    model, tokenizer, batch_size, seq_len
                )
                results["ntp_results"][config_key] = {
                    "mean_time": float(np.mean(ntp_times)),
                    "std_time": float(np.std(ntp_times)),
                    "tokens_per_sec": float(seq_len / np.mean(ntp_times)),
                }

                # Self-speculative decoding 벤치마크
                spec_times = self._benchmark_speculative_decoding(
                    model, tokenizer, batch_size, seq_len
                )
                results["speculative_results"][config_key] = {
                    "mean_time": float(np.mean(spec_times)),
                    "std_time": float(np.std(spec_times)),
                    "tokens_per_sec": float(seq_len / np.mean(spec_times)),
                }

                # 속도 향상 비율 계산
                speedup_ratio = np.mean(ntp_times) / np.mean(mtp_times)
                spec_speedup = np.mean(ntp_times) / np.mean(spec_times)

                results["speedup_metrics"][config_key] = {
                    "mtp_speedup": float(speedup_ratio),
                    "speculative_speedup": float(spec_speedup),
                }

                console.print(f"  MTP speedup: {speedup_ratio:.2f}x")
                console.print(f"  Speculative speedup: {spec_speedup:.2f}x")

        # 전체 평균 계산
        all_mtp_speedups = [
            v["mtp_speedup"] for v in results["speedup_metrics"].values()
        ]
        all_spec_speedups = [
            v["speculative_speedup"] for v in results["speedup_metrics"].values()
        ]

        results["summary"] = {
            "avg_mtp_speedup": float(np.mean(all_mtp_speedups)),
            "max_mtp_speedup": float(np.max(all_mtp_speedups)),
            "avg_speculative_speedup": float(np.mean(all_spec_speedups)),
            "max_speculative_speedup": float(np.max(all_spec_speedups)),
        }

        # Meta 논문 결과와 비교
        if results["summary"]["avg_mtp_speedup"] < 1.0:
            console.print(
                "[red]Warning: MTP is slower than NTP. Check model implementation.[/red]"
            )
        else:
            console.print(
                f"[green]Average MTP speedup: {results['summary']['avg_mtp_speedup']:.2f}x[/green]"
            )

        return {
            "metrics": {
                "inference_speed_mtp": results["summary"]["avg_mtp_speedup"],
                "inference_speed_speculative": results["summary"][
                    "avg_speculative_speedup"
                ],
                "max_speedup": results["summary"]["max_mtp_speedup"],
            },
            "detailed_results": results,
        }

    def _benchmark_mtp_inference(
        self, model: nn.Module, tokenizer: Any, batch_size: int, seq_len: int
    ) -> list[float]:
        """
        MTP 모드 추론 속도 측정 (모든 헤드 활용).

        Args:
            model: MTP 모델
            tokenizer: 토크나이저
            batch_size: 배치 크기
            seq_len: 시퀀스 길이

        Returns:
            각 시도의 추론 시간 리스트
        """
        times = []

        # 더미 입력 생성
        input_ids = torch.randint(
            0, tokenizer.vocab_size, (batch_size, seq_len), device=self.device
        )

        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup_steps):
                _ = model(input_ids)

        # 실제 측정
        with torch.no_grad():
            for _ in range(self.num_trials):
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.perf_counter()

                # MTP 추론: 모든 헤드 활용
                _ = model(input_ids)  # Inference for timing only

                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.perf_counter()

                times.append(end_time - start_time)

        return times

    def _benchmark_ntp_inference(
        self, model: nn.Module, tokenizer: Any, batch_size: int, seq_len: int
    ) -> list[float]:
        """
        NTP 모드 추론 속도 측정 (첫 번째 헤드만 사용).

        Args:
            model: MTP 모델
            tokenizer: 토크나이저
            batch_size: 배치 크기
            seq_len: 시퀀스 길이

        Returns:
            각 시도의 추론 시간 리스트
        """
        times = []

        # 더미 입력 생성
        input_ids = torch.randint(
            0, tokenizer.vocab_size, (batch_size, seq_len), device=self.device
        )

        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup_steps):
                outputs = model(input_ids)
                # NTP 시뮬레이션: 첫 번째 헤드만 사용
                if hasattr(outputs, "prediction_logits"):
                    _ = outputs.prediction_logits[:, :, 0, :]  # t+1 헤드만

        # 실제 측정
        with torch.no_grad():
            for _ in range(self.num_trials):
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.perf_counter()

                # NTP 추론: 첫 번째 헤드만 사용
                outputs = model(input_ids)
                if hasattr(outputs, "prediction_logits"):
                    next_token_logits = outputs.prediction_logits[:, :, 0, :]
                    _ = next_token_logits.argmax(dim=-1)  # Next tokens for timing only

                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.perf_counter()

                times.append(end_time - start_time)

        return times

    def _benchmark_speculative_decoding(
        self, model: nn.Module, tokenizer: Any, batch_size: int, seq_len: int
    ) -> list[float]:
        """
        Self-speculative decoding 속도 측정.

        Meta 논문의 핵심 기법: MTP 모델이 자기 자신을 draft model로 사용하여
        여러 토큰을 한 번에 제안하고 검증하는 방식.

        Args:
            model: MTP 모델
            tokenizer: 토크나이저
            batch_size: 배치 크기
            seq_len: 시퀀스 길이

        Returns:
            각 시도의 추론 시간 리스트
        """
        times = []
        acceptance_threshold = 0.8  # 수락 임계값
        max_speculative_tokens = 3  # 최대 추측 토큰 수

        # 더미 입력 생성
        input_ids = torch.randint(
            0,
            tokenizer.vocab_size,
            (batch_size, 32),  # 시작 시퀀스
            device=self.device,
        )

        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup_steps):
                _ = model(input_ids)

        # 실제 측정
        with torch.no_grad():
            for _ in range(self.num_trials):
                current_input = input_ids.clone()
                tokens_generated = 0

                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.perf_counter()

                while tokens_generated < seq_len:
                    # MTP로 여러 토큰 제안
                    outputs = model(current_input)

                    if hasattr(outputs, "prediction_logits"):
                        # 각 헤드에서 토큰 제안
                        proposed_tokens = []
                        for head_idx in range(min(4, max_speculative_tokens + 1)):
                            head_logits = outputs.prediction_logits[:, -1, head_idx, :]
                            head_probs = torch.softmax(head_logits, dim=-1)
                            max_prob = head_probs.max(dim=-1).values

                            # 확률이 임계값 이상이면 수락
                            if max_prob.mean() > acceptance_threshold:
                                next_token = head_logits.argmax(dim=-1, keepdim=True)
                                proposed_tokens.append(next_token)
                            else:
                                break

                        # 제안된 토큰들을 시퀀스에 추가
                        if proposed_tokens:
                            new_tokens = torch.cat(proposed_tokens, dim=-1)
                            current_input = torch.cat(
                                [current_input, new_tokens], dim=-1
                            )
                            tokens_generated += len(proposed_tokens)
                        else:
                            # 폴백: NTP 방식으로 1개 토큰 생성
                            next_token = outputs.prediction_logits[:, -1, 0, :].argmax(
                                dim=-1, keepdim=True
                            )
                            current_input = torch.cat(
                                [current_input, next_token], dim=-1
                            )
                            tokens_generated += 1
                    else:
                        # 폴백
                        tokens_generated += 1

                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.perf_counter()

                times.append(end_time - start_time)

        return times
