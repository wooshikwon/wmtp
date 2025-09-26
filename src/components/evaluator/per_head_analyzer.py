"""
Meta 2024 MTP 논문 헤드별 성능 분석

WMTP 연구 맥락:
Multi-Token Prediction에서 각 예측 헤드(t+1, t+2, t+3, t+4)는
서로 다른 난이도의 예측 태스크를 수행합니다.
일반적으로 가까운 미래(t+1)가 먼 미래(t+4)보다 정확합니다.

Meta 논문 통찰:
- 가까운 헤드(t+1)는 국소적 패턴 학습
- 먼 헤드(t+3, t+4)는 장기 의존성 학습
- 헤드별 성능 차이가 모델의 학습 품질 지표

분석 목적:
1. 각 헤드의 예측 정확도 측정
2. 헤드 간 성능 패턴 분석
3. WMTP 가중치가 각 헤드에 미치는 영향 평가
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
from rich.console import Console
from torch import nn
from tqdm import tqdm

from ...components.base import BaseComponent
from ..registry import evaluator_registry

console = Console()


@evaluator_registry.register("per-head-analysis", category="evaluator", version="1.0.0")
class PerHeadAnalyzer(BaseComponent):
    """
    MTP 모델의 각 예측 헤드별 성능을 분석하는 평가기.

    각 헤드(t+1, t+2, t+3, t+4)의 예측 정확도를 개별적으로 측정하고,
    헤드 간 성능 패턴을 분석하여 모델의 학습 품질을 평가합니다.

    분석 항목:
    - 헤드별 토큰 예측 정확도
    - 헤드별 perplexity
    - 헤드 간 confidence 분포
    - 위치별 헤드 성능 변화

    설정 예시:
    ```yaml
    per-head-analysis:
      analyze_positions: true
      compute_confidence: true
      head_comparison: true
      position_buckets: [0-128, 128-512, 512-1024, 1024-2048]
    ```
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.analyze_positions = self.config.get("analyze_positions", True)
        self.compute_confidence = self.config.get("compute_confidence", True)
        self.head_comparison = self.config.get("head_comparison", True)
        self.position_buckets = self.config.get(
            "position_buckets", [(0, 128), (128, 512), (512, 1024), (1024, 2048)]
        )
        self.device = self.config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.batch_size = self.config.get("batch_size", 8)

    def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """
        헤드별 성능 분석 실행.

        Args:
            ctx: 평가 컨텍스트
                - model: MTP 모델
                - tokenizer: 토크나이저
                - dataset: 평가 데이터셋 (선택)

        Returns:
            헤드별 성능 메트릭
        """
        self.validate_initialized()

        model = ctx.get("model")
        tokenizer = ctx.get("tokenizer")
        dataset = ctx.get("dataset")

        if not model or not tokenizer:
            raise ValueError("Model and tokenizer are required for per-head analysis")

        # 모델을 평가 모드로 설정
        model.eval()

        # 데이터셋이 없으면 더미 데이터 생성
        if dataset is None:
            dataset = self._create_dummy_dataset(tokenizer)

        console.print("[cyan]Starting per-head performance analysis...[/cyan]")

        # 헤드별 메트릭 초기화
        head_metrics = {
            f"head_{i+1}": {
                "accuracies": [],
                "perplexities": [],
                "confidences": [],
                "position_accuracies": defaultdict(list),
            }
            for i in range(4)
        }

        # 배치별로 평가 수행
        total_samples = 0
        with torch.no_grad():
            for batch in tqdm(self._create_batches(dataset), desc="Analyzing heads"):
                batch_results = self._analyze_batch(model, batch)

                # 각 헤드별 결과 수집
                for head_idx in range(4):
                    head_key = f"head_{head_idx+1}"

                    # 정확도 수집
                    if batch_results[head_key]["accuracy"] is not None:
                        head_metrics[head_key]["accuracies"].append(
                            batch_results[head_key]["accuracy"]
                        )

                    # Perplexity 수집
                    if batch_results[head_key]["perplexity"] is not None:
                        head_metrics[head_key]["perplexities"].append(
                            batch_results[head_key]["perplexity"]
                        )

                    # Confidence 수집
                    if (
                        self.compute_confidence
                        and batch_results[head_key]["confidence"] is not None
                    ):
                        head_metrics[head_key]["confidences"].extend(
                            batch_results[head_key]["confidence"]
                        )

                    # 위치별 정확도 수집
                    if self.analyze_positions:
                        for pos_key, acc in batch_results[head_key][
                            "position_accuracies"
                        ].items():
                            head_metrics[head_key]["position_accuracies"][
                                pos_key
                            ].append(acc)

                total_samples += batch["input_ids"].size(0)

        # 결과 집계
        results = self._aggregate_results(head_metrics)

        # 헤드 간 비교 분석
        if self.head_comparison:
            comparison = self._compare_heads(results)
            results["head_comparison"] = comparison

        # Meta 논문 패턴 검증
        self._validate_meta_patterns(results)

        console.print(f"[green]Analyzed {total_samples} samples across 4 heads[/green]")

        return {"metrics": self._extract_metrics(results), "detailed_results": results}

    def _analyze_batch(
        self, model: nn.Module, batch: dict[str, torch.Tensor]
    ) -> dict[str, Any]:
        """
        단일 배치에 대한 헤드별 분석 수행.

        Args:
            model: MTP 모델
            batch: 입력 배치

        Returns:
            헤드별 분석 결과
        """
        input_ids = batch["input_ids"].to(self.device)
        labels = batch.get("labels", input_ids.clone())

        # 모델 추론
        outputs = model(input_ids)

        batch_results = {}

        if hasattr(outputs, "prediction_logits"):
            # 각 헤드별 분석
            for head_idx in range(4):
                head_key = f"head_{head_idx+1}"
                head_logits = outputs.prediction_logits[:, :, head_idx, :]

                # 타겟 위치 계산 (t+head_idx+1)
                target_start = head_idx + 1
                if target_start < labels.size(1):
                    target_labels = labels[:, target_start:]
                    valid_length = min(head_logits.size(1), target_labels.size(1))

                    if valid_length > 0:
                        # 유효한 부분만 사용
                        head_logits_valid = head_logits[:, :valid_length, :]
                        target_labels_valid = target_labels[:, :valid_length]

                        # 정확도 계산
                        predictions = head_logits_valid.argmax(dim=-1)
                        accuracy = (
                            (predictions == target_labels_valid).float().mean().item()
                        )

                        # Perplexity 계산
                        loss = nn.functional.cross_entropy(
                            head_logits_valid.reshape(-1, head_logits_valid.size(-1)),
                            target_labels_valid.reshape(-1),
                            reduction="mean",
                        )
                        perplexity = torch.exp(loss).item()

                        # Confidence 계산
                        confidence_scores = []
                        if self.compute_confidence:
                            probs = torch.softmax(head_logits_valid, dim=-1)
                            max_probs = probs.max(dim=-1).values
                            confidence_scores = max_probs.mean(dim=1).cpu().tolist()

                        # 위치별 정확도 계산
                        position_accuracies = {}
                        if self.analyze_positions:
                            for start, end in self.position_buckets:
                                mask = torch.zeros_like(predictions, dtype=torch.bool)
                                mask[:, start : min(end, valid_length)] = True
                                if mask.any():
                                    pos_acc = (
                                        (predictions[mask] == target_labels_valid[mask])
                                        .float()
                                        .mean()
                                        .item()
                                    )
                                    position_accuracies[f"pos_{start}_{end}"] = pos_acc

                        batch_results[head_key] = {
                            "accuracy": accuracy,
                            "perplexity": perplexity,
                            "confidence": confidence_scores,
                            "position_accuracies": position_accuracies,
                        }
                    else:
                        batch_results[head_key] = {
                            "accuracy": None,
                            "perplexity": None,
                            "confidence": None,
                            "position_accuracies": {},
                        }
                else:
                    batch_results[head_key] = {
                        "accuracy": None,
                        "perplexity": None,
                        "confidence": None,
                        "position_accuracies": {},
                    }
        else:
            # MTP 출력이 아닌 경우
            for head_idx in range(4):
                batch_results[f"head_{head_idx+1}"] = {
                    "accuracy": None,
                    "perplexity": None,
                    "confidence": None,
                    "position_accuracies": {},
                }

        return batch_results

    def _aggregate_results(self, head_metrics: dict[str, dict]) -> dict[str, Any]:
        """
        헤드별 메트릭을 집계.

        Args:
            head_metrics: 헤드별 수집된 메트릭

        Returns:
            집계된 결과
        """
        results = {}

        for head_key, metrics in head_metrics.items():
            # 정확도 평균
            if metrics["accuracies"]:
                avg_accuracy = float(np.mean(metrics["accuracies"]))
                std_accuracy = float(np.std(metrics["accuracies"]))
            else:
                avg_accuracy = 0.0
                std_accuracy = 0.0

            # Perplexity 평균
            if metrics["perplexities"]:
                avg_perplexity = float(np.mean(metrics["perplexities"]))
                std_perplexity = float(np.std(metrics["perplexities"]))
            else:
                avg_perplexity = float("inf")
                std_perplexity = 0.0

            # Confidence 분포
            confidence_stats = {}
            if metrics["confidences"]:
                confidence_stats = {
                    "mean": float(np.mean(metrics["confidences"])),
                    "std": float(np.std(metrics["confidences"])),
                    "min": float(np.min(metrics["confidences"])),
                    "max": float(np.max(metrics["confidences"])),
                    "percentiles": {
                        "p25": float(np.percentile(metrics["confidences"], 25)),
                        "p50": float(np.percentile(metrics["confidences"], 50)),
                        "p75": float(np.percentile(metrics["confidences"], 75)),
                    },
                }

            # 위치별 정확도 평균
            position_accuracy_avg = {}
            for pos_key, accs in metrics["position_accuracies"].items():
                if accs:
                    position_accuracy_avg[pos_key] = float(np.mean(accs))

            results[head_key] = {
                "accuracy": {"mean": avg_accuracy, "std": std_accuracy},
                "perplexity": {"mean": avg_perplexity, "std": std_perplexity},
                "confidence": confidence_stats,
                "position_accuracies": position_accuracy_avg,
            }

        return results

    def _compare_heads(self, results: dict[str, Any]) -> dict[str, Any]:
        """
        헤드 간 성능 비교 분석.

        Args:
            results: 헤드별 집계 결과

        Returns:
            비교 분석 결과
        """
        comparison = {
            "accuracy_trend": [],
            "perplexity_trend": [],
            "confidence_trend": [],
            "relative_performance": {},
        }

        # 헤드별 트렌드 추출
        for i in range(4):
            head_key = f"head_{i+1}"
            if head_key in results:
                comparison["accuracy_trend"].append(
                    results[head_key]["accuracy"]["mean"]
                )
                comparison["perplexity_trend"].append(
                    results[head_key]["perplexity"]["mean"]
                )
                if results[head_key]["confidence"]:
                    comparison["confidence_trend"].append(
                        results[head_key]["confidence"]["mean"]
                    )

        # 상대적 성능 계산 (head_1 대비)
        if comparison["accuracy_trend"]:
            base_accuracy = comparison["accuracy_trend"][0]
            for i in range(1, 4):
                if i < len(comparison["accuracy_trend"]):
                    relative = (
                        comparison["accuracy_trend"][i] / base_accuracy
                        if base_accuracy > 0
                        else 0
                    )
                    comparison["relative_performance"][f"head_{i+1}_vs_head_1"] = float(
                        relative
                    )

        # 성능 감소율 계산
        if len(comparison["accuracy_trend"]) > 1:
            decay_rates = []
            for i in range(1, len(comparison["accuracy_trend"])):
                decay = 1 - (
                    comparison["accuracy_trend"][i]
                    / comparison["accuracy_trend"][i - 1]
                    if comparison["accuracy_trend"][i - 1] > 0
                    else 0
                )
                decay_rates.append(float(decay))
            comparison["accuracy_decay_rates"] = decay_rates

        return comparison

    def _validate_meta_patterns(self, results: dict[str, Any]) -> None:
        """
        Meta 논문의 예상 패턴과 비교 검증.

        Meta 논문에 따르면:
        - 가까운 헤드(t+1)가 먼 헤드(t+4)보다 정확해야 함
        - 성능 감소가 단조적이어야 함
        """
        if (
            "head_comparison" in results
            and results["head_comparison"]["accuracy_trend"]
        ):
            trend = results["head_comparison"]["accuracy_trend"]

            # 패턴 1: head_1이 가장 정확해야 함
            if trend[0] == max(trend):
                console.print(
                    "[green]✓ Pattern confirmed: Head 1 (t+1) has highest accuracy[/green]"
                )
            else:
                console.print(
                    "[yellow]⚠ Warning: Head 1 is not the most accurate[/yellow]"
                )

            # 패턴 2: 단조 감소 패턴이어야 함
            is_monotonic = all(trend[i] >= trend[i + 1] for i in range(len(trend) - 1))
            if is_monotonic:
                console.print(
                    "[green]✓ Pattern confirmed: Monotonic accuracy decrease[/green]"
                )
            else:
                console.print(
                    "[yellow]⚠ Warning: Non-monotonic accuracy pattern[/yellow]"
                )

    def _extract_metrics(self, results: dict[str, Any]) -> dict[str, float]:
        """
        MLflow 로깅용 메트릭 추출.

        Args:
            results: 분석 결과

        Returns:
            메트릭 딕셔너리
        """
        metrics = {}

        # 헤드별 정확도
        for i in range(4):
            head_key = f"head_{i+1}"
            if head_key in results:
                metrics[f"head_{i+1}_accuracy"] = results[head_key]["accuracy"]["mean"]
                metrics[f"head_{i+1}_perplexity"] = results[head_key]["perplexity"][
                    "mean"
                ]

        # 비교 메트릭
        if "head_comparison" in results:
            comparison = results["head_comparison"]
            if comparison["accuracy_trend"]:
                metrics["head_accuracy_range"] = float(
                    max(comparison["accuracy_trend"])
                    - min(comparison["accuracy_trend"])
                )
            if "accuracy_decay_rates" in comparison:
                metrics["avg_accuracy_decay"] = float(
                    np.mean(comparison["accuracy_decay_rates"])
                )

        return metrics

    def _create_batches(self, dataset: Any) -> list[dict[str, torch.Tensor]]:
        """
        데이터셋을 배치로 분할.

        Args:
            dataset: 입력 데이터셋

        Returns:
            배치 리스트
        """
        # 간단한 구현 - 실제로는 DataLoader 사용 권장
        batches = []

        if isinstance(dataset, list):
            for i in range(0, len(dataset), self.batch_size):
                batch_data = dataset[i : i + self.batch_size]
                # 텐서로 변환
                if batch_data and isinstance(batch_data[0], dict):
                    batch = {
                        "input_ids": torch.stack([d["input_ids"] for d in batch_data]),
                        "labels": torch.stack(
                            [d.get("labels", d["input_ids"]) for d in batch_data]
                        ),
                    }
                    batches.append(batch)

        return batches

    def _create_dummy_dataset(self, tokenizer: Any) -> list[dict[str, torch.Tensor]]:
        """
        테스트용 더미 데이터셋 생성.

        Args:
            tokenizer: 토크나이저

        Returns:
            더미 데이터셋
        """
        console.print(
            "[yellow]No dataset provided, using dummy data for testing[/yellow]"
        )

        dataset = []
        for _ in range(10):  # 10개 샘플
            input_ids = torch.randint(0, tokenizer.vocab_size, (512,))
            dataset.append({"input_ids": input_ids, "labels": input_ids.clone()})

        return dataset
