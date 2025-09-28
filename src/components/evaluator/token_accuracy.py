"""
토큰 위치별 예측 정확도 분석기

WMTP 연구 맥락:
시퀀스 내 토큰 위치에 따른 예측 정확도를 분석합니다.
일반적으로 시퀀스 시작 부분이 끝 부분보다 예측하기 쉽습니다.

Meta 논문 통찰:
- 문맥이 누적될수록 예측 정확도 향상
- 긴 시퀀스에서 MTP의 장점이 더 크게 나타남
- 특정 토큰 유형(코드, 주석, 문자열)에 따른 성능 차이

분석 목적:
1. 시퀀스 위치별 예측 패턴 발견
2. 토큰 유형별 성능 차이 측정
3. WMTP 가중치가 위치별 정확도에 미치는 영향 평가
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


@evaluator_registry.register("token-accuracy", category="evaluator", version="1.0.0")
class TokenAccuracyAnalyzer(BaseComponent):
    """
    토큰 위치별 예측 정확도를 분석하는 평가기.

    시퀀스 내 각 위치에서의 예측 정확도를 측정하고,
    토큰 유형과 위치에 따른 성능 패턴을 분석합니다.

    분석 항목:
    - 절대 위치별 정확도
    - 상대 위치별 정확도 (시퀀스 대비 비율)
    - 토큰 유형별 정확도 (코드, 텍스트, 특수)
    - 헤드-위치 상호작용 분석

    설정 예시:
    ```yaml
    token-accuracy:
      position_range: [0, 100]
      token_types: ["code", "text", "special"]
      accuracy_threshold: 0.5
      granularity: 10  # 위치를 10개 단위로 그룹화
    ```
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.position_range = self.config.get("position_range", (0, 100))
        self.token_types = self.config.get("token_types", ["code", "text", "special"])
        self.accuracy_threshold = self.config.get("accuracy_threshold", 0.5)
        self.granularity = self.config.get("granularity", 10)
        self.analyze_token_types = self.config.get("analyze_token_types", True)
        self.device = self.config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.batch_size = self.config.get("batch_size", 8)

    def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """
        토큰 위치별 정확도 분석 실행.

        Args:
            ctx: 평가 컨텍스트
                - model: MTP 모델
                - tokenizer: 토크나이저
                - dataset: 평가 데이터셋 (선택)

        Returns:
            위치별 정확도 메트릭
        """
        self.validate_initialized()

        model = ctx.get("model")
        tokenizer = ctx.get("tokenizer")
        dataset = ctx.get("dataset")

        if not model or not tokenizer:
            raise ValueError(
                "Model and tokenizer are required for token accuracy analysis"
            )

        # 모델을 평가 모드로 설정
        model.eval()

        # 데이터셋이 없으면 더미 데이터 생성
        if dataset is None:
            dataset = self._create_dummy_dataset(tokenizer)

        console.print("[cyan]Starting token position accuracy analysis...[/cyan]")

        # 위치별 정확도 수집
        position_accuracies = defaultdict(list)  # position -> accuracy list
        token_type_accuracies = defaultdict(
            lambda: defaultdict(list)
        )  # type -> position -> accuracy list
        head_position_accuracies = defaultdict(
            lambda: defaultdict(list)
        )  # head -> position -> accuracy list

        # 배치별로 평가 수행
        total_samples = 0
        with torch.no_grad():
            for batch in tqdm(
                self._create_batches(dataset), desc="Analyzing token positions"
            ):
                batch_results = self._analyze_batch(model, tokenizer, batch)

                # 위치별 정확도 수집
                for pos, accuracies in batch_results["position_accuracies"].items():
                    position_accuracies[pos].extend(accuracies)

                # 토큰 유형별 정확도 수집
                if self.analyze_token_types:
                    for token_type, type_positions in batch_results[
                        "token_type_accuracies"
                    ].items():
                        for pos, accuracies in type_positions.items():
                            token_type_accuracies[token_type][pos].extend(accuracies)

                # 헤드-위치 상호작용 수집
                for head_idx, head_positions in batch_results[
                    "head_position_accuracies"
                ].items():
                    for pos, accuracies in head_positions.items():
                        head_position_accuracies[head_idx][pos].extend(accuracies)

                total_samples += batch["input_ids"].size(0)

        # 결과 집계
        results = self._aggregate_results(
            position_accuracies, token_type_accuracies, head_position_accuracies
        )

        # 패턴 분석
        patterns = self._analyze_patterns(results)
        results["patterns"] = patterns

        console.print(
            f"[green]Analyzed {total_samples} samples for token position accuracy[/green]"
        )

        return {"metrics": self._extract_metrics(results), "detailed_results": results}

    def _analyze_batch(
        self, model: nn.Module, tokenizer: Any, batch: dict[str, torch.Tensor]
    ) -> dict[str, Any]:
        """
        단일 배치에 대한 위치별 정확도 분석.

        Args:
            model: MTP 모델
            tokenizer: 토크나이저
            batch: 입력 배치

        Returns:
            위치별 분석 결과
        """
        input_ids = batch["input_ids"].to(self.device)
        labels = batch.get("labels", input_ids.clone())

        # 모델 추론
        outputs = model(input_ids)

        batch_results = {
            "position_accuracies": defaultdict(list),
            "token_type_accuracies": defaultdict(lambda: defaultdict(list)),
            "head_position_accuracies": defaultdict(lambda: defaultdict(list)),
        }

        if hasattr(outputs, "prediction_logits"):
            batch_size, seq_len = input_ids.shape

            # 각 헤드별로 분석
            for head_idx in range(4):
                head_logits = outputs.prediction_logits[:, :, head_idx, :]
                target_start = head_idx + 1

                if target_start < labels.size(1):
                    target_labels = labels[:, target_start:]
                    valid_length = min(head_logits.size(1), target_labels.size(1))

                    if valid_length > 0:
                        # 예측 계산
                        predictions = head_logits[:, :valid_length, :].argmax(dim=-1)
                        target_labels_valid = target_labels[:, :valid_length]

                        # 각 위치별 정확도 계산
                        for pos in range(min(valid_length, self.position_range[1])):
                            if pos >= self.position_range[0]:
                                # 위치별 정확도
                                pos_correct = (
                                    predictions[:, pos] == target_labels_valid[:, pos]
                                ).float()
                                batch_results["position_accuracies"][pos].extend(
                                    pos_correct.cpu().tolist()
                                )

                                # 헤드-위치 상호작용
                                batch_results["head_position_accuracies"][head_idx][
                                    pos
                                ].extend(pos_correct.cpu().tolist())

                                # 토큰 유형별 정확도 (선택적)
                                if self.analyze_token_types:
                                    for b in range(batch_size):
                                        token_id = target_labels_valid[b, pos].item()
                                        token_type = self._classify_token_type(
                                            token_id, tokenizer
                                        )
                                        batch_results["token_type_accuracies"][
                                            token_type
                                        ][pos].append(pos_correct[b].item())

        return batch_results

    def _classify_token_type(self, token_id: int, tokenizer: Any) -> str:
        """
        토큰을 유형별로 분류.

        Args:
            token_id: 토큰 ID
            tokenizer: 토크나이저

        Returns:
            토큰 유형 ("code", "text", "special")
        """
        # 토크나이저에서 토큰 텍스트 가져오기
        try:
            token_text = tokenizer.decode([token_id])
        except Exception:
            return "special"

        # 간단한 휴리스틱 기반 분류
        # 실제로는 더 정교한 분류기 사용 권장
        if token_id < 100:  # 특수 토큰 (보통 낮은 ID)
            return "special"
        elif any(c in token_text for c in "{}()[];=<>"):  # 코드 관련 문자
            return "code"
        elif token_text.strip() and token_text[0].isalpha():  # 텍스트
            return "text"
        else:
            return "special"

    def _aggregate_results(
        self,
        position_accuracies: dict[int, list[float]],
        token_type_accuracies: dict[str, dict[int, list[float]]],
        head_position_accuracies: dict[int, dict[int, list[float]]],
    ) -> dict[str, Any]:
        """
        위치별 정확도 결과 집계.

        Args:
            position_accuracies: 위치별 정확도
            token_type_accuracies: 토큰 유형별 정확도
            head_position_accuracies: 헤드-위치별 정확도

        Returns:
            집계된 결과
        """
        results = {}

        # 전체 위치별 정확도 평균
        position_avg = {}
        for pos in sorted(position_accuracies.keys()):
            if position_accuracies[pos]:
                position_avg[pos] = float(np.mean(position_accuracies[pos]))

        results["position_accuracies"] = position_avg

        # 그룹화된 위치별 정확도 (granularity 적용)
        grouped_accuracies = defaultdict(list)
        for pos, acc in position_avg.items():
            group = pos // self.granularity * self.granularity
            grouped_accuracies[f"pos_{group}_{group+self.granularity}"].append(acc)

        results["grouped_position_accuracies"] = {
            group: float(np.mean(accs)) for group, accs in grouped_accuracies.items()
        }

        # 토큰 유형별 정확도
        if token_type_accuracies:
            type_results = {}
            for token_type, type_positions in token_type_accuracies.items():
                type_avg = {}
                for pos, accs in type_positions.items():
                    if accs:
                        type_avg[pos] = float(np.mean(accs))
                if type_avg:
                    type_results[token_type] = {
                        "position_accuracies": type_avg,
                        "overall_accuracy": float(np.mean(list(type_avg.values()))),
                    }
            results["token_type_accuracies"] = type_results

        # 헤드-위치 상호작용
        head_results = {}
        for head_idx, head_positions in head_position_accuracies.items():
            head_avg = {}
            for pos, accs in head_positions.items():
                if accs:
                    head_avg[pos] = float(np.mean(accs))
            if head_avg:
                head_results[f"head_{head_idx+1}"] = {
                    "position_accuracies": head_avg,
                    "overall_accuracy": float(np.mean(list(head_avg.values()))),
                }
        results["head_position_interactions"] = head_results

        return results

    def _analyze_patterns(self, results: dict[str, Any]) -> dict[str, Any]:
        """
        위치별 정확도 패턴 분석.

        Args:
            results: 집계된 결과

        Returns:
            발견된 패턴
        """
        patterns = {}

        # 위치별 정확도 트렌드 분석
        if "position_accuracies" in results and results["position_accuracies"]:
            positions = sorted(results["position_accuracies"].keys())
            accuracies = [results["position_accuracies"][p] for p in positions]

            if len(accuracies) > 1:
                # 선형 회귀를 통한 트렌드 계산
                x = np.array(positions)
                y = np.array(accuracies)
                if len(x) > 1:
                    # 간단한 선형 회귀
                    A = np.vstack([x, np.ones(len(x))]).T
                    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
                    patterns["accuracy_trend"] = {
                        "slope": float(slope),
                        "intercept": float(intercept),
                        "trend": "increasing"
                        if slope > 0.001
                        else "decreasing"
                        if slope < -0.001
                        else "stable",
                    }

                # 정확도 범위
                patterns["accuracy_range"] = {
                    "min": float(min(accuracies)),
                    "max": float(max(accuracies)),
                    "mean": float(np.mean(accuracies)),
                    "std": float(np.std(accuracies)),
                }

                # 임계값 이하 위치 찾기
                below_threshold = [
                    p
                    for p, a in zip(positions, accuracies)
                    if a < self.accuracy_threshold
                ]
                if below_threshold:
                    patterns["below_threshold_positions"] = below_threshold[
                        :10
                    ]  # 처음 10개만

        # 토큰 유형별 패턴
        if "token_type_accuracies" in results and results["token_type_accuracies"]:
            type_comparison = {}
            for token_type, type_data in results["token_type_accuracies"].items():
                type_comparison[token_type] = type_data["overall_accuracy"]

            if type_comparison:
                best_type = max(type_comparison, key=type_comparison.get)
                worst_type = min(type_comparison, key=type_comparison.get)
                patterns["token_type_performance"] = {
                    "best_type": best_type,
                    "best_accuracy": type_comparison[best_type],
                    "worst_type": worst_type,
                    "worst_accuracy": type_comparison[worst_type],
                    "all_types": type_comparison,
                }

        # 헤드별 위치 성능 패턴
        if (
            "head_position_interactions" in results
            and results["head_position_interactions"]
        ):
            head_comparison = {}
            for head_name, head_data in results["head_position_interactions"].items():
                head_comparison[head_name] = head_data["overall_accuracy"]

            if head_comparison:
                patterns["head_position_performance"] = {
                    "head_accuracies": head_comparison,
                    "best_head": max(head_comparison, key=head_comparison.get),
                    "consistency": float(1.0 - np.std(list(head_comparison.values()))),
                }

        return patterns

    def _extract_metrics(self, results: dict[str, Any]) -> dict[str, float]:
        """
        MLflow 로깅용 메트릭 추출.

        Args:
            results: 분석 결과

        Returns:
            메트릭 딕셔너리
        """
        metrics = {}

        # 전체 평균 정확도
        if "position_accuracies" in results and results["position_accuracies"]:
            all_accuracies = list(results["position_accuracies"].values())
            metrics["avg_position_accuracy"] = float(np.mean(all_accuracies))
            metrics["std_position_accuracy"] = float(np.std(all_accuracies))

        # 패턴 메트릭
        if "patterns" in results:
            patterns = results["patterns"]
            if "accuracy_trend" in patterns:
                metrics["accuracy_trend_slope"] = patterns["accuracy_trend"]["slope"]
            if "accuracy_range" in patterns:
                metrics["accuracy_range_span"] = (
                    patterns["accuracy_range"]["max"]
                    - patterns["accuracy_range"]["min"]
                )
            if "token_type_performance" in patterns:
                metrics["best_token_type_accuracy"] = patterns[
                    "token_type_performance"
                ]["best_accuracy"]

        # 그룹화된 위치별 메트릭
        if "grouped_position_accuracies" in results:
            for group, acc in results["grouped_position_accuracies"].items():
                metrics[f"accuracy_{group}"] = acc

        return metrics

    def _create_batches(self, dataset: Any) -> list[dict[str, torch.Tensor]]:
        """
        데이터셋을 배치로 분할.

        Args:
            dataset: 입력 데이터셋

        Returns:
            배치 리스트
        """
        batches = []

        if isinstance(dataset, list):
            for i in range(0, len(dataset), self.batch_size):
                batch_data = dataset[i : i + self.batch_size]
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
            # 다양한 토큰 유형을 포함한 더미 데이터
            input_ids = torch.randint(0, tokenizer.vocab_size, (256,))
            dataset.append({"input_ids": input_ids, "labels": input_ids.clone()})

        return dataset
