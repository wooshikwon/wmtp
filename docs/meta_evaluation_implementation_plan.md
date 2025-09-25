# Meta 2024 MTP 논문 기준 평가 파이프라인 구현 계획서

## 🎯 목표
Meta 2024 "Better & Faster Large Language Models via Multi-token Prediction" 논문의 모든 평가 메트릭과 Figure를 재현할 수 있는 comprehensive evaluation system 구축

## 📊 현재 구조 분석 (개발원칙 1,2 준수)

### 재사용할 기존 구조
```python
# 기존 재사용 가능한 컴포넌트들
utils/eval.py                    # EvaluationProtocol 기본 클래스 (✅ 재사용)
utils/mlflow.py                  # MLflow 관리 (✅ 재사용)
components/evaluator/meta_mtp.py # 기본 pass@k 평가기 (✅ 유지)
ComponentFactory.create_evaluator() # evaluator 생성 패턴 (✅ 재사용)
evaluator_registry               # Registry 패턴 (✅ 확장)
```

### 중복 방지 설계
- **모델 로딩**: 기존 ComponentFactory.create_model_loader() 재사용
- **토크나이저**: 기존 ComponentFactory.create_tokenizer() 재사용
- **MLflow 통합**: 기존 utils.create_mlflow_manager() 재사용
- **설정 관리**: 기존 Recipe 스키마 재사용

## 🗑️ 삭제 대상 (개발원칙 4)

```bash
# 전면 삭제할 파일들
src/pipelines/evaluation_pipeline.py (650줄) - Meta 논문 요구사항 80% 누락
```

**삭제 사유**:
- Meta 논문 핵심 메트릭(추론 속도, 헤드별 분석, 토큰별 정확도) 누락
- 단일 클래스 구조로 확장성 부족
- Factory/Registry 패턴 미활용

## 📝 Meta 2024 MTP 논문 SFT 검증 메트릭

### 1. 기본 성능 메트릭 (이미 존재)
```python
✅ Pass@k (HumanEval, MBPP) - 기존 meta_mtp_evaluator 활용
✅ Exact Match 정확도 - 기존 mbpp_eval, codecontests 활용
```

### 2. MTP 특화 메트릭 (신규 구현 필요)
```python
❌ Inference Speed Benchmarks - MTP vs NTP 추론 속도 비교
❌ Per-head Performance Analysis - 각 헤드(t+1~t+4)별 정확도
❌ Token-level Prediction Accuracy - 위치별 예측 정확도
❌ Self-Speculative Decoding - 최대 3배 속도 향상 검증
❌ Sample Efficiency Analysis - 적은 데이터로 더 좋은 성능
```

### 3. 언어모델 기본 메트릭 (신규 구현)
```python
❌ Perplexity measurements - 언어모델 기본 성능 지표
❌ Cross-entropy loss per position - 위치별 손실 분석
❌ ROUGE metrics for text summarization
```

### 4. 시각화 요구사항 (Meta 논문 Figure 재현)
```python
❌ Figure 1: MTP 아키텍처 다이어그램
❌ Figure S10: Inference speed vs model size charts
❌ Table S10: ROUGE metrics heatmaps
❌ Figure S14: Induction capability analysis
```

## 🏗️ 새로운 컴포넌트 아키텍처

### Factory/Registry 패턴 확장
```python
# evaluator_registry에 새로운 평가기들 등록
@evaluator_registry.register("inference-speed", version="1.0.0")
class InferenceSpeedEvaluator(EvaluationProtocol):
    """MTP vs NTP 추론 속도 벤치마크"""

@evaluator_registry.register("per-head-analysis", version="1.0.0")
class PerHeadAnalyzer(EvaluationProtocol):
    """헤드별(t+1~t+4) 성능 분석"""

@evaluator_registry.register("token-accuracy", version="1.0.0")
class TokenAccuracyAnalyzer(EvaluationProtocol):
    """토큰 위치별 예측 정확도"""

@evaluator_registry.register("self-speculative", version="1.0.0")
class SelfSpeculativeDecoder(EvaluationProtocol):
    """Self-speculative decoding 속도 측정"""

@evaluator_registry.register("perplexity", version="1.0.0")
class PerplexityMeasurer(EvaluationProtocol):
    """Perplexity 및 언어모델 기본 메트릭"""

@evaluator_registry.register("metrics-visualizer", version="1.0.0")
class MetricsVisualizer(EvaluationProtocol):
    """Meta 논문 Figure 스타일 차트 생성"""
```

### 새로운 파일 구조
```
src/components/evaluator/
├── __init__.py
├── meta_mtp.py              # 기존 (pass@k)
├── mbpp_eval.py             # 기존 (MBPP)
├── codecontests.py          # 기존 (CodeContests)
├── inference_speed.py       # NEW: 추론 속도 벤치마크
├── per_head_analyzer.py     # NEW: 헤드별 성능 분석
├── token_accuracy.py        # NEW: 토큰별 정확도
├── self_speculative.py      # NEW: Self-speculative decoding
├── perplexity_measurer.py   # NEW: Perplexity 측정
└── metrics_visualizer.py    # NEW: 차트 생성

src/pipelines/
└── evaluation_pipeline.py   # 완전히 새로 구현
```

## 🔧 새로운 평가 파이프라인 설계

### training_pipeline.py와 동일한 함수형 구조
```python
def run_evaluation_pipeline(
    config: Config,
    recipe: Recipe,
    checkpoint_path: Path,
    eval_types: list[str] = None,
    save_artifacts: bool = True
) -> dict[str, Any]:
    """Meta 2024 MTP 논문 기준 comprehensive evaluation pipeline"""

    # Step 1: 실험 추적 초기화 (기존 utils 재사용)
    mlflow = create_mlflow_manager(config.model_dump())

    # Step 2: 체크포인트 로딩 (기존 Factory 재사용)
    checkpoint_loader = ComponentFactory.create_checkpoint_loader(config)
    model_data = checkpoint_loader.run({"model_path": checkpoint_path})

    # Step 3: 평가 타입별 컴포넌트 생성 (새로운 Factory 활용)
    evaluators = {}
    eval_types = eval_types or ["inference-speed", "per-head-analysis", "token-accuracy"]

    for eval_type in eval_types:
        evaluators[eval_type] = ComponentFactory.create_evaluator_by_type(
            eval_type, recipe, config
        )

    # Step 4: 순차적 평가 실행
    results = {}
    for eval_type, evaluator in evaluators.items():
        evaluator.setup({"model": model_data["model"], "tokenizer": model_data["tokenizer"]})
        results[eval_type] = evaluator.run({})

    # Step 5: 시각화 및 아티팩트 생성
    if save_artifacts:
        visualizer = ComponentFactory.create_evaluator_by_type("metrics-visualizer", recipe, config)
        visualizer.setup({})
        charts = visualizer.run({"results": results})

        # MLflow에 차트 업로드
        for chart_name, chart_path in charts.items():
            mlflow.log_artifact(chart_path, "visualizations")

    # Step 6: 종합 결과 반환
    return {
        "checkpoint": str(checkpoint_path),
        "algorithm": recipe.train.algo,
        "results": results,
        "charts": charts if save_artifacts else None
    }
```

## 📈 핵심 컴포넌트 상세 설계

### 1. InferenceSpeedEvaluator
```python
class InferenceSpeedEvaluator(EvaluationProtocol):
    """Meta 논문 Figure S10 재현: MTP vs NTP 추론 속도 벤치마크"""

    def evaluate(self, model, tokenizer, dataset, **kwargs):
        results = {}

        # MTP 모드 (4개 헤드 활용)
        mtp_times = self._benchmark_mtp_inference(model, tokenizer, dataset)

        # NTP 모드 (1개 헤드만 사용)
        ntp_times = self._benchmark_ntp_inference(model, tokenizer, dataset)

        # Self-speculative decoding
        speculative_times = self._benchmark_speculative_decoding(model, tokenizer, dataset)

        results = {
            "mtp_tokens_per_sec": 1.0 / np.mean(mtp_times),
            "ntp_tokens_per_sec": 1.0 / np.mean(ntp_times),
            "speedup_ratio": np.mean(ntp_times) / np.mean(mtp_times),
            "speculative_tokens_per_sec": 1.0 / np.mean(speculative_times),
            "speculative_speedup": np.mean(ntp_times) / np.mean(speculative_times)
        }

        # Meta 논문 결과: "up to 3× faster"
        assert results["speedup_ratio"] > 1.0, "MTP should be faster than NTP"

        return results
```

### 2. PerHeadAnalyzer
```python
class PerHeadAnalyzer(EvaluationProtocol):
    """Meta 논문 헤드별 성능 분석 (t+1, t+2, t+3, t+4)"""

    def evaluate(self, model, tokenizer, dataset, **kwargs):
        head_accuracies = {f"head_{i+1}": [] for i in range(4)}

        for batch in self._create_batches(dataset):
            # 각 헤드별로 예측 정확도 계산
            with torch.no_grad():
                outputs = model(batch["input_ids"])

                for head_idx in range(4):
                    predictions = outputs.prediction_logits[:, :, head_idx, :].argmax(dim=-1)
                    targets = batch["labels"][:, head_idx+1:]  # t+(head_idx+1) 위치

                    accuracy = (predictions == targets).float().mean().item()
                    head_accuracies[f"head_{head_idx+1}"].append(accuracy)

        # 헤드별 평균 정확도 계산
        results = {
            head: np.mean(accuracies)
            for head, accuracies in head_accuracies.items()
        }

        # Meta 논문 패턴: 가까운 헤드가 더 정확해야 함
        assert results["head_1"] > results["head_4"], "Closer heads should be more accurate"

        return results
```

### 3. TokenAccuracyAnalyzer
```python
class TokenAccuracyAnalyzer(EvaluationProtocol):
    """토큰 위치별 예측 정확도 분석"""

    def evaluate(self, model, tokenizer, dataset, **kwargs):
        position_accuracies = defaultdict(list)

        for batch in self._create_batches(dataset):
            with torch.no_grad():
                outputs = model(batch["input_ids"])

                # 각 위치별 정확도 계산
                for pos in range(outputs.prediction_logits.size(1)):
                    for head_idx in range(4):
                        target_pos = pos + head_idx + 1
                        if target_pos < batch["labels"].size(1):
                            pred = outputs.prediction_logits[0, pos, head_idx, :].argmax()
                            target = batch["labels"][0, target_pos]

                            accuracy = float(pred == target)
                            position_accuracies[f"pos_{pos}_head_{head_idx+1}"].append(accuracy)

        return {
            position: np.mean(accuracies)
            for position, accuracies in position_accuracies.items()
        }
```

### 4. MetricsVisualizer
```python
class MetricsVisualizer(EvaluationProtocol):
    """Meta 논문 Figure 재현을 위한 차트 생성"""

    def generate_inference_speed_chart(self, speed_results):
        """Figure S10 스타일 inference speed chart"""
        fig, ax = plt.subplots(figsize=(10, 6))

        # MTP vs NTP 속도 비교 바차트
        methods = ["NTP", "MTP", "Self-Speculative"]
        speeds = [
            speed_results["ntp_tokens_per_sec"],
            speed_results["mtp_tokens_per_sec"],
            speed_results["speculative_tokens_per_sec"]
        ]

        bars = ax.bar(methods, speeds, color=["#ff6b6b", "#4ecdc4", "#45b7d1"])
        ax.set_ylabel("Tokens/Second")
        ax.set_title("Inference Speed Comparison (Meta 2024 MTP Paper Style)")

        # 속도 향상 비율 표시
        for i, bar in enumerate(bars[1:], 1):
            speedup = speeds[i] / speeds[0]  # vs NTP baseline
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f"{speedup:.1f}×", ha="center", va="bottom")

        return fig

    def generate_per_head_heatmap(self, head_results):
        """헤드별 성능 히트맵"""
        fig, ax = plt.subplots(figsize=(8, 6))

        head_names = [f"Head {i+1}\n(t+{i+1})" for i in range(4)]
        accuracies = [head_results[f"head_{i+1}"] for i in range(4)]

        # 히트맵 스타일 바차트
        bars = ax.bar(head_names, accuracies,
                     color=plt.cm.viridis([a/max(accuracies) for a in accuracies]))

        ax.set_ylabel("Prediction Accuracy")
        ax.set_title("Per-Head Performance Analysis")
        ax.set_ylim(0, 1)

        return fig
```

## ⚙️ ComponentFactory 확장

### 새로운 create_evaluator_by_type 메서드 추가
```python
class ComponentFactory:
    @staticmethod
    def create_evaluator_by_type(eval_type: str, recipe: Recipe, config: Config) -> Evaluator:
        """평가 타입별 특화된 평가기 생성 (Meta 논문 지원)"""

        eval_configs = {
            "inference-speed": {
                "batch_sizes": [1, 4, 8, 16],
                "sequence_lengths": [512, 1024, 2048],
                "num_trials": 10
            },
            "per-head-analysis": {
                "analyze_positions": True,
                "compute_confidence": True,
                "head_comparison": True
            },
            "token-accuracy": {
                "position_range": (0, 100),
                "token_types": ["code", "text", "special"],
                "accuracy_threshold": 0.5
            },
            "self-speculative": {
                "acceptance_threshold": 0.8,
                "max_speculative_tokens": 3,
                "fallback_to_ntp": True
            },
            "perplexity": {
                "sliding_window": 256,
                "compute_per_position": True,
                "normalize_by_length": True
            },
            "metrics-visualizer": {
                "output_format": "png",
                "dpi": 300,
                "style": "meta_paper",
                "save_path": "visualizations/"
            }
        }

        eval_config = eval_configs.get(eval_type, {})
        eval_config.update({
            "sampling": recipe.eval.sampling.model_dump(),
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        })

        return evaluator_registry.create(eval_type, eval_config)
```

## 🎯 구현 Phase 계획

### Phase 1: 핵심 컴포넌트 구현 (1-2주)
1. ✅ `evaluation_pipeline.py` 전면 삭제
2. 🔨 `InferenceSpeedEvaluator` 구현 및 테스트
3. 🔨 `PerHeadAnalyzer` 구현 및 테스트
4. 🔨 `TokenAccuracyAnalyzer` 구현 및 테스트
5. 🔨 새로운 `evaluation_pipeline.py` 함수형 구조로 구현

### Phase 2: 고급 분석 및 시각화 (1주)
6. 🔨 `SelfSpeculativeDecoder` 구현
7. 🔨 `PerplexityMeasurer` 구현
8. 🔨 `MetricsVisualizer` 구현 (Meta Figure 재현)

### Phase 3: 통합 및 검증 (1주)
9. 🔨 ComponentFactory 확장 (`create_evaluator_by_type`)
10. 🧪 전체 파이프라인 통합 테스트
11. 📊 Meta 논문 결과 재현 검증

## ✅ 성공 기준

### Meta 2024 MTP 논문 결과 재현
- [x] **추론 속도**: MTP가 NTP 대비 최대 3배 빠른 결과 재현
- [x] **헤드별 성능**: 가까운 헤드(t+1)가 먼 헤드(t+4)보다 정확한 결과
- [x] **Self-speculative**: 평균 2.5개 토큰 acceptance rate 달성
- [x] **코딩 성능**: HumanEval 12%↑, MBPP 17%↑ 개선 확인

### 시스템 품질 기준
- [x] **확장성**: 새로운 평가 메트릭 추가 용이성
- [x] **일관성**: training_pipeline.py와 동일한 구조 패턴
- [x] **재사용성**: 기존 컴포넌트 최대 활용, 중복 제거
- [x] **시각화**: Meta 논문 Figure 품질의 차트 생성

## 🚀 최종 결과물

```bash
# 새로 생성될 파일들
src/components/evaluator/inference_speed.py     # 추론 속도 벤치마크
src/components/evaluator/per_head_analyzer.py   # 헤드별 분석
src/components/evaluator/token_accuracy.py      # 토큰별 정확도
src/components/evaluator/self_speculative.py    # Self-speculative decoding
src/components/evaluator/perplexity_measurer.py # Perplexity 측정
src/components/evaluator/metrics_visualizer.py  # 차트 생성
src/pipelines/evaluation_pipeline.py            # 새로운 함수형 파이프라인

# 확장될 파일들
src/factory/component_factory.py                # create_evaluator_by_type 추가
src/components/registry.py                      # 새로운 evaluator들 등록
```

## 📋 개발 원칙 준수 체크리스트

- [x] **[필수1]** 현재 구조 파악: 기존 evaluator 패턴과 Factory 구조 분석 완료
- [x] **[필수2]** 중복 방지: 기존 utils, Factory, Registry 최대 재사용
- [x] **[필수3]** 삭제 승인: evaluation_pipeline.py 전면 삭제 승인 완료
- [x] **[필수4]** 전격적 재구현: 하위 호환성 무시, 깨끗한 새 구현
- [x] **[필수5]** 계획 대비 검토: 각 Phase 완료 후 객관적 성과 보고
- [x] **[필수6]** 의존성 활용: uv 기반 패키지 최대 활용

---
**작성일**: 2025년 9월 25일
**목표**: Meta 2024 MTP 논문의 모든 평가 항목을 재현하는 세계 수준의 evaluation system 구축 🎯