물론입니다. Phase 5부터 Phase 15까지, 전체 개발 계획을 완수할 수 있도록 각 단계별 LLM Agent용 '바이브 코딩' 프롬프트를 순차적으로 생성해 드리겠습니다.

---

### **Phase 5: Data & Model Loaders**

**Context:** 실험의 재료인 데이터와 모델을 가져오는 로더를 구현합니다. `DEV_PRINCIPLE`의 '로컬 우선 → S3 미러링' 정책을 구현하여 개발 생산성과 재현성을 모두 잡는 것이 핵심입니다.

**Your Task:**
* `components/loader/`에 `BLUEPRINT`에 명시된 데이터셋 로더(`dataset_mbpp_loader.py` 등)와 모델 로더(`hf_local_s3_loader.py`)를 구현해주세요.
* 로더는 지정된 로컬 경로에 파일이 있으면 즉시 사용하고, 없으면 S3에서 `.cache/` 디렉토리로 다운로드(미러링)한 후 사용해야 합니다.
* 데이터셋 캐시 디렉토리는 `(데이터 버전 + 전처리 옵션 + 스플릿 시드)`를 조합한 해시 키로 결정되도록 구현해주세요.

**Remember to:**
* 스플릿 시드는 42로 고정합니다.
* 모든 S3 관련 로직은 `src/utils/s3.py`에 구현된 헬퍼 함수를 통해 호출되어야 합니다.

**DoD (Definition of Done):**
* 로컬에만 파일이 있는 경우, S3에만 파일이 있는 경우, 둘 다 있는 경우(로컬 우선) 모두에서 로더가 정상적으로 데이터를 로드해야 합니다.

> 좋아, 이제 어떤 환경에서도 동일한 데이터를 안정적으로 공급할 수 있게 됐어. 실험의 재현성을 위한 첫 단추를 잘 끼웠네!

---

### **Phase 6: Scorers (Critic & Rho-1)**

**Context:** 우리 연구의 핵심 아이디어를 코드로 구현하는 가장 중요한 단계입니다. 두 가지 토큰 중요도 산출 알고리즘을 동일한 인터페이스를 따르는 Scorer 컴포넌트로 각각 구현합니다.

**Your Task:**
* `components/scorer/critic_delta_v1.py`를 구현해주세요. RM의 시퀀스 보상을 GAE 방식으로 토큰별로 분배하고, Value Head 회귀를 통해 $\delta_t$를 계산한 뒤, z-score, softmax(T), mean=1.0 정규화 및 클리핑을 적용해야 합니다.
* `components/scorer/rho1_excess_v1.py`를 구현해주세요. 참조 모델과 베이스 모델의 CE 값 차이($|CE_{ref} - CE_{base}|$)를 계산하고, 마찬가지로 정규화 및 클리핑 과정을 거쳐 가중치를 산출해야 합니다.
* 두 Scorer의 출력 가중치는 **평균 1.0±ε을 유지하고, NaN/Inf가 없으며, 지정된 범위 내([ε, Wmax])** 여야 한다는 통계적 불변식을 반드시 만족해야 합니다.

**Remember to:**
* 이 단계에서 구현되는 로직은 연구의 독창성과 직결되므로, `BLUEPRINT`의 수식을 정확하게 코드로 옮겨야 합니다.
* 통계적 불변식은 나중에 `Phase 12`에서 단위 테스트로 검증할 것입니다.

**DoD (Definition of Done):**
* 두 Scorer는 동일한 인터페이스를 가지므로, `recipe.yaml`에서 키 변경만으로 Trainer 코드 수정 없이 서로 교체되어 적용될 수 있어야 합니다.

> 굉장해! 드디어 연구 아이디어가 살아 숨 쉬는 코드가 됐어. 이제 이 가중치를 모델에 먹여줄 일만 남았네.

---

### **Phase 7: Trainer (MTP Weighted CE) + Optimizer**

**Context:** Scorer가 계산한 토큰 가중치를 실제 MTP 모델의 손실 함수에 적용하는 Trainer를 구현합니다. FSDP 분산 학습, 자동 혼합 정밀도(AMP), 안정적인 학습을 위한 모든 장치가 여기에 통합됩니다.

**Your Task:**
* `components/trainer/mtp_weighted_ce_trainer.py`를 구현해주세요. 이 Trainer는 Scorer로부터 받은 토큰별 가중치를 MTP의 병렬 헤드에서 계산된 Cross-Entropy 손실에 곱하여 최종 가중 손실을 계산해야 합니다.
* `recipe.yaml` 설정에 따라 AdamW 옵티마이저와 코사인 스케줄러(warmup 3%)를 생성하는 팩토리를 구현하고, 그래디언트 클리핑(1.0)을 적용해주세요.
* `train.lora.enabled` 값에 따라 PEFT(LoRA) 로직으로 분기하도록 구현해주세요.

**Remember to:**
* '안전한 성능' 원칙에 따라, **OOM 발생 시 그래디언트 축적(accumulation) 스텝을 자동으로 늘리거나, NaN 발생 시 학습률을 줄이는 등의 폴백(fallback) 로직**을 포함해야 합니다.

**DoD (Definition of Done):**
* 학습 중 NaN이나 OOM이 발생했을 때, 프로그램이 그냥 죽는 게 아니라 정의된 안정화 폴백 로직이 정상적으로 작동해야 합니다.

> 좋아, 우리 모델의 엔진이 완성됐어. 강력할 뿐만 아니라 어떤 상황에서도 안정적으로 돌아가는 심장을 가졌네!

---

### **Phase 8: Pipelines Orchestration**

**Context:** 지금까지 만든 레고 블록들(Loader, Scorer, Trainer, Evaluator)을 조립하여 `critic-wmtp`와 `rho1-wmtp`라는 두 개의 완전한 파이프라인을 만듭니다.

**Your Task:**
* `src/pipelines/` 디렉토리에 `critic_wmtp.py`와 `rho1_wmtp.py`를 구현해주세요.
* 각 파이프라인은 설정 파일을 읽고, 팩토리를 통해 필요한 컴포넌트들을 생성한 뒤, `데이터 로딩 → 학습 → 평가`의 흐름으로 오케스트레이션해야 합니다.
* 컴포넌트 간 데이터 전달은 `ctx` (context)라는 공통 딕셔너리를 통해 명시적으로 이루어져야 합니다.

**Remember to:**
* 파이프라인의 역할은 **오케스트레이션**에만 집중해야 합니다. 실제 비즈니스 로직은 각 컴포넌트 내부에 캡슐화되어 있어야 합니다.

**DoD (Definition of Done):**
* 아주 작은 더미 모델과 데이터셋을 사용한 100 step짜리 스모크 테스트에서 두 파이프라인 모두 에러 없이 시작부터 끝까지 완주해야 합니다.

> 모든 조각이 제자리를 찾았어! 이제 버튼 하나만 누르면 전체 실험이 자동으로 흘러가는 멋진 자동화 시스템이 완성됐네.

---

### **Phase 9: Evaluation Harness (Meta MTP Protocol)**

**Context:** 실험의 성공 여부를 판단할 평가 장치를 구현합니다. Meta의 MTP 논문에서 사용한 평가 프로토콜을 그대로 재현하여 연구 결과의 신뢰도를 확보하는 것이 목표입니다.

**Your Task:**
* `components/evaluator/`에 `mbpp_eval_v1.py`와 `contest_eval_v1.py`를 구현해주세요.
* 평가는 `BLUEPRINT`에 명시된 샘플링 파라미터(T=0.2, top-p=0.95)를 고정하여 사용해야 합니다.
* 평가 지표는 **MBPP의 exact-match 정확도**와 **CodeContests의 pass@k**를 계산해야 합니다.

**Remember to:**
* 평가 결과는 수치뿐만 아니라, 일부 예측/정답 샘플, 헤드별 CE 통계, 가중치 분포 히스토그램 등 정성적 분석이 가능한 아티팩트와 함께 MLflow에 로깅되어야 합니다.

**DoD (Definition of Done):**
* 미리 준비된 고정된 샘플 예측 결과에 대해, 평가 스크립트가 항상 동일한 점수(예: pass@1 = 0.5)를 출력하는 리그레션 테스트를 통과해야 합니다.

> 이제 우리의 모델이 얼마나 똑똑해졌는지 정확하게 측정할 수 있는 자(ruler)가 생겼어. 모든 비교는 이 공정한 기준 위에서 이루어질 거야.

---

*이후 프롬프트는 아래와 같이 계속됩니다.*

### **Phase 10: MLflow (S3) Integration & Experiment Taxonomy**

**Context:** 모든 실험 기록을 체계적으로 관리하기 위해 MLflow 연동을 완성합니다. 로컬에서 실행하든, VESSL 클러스터에서 실행하든 모든 결과가 중앙 S3 저장소에 동일한 구조로 기록되도록 합니다.

**Your Task:**
* `utils/mlflow.py`의 로깅 함수들을 사용하여 파이프라인 곳곳에 파라미터, 메트릭, 아티팩트 로깅을 추가해주세요.
* 실험 이름은 `mtp/{algo}/{dataset}` 규칙을 따르도록 `utils/mlflow.py`의 초기화 함수를 수정해주세요.
* 아티팩트는 `checkpoints/`, `reports/` 등 표준화된 디렉토리 구조로 저장되도록 해주세요.

**DoD (Definition of Done):**
* 로컬 PC에서 실행한 스모크 테스트와 VESSL에서 실행한 스모크 테스트의 결과가 MLflow UI에서 동일한 실험(Experiment) 트리 구조 아래에 각각의 실행(Run)으로 기록되어야 합니다.

> 완벽해! 이제 우리의 모든 실험은 투명하게 기록되고, 언제든 추적하고 재현할 수 있어. 더 이상 "그때 그 실험 어떻게 했더라?" 하는 일은 없을 거야.

---

### **Phase 11: Dockerization (CUDA 12.1) & VESSL Spec**

**Context:** 개발 환경을 그대로 클러스터 환경으로 복제하기 위해 Docker 이미지를 만들고, VESSL 실행 스펙을 정의합니다. '내 PC에선 됐는데...'라는 문제를 원천 차단합니다.

**Your Task:**
* `docker/Dockerfile`을 작성해주세요. `pytorch/pytorch:2.4.0-cuda12.1-cudnn9` 이미지를 기반으로, `uv`를 설치하고 `uv sync --frozen` 명령으로 의존성을 설치해야 합니다.
* `docker/vessl.yaml` 파일을 작성해주세요. A100 GPU 4개, CPU, 메모리 등 리소스 요구사항과 S3 접근을 위한 시크릿 변수 설정, 그리고 실행 명령어를 명시해야 합니다.

**DoD (Definition of Done):**
* `docker/vessl.yaml` 스펙을 사용하여 VESSL에 실행(run)을 제출했을 때, 컨테이너가 정상적으로 시작되고 `uv run python -m src.cli.train ...` 명령어가 에러 없이 실행되어야 합니다.

> 좋아, 이제 우리 코드는 어디든 갈 수 있는 튼튼한 집(컨테이너)을 갖게 됐어. 개발부터 운영까지 동일한 환경을 보장할 수 있게 됐네.

---

### **Phase 12: Test Pyramid & Quality Gates**

**Context:** 프로젝트의 안정성과 코드 품질을 보장하기 위한 자동화된 테스트 스위트를 구축합니다. '테스트는 기능이다'라는 원칙을 실현합니다.

**Your Task:**
* `tests/` 디렉토리 아래에 단위/통합 테스트를 추가해주세요.
    * **Unit:** Pydantic 스키마 검증, Scorer의 통계적 불변식(평균 1.0, NaN 없음) 테스트.
    * **Integration:** 작은 설정으로 파이프라인 전체가 도는지 확인하는 스모크 테스트, MLflow에 실제로 로그가 남는지 확인하는 테스트.
* 오래 걸리는 테스트는 `@pytest.mark.slow` 마커를 붙여 분리해주세요.
* GitHub Actions 같은 CI(지속적 통합) 시스템에 연동하여, PR이 `develop` 브랜치에 합쳐지기 전에 모든 테스트와 린트 검사를 통과하도록 강제해주세요.

**DoD (Definition of Done):**
* CI에서 린트와 필수 테스트(slow 제외)를 통과하지 못한 PR은 `main` 또는 `develop` 브랜치에 병합(merge)될 수 없어야 합니다.

> 훌륭해! 이제 우리는 코드가 언제나 최상의 품질을 유지하도록 지켜주는 든든한 자동화된 수호자를 갖게 됐어.

---

### **Phase 13: First Real Run (Local → Single GPU → VESSL x4 A100)**

**Context:** 드디어 실제 데이터와 모델로 첫 실험을 진행합니다. 작은 규모에서 시작하여 점진적으로 스케일을 키워나가며 파이프라인의 모든 부분이 실제 환경에서 잘 작동하는지 최종 점검합니다.

**Your Task:**
* `exp/` 디렉토리에 `critic_mbpp_small.yaml` 같은 작은 규모의 실험 레시피를 작성해주세요.
* 먼저 로컬 CPU 환경에서 dry-run을, 다음으로 단일 GPU 환경에서 작은 데이터셋으로, 마지막으로 `vessl.yaml`을 이용해 A100 4대 환경에서 전체 데이터셋으로 학습을 실행해주세요.
* 이 과정에서 VRAM 사용량, 처리 속도(tok/s) 등 성능을 프로파일링하고, 토큰 예산이나 그래디언트 축적 스텝을 최적화해주세요.

**DoD (Definition of Done):**
* 두 파이프라인(`critic-wmtp`, `rho1-wmtp`) 모두 VESSL A100 4대 환경에서 MBPP 데이터셋 전체에 대한 end-to-end 학습 및 평가를 성공적으로 완료하고, 모든 결과가 MLflow에 정상적으로 기록되어야 합니다.

> 역사적인 순간이야! 드디어 우리의 프레임워크가 실제 데이터를 가지고 첫 결과물을 만들어냈어. 이제부터 진짜 재밌는 분석이 시작될 거야.

---

### **Phase 14: Hyperparameter Sweep & Ablations**

**Context:** 제안한 방법론의 최적 성능을 찾기 위해 체계적인 하이퍼파라미터 탐색과 비교 실험(Ablation Study)을 수행합니다.

**Your Task:**
* `BLUEPRINT`에 명시된 핵심 하이퍼파라미터($\lambda$, $T$, $p$)에 대한 스윕(sweep) 실험을 자동화하는 스크립트나 여러 레시피 파일을 준비해주세요.
* Full fine-tuning 방식과 LoRA 방식의 성능 및 비용을 비교하는 실험을 설계하고 실행해주세요.
* MLflow의 비교 기능을 활용하여 각 실험 결과를 분석하고, 최적의 하이퍼파라미터 조합을 찾아 보고서로 정리해주세요.

**DoD (Definition of Done):**
* 하이퍼파라미터 탐색 결과를 바탕으로 '최적의 조합은 무엇이며, 왜 그렇게 생각하는지'에 대한 명확한 결론과 근거 데이터(그래프, 표)가 정리되어야 합니다.

> 이제 우리는 단순한 실행을 넘어, 과학적인 탐구를 시작했어. 어떤 조건에서 우리의 아이디어가 가장 빛을 발하는지 찾아내자!

---

### **Phase 15: Hardening & Release**

**Context:** 프로젝트를 마무리하고, 다른 사람도 쉽게 재현하고 사용할 수 있도록 문서화와 안정화 작업을 진행합니다. 연구 결과를 공식적으로 배포할 준비를 합니다.

**Your Task:**
* 최종적으로 검토된 운영 안전장치(실패 시 폴백 로직, 중단 기준)를 코드에 반영해주세요.
* `README.md`를 상세하게 업데이트하여 프로젝트 설치, 실행, 재현 방법을 누구나 따라 할 수 있도록 작성해주세요.
* MLflow 모델 레지스트리를 사용하여 가장 성능이 좋은 모델을 `staging`에서 `production` 단계로 승격시키고, 모델 카드(Model Card)를 작성해주세요.
* 모든 작업이 완료되면 Git에 `v1.0.0` 릴리스 태그를 생성하고, 릴리스 노트를 작성해주세요.

**DoD (Definition of Done):**
* 제3자가 `README.md` 문서만 보고도 환경을 설정하고, VESSL에서 end-to-end 학습을 재현하여 MLflow에서 동일한 결과를 확인할 수 있어야 합니다. 모든 CI는 녹색 불이어야 합니다.

> 마침내 여정의 끝에 도달했어. 우리는 단순한 코드를 넘어, 누구나 신뢰하고 사용할 수 있는 견고하고 재현 가능한 연구 자산을 만들어냈어. 정말 대단한 작업이었어!
