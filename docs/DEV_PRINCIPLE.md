# DEV_PRINCIPLES.md

*WMTP 파인튜닝 프레임워크 · Vibe Coding 가이드라인*

본 문서는 **Meta MTP 기반 WMTP 파인튜닝 프레임워크**를 구현·운영하는 동안 개발자 전원이 일관되게 따를 **개발 원칙(코딩/테스트/운영/협업)** 을 정의한다. 목표는 **빠른 실험 반복**, **강한 재현성**, **안전한 운영**, **명료한 코드 구조**다.

---

## 1) 철학 & 목표

* **실험의 주인은 데이터**: 모든 결정은 **평가 하네스(메타 MTP 프로토콜)** 결과로 검증한다.
* **설정 주도(Declarative)**: 코드는 유연하되, **행동은 `config.yaml`+`recipe.yaml`로 결정**된다.
* **단일 책임(SRP)**: 컴포넌트는 **하나의 역할**만 수행한다. 로더는 로딩만, 스코어러는 점수만.
* **관찰 가능성(Observability)**: **MLflow**에 파라미터/지표/아티팩트를 **반드시 기록**한다.
* **유틸 단일화**: S3/HF/분산/평가/MLflow 도우미는 **`src/utils/`에만 존재**한다(흩뿌리지 않는다).
* **로컬 우선 → S3 미러링**: 개발 생산성을 위해 **로컬 캐시 우선**, 부재 시 S3에서 당겨온다.
* **안전한 성능**: A100·FSDP·bf16을 기본으로, **OOM/NaN 안전장치**를 갖춘다.
* **코드는 글**: 함수·클래스·모듈은 **읽기 쉬운 문장**이 되어야 한다(이름·주석·Docstring).

---

## 2) 리포 구조 & 경계

```
src/
 ├─ cli/           # 엔트리포인트 (train/eval)
 ├─ pipelines/     # 파이프라인 DAG (critic-wmtp, rho1-wmtp)
 ├─ components/    # registry 컴포넌트(Loader/Scorer/Trainer/Optimizer/Evaluator)
 ├─ factory/       # settings(dict) -> components 조립
 ├─ settings/      # Pydantic 스키마/로더/검증
 └─ utils/         # s3/mlflow/hf/dist/eval 공용 유틸 (유일한 유틸 위치)
```

### 경계 원칙

* **`utils/` 외부**에서 S3/HF/MLflow/분산 호출 **금지**. (중복/누수 방지)
* **`components/*`** 는 **상태 없는 빌더**로 설계(파이프라인이 주입한 `ctx` 기반).
* **`pipelines/*`** 는 **비즈니스 로직의 실제 오케스트레이션만** 담당.
* **`settings/*`** 는 **입력 검증 및 변환** 외의 일을 하지 않는다(파일 I/O 최소화).

### 구조 일관성(Non-Negotiable)

* **설정 주도 실행**: 동작은 오직 `config.yaml`+`recipe.yaml`로 결정. 코드에 상수/매직 넘버 삽입 금지.
* **플러그인 규약 준수**: 신규 기능은 반드시 레지스트리에 `kebab-case` 키로 등록하고 팩토리에서 조립한다.
* **경계 단일화**: 외부 IO/런타임 접근은 `src/utils/`로만 집중. 다른 계층에서 직접 호출 금지.
* **파이프라인 최소 책임**: 파이프라인은 컴포넌트 조립과 수명주기 관리만 담당(학습/평가 로직은 컴포넌트 내부).
* **테스트 가능한 계약**: 컴포넌트는 `setup(ctx)->run(ctx)->dict` 계약을 반드시 지키고, 상태는 `ctx`로 명시 전달.
* **문서-코드 동기화**: 구조/원칙 변경 시 `DEV_PLANS.md`/`DEV_PRINCIPLE.md`를 먼저 갱신 후 구현한다.

---

## 3) 코딩 스타일 & 품질

* **언어/버전**: Python **3.11**.
* **스타일**: `ruff`(lint/format 일원화) + `pyproject.toml`에서 룰 고정.
* **Docstring**: **Google 스타일**. 함수 시그니처에 타입힌트 필수.
* **명명 규칙**

  * 파일/모듈: `snake_case.py`
  * 클래스: `PascalCase`
  * 함수/변수: `snake_case`
  * 레지스트리 키: `kebab-case` (예: `critic-delta-v1`, `rho1-excess-v1`)
* **함수 길이**: 50줄 내외 권장(복잡도 ↑ 시 분리).
* **예외 처리**: 사용자 입력/IO/분산 초기화/모델 로딩 등 **경계 지점에 명시적 try/except**.
* **로그**: 인프라 로그는 `logging` 표준 사용, 실험 로그는 `utils/mlflow.py`에 위임.

**DO**

* 작은 함수/짧은 클래스, 명시적 데이터 흐름(`ctx`), 불변 데이터 우선(dict copy)
* 빠른 실패(Fail Fast): 필수 전제 조건은 시작 시 검증

**DON’T**

* 전역 변이 상태, 암묵적 환경 의존, 컴포넌트 내부에서 S3/HF 직접 호출

---

## 4) 의존성·환경(uv) & Docker/VESSL

* **uv**: `uv lock` 기반 고정. `uv sync --frozen`으로 재현성 확보.
* **런처**: `launcher.target`이 `local|vessl`을 결정. 분산 옵션은 설정에만 둔다.
* **Docker**: 베이스 이미지는 PyTorch CUDA 12.1, 런타임은 **A100 호환**.
* **VESSL**: GPU/CPU/Mem/Disk/Secret 정의는 `docker/vessl.yaml`로 명시.

**명령 예시**

```bash
uv run python -m src.cli.train --config configs/config.yaml --recipe configs/recipe.yaml
uv run python -m src.cli.eval  --config configs/config.yaml --recipe configs/recipe.yaml
```

---

## 5) 설정(설계 주도)

* **`config.yaml`**: 환경(스토리지/경로/MLflow/런처/장비/시드)
* **`recipe.yaml`**: **모델/학습/배치/손실/스코어러/평가**
* **Pydantic 검증**:

  * base 모델은 **MTP 사전학습 모델**이어야 함
  * `algo in {critic-wmtp, rho1-wmtp}`
  * 가중치 정규화 정책(`mean1.0_clip`) 유효성 등

**원칙**: **코드를 바꾸지 않고 YAML만 바꿔 실험**할 수 있어야 한다.

---

## 6) 레지스트리·팩토리

* **Registry**: `register("critic-delta-v1")(cls)` 데코레이터로 플러그인 등록.
* **Factory**: Pydantic 통과한 `settings: dict` → 필요한 컴포넌트 인스턴스 생성.
* **교체 가능성**: 스코어러/트레이너/옵티마이저는 **키만 바꾸면 교체** 가능해야 한다.

**인터페이스 규약**

```python
class Component(Protocol):
    def setup(self, ctx: dict) -> None: ...
    def run(self, ctx: dict) -> dict: ...
```

---

## 7) 학습 파이프라인(안정장치 포함)

### 7.1 Critic-Weighted MTP (선택/옵션)

* Stage-1: RM 시퀀스 보상 → 토큰 분배(`gae` 권장) → Value Head 회귀
* Stage-2: δ\_t → z-score → softmax(T) → mean-1.0 정규화 + \[ε, Wmax] 클립 → MTP CE 가중
* **안정화**: `bf16`, FSDP(full), activation checkpointing, grad clip=1.0, warmup 3%
* **비고**: 연구개선안 취지에 따라 **critic-free 경로(Rho-1)** 를 **기본(Default)** 으로 우선 적용하고, 본 경로는 선택적으로 사용한다.

### 7.2 Rho-1 Weighted MTP (기본/Default)

* `|CE_ref - CE_base|` → z-score → softmax(T) → mean-1.0 + 클립
* 상위 p% 연속 가중 강화(하드 드롭 지양), 대규모면 사전 프리컴퓨트

**공통 품질 게이트**

* 가중 평균 1.0±ε 유지(테스트로 검증)
* NaN/Inf 감지 시 **즉시 중단** + 러닝 레이트/배치/정밀도 폴백
* OOM 시 토큰 예산/accumulation 자동 조정

---

## 8) 데이터·평가

* **로딩 정책**: 로컬 존재 → 로컬 사용, 미존재 → **S3→`.cache/` 미러링**
* **해시 키**: (데이터 버전+전처리+스플릿 시드)로 캐시 디렉토리 결정
* **스플릿 시드**: 42 고정
* **평가 하네스**: Meta MTP 프로토콜( MBPP exact, CodeContests pass\@k )
* **샘플링**: T=0.2, top-p=0.95, n=1(기본)
* **로그**: 예측·정답 일부, head별 CE 통계, 가중치 분포 히스토그램

---

## 9) MLflow 규율

* **Experiments**: `mtp/{algo}/{dataset}`
* **Params**: 모델 id, n\_heads, λ, T, optimizer, lr, batch, FSDP 옵션, data hash 등
* **Metrics**: train/val loss, `mbpp_exact`, `contest_pass@1,5`, tok/s
* **Artifacts**: 체크포인트(last/best/periodic), `reports/{run_id}.md`, 중요토큰 통계
* **모델 레지스트리**: `staging → production` 전환 기록(버전·릴리스 노트)

---

## 10) 보안·시크릿·컴플라이언스

* **시크릿**: VESSL Secret(또는 환경변수)로만 주입. **코드 하드코딩 금지**.
* **S3**: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`.
* **라이선스**: HF 모델/데이터셋 라이선스 체크 후 캐시.
* **데이터 유출 방지**: 평가셋 텍스트/정답 학습 경로 차단, 로깅 마스킹.

---

## 11) 테스트 전략(테스트는 기능이다)

### 11.1 레벨별

* **Unit**

  * Pydantic 스키마(필수/제약/기본값)
  * Scorer(critic/rho1) 출력 **통계 불변식**: 평균 1.0, 클립 범위 내, NaN 금지
  * Utils(s3/hf/mlflow/dist): Mock으로 API 안정성 검증
* **Integration**

  * `pipelines/*`: tiny 설정으로 end-to-end 100 step 스모크
  * MLflow에 파라미터/메트릭/아티팩트 기록 여부 확인
* **E2E (옵션)**

  * MBPP 소셋으로 두 파이프라인 모두 학습→평가까지 1회 회전

### 11.2 속도·결정론성

* **빠르게**: 기본 테스트는 3분 이내, 대형은 마커 `@pytest.mark.slow`
* **결정론성**: seed=42, 주요 난수 시드 고정(단, CUDA 완전 결정론은 성능 저하 시 완화 가능)

### 11.3 필수 테스트 케이스

* 잘못된 설정(미지원 algo/모델 id) → **명확한 에러 메시지**
* 가중치 정규화 파이프 → 평균 1.0 유지
* OOM/NaN 시 폴백 로직 작동 여부(단위 레벨에서 시뮬레이션)

---

## 12) 커밋·브랜치·PR

* **브랜치**: `feature/*`, `fix/*`, `exp/*` (실험 스크립트/레시피는 `exp/*`)
* **커밋 메시지**:

  * `feat: …`, `fix: …`, `refactor: …`, `docs: …`, `test: …`, `exp: …`
* **PR 규칙**

  * 요약 + 변경 파일 개요 + 스크린샷/MLflow 링크
  * **체크리스트**: 테스트 통과, 린트 통과, 문서 갱신, 설정 예제 업데이트
  * 리뷰어 1+ 승인 필요, 실험 결과(PR 설명에 첨부)

---

## 13) 관찰·운영

* **중단 기준**: NaN/Inf/폭주(loss↑) 감지 시 즉시 중단 → 하이퍼/정밀도/배치 자동 완화
* **지표 감시**: tok/s 변동, head별 CE 분해, 가중치 분포 드리프트
* **리포트 생성**: 각 러닝 후 `reports/{run_id}.md` 자동 생성(지표/설정/샘플 포함)

---

## 14) 성능·메모리 가이드

* **우선순위**: FSDP(full) → Activation Checkpointing → token-budget 조정 → grad-accum↑
* **정밀도**: bf16 기본, 필요 시 fp16 폴백(스케일러 필수)
* **Flash-Attn**: 가능 시 활성화(모델/라이센스/환경 제약 검토)
* **프로파일링**: 첫 대형 실험 전 **샘플 배치**로 **VRAM 프로파일** 필수

---

## 15) 실패에서 배우기

* **실패 로그 보존**: OOM/NaN/중단 시 **MLflow에 원인·환경·스택** 기록
* **실패 재현 레시피**: 문제 재현을 위한 최소 `recipe.yaml` 스냅샷을 아티팩트로 첨부
* **교훈 기록**: `reports/`에 “What went wrong / Next action” 섹션 유지

---

## 16) “Vibe Coding” 운용 팁

* **작은 루프**: (문제 가설) → (아주 작은 변경) → (스모크) → (MLflow 확인)
* **문서 우선**: 기능 변경 전 **레시피·설정·README·본 문서** 동기화
* **도구 일관성**: 모든 실행은 `uv run`을 통해, 수동 가상환경/패키지 설치 금지
* **레시피 복제**: 새 실험은 기존 레시피 복제 후 변경점만 수정(명확한 diff 보장)
* **Checklists**: PR/릴리스/대형 실험 전 **체크리스트**를 강제한다.

---

## 부록 A. 체크리스트

* [ ] `config.yaml`/`recipe.yaml` 스키마 검증 통과
* [ ] 레지스트리 키 존재/오탈자 없음
* [ ] 스모크 테스트 통과(critic/rho1 모두)
* [ ] MLflow에 params/metrics/artifacts 기록 확인
* [ ] 가중치 평균 1.0±ε, NaN/Inf 없음
* [ ] README/문서 업데이트
* [ ] VESSL 리소스/시크릿 확인(A100, S3 자격)

---

## 부록 B. 코드 스니펫(요지)

**레지스트리**

```python
# src/components/registry.py
_REG = {}
def register(name: str):
    def deco(cls):
        _REG[name] = cls
        return cls
    return deco

def get(name: str):
    if name not in _REG:
        raise KeyError(f"Unknown component: {name}")
    return _REG[name]
```

**컴포넌트 인터페이스**

```python
class Component(Protocol):
    def setup(self, ctx: dict) -> None: ...
    def run(self, ctx: dict) -> dict: ...
```

**가중치 정규화 불변식(테스트 예)**

```python
def test_weight_norm_invariant():
    w = scorer_outputs  # shape [T]
    assert np.isfinite(w).all()
    m = float(np.mean(w))
    assert 0.95 <= m <= 1.05  # mean~1.0
    assert (w >= EPS).all() and (w <= WMAX).all()
```

---

### 결론

이 문서는 **코드 품질/실험 재현성/운영 안전성**의 3축을 지키면서, **Vibe Coding** 흐름에서 빠르고 반복 가능한 실험을 보장한다.
문서 변경 없이 코드가 바뀌지 않도록, **설정 주도** 원칙을 끝까지 지키자.
