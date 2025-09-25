# WMTP 파인튜닝: VESSL GPU 실행 가이드

아주 간략한 프로젝트 설명은 다음 문서를 참고하세요:
- 연구 제안과 이론 배경: `docs/WMTP_학술_연구제안서.md`
- 실제 코드 아키텍처: `docs/WMTP_시스템_아키텍처.md`

아래는 VESSL GPU 환경에서 `configs/`의 YAML과 Docker 컨테이너를 사용해 학습을 실행하는 순차 가이드입니다.

---

## 0) 사전 준비

- Docker / 컨테이너 레지스트리(예: GHCR) 접근 권한
- VESSL 계정 및 VESSL CLI 설치
- HuggingFace 토큰(HF_TOKEN)
- S3 자격증명(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION)

구성 파일과 도커 스펙은 여기에서 확인할 수 있습니다:
- 도커/베슬 배포 설명: `docker/README.md`
- 도커 스펙: `docker/Dockerfile`, `docker/vessl.yaml`
- 학습/평가 설정: `configs/*.yaml`

---

## 1) Docker 이미지 빌드 및 푸시

프로젝트 루트에서:

```bash
# 이미지 빌드 (pyproject와 일치하는 의존성 설치)
make build IMAGE_TAG=latest

# 레지스트리에 푸시 (예: GHCR)
make push REGISTRY=ghcr.io/wooshikwon IMAGE_TAG=latest
```

- 기본 베이스: PyTorch 2.4.0 + CUDA 12.1
- 패키지 관리: uv (락파일 기반, `uv sync --frozen`)

---

## 2) VESSL 시크릿 설정

VESSL UI/CLI에서 다음 시크릿을 등록하세요.
- `HF_TOKEN`
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`

`docker/vessl.yaml`는 시크릿을 `${secret:...}`로 참조합니다.

---

## 3) 알고리즘/레시피 선택

아래 중 하나를 선택합니다. 알고리즘은 `WMTP_ALGO` 환경변수로도 제어할 수 있습니다.
- 기준선 MTP: `configs/config.mtp_baseline.yaml` + `configs/recipe.mtp_baseline.yaml`
- Critic WMTP: `configs/config.critic_wmtp.yaml` + `configs/recipe.critic_wmtp.yaml`
- Rho‑1 WMTP: `configs/config.rho1_wmtp.yaml` + `configs/recipe.rho1_wmtp.yaml`

`docker/vessl.yaml`는 기본적으로 `WMTP_ALGO=rho1-wmtp`로 설정되어 있으며, 알고리즘에 맞춰 config/recipe를 자동 선택합니다.

---

## 4) VESSL 리소스 설정

`docker/vessl.yaml`에서 클러스터/프리셋을 환경에 맞게 수정하세요.

```yaml
resources:
  cluster: default           # 예: vessl-gcp-oregon
  preset: v1-a100-1-pod      # 테스트: 1x A100,  프로덕션: v1-a100-4-pod
```

S3/MLflow 경로는 각 `configs/config.*.yaml`의 `storage`, `mlflow` 섹션을 사용합니다.

---

## 5) 학습 실행 (VESSL 제출)

```bash
# VESSL 제출 (기본: Rho‑1)
make vessl-run
```

필요시 VESSL UI에서 `WMTP_ALGO`를 `baseline-mtp | critic-wmtp | rho1-wmtp` 중 하나로 변경하거나, `docker/vessl.yaml`의 `env`를 수정하세요.

제출 시 실행되는 명령(발췌):

```yaml
command: |
  # 알고리즘에 따라 config/recipe 자동 선택
  uv run python -m src.cli.train \
    --config ${CONFIG} \
    --recipe ${RECIPE}
```

추가 참고 (uv 환경):
- Docker 이미지 빌드시 `uv sync --frozen`으로 의존성을 고정 설치합니다.
- 런타임 실행은 항상 `uv run`으로 동일한 해석기/환경에서 실행합니다.
- 로컬에서 빠른 확인이 필요한 경우:
  ```bash
  uv pip install --system boto3 sentencepiece  # 필요한 추가 패키지
  uv run python -m tests.script.test_m3_pipeline --algo mtp-baseline --tiny
  ```

---

## 6) 로그/모니터링 & 결과

- GPU/메모리 모니터링: `docker/vessl.yaml`의 `monitoring` 섹션 참조
- MLflow 추적: 각 `configs/config.*.yaml`의 `mlflow.tracking_uri`/`registry_uri`
- 체크포인트/모델: 각 `configs/config.*.yaml`의 `storage`/`paths`에 정의된 S3 프리픽스

(옵션) VESSL CLI로 실행 로그 보기:

```bash
# 실행 ID가 있을 때
make vessl-logs RUN_ID=<your-run-id>
```

---

## 7) 로컬 테스트 (선택)

간단한 로컬 확인이 필요하면:

```bash
# GPU 셸
make run-bash

# 직접 실행 (예: Rho‑1)
docker run --rm --gpus all \
  -e HF_TOKEN=$HF_TOKEN \
  -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  -e AWS_DEFAULT_REGION=eu-north-1 \
  -v $(pwd)/configs:/app/configs \
  ghcr.io/wooshikwon/wmtp:latest \
  uv run python -m src.cli.train \
    --config configs/config.rho1_wmtp.yaml \
    --recipe configs/recipe.rho1_wmtp.yaml
```

---

## 8) 문제 해결 팁

- 모델/데이터 경로: 각 `configs/config.*.yaml`의 `paths.models`, `paths.datasets` 확인
- 토크나이저/시점 정렬: Rho‑1/critic는 동일 토크나이저 및 올바른 시점 정렬이 전제(세부는 아키텍처 문서 참조)
- CUDA OOM: `configs/recipe.*.yaml`의 배치/길이 조정 또는 preset 축소/확대
- MLflow 접근오류: S3 권한/URI 확인

---

필요한 상세 이론/구현 설명은 다음을 참고하세요:
- `docs/WMTP_학술_연구제안서.md`
- `docs/WMTP_시스템_아키텍처.md`
