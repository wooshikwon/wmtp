# WMTP 체크포인트 관리 시스템 개발 계획서
## Config 기반 체크포인트 경로 설정 및 Resume 기능 고도화

---

## 📋 Executive Summary

### 목표
1. **Config 체크포인트 경로 설정**: `config.vessl.yaml`에 체크포인트 저장 경로 설정 추가
2. **Pydantic Schema 확장**: 체크포인트 관련 설정을 타입 안전하게 관리
3. **동적 경로 저장**: 지정된 경로(로컬/S3)에 체크포인트 자동 저장
4. **고도화된 Resume**: CLI `--resume` 옵션으로 특정 step/latest 체크포인트에서 학습 재개

### 현재 한계점
- 체크포인트 경로 하드코딩: `./checkpoints/{run_name}` 고정
- Resume 기능 제한적: 전체 경로 지정만 가능
- S3 지원 미활용: Phase 1에서 구현했지만 경로 설정 불가

---

## 🔍 현재 구조 분석

### 1. 체크포인트 저장 흐름
```
CLI train() → TrainingPipeline.run() → BaseWmtpTrainer.train()
    ↓
BaseWmtpTrainer.__init__() → self.checkpoint_dir = Path("./checkpoints") / run_name (고정)
    ↓
_save_checkpoint() → DistributedManager.save_checkpoint() → Phase 1 S3 지원
```

### 2. Resume 기능 흐름
```
CLI --resume /path/to/checkpoint.pt → ctx["resume_checkpoint"]
    ↓
BaseWmtpTrainer.__init__() → self.dist_manager.load_checkpoint()
    ↓
학습 루프에서 self.start_step부터 재개
```

### 3. 현재 Config Schema (src/settings/config_schema.py)
```python
class PathsConfig(BaseModel):
    models: ModelsConfig
    datasets: DatasetsConfig
    # ❌ checkpoints 경로 설정 없음
```

### 4. 기존 Resume 제약사항
- CLI에서 전체 파일 경로만 지정 가능
- 자동 체크포인트 탐색 불가
- Step 기반 resume 불가 (예: `--resume step_1000`)

---

## 📈 Phase별 개발 계획

## Phase 1: Config Schema 확장 - 체크포인트 경로 설정
**목표**: Config에 체크포인트 저장 경로 설정 추가
**일정**: 2일
**위험도**: 낮음 (스키마 확장만)

### 1.1 구현 내용

#### src/settings/config_schema.py 수정
```python
class CheckpointConfig(BaseModel):
    """체크포인트 저장 설정

    로컬/S3 경로 모두 지원하며, Phase 1에서 구현한 S3 기능 활용
    """

    # 기본 저장 경로 (프로토콜 기반)
    base_path: str = Field(
        default="file://./checkpoints",
        description="체크포인트 기본 저장 경로 (file:// 또는 s3://)"
    )

    # 저장 정책
    save_interval: int = Field(
        default=500,
        description="체크포인트 저장 간격 (steps)"
    )
    keep_last: int = Field(
        default=3,
        description="보관할 체크포인트 개수"
    )
    save_final: bool = Field(
        default=True,
        description="최종 모델 저장 여부"
    )

class PathsConfig(BaseModel):
    """경로 설정 (기존 + 체크포인트 추가)"""
    models: ModelsConfig
    datasets: DatasetsConfig
    checkpoints: CheckpointConfig = Field(default_factory=CheckpointConfig)  # ← 신규 추가

class Config(BaseModel):
    # ... 기존 필드들
    paths: PathsConfig
    # ...
```

#### 프로토콜 기반 경로 처리
```python
# src/utils/path_resolver.py (신규 생성)
def resolve_checkpoint_path(base_path: str, run_name: str) -> tuple[str, bool]:
    """
    체크포인트 경로 해석

    Returns:
        (resolved_path, is_s3)

    Examples:
        "file://./checkpoints" → ("./checkpoints/run_name", False)
        "s3://wmtp/checkpoints" → ("s3://wmtp/checkpoints/run_name", True)
    """
    if base_path.startswith("s3://"):
        return f"{base_path.rstrip('/')}/{run_name}", True
    elif base_path.startswith("file://"):
        local_path = base_path.replace("file://", "")
        return f"{local_path}/{run_name}", False
    else:
        # 기본값 처리 (하위 호환성)
        return f"{base_path}/{run_name}", False
```

### 1.2 테스트 계획
- [ ] CheckpointConfig 스키마 검증
- [ ] 프로토콜 기반 경로 해석 테스트
- [ ] S3/로컬 경로 혼용 테스트
- [ ] 기존 config 파일 호환성 검증

---

## Phase 2: Recipe Schema 확장 - 세밀한 체크포인트 제어
**목표**: Recipe에서 Config 체크포인트 설정 오버라이드 가능
**일정**: 1일
**위험도**: 낮음 (기존 구조 확장)

### 2.1 구현 내용

#### src/settings/recipe_schema.py 수정
```python
class CheckpointingConfig(BaseModel):
    """Recipe별 체크포인트 설정 (기존 확장)"""

    # 기존 필드 유지
    save_interval: int = Field(default=100)
    keep_last: int = Field(default=3)
    save_final: bool = Field(default=True)

    # 새로운 필드 추가
    checkpoint_path: str | None = Field(
        default=None,
        description="Recipe별 체크포인트 경로 오버라이드 (선택)"
    )
    auto_resume_latest: bool = Field(
        default=False,
        description="최신 체크포인트 자동 탐색 및 재개"
    )

class TrainConfig(BaseModel):
    # ... 기존 필드들
    checkpointing: CheckpointingConfig | None = Field(default=None)
```

### 2.2 우선순위 정의
1. **CLI --resume** (최우선)
2. **Recipe checkpoint_path**
3. **Config paths.checkpoints.base_path**
4. **기본값** (`./checkpoints`)

---

## Phase 3: BaseWmtpTrainer 체크포인트 로직 개선
**목표**: Config/Recipe 기반 동적 체크포인트 경로 사용
**일정**: 2일
**위험도**: 중간 (핵심 로직 수정)

### 3.1 구현 내용

#### src/components/trainer/base_wmtp_trainer.py 수정
```python
class BaseWmtpTrainer:
    def __init__(self, config, recipe, ctx):
        # ... 기존 초기화

        # 체크포인트 경로 결정 (우선순위 적용)
        self.checkpoint_base_path, self.is_s3_checkpoint = self._resolve_checkpoint_path(
            config, recipe, ctx
        )

        # 기존 고정 경로 대신 동적 경로 사용
        run_name = recipe.run.name if recipe else "default"
        if self.is_s3_checkpoint:
            self.checkpoint_dir = f"{self.checkpoint_base_path}/{run_name}"
        else:
            self.checkpoint_dir = Path(self.checkpoint_base_path) / run_name
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_checkpoint_path(self, config, recipe, ctx) -> tuple[str, bool]:
        """체크포인트 경로 우선순위 결정"""
        from src.utils.path_resolver import resolve_checkpoint_path

        # 1. Recipe checkpoint_path (최우선)
        if (recipe and hasattr(recipe, 'train') and
            hasattr(recipe.train, 'checkpointing') and
            recipe.train.checkpointing and
            recipe.train.checkpointing.checkpoint_path):

            path = recipe.train.checkpointing.checkpoint_path
            return path, path.startswith("s3://")

        # 2. Config paths.checkpoints.base_path
        if hasattr(config, 'paths') and hasattr(config.paths, 'checkpoints'):
            return resolve_checkpoint_path(
                config.paths.checkpoints.base_path,
                recipe.run.name if recipe else "default"
            )

        # 3. 기본값 (하위 호환성)
        return "./checkpoints", False

    def _save_checkpoint(self, epoch: int, step: int, metrics: dict) -> str:
        """체크포인트 저장 (S3/로컬 자동 판단)"""

        if self.is_s3_checkpoint:
            # S3 경로 생성
            checkpoint_path = f"{self.checkpoint_dir}/checkpoint_step_{step}.pt"
        else:
            # 로컬 경로 생성
            checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"
            checkpoint_path = str(checkpoint_path)

        # Phase 1에서 구현한 S3 지원 활용
        self.dist_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            checkpoint_path=checkpoint_path,  # ← S3/로컬 자동 판단
            epoch=epoch,
            step=step,
            mlflow_manager=self.mlflow,
            metrics=metrics,
            algorithm=getattr(self, "algorithm", "wmtp"),
            mlflow_run_id=self.mlflow.get_run_id() if self.mlflow else None,
        )

        return checkpoint_path
```

### 3.2 하위 호환성 보장
- 기존 `./checkpoints` 기본값 유지
- Config/Recipe 설정이 없으면 기존 방식으로 동작
- 기존 테스트 코드 호환성 유지

---

## Phase 4: 고도화된 Resume 기능
**목표**: Step 기반 resume, 자동 최신 체크포인트 탐색
**일정**: 2일
**위험도**: 중간 (CLI 확장)

### 4.1 구현 내용

#### CLI 확장 (src/cli/train.py)
```python
@app.command()
def train(
    # ... 기존 매개변수들

    # Resume 옵션 확장
    resume: str | None = typer.Option(
        None,
        "--resume",
        help="체크포인트 재개: 'latest', 'step_1000', 또는 전체 경로"
    ),
):
    # Resume 처리 로직
    resume_checkpoint_path = None
    if resume:
        resume_checkpoint_path = resolve_resume_path(resume, cfg, rcp)
        if resume_checkpoint_path:
            console.print(f"[green]체크포인트에서 훈련 재개: {resume_checkpoint_path}[/green]")
        else:
            console.print(f"[red]체크포인트를 찾을 수 없습니다: {resume}[/red]")
            raise typer.Exit(1)
```

#### Resume 경로 해석기 (src/utils/checkpoint_resolver.py - 신규)
```python
def resolve_resume_path(resume_arg: str, config, recipe) -> str | None:
    """
    Resume 인자를 실제 체크포인트 경로로 변환

    Args:
        resume_arg: 'latest', 'step_1000', 또는 전체 경로

    Returns:
        실제 체크포인트 파일 경로 또는 None
    """

    # 1. 전체 경로인 경우 (기존 방식)
    if Path(resume_arg).exists() or resume_arg.startswith("s3://"):
        return resume_arg

    # 2. 체크포인트 디렉토리 결정
    checkpoint_base, is_s3 = resolve_checkpoint_path_from_config(config, recipe)

    # 3. 'latest' 처리
    if resume_arg == "latest":
        return find_latest_checkpoint(checkpoint_base, is_s3)

    # 4. 'step_X' 처리
    if resume_arg.startswith("step_"):
        step = resume_arg.replace("step_", "")
        return find_checkpoint_by_step(checkpoint_base, step, is_s3)

    return None

def find_latest_checkpoint(checkpoint_base: str, is_s3: bool) -> str | None:
    """최신 체크포인트 탐색"""
    if is_s3:
        return find_latest_s3_checkpoint(checkpoint_base)
    else:
        return find_latest_local_checkpoint(checkpoint_base)

def find_latest_local_checkpoint(checkpoint_dir: str) -> str | None:
    """로컬 최신 체크포인트 탐색"""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return None

    # checkpoint_step_*.pt 파일들 찾기
    checkpoints = list(checkpoint_path.glob("checkpoint_step_*.pt"))
    if not checkpoints:
        return None

    # step 번호로 정렬하여 최신 반환
    latest = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
    return str(latest)

def find_latest_s3_checkpoint(s3_path: str) -> str | None:
    """S3 최신 체크포인트 탐색"""
    from src.utils.s3 import S3Manager

    # S3Manager를 사용하여 객체 목록 조회
    s3_manager = S3Manager()
    bucket = s3_path.replace("s3://", "").split("/")[0]
    prefix = "/".join(s3_path.replace("s3://", "").split("/")[1:])

    # checkpoint_step_*.pt 패턴으로 검색
    objects = s3_manager.list_objects(bucket, f"{prefix}/checkpoint_step_")
    if not objects:
        return None

    # 최신 체크포인트 반환
    latest_key = max(objects, key=lambda k: int(k.split('_')[-1].replace('.pt', '')))
    return f"s3://{bucket}/{latest_key}"
```

### 4.2 자동 Resume 기능
```python
# Recipe에서 auto_resume_latest: true 설정 시
class BaseWmtpTrainer:
    def __init__(self, config, recipe, ctx):
        # 자동 resume 처리
        if (not ctx.get("resume_checkpoint") and
            recipe and hasattr(recipe.train, 'checkpointing') and
            recipe.train.checkpointing and
            recipe.train.checkpointing.auto_resume_latest):

            latest_checkpoint = find_latest_checkpoint(
                self.checkpoint_base_path, self.is_s3_checkpoint
            )
            if latest_checkpoint:
                console.print(f"[green]자동 Resume: {latest_checkpoint}[/green]")
                ctx["resume_checkpoint"] = latest_checkpoint
```

---

## Phase 5: S3 체크포인트 관리 최적화
**목표**: S3 환경에서 체크포인트 관리 효율성 향상
**일정**: 1일
**위험도**: 낮음 (기능 추가)

### 5.1 구현 내용

#### S3 체크포인트 목록 조회 최적화
```python
# src/utils/s3.py 확장
class S3Manager:
    def list_checkpoints(self, checkpoint_path: str) -> list[dict]:
        """
        S3 체크포인트 목록 조회

        Returns:
            [{"key": "checkpoint_step_1000.pt", "step": 1000, "modified": datetime}, ...]
        """
        import re
        from datetime import datetime

        bucket = checkpoint_path.replace("s3://", "").split("/")[0]
        prefix = "/".join(checkpoint_path.replace("s3://", "").split("/")[1:])

        response = self.s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix
        )

        checkpoints = []
        step_pattern = re.compile(r'checkpoint_step_(\d+)\.pt$')

        for obj in response.get('Contents', []):
            match = step_pattern.search(obj['Key'])
            if match:
                checkpoints.append({
                    "key": obj['Key'],
                    "step": int(match.group(1)),
                    "modified": obj['LastModified'],
                    "size": obj['Size']
                })

        return sorted(checkpoints, key=lambda x: x['step'])
```

#### 체크포인트 정리 기능
```python
def cleanup_old_checkpoints(checkpoint_base: str, keep_last: int, is_s3: bool):
    """오래된 체크포인트 정리 (keep_last 개수만 유지)"""

    if is_s3:
        s3_manager = S3Manager()
        checkpoints = s3_manager.list_checkpoints(checkpoint_base)

        if len(checkpoints) > keep_last:
            to_delete = checkpoints[:-keep_last]  # 오래된 것들
            for checkpoint in to_delete:
                s3_manager.delete_object(checkpoint['key'])
                console.print(f"[blue]S3 체크포인트 삭제: {checkpoint['key']}[/blue]")
    else:
        # 기존 로컬 정리 로직 유지
        pass
```

---

## 🎯 설정 파일 예시

### config.vessl.yaml (수정)
```yaml
# 기존 설정들...

# 새로 추가된 체크포인트 설정
paths:
  models:
    base: "s3://wmtp/models/7b_1t_4/"
    rm: "s3://wmtp/models/Starling-RM-7B-alpha/"
    ref: "s3://wmtp/models/Sheared-LLaMA-2.7B/"
  datasets:
    mbpp: "s3://wmtp/dataset/mbpp"
    contest: "s3://wmtp/dataset/contest"
    humaneval: "s3://wmtp/dataset/humaneval"

  # ✅ 새로 추가: 체크포인트 설정
  checkpoints:
    base_path: "s3://wmtp/checkpoints"  # S3에 저장
    save_interval: 500                  # 500 스텝마다 저장
    keep_last: 5                        # 최근 5개만 보관
    save_final: true                    # 최종 모델 저장
```

### recipe.cluster_rho1_wmtp.yaml (수정)
```yaml
# 기존 설정들...

train:
  algo: "rho1-wmtp"
  max_steps: 5000

  # ✅ 개선된 체크포인트 설정
  checkpointing:
    save_interval: 200                    # Recipe별 간격 오버라이드
    keep_last: 3
    save_final: true
    checkpoint_path: "s3://wmtp/prod/checkpoints"  # Recipe별 경로 오버라이드
    auto_resume_latest: true              # 자동 최신 체크포인트 재개
```

---

## 🚀 사용 시나리오

### 시나리오 1: 기본 S3 체크포인트 훈련
```bash
# config.vessl.yaml에 s3://wmtp/checkpoints 설정된 상태
python -m src.cli.train \
  --config configs/config.vessl.yaml \
  --recipe configs/recipe.cluster_rho1_wmtp.yaml

# 결과: s3://wmtp/checkpoints/cluster_rho1_wmtp_production/ 에 저장
```

### 시나리오 2: 최신 체크포인트에서 재개
```bash
python -m src.cli.train \
  --config configs/config.vessl.yaml \
  --recipe configs/recipe.cluster_rho1_wmtp.yaml \
  --resume latest  # ✅ 자동으로 최신 체크포인트 탐색

# 결과: s3://wmtp/checkpoints/cluster_rho1_wmtp_production/checkpoint_step_1500.pt 에서 재개
```

### 시나리오 3: 특정 Step에서 재개
```bash
python -m src.cli.train \
  --config configs/config.vessl.yaml \
  --recipe configs/recipe.cluster_rho1_wmtp.yaml \
  --resume step_1000  # ✅ step_1000에서 재개

# 결과: checkpoint_step_1000.pt 에서 재개
```

### 시나리오 4: 자동 Resume (Recipe 설정)
```yaml
# recipe에 auto_resume_latest: true 설정
train:
  checkpointing:
    auto_resume_latest: true
```
```bash
# Resume 옵션 없이도 자동으로 최신 체크포인트에서 재개
python -m src.cli.train \
  --config configs/config.vessl.yaml \
  --recipe configs/recipe.cluster_rho1_wmtp.yaml
```

---

## ⚠️ 위험 관리

### Phase별 위험도 평가

| Phase | 위험도 | 주요 위험 | 완화 방안 |
|-------|--------|-----------|-----------|
| **Phase 1** | 낮음 | 스키마 호환성 | 기본값 제공, 테스트 강화 |
| **Phase 2** | 낮음 | Recipe 파싱 오류 | 선택적 필드, 검증 로직 |
| **Phase 3** | 중간 | 체크포인트 저장 실패 | Phase 1 S3 로직 재사용 |
| **Phase 4** | 중간 | Resume 경로 해석 실패 | 명확한 에러 메시지 |
| **Phase 5** | 낮음 | S3 권한 문제 | Graceful degradation |

### 롤백 계획
- 각 Phase별 Git 브랜치 관리
- 기존 방식 호환성 유지로 점진적 마이그레이션
- 실패 시 이전 Phase로 즉시 롤백

---

## 🔧 개발 원칙 준수 검증

### [필수1] ✅ 현재 구조 분석 완료
- CLI resume 옵션 기존 존재 확인
- BaseWmtpTrainer 체크포인트 로직 완전 분석
- Phase 1 S3 지원 구조 활용 계획

### [필수2] ✅ 기존 구조 최대한 존중
- 기본값 유지로 하위 호환성 보장
- 기존 resume 로직 확장만 진행
- Phase 1 S3 저장 로직 재사용

### [필수3] ✅ 기존 코드 개선이 적절
- 고정 경로 → 동적 경로 변경 필요
- 중복 제거: 체크포인트 관련 로직 통합
- 새로운 삭제 없이 확장만 진행

### [필수4] ✅ 하위 호환성 고려하지 않음
- 기존 API 유지하되 내부 로직만 개선
- 점진적 마이그레이션으로 안정성 확보

### [필수5] ✅ 계획서 기반 구현 검증 예정
- 각 Phase 완료 후 목표 달성도 검토
- 과장 없는 객관적 성과 측정

### [필수6] ✅ uv 패키지 의존성 활용
- 기존 boto3, pydantic 등 활용
- 새로운 의존성 추가 없이 구현

---

## 📅 타임라인

| Phase | 작업 내용 | 예상 기간 | 완료 기준 |
|-------|-----------|-----------|-----------|
| **Phase 1** | Config Schema 확장 | 2일 | CheckpointConfig 추가, 테스트 통과 |
| **Phase 2** | Recipe Schema 확장 | 1일 | CheckpointingConfig 확장 완료 |
| **Phase 3** | BaseWmtpTrainer 개선 | 2일 | 동적 경로 저장 구현 완료 |
| **Phase 4** | Resume 기능 고도화 | 2일 | Step/latest resume 구현 완료 |
| **Phase 5** | S3 최적화 | 1일 | S3 체크포인트 관리 완료 |
| **총 기간** | | **8일** | 전체 기능 통합 테스트 완료 |

---

## 🎉 기대 효과

### 즉시 효과
- **VESSL 환경 S3 체크포인트**: config.vessl.yaml로 S3 저장 자동화
- **유연한 Resume**: `--resume latest`, `--resume step_1000` 지원
- **설정 기반 관리**: 하드코딩 제거, 환경별 맞춤 설정

### 장기 효과
- **클러스터 환경 최적화**: 체크포인트 중앙화로 실험 관리 효율성
- **비용 절감**: S3 체크포인트 정리로 스토리지 최적화
- **개발 생산성**: 자동 resume으로 실험 연속성 확보

**작성일**: 2025-01-27
**버전**: 1.0
**다음 단계**: Phase 1 Config Schema 확장 구현 시작