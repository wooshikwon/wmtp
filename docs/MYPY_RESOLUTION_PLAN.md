# mypy 타입 에러 완전 해결 계획

## 현재 상태
- **전체 에러**: 153개
- **최근 진행**: 173 → 153개 (약 12% 감소)
- **목표**: 0개 (완전 해결)

### 최근 변경 요약
- 토크나이저(HF-SP): 오버로드 추가, `SentencePieceProcessor` 타입 지정, 안전 접근자 도입
- 토크나이저(HF-Transformers): 오버로드/반환 타입 `cast(Dict[str, Any], ...)`, `model_id` 반환 `str` 강제
- stubs: `datasets.pyi`에 `select(Sequence[int])`, `add_column`, `from_list` 추가; `sentencepiece.pyi` 확인
- 외부 스텁: `types-PyYAML` 설치로 `yaml` 타입 경고 해소
- 레지스트리: 동적 속성 `setattr`로 타입 안전화, `create()` 호출 인자 보정
- 유틸: `mps_optimizer.py`의 `callable` → `Callable[..., Any]` 변경

## 에러 분류 및 분포

### 1. 외부 라이브러리 타입 스텁 부재 (22개, 14%)
**문제**: datasets, sentencepiece 등 타사 라이브러리의 타입 정의 누락
```python
# Example errors
Module "datasets" has no attribute "Dataset"  [attr-defined]
Module "datasets" has no attribute "load_dataset"  [attr-defined]
```

### 2. None 타입 체크 누락 (45개, 28%)
**문제**: None 체크 없이 속성 접근
```python
# Example errors
"None" has no attribute "encode"  [attr-defined]
"None" has no attribute "setup"  [attr-defined]
```

### 3. 반환값 타입 불일치 (9개, 6%)
**문제**: 선언된 반환 타입과 실제 반환값 불일치
```python
# Example errors
Incompatible return value type (got "None", expected "int")  [return-value]
Incompatible return value type (got "dict[str, list[list[int]]]", expected "dict[str, list[int]]")
```

### 4. 동적 속성 접근 (15개, 9%)
**문제**: Registry 패턴의 동적 속성 설정
```python
# Example errors
"type[T]" has no attribute "_registry_key"  [attr-defined]
"type[T]" has no attribute "_registry_category"  [attr-defined]
```

### 5. 기타 타입 불일치 (69개, 43%)
**문제**: 다양한 타입 불일치 및 제네릭 타입 문제

## 단계별 해결 전략

### Step 1: 타입 스텁 추가 [우선순위: 높음]
**목표**: 외부 라이브러리 타입 정의 추가

#### 1.1 datasets 라이브러리 타입 스텁
```python
# src/stubs/datasets.pyi
from typing import Any, Iterator, Optional
import torch

class Dataset:
    def __getitem__(self, key: str | int) -> Any: ...
    def __len__(self) -> int: ...
    def map(self, function: Any, **kwargs: Any) -> "Dataset": ...
    def filter(self, function: Any, **kwargs: Any) -> "Dataset": ...
    def select(self, indices: list[int]) -> "Dataset": ...

def load_dataset(
    path: str,
    name: Optional[str] = None,
    split: Optional[str] = None,
    **kwargs: Any
) -> Dataset: ...

def load_from_disk(path: str) -> Dataset: ...
```

#### 1.2 sentencepiece 타입 스텁
```python
# src/stubs/sentencepiece.pyi
from typing import List, Optional

class SentencePieceProcessor:
    def Load(self, model_file: str) -> None: ...
    def encode(
        self, text: str,
        out_type: type = int,
        add_bos: bool = False,
        add_eos: bool = False
    ) -> List[int]: ...
    def decode(self, ids: List[int]) -> str: ...
    def get_piece_size(self) -> int: ...
    def bos_id(self) -> int: ...
    def eos_id(self) -> int: ...
    def pad_id(self) -> int: ...
    def unk_id(self) -> int: ...
```

**실행 방법**:
```bash
# stubs 디렉토리 생성
mkdir -p src/stubs
touch src/stubs/__init__.py

# PYTHONPATH에 stubs 추가
export MYPYPATH=src/stubs:$MYPYPATH
```

### Step 2: None 체크 강화 [우선순위: 높음]
**목표**: Optional 타입 및 None 체크 추가

#### 2.1 SentencePieceTokenizer 수정
```python
# Before
def encode(self, text: str) -> List[int]:
    return self.processor.encode(text)

# After
def encode(self, text: str) -> List[int]:
    if self.processor is None:
        raise RuntimeError("Processor not initialized")
    return self.processor.encode(text)
```

#### 2.2 타입 가드 패턴 적용
```python
from typing import Optional, TypeGuard

def has_processor(self) -> TypeGuard[SentencePieceProcessor]:
    return self.processor is not None
```

### Step 3: 반환 타입 정확화 [우선순위: 중간]
**목표**: 함수 시그니처와 실제 반환값 일치

#### 3.1 Union 타입 사용
```python
# Before
def get_id(self) -> int:
    return None  # Error

# After
def get_id(self) -> Optional[int]:
    return None  # OK
```

#### 3.2 오버로드 활용
```python
from typing import overload

@overload
def tokenize(self, text: str) -> list[int]: ...

@overload
def tokenize(self, text: list[str]) -> list[list[int]]: ...

def tokenize(self, text: str | list[str]) -> list[int] | list[list[int]]:
    ...
```

### Step 4: Registry 패턴 타입 개선 [우선순위: 낮음]
**목표**: 동적 속성을 타입 안전하게 처리

#### 4.1 Protocol 사용
```python
from typing import Protocol

class Registrable(Protocol):
    _registry_key: str
    _registry_category: str
    _registry: dict[str, type]
```

#### 4.2 setattr/getattr 대신 명시적 속성
```python
class Component:
    __registry_key__: str = ""
    __registry_category__: str = ""
```

### Step 5: 설정 파일 추가 [우선순위: 중간]
**목표**: mypy 설정 최적화

#### 5.1 mypy.ini 생성
```ini
[mypy]
python_version = 3.12
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = False
ignore_missing_imports = True
follow_imports = silent
show_error_codes = True

# Per-module options
[mypy-datasets.*]
ignore_errors = True

[mypy-sentencepiece.*]
ignore_errors = True

[mypy-matplotlib.*]
ignore_errors = True

[mypy-transformers.*]
ignore_missing_imports = True
```

## 실행 계획

### Phase 1: 빠른 개선 (1-2시간)
1. [x] mypy.ini 설정 파일 생성
2. [x] 타입 스텁 파일 추가 (datasets, sentencepiece)
3. [x] None 체크가 명백히 누락된 부분 수정 (토크나이저 중심)

**예상 개선**: 160 → 80개 (50% 감소)

### Phase 2: 구조적 개선 (2-3시간)
1. [ ] 반환 타입 정확화
2. [ ] Optional 타입 전체 검토
3. [ ] TypeAlias 추가 확장

**예상 개선**: 80 → 30개 (추가 60% 감소)

### Phase 3: 완전 해결 (2-3시간)
1. [ ] Registry 패턴 타입 개선
2. [ ] 복잡한 제네릭 타입 처리
3. [ ] 남은 edge case 해결

**예상 개선**: 30 → 0개 (완전 해결)

## 검증 명령어

```bash
# 전체 검증
uv run mypy src/ --config-file mypy.ini

# 특정 모듈 검증
uv run mypy src/components/tokenizer/ --strict

# 에러 타입별 카운트
uv run mypy src/ | grep -E "\[.*\]" -o | sort | uniq -c

# 진행률 확인
uv run mypy src/ 2>&1 | grep -c "error:"
```

## 성공 지표

1. **단기 목표** (1주일)
   - mypy 에러 80% 감소
   - CI/CD 파이프라인에서 타입 체크 통과

2. **장기 목표** (2주일)
   - mypy 에러 0개
   - --strict 모드에서도 주요 모듈 통과

## 추가 권장사항

### 1. 점진적 타입 도입
```python
# pyproject.toml
[tool.mypy]
incremental = true
cache_dir = ".mypy_cache"
```

### 2. Pre-commit Hook 추가
```yaml
# .pre-commit-config.yaml
- repo: https://github.com/pre-commit/mirrors-mypy
  rev: v1.18.2
  hooks:
    - id: mypy
      args: [--config-file=mypy.ini]
      additional_dependencies: [types-requests]
```

### 3. IDE 통합
- VSCode: Pylance strict mode 활성화
- PyCharm: Type checker inspection 활성화

## 참고 자료
- [mypy 공식 문서](https://mypy.readthedocs.io/)
- [Python Type Hints PEP 484](https://www.python.org/dev/peps/pep-0484/)
- [typing 모듈 문서](https://docs.python.org/3/library/typing.html)