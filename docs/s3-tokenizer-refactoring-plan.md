# S3 기반 통합 토크나이저 리팩토링 계획

## 📋 개요

### 목표
파일시스템 의존성을 완전히 제거하고 S3에서 메모리로 직접 토크나이저를 로드하는 통합 시스템 구축

### 핵심 변경사항
- ❌ **제거**: `unified_tokenizer.py` 완전 삭제
- ✅ **통합**: `sentence_piece.py`가 모든 토크나이저 기능 담당
- ✅ **S3 우선**: 파일시스템 대신 S3에서 메모리로 직접 로드
- ✅ **VESSL 최적화**: 컨테이너 재시작에도 안정적 동작

## 🔍 현재 구조 분석 (개발 원칙 필수1)

### 현재 토크나이저 의존성
```
1. unified_tokenizer.py
   - 싱글톤 패턴으로 토크나이저 관리
   - 파일시스템에서 tokenizer.model 로드
   - get_unified_tokenizer() 함수 제공

2. sentence_piece.py
   - unified_tokenizer의 wrapper 역할
   - Registry 시스템과 호환

3. 사용처
   - base_loader.py: ModelLoader.load_tokenizer()
   - sentence_piece.py: 내부에서 get_unified_tokenizer() 호출
```

### 기존 S3 인프라 (재사용 대상)
```python
# src/utils/s3.py
- S3Manager 클래스 존재
- download_file(), upload_file() 메서드 지원
- 체크포인트 시스템 구축됨
```

### 문제점
1. **파일시스템 의존**: models/7b_1t_4/tokenizer.model 경로 하드코딩
2. **VESSL 비호환**: 컨테이너에 models/ 마운트 불가
3. **중복 구조**: unified_tokenizer와 sentence_piece 이중 구조

---

## 📐 설계 방향 (개발 원칙 필수2)

### 구조 단순화
```python
# 기존 (복잡)
ComponentFactory → sentence_piece → unified_tokenizer → 파일시스템

# 개선 (단순)
ComponentFactory → sentence_piece → S3/메모리
```

### 기존 구조 존중
- ✅ **싱글톤 패턴 유지**: 메모리 효율성 보장
- ✅ **S3Manager 재사용**: 기존 인프라 활용
- ✅ **Registry 패턴 유지**: ComponentFactory 일관성

---

## 🚀 Phase별 구현 계획

### Phase 1: 현재 구조 상세 분석

**목표**: 토크나이저 의존성 완전 파악

**작업 내용**:
1. `unified_tokenizer.py` 사용처 전체 조사
2. `base_loader.py`의 `load_tokenizer()` 메서드 분석
3. S3Manager의 바이트 스트림 지원 여부 확인

**검증 기준**:
- 모든 의존성 문서화 완료
- 영향받는 파일 목록 확정
- S3 인프라 활용 방안 확정

---

### Phase 2: 새로운 SentencePieceTokenizer 설계

**목표**: S3 기반 통합 토크나이저 설계

**설계 내용**:
```python
class SentencePieceTokenizer(BaseComponent):
    """S3 기반 통합 SentencePiece 토크나이저"""

    # 클래스 변수로 싱글톤 관리
    _processor = None
    _instance = None

    # S3 경로 상수
    S3_TOKENIZER_PATH = "models/shared/tokenizer.model"

    def __new__(cls, *args, **kwargs):
        """싱글톤 패턴 구현"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _load_from_s3(self) -> bytes:
        """S3에서 tokenizer.model 바이트 다운로드"""
        # S3Manager 활용하여 메모리로 직접 로드

    def _initialize_processor(self) -> None:
        """SentencePieceProcessor 초기화 (1회만)"""
        if self._processor is None:
            model_bytes = self._load_from_s3()
            sp = SentencePieceProcessor()
            sp.LoadFromSerializedProto(model_bytes)
            self._processor = sp
```

**핵심 특징**:
- ✅ **파일시스템 미사용**: 메모리 직접 로드
- ✅ **싱글톤 보장**: 클래스 레벨 인스턴스 관리
- ✅ **S3 우선**: VESSL 환경 최적화
- ✅ **로컬 개발 지원**: 환경변수로 로컬 경로 옵션

---

### Phase 3: unified_tokenizer 제거 및 의존성 정리 (개발 원칙 필수3)

**목표**: 기존 코드 정리 및 새 구조로 전환

**승인 요청 사항**:
1. `unified_tokenizer.py` 완전 삭제 승인
2. `base_loader.py`의 `load_tokenizer()` 수정 방향:
   - 옵션 A: `SentencePieceTokenizer` 직접 사용
   - 옵션 B: ComponentFactory.create_tokenizer() 사용

**작업 내용**:
1. `unified_tokenizer.py` 삭제
2. `base_loader.py` 수정:
```python
def load_tokenizer(self, path: str, **kwargs) -> Any:
    """통합 SentencePiece 토크나이저 로드"""
    from src.components.tokenizer.sentence_piece import SentencePieceTokenizer

    # Registry 시스템 활용
    tokenizer = SentencePieceTokenizer({"s3_manager": self.s3_manager})
    tokenizer.setup({})
    result = tokenizer.run({})

    return result["tokenizer"]
```

3. import 정리:
   - `tokenizer/__init__.py`에서 UnifiedSentencePieceTokenizer 제거
   - 관련 import 문 모두 정리

---

### Phase 4: S3 통합 구현 (개발 원칙 필수4)

**목표**: S3 기반 토크나이저 완전 구현

**구현 내용**:

#### 4.1 S3Manager 확장 (필요시)
```python
# src/utils/s3.py에 추가
def download_to_bytes(self, s3_key: str) -> bytes:
    """S3 객체를 바이트로 다운로드"""
    obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
    return obj['Body'].read()
```

#### 4.2 SentencePieceTokenizer 완전 구현
```python
@tokenizer_registry.register("unified-sentencepiece", category="tokenizer", version="2.0.0")
class SentencePieceTokenizer(BaseComponent):
    """S3 기반 통합 SentencePiece 토크나이저"""

    _processor = None  # 클래스 레벨 싱글톤

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.s3_manager = config.get("s3_manager") or create_s3_manager(config)

    def setup(self, ctx: dict[str, Any]) -> None:
        """토크나이저 초기화"""
        self.validate_initialized()
        self._ensure_processor_loaded()

    def _ensure_processor_loaded(self) -> None:
        """프로세서 로드 보장 (싱글톤)"""
        if SentencePieceTokenizer._processor is not None:
            return  # 이미 로드됨

        # S3에서 로드 시도
        if self.s3_manager and self.s3_manager.connected:
            try:
                # S3에서 바이트로 다운로드
                model_bytes = self.s3_manager.download_to_bytes(
                    "models/shared/tokenizer.model"
                )

                # SentencePiece 초기화
                from sentencepiece import SentencePieceProcessor
                sp = SentencePieceProcessor()
                sp.LoadFromSerializedProto(model_bytes)

                SentencePieceTokenizer._processor = sp
                logger.info("✅ S3에서 토크나이저 로드 성공")
                return
            except Exception as e:
                logger.warning(f"S3 로드 실패: {e}")

        # 로컬 개발 환경 폴백 (선택적)
        if local_path := os.getenv("TOKENIZER_MODEL_PATH"):
            from sentencepiece import SentencePieceProcessor
            sp = SentencePieceProcessor(model_file=local_path)
            SentencePieceTokenizer._processor = sp
            logger.info(f"✅ 로컬 경로에서 토크나이저 로드: {local_path}")
            return

        raise RuntimeError(
            "토크나이저 로드 실패. "
            "S3 연결을 확인하거나 TOKENIZER_MODEL_PATH 환경변수를 설정하세요."
        )

    def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """토크나이저 반환"""
        self.validate_initialized()

        if self._processor is None:
            raise RuntimeError("토크나이저가 초기화되지 않았습니다")

        return {
            "tokenizer": self._processor,
            "vocab_size": self._processor.get_piece_size(),
            "special_tokens": {
                "bos_id": self._processor.bos_id(),
                "eos_id": self._processor.eos_id(),
                "pad_id": self._processor.pad_id(),
                "unk_id": self._processor.unk_id(),
            }
        }
```

---

### Phase 5: 검증 및 테스트 (개발 원칙 필수5)

**목표**: 구현 완료 후 계획 대비 검증

**검증 항목**:

1. **기능 검증**
   - [ ] S3에서 토크나이저 로드 성공
   - [ ] 싱글톤 패턴 동작 확인
   - [ ] 토크나이징 결과 동일성 검증

2. **VESSL 환경 검증**
   - [ ] models/ 폴더 없이 정상 동작
   - [ ] 컨테이너 재시작 후 정상 로드
   - [ ] S3 연결 실패 시 적절한 에러 메시지

3. **하위 호환성**
   - [ ] base_loader.py 정상 동작
   - [ ] ComponentFactory.create_tokenizer() 정상 동작
   - [ ] training_pipeline.py 영향 없음

**성과 측정 기준**:
- ✅ 파일시스템 의존성 100% 제거
- ✅ S3 기반 로드 구현 완료
- ✅ 메모리 효율성 유지 (싱글톤)
- ✅ VESSL 환경 완전 지원

**계획 대비 평가**:
각 Phase 완료 후 목표 달성도를 객관적으로 평가하고 보고

---

## 🔧 기술적 고려사항 (개발 원칙 필수6)

### 의존성 관리
- **기존 유지**: `sentencepiece` 패키지 (uv 관리)
- **기존 활용**: `boto3` (S3Manager에서 이미 사용)
- **추가 불필요**: 새로운 패키지 없음

### 환경 설정
```bash
# VESSL 환경 (S3 자동 사용)
export AWS_ACCESS_KEY_ID=xxx
export AWS_SECRET_ACCESS_KEY=xxx
export S3_BUCKET=wmtp-models

# 로컬 개발 환경 (선택적)
export TOKENIZER_MODEL_PATH=/path/to/tokenizer.model
```

### 사전 준비 (1회만)
```bash
# tokenizer.model을 S3에 업로드
aws s3 cp models/7b_1t_4/tokenizer.model s3://wmtp-models/models/shared/tokenizer.model
```

---

## 📊 예상 효과

### 즉시 효과
- 🎯 **VESSL 완전 지원**: models/ 마운트 불필요
- 📦 **구조 단순화**: unified_tokenizer 제거로 복잡도 감소
- 🚀 **로드 속도**: 488KB 메모리 직접 로드 (1초 이내)

### 장기 효과
- 🔧 **유지보수성**: 단일 파일로 관리 용이
- 🌐 **확장성**: 다른 클라우드 스토리지 지원 가능
- 💾 **메모리 효율**: 싱글톤 패턴 유지

---

## 📝 실행 체크리스트

- [ ] Phase 1: 현재 구조 분석 완료
- [ ] Phase 2: 새 토크나이저 설계 검토
- [ ] Phase 3: unified_tokenizer 제거 승인
- [ ] Phase 4: S3 통합 구현 완료
- [ ] Phase 5: 검증 및 테스트 통과
- [ ] S3에 tokenizer.model 업로드 완료
- [ ] 문서 업데이트 완료

---

## 🚨 리스크 및 대응

### 리스크 1: S3 연결 실패
- **대응**: 로컬 경로 폴백 옵션 제공 (환경변수)

### 리스크 2: 메모리 로드 실패
- **대응**: 명확한 에러 메시지 및 복구 가이드

### 리스크 3: 기존 코드 호환성
- **대응**: base_loader.py 신중하게 수정

---

## 💡 결론

이 계획은 파일시스템 의존성을 완전히 제거하고 S3 기반의 깔끔한 토크나이저 시스템을 구축합니다.

**핵심 철학**:
"단순함이 최고의 설계 - S3에서 메모리로 직접, 파일시스템 제거"

각 Phase는 개발 원칙을 철저히 준수하며, VESSL 환경에 최적화된 안정적인 시스템을 보장합니다.
