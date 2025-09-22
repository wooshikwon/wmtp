# Contributing to WMTP Framework

WMTP 프레임워크에 기여해주셔서 감사합니다! 이 가이드는 프로젝트에 기여하는 방법을 안내합니다.

## 기여 방식

### 이슈 제출
- 버그 리포트, 기능 제안, 질문 등을 GitHub Issues를 통해 제출
- 이슈 템플릿을 사용하여 필요한 정보 제공

### Pull Request
1. 레포지토리를 Fork
2. Feature 브랜치 생성 (`git checkout -b feature/amazing-feature`)
3. 변경사항 커밋 (`git commit -m 'feat: Add amazing feature'`)
4. 브랜치에 Push (`git push origin feature/amazing-feature`)
5. Pull Request 생성

## 개발 환경 설정

### 필수 요구사항
- Python 3.11+
- uv 패키지 관리자
- Git

### 로컬 개발 환경 구축

```bash
# 레포지토리 클론
git clone https://github.com/yourusername/wmtp.git
cd wmtp

# uv로 의존성 설치
uv sync --frozen

# pre-commit 훅 설치
uv run pre-commit install

# 테스트 실행
uv run pytest
```

## 코딩 규칙

### 코드 스타일
- Python 코드는 ruff로 포맷팅 및 린팅
- 모든 함수와 클래스에 Google 스타일 docstring 작성
- 타입 힌트 필수

### 명명 규칙
- 파일/모듈: `snake_case.py`
- 클래스: `PascalCase`
- 함수/변수: `snake_case`
- 상수: `UPPER_SNAKE_CASE`

### 코드 품질 원칙
- 단일 책임 원칙(SRP) 준수
- 함수는 50줄 이내 권장
- 명확하고 자기 설명적인 코드 작성
- 복잡한 로직에는 주석 추가

## 커밋 메시지 규칙

커밋 메시지는 다음 형식을 따릅니다:

```
<type>: <subject>

<body>

<footer>
```

### 타입
- `feat`: 새로운 기능 추가
- `fix`: 버그 수정
- `refactor`: 리팩토링
- `docs`: 문서 변경
- `test`: 테스트 추가/수정
- `perf`: 성능 개선
- `style`: 코드 스타일 변경
- `chore`: 빌드 프로세스 등 기타 변경
- `exp`: 실험 코드

### 예시
```
feat: Add Rho-1 scorer implementation

- Implement absolute excess CE scoring
- Add percentile-based weighting
- Include z-score normalization

Closes #123
```

## 브랜치 전략

- `main`: 프로덕션 준비 코드 (보호됨)
- `develop`: 개발 브랜치
- `feature/*`: 기능 개발 브랜치
- `fix/*`: 버그 수정 브랜치
- `exp/*`: 실험 브랜치

## 테스트

### 테스트 작성
- 새로운 기능에는 반드시 테스트 포함
- 유닛 테스트와 통합 테스트 구분
- 테스트 이름은 명확하게 작성

### 테스트 실행
```bash
# 전체 테스트
uv run pytest

# 특정 테스트 파일
uv run pytest tests/test_scorer.py

# 커버리지 포함
uv run pytest --cov=src
```

## Pull Request 체크리스트

PR 제출 전 확인사항:

- [ ] 코드가 프로젝트 스타일 가이드를 따르는가?
- [ ] 모든 테스트가 통과하는가?
- [ ] 새로운 기능에 대한 테스트가 추가되었는가?
- [ ] 문서가 업데이트되었는가?
- [ ] 커밋 메시지가 규칙을 따르는가?
- [ ] PR 설명이 명확한가?

## 문서 기여

### 문서 종류
- API 문서: 코드 내 docstring
- 사용자 가이드: `docs/` 디렉토리
- 예제 코드: `examples/` 디렉토리

### 문서 작성 원칙
- 명확하고 간결한 설명
- 실제 동작하는 예제 포함
- 필요시 다이어그램 추가

## 이슈 라벨

- `bug`: 버그 리포트
- `enhancement`: 기능 개선 제안
- `documentation`: 문서 관련
- `question`: 질문
- `good first issue`: 처음 기여하기 좋은 이슈
- `help wanted`: 도움이 필요한 이슈

## 행동 강령

### 존중과 포용
- 모든 기여자를 존중하고 환영
- 건설적인 피드백 제공
- 다양한 관점과 경험 존중

### 협력
- 질문과 토론 환영
- 지식 공유 장려
- 함께 배우고 성장

## 질문 및 지원

- GitHub Issues: 버그 리포트 및 기능 제안
- Discussions: 일반적인 질문 및 토론
- Email: [contact@example.com]

## 라이선스

이 프로젝트에 기여함으로써, 귀하의 기여가 MIT 라이선스 하에 배포됨에 동의합니다.
