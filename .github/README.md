# GitHub Actions CI/CD 설정 가이드

WMTP 프로젝트의 자동화된 빌드, 테스트, 배포를 위한 GitHub Actions 설정 가이드입니다.

## 🔐 필수 GitHub Secrets 설정

GitHub 저장소의 Settings → Secrets and variables → Actions에서 다음 시크릿을 추가하세요:

### 필수 Secrets
```yaml
# AWS S3 (MLflow 추적용)
AWS_ACCESS_KEY_ID: "AKIAXXXXXXXXXXXXXX"
AWS_SECRET_ACCESS_KEY: "wJalrXUtnFEMI/K7MDENG/XXXXXXXXXXXX"

# HuggingFace (모델 다운로드용)
HF_TOKEN: "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# VESSL (선택적, 자동 배포용)
VESSL_API_TOKEN: "vsl_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### Secrets 추가 방법
1. GitHub 저장소 → Settings
2. Secrets and variables → Actions
3. New repository secret 클릭
4. Name과 Secret 입력 후 저장

## 📋 워크플로우 구성

### 1. `docker-build-push.yml`
**목적**: 코드 변경 시 자동으로 Docker 이미지를 빌드하고 GitHub Container Registry에 푸시

**트리거**:
- main/master 브랜치 푸시
- Pull Request
- 수동 실행 (workflow_dispatch)

**주요 단계**:
1. 코드 린트 및 타입 체크
2. 4개 알고리즘 dry-run 테스트
3. Docker 이미지 빌드
4. ghcr.io에 이미지 푸시
5. (선택적) VESSL 자동 배포

### 2. `test-algorithms.yml`
**목적**: PR 또는 수동으로 모든 알고리즘 테스트 실행

**트리거**:
- Pull Request
- 수동 실행 (특정 알고리즘 선택 가능)

**테스트 매트릭스**:
- baseline-mtp
- critic-wmtp
- rho1-weighted
- rho1-tokenskip

## 🚀 사용 방법

### 자동 실행
```bash
# 1. 브랜치에서 작업
git checkout -b feature/my-feature

# 2. 변경사항 커밋
git add .
git commit -m "feat: Add new feature"

# 3. 푸시하면 자동으로 CI 실행
git push origin feature/my-feature

# 4. PR 생성 시 테스트 자동 실행
```

### 수동 실행
1. GitHub 저장소 → Actions 탭
2. 원하는 워크플로우 선택
3. "Run workflow" 버튼 클릭
4. 파라미터 선택 후 실행

### 특정 알고리즘만 테스트
```yaml
# Actions 탭에서 "Test All Algorithms" 선택
# algorithm 드롭다운에서 선택:
- baseline-mtp
- critic-wmtp
- rho1-weighted
- rho1-tokenskip
```

## 🐳 생성되는 Docker 이미지 태그

자동으로 생성되는 태그 형식:

```bash
# 브랜치 기반
ghcr.io/wooshikwon/wmtp:main
ghcr.io/wooshikwon/wmtp:feature-branch-name

# SHA 기반
ghcr.io/wooshikwon/wmtp:main-abc1234

# 날짜 기반
ghcr.io/wooshikwon/wmtp:20241228-abc1234

# 최신 (main 브랜치만)
ghcr.io/wooshikwon/wmtp:latest

# 버전 태그 (v1.0.0 태그 푸시 시)
ghcr.io/wooshikwon/wmtp:1.0.0
ghcr.io/wooshikwon/wmtp:1.0
```

## 📊 CI/CD 상태 확인

### Actions 탭에서 확인
- ✅ 성공: 녹색 체크마크
- ❌ 실패: 빨간색 X
- 🔄 진행 중: 노란색 원

### 상세 로그 확인
1. Actions 탭 → 워크플로우 실행 클릭
2. Job 선택 → 단계별 로그 확인

### 이미지 확인
```bash
# GitHub Packages 페이지에서 확인
https://github.com/wooshikwon/wmtp/pkgs/container/wmtp

# Docker로 확인
docker pull ghcr.io/wooshikwon/wmtp:latest
docker images | grep wmtp
```

## 🔧 문제 해결

### 권한 오류
```yaml
# .github/workflows/docker-build-push.yml에 권한 추가
permissions:
  contents: read
  packages: write
```

### 시크릿 누락
```
Error: GITHUB_TOKEN is not set
```
→ Settings → Actions → General → Workflow permissions → Read and write permissions 선택

### 이미지 푸시 실패
```bash
# 수동으로 로그인 테스트
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin
```

## 🏷️ 버전 태그 관리

### 새 버전 릴리즈
```bash
# Semantic Versioning 사용
git tag v1.0.0
git push origin v1.0.0

# 자동으로 다음 태그 생성:
# - ghcr.io/wooshikwon/wmtp:1.0.0
# - ghcr.io/wooshikwon/wmtp:1.0
# - ghcr.io/wooshikwon/wmtp:latest
```

## 📈 모니터링

### GitHub Actions 대시보드
- 실행 시간 추이
- 성공/실패 비율
- 리소스 사용량

### 이메일 알림 설정
Settings → Notifications → GitHub Actions → Email 알림 활성화

## 🔄 로컬에서 워크플로우 테스트

[act](https://github.com/nektos/act) 도구를 사용하여 로컬에서 테스트:

```bash
# act 설치
brew install act  # macOS
# 또는
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | bash

# 워크플로우 테스트
act -W .github/workflows/docker-build-push.yml

# 특정 이벤트 테스트
act push
act pull_request
```

## 📚 참고 문서

- [GitHub Actions 공식 문서](https://docs.github.com/en/actions)
- [GitHub Container Registry 가이드](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)
- [VESSL CLI 문서](https://docs.vessl.ai/)