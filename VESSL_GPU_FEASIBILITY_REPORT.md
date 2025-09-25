# VESSL GPU 환경 실행 가능성 검토 보고서

## 📊 종합 평가: **조건부 실행 가능** ⚠️

## 1. 환경 구성 분석

### 1.1 Docker 환경
```yaml
Base Image: pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
GPU Support: CUDA 12.1 + cuDNN 9
Package Manager: uv (최신 고속 패키지 관리자)
```

### 1.2 VESSL 리소스 설정
```yaml
GPU: 4x NVIDIA A100 (80GB VRAM each)
CPU: 32 cores
Memory: 256GB RAM
Storage: 500GB disk
```

### 1.3 스토리지 아키텍처
- **S3 버킷**: `wmtp` (eu-north-1 region)
- **모델 경로**: S3에서 직접 다운로드
- **캐시 디렉토리**: `/tmp/models`, `/tmp/datasets`

## 2. 모델 호환성 검토

### 2.1 Recipe 별 모델 요구사항

#### **MTP Baseline** (recipe.mtp_baseline.yaml)
- Base Model: `facebook/multi-token-prediction` ✅
- Reference Model: `codellama/CodeLlama-7b-Python-hf` ✅
- 메모리 요구: ~30GB VRAM
- **상태: 실행 가능**

#### **Critic-WMTP** (recipe.critic.yaml)
- Base Model: `facebook/multi-token-prediction` ✅
- RM Model: `sfair/Llama-3-8B-RM-Reward-Model` ❓ (S3 미확인)
- Reference Model: `codellama/CodeLlama-7b-Python-hf` ✅
- 메모리 요구: ~60GB VRAM
- **상태: RM 모델 업로드 필요**

#### **Rho1-WMTP** (recipe.rho1.yaml)
- Base Model: `facebook/multi-token-prediction` ✅
- Reference Model: `codellama/CodeLlama-7b-Python-hf` ✅
- 메모리 요구: ~40GB VRAM
- **상태: 실행 가능**

### 2.2 S3 업로드 현황
✅ **업로드 완료:**
- `models/Starling-RM-7B-alpha` (테스트용 RM 모델)
- `models/Sheared-LLaMA-2.7B` (경량 테스트 모델)

⚠️ **업로드 필요:**
- `models/7b_1t_4` (Facebook MTP 모델)
- `models/Llama_3_8B_RM` (Critic용 RM 모델)
- `models/codellama_7b_python` (Reference 모델)

## 3. 잠재적 이슈 및 해결방안

### 🔴 **Critical Issue 1: PyTorch 2.4.0 Safetensors 취약점**
- **문제**: PyTorch 2.4.0에서 `.bin` 파일 로딩 시 보안 취약점
- **영향**: Meta MTP 모델이 `.bin` 형식일 경우 로딩 실패 가능
- **해결방안**:
  1. Docker 이미지를 PyTorch 2.6.0+로 업그레이드 ✅ (권장)
  2. 또는 모델을 safetensors 형식으로 변환

### 🟡 **Issue 2: 모델 경로 불일치**
- **문제**: Recipe 파일의 모델 ID와 실제 S3 경로 매칭 필요
- **해결방안**:
  ```python
  # config.vessl.yaml 수정
  paths:
    models:
      base_s3: "models/7b_1t_4"  # Facebook MTP 실제 경로로 수정
      rm_s3: "models/Starling-RM-7B-alpha"  # 업로드된 모델로 변경
  ```

### 🟡 **Issue 3: HuggingFace 모델 다운로드**
- **문제**: `facebook/multi-token-prediction` 접근 권한
- **해결방안**:
  1. HF_TOKEN 설정 확인
  2. 또는 사전 다운로드 후 S3 업로드

## 4. 필수 조치 사항

### 4.1 즉시 필요한 작업
1. **Dockerfile 수정**: PyTorch 버전 업그레이드
   ```dockerfile
   FROM pytorch/pytorch:2.6.0-cuda12.1-cudnn9-runtime
   ```

2. **모델 업로드 완료**:
   ```bash
   # Facebook MTP 모델 업로드
   aws s3 sync models/7b_1t_4 s3://wmtp/models/7b_1t_4

   # CodeLlama Reference 모델 (HuggingFace에서 직접 다운로드 가능)
   ```

3. **Config 파일 검증**:
   - S3 경로와 실제 업로드된 모델 경로 일치 확인
   - AWS credentials 설정 확인

### 4.2 Recipe 별 권장 실행 순서
1. **Rho1-WMTP** 먼저 테스트 (가장 단순, RM 불필요)
2. **MTP Baseline** 테스트 (비교 기준선)
3. **Critic-WMTP** (RM 모델 준비 후)

## 5. CPU 테스트 성공 내역 반영

### 성공 요인
- ✅ 로더 레지스트리 시스템 정상 작동
- ✅ MTP 헤드 구조 검증 완료
- ✅ WMTP 손실 계산 로직 확인
- ✅ Safetensors 변환으로 PyTorch 이슈 해결

### GPU 환경 장점
- 메모리 제약 해소 (A100 80GB x 4 = 320GB VRAM)
- CUDA 가속으로 빠른 학습
- Mixed Precision (bf16) 지원으로 효율성 증대

## 6. 최종 권고사항

### ✅ **실행 가능 조건**
1. PyTorch 버전 업그레이드 (2.4.0 → 2.6.0+)
2. 필수 모델 S3 업로드 완료
3. Config 경로 검증

### ⚠️ **주의 사항**
1. 첫 실행 시 모델 다운로드로 시간 소요 (약 30분)
2. S3 비용 고려 (대용량 모델 전송)
3. VESSL secrets 설정 필수 (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, HF_TOKEN)

### 🎯 **추천 테스트 프로세스**
```bash
# 1. Docker 이미지 빌드 및 푸시
docker build -t wmtp:latest docker/
docker tag wmtp:latest your-registry/wmtp:latest
docker push your-registry/wmtp:latest

# 2. VESSL에서 Rho1 테스트 실행
vessl run -f docker/vessl.yaml --recipe configs/recipe.rho1.yaml

# 3. 로그 모니터링
vessl logs -f <run-id>
```

## 7. 결론

**VESSL GPU 환경에서 WMTP 실험은 기술적으로 실행 가능합니다.**
단, PyTorch 버전 업그레이드와 모델 업로드 완료가 선행되어야 합니다.

CPU 테스트에서 검증된 코드 구조와 로직은 GPU 환경에서도 정상 작동할 것으로 예상되며,
A100 GPU의 충분한 메모리와 연산 능력으로 대규모 모델 학습이 가능합니다.

---
*작성일: 2024-09-25*
*검토 기준: CPU 테스트 성공 결과 + Docker/VESSL 설정 분석*
