# WMTP Pipeline Test Setup for MacBook M3

## 📋 개요

WMTP 파이프라인을 MacBook M3 Pro (64GB)에서 테스트하기 위한 경량화 설정입니다.
원본 7B MTP 모델 대신 작은 모델을 MTP 구조로 래핑하여 사용합니다.

## 🏗️ 구현 구조

### 1. **MTP 모델 래퍼** (`src/components/model/mtp_wrapper.py`)
- Sheared-LLaMA-2.7B를 기반으로 MTP 헤드 추가
- 4개의 예측 헤드 (t+1, t+2, t+3, t+4)
- MPS (Metal Performance Shaders) 지원

### 2. **테스트 모델 로더** (`src/components/loader/test_mtp_loader.py`)
- `TestMTPLoader`: Sheared-LLaMA-2.7B 기반 (2.7B 파라미터)
- `TinyMTPLoader`: DistilGPT2 기반 (82M 파라미터)
- 캐싱 지원으로 재로딩 최적화

### 3. **M3 최적화 설정**
- `configs/config.m3_test.yaml`: M3 환경 설정
- `configs/recipe.m3_test.yaml`: 테스트용 간소화 레시피
- MPS 자동 감지 및 활용
- 메모리 최적화 (gradient checkpointing)

### 4. **테스트 데이터셋**
- `dataset/test_samples/`: 5개 샘플만 포함
- 빠른 파이프라인 검증용

## 🚀 실행 방법

### 기본 실행 (Sheared-LLaMA-2.7B)
```bash
python test_m3_pipeline.py
```

### 초경량 모드 (DistilGPT2, 82M)
```bash
python test_m3_pipeline.py --tiny
```

### Dry Run (실제 학습 없이 검증)
```bash
python test_m3_pipeline.py --dry-run
```

### 커스텀 설정
```bash
python test_m3_pipeline.py \
    --config configs/config.m3_test.yaml \
    --recipe configs/recipe.m3_test.yaml
```

## 💾 메모리 요구사항

| 모델 | 파라미터 | 예상 메모리 | 권장 환경 |
|------|---------|------------|-----------|
| 7B MTP (원본) | 7B | ~28GB | A100 GPU |
| Sheared-LLaMA-2.7B + MTP | 2.7B | ~11GB | M3 Pro (64GB) |
| DistilGPT2 + MTP | 82M | ~0.5GB | 모든 M3 |

## ⚙️ MPS (Metal) 지원

코드베이스는 이미 MPS를 지원합니다:
```python
# src/components/loader/unified_model_loader.py
if compute_backend == "auto":
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"  # ✅ M3 GPU 자동 감지
    else:
        return "cpu"
```

## 🔧 트러블슈팅

### 1. MPS 관련 오류
```bash
# MPS 사용 불가시 CPU로 전환
PYTORCH_ENABLE_MPS_FALLBACK=1 python test_m3_pipeline.py
```

### 2. 메모리 부족
```bash
# 초경량 모드 사용
python test_m3_pipeline.py --tiny

# 또는 배치 크기 줄이기 (recipe.m3_test.yaml)
batch_size: 1  # 이미 1로 설정됨
```

### 3. 모델 다운로드 실패
```bash
# HuggingFace 토큰 설정 (.env 파일 확인)
export HF_TOKEN=your_huggingface_token_here
```

## 📊 예상 결과

테스트 실행 시:
1. ✅ 환경 체크 (PyTorch, MPS, Memory)
2. ✅ 모델 래핑 (Base Model + MTP Heads)
3. ✅ 10 스텝 학습 진행
4. ✅ 손실값 감소 확인
5. ✅ 체크포인트 저장

## 🎯 테스트 목적

1. **파이프라인 검증**: 전체 학습 파이프라인이 정상 작동하는지 확인
2. **MTP 구조 검증**: 가중치 계산 및 손실 함수 구현 확인
3. **M3 호환성**: Apple Silicon에서의 실행 가능성 확인
4. **메모리 효율성**: 제한된 리소스에서의 학습 가능성 확인

## 📝 주의사항

- 이는 **테스트 전용** 설정입니다
- 실제 성능 평가는 원본 7B MTP 모델과 전체 데이터셋 필요
- MPS는 아직 실험적 기능이므로 일부 연산에서 CPU fallback 발생 가능

## 🔍 검증 체크리스트

- [ ] MPS 디바이스 감지 확인
- [ ] 모델 로딩 성공
- [ ] MTP 헤드 생성 확인 (4개)
- [ ] 손실 계산 정상 작동
- [ ] 그래디언트 업데이트 확인
- [ ] 체크포인트 저장 성공
- [ ] MLflow 로깅 작동

## 🚦 다음 단계

파이프라인 검증 후:
1. 더 큰 데이터셋으로 확장
2. Critic-WMTP, Rho1-WMTP 알고리즘 테스트
3. 실제 7B 모델로 전환 (적절한 GPU 환경에서)
