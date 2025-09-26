#!/usr/bin/env python3
"""
AWS CLI를 이용한 간단한 S3 모델 디렉토리 교체
"""

import os
import subprocess
from dotenv import load_dotenv

load_dotenv()

# AWS CLI를 이용한 빠른 복사 및 삭제
print("🚀 AWS CLI를 이용한 S3 모델 디렉토리 교체")
print("=" * 60)

# Step 1: 기존 models/ 삭제
print("\n[Step 1] 기존 models/ 삭제...")
cmd1 = ["aws", "s3", "rm", "s3://wmtp/models/", "--recursive", "--region", "eu-north-1"]
result1 = subprocess.run(cmd1, capture_output=True, text=True)
if result1.returncode == 0:
    print("✅ 기존 models/ 삭제 완료")
else:
    print(f"❌ 삭제 오류: {result1.stderr}")

# Step 2: models_v2/ → models/ 복사
print("\n[Step 2] models_v2/ → models/ 복사...")
cmd2 = ["aws", "s3", "cp", "s3://wmtp/models_v2/", "s3://wmtp/models/",
        "--recursive", "--region", "eu-north-1"]
result2 = subprocess.run(cmd2, capture_output=True, text=True)
if result2.returncode == 0:
    print("✅ models_v2/ → models/ 복사 완료")
else:
    print(f"❌ 복사 오류: {result2.stderr}")

# Step 3: models_v2/ 삭제
print("\n[Step 3] models_v2/ 삭제...")
cmd3 = ["aws", "s3", "rm", "s3://wmtp/models_v2/", "--recursive", "--region", "eu-north-1"]
result3 = subprocess.run(cmd3, capture_output=True, text=True)
if result3.returncode == 0:
    print("✅ models_v2/ 삭제 완료")
else:
    print(f"❌ 삭제 오류: {result3.stderr}")

print("\n" + "=" * 60)
print("🎉 모델 디렉토리 교체 완료!")
print("\n최종 구조:")
print("  s3://wmtp/models/sheared-llama-2.7b/")
print("  s3://wmtp/models/llama-7b-mtp/")
print("  s3://wmtp/models/starling-rm-7b/")