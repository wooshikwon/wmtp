#!/usr/bin/env python3
"""
S3 models/ 디렉토리 교체
1. 기존 models/ 삭제
2. models_v2/ → models/ 이름 변경
"""

import boto3
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name='eu-north-1'
)

bucket = 'wmtp'

print(f"[{datetime.now()}] S3 모델 디렉토리 교체 시작")
print("=" * 60)

# Step 1: 기존 models/ 삭제
print("\n[Step 1] 기존 models/ 디렉토리 삭제")
paginator = s3.get_paginator('list_objects_v2')

# 삭제할 객체 수집
delete_count = 0
for page in paginator.paginate(Bucket=bucket, Prefix='models/'):
    if 'Contents' in page:
        delete_keys = [{'Key': obj['Key']} for obj in page['Contents']]

        # 1000개씩 배치 삭제 (AWS 제한)
        for i in range(0, len(delete_keys), 1000):
            batch = delete_keys[i:i+1000]
            s3.delete_objects(Bucket=bucket, Delete={'Objects': batch})
            delete_count += len(batch)
            print(f"  삭제됨: {delete_count}개 파일...")

print(f"✅ 총 {delete_count}개 파일 삭제 완료")

# Step 2: models_v2/ → models/ 복사
print("\n[Step 2] models_v2/ → models/ 복사")
copy_count = 0

for page in paginator.paginate(Bucket=bucket, Prefix='models_v2/'):
    if 'Contents' in page:
        for obj in page['Contents']:
            old_key = obj['Key']
            new_key = old_key.replace('models_v2/', 'models/', 1)

            # S3 내부 복사
            copy_source = {'Bucket': bucket, 'Key': old_key}
            s3.copy_object(
                CopySource=copy_source,
                Bucket=bucket,
                Key=new_key,
                MetadataDirective='COPY',
                TaggingDirective='COPY'
            )
            copy_count += 1

            if copy_count % 10 == 0:
                print(f"  복사됨: {copy_count}개 파일...")

print(f"✅ 총 {copy_count}개 파일 복사 완료")

# Step 3: models_v2/ 삭제
print("\n[Step 3] models_v2/ 디렉토리 삭제")
delete_v2_count = 0

for page in paginator.paginate(Bucket=bucket, Prefix='models_v2/'):
    if 'Contents' in page:
        delete_keys = [{'Key': obj['Key']} for obj in page['Contents']]

        for i in range(0, len(delete_keys), 1000):
            batch = delete_keys[i:i+1000]
            s3.delete_objects(Bucket=bucket, Delete={'Objects': batch})
            delete_v2_count += len(batch)

print(f"✅ models_v2/ 삭제 완료: {delete_v2_count}개 파일")

# 최종 확인
print("\n" + "=" * 60)
print("🎉 모델 디렉토리 교체 완료!")
print("\n최종 구조:")
print("  s3://wmtp/models/sheared-llama-2.7b/")
print("  s3://wmtp/models/llama-7b-mtp/")
print("  s3://wmtp/models/starling-rm-7b/")
print("\n모든 모델이 새로운 표준 구조로 업데이트되었습니다!")