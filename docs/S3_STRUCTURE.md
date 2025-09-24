# S3 Bucket Structure for WMTP

본 문서는 WMTP 프로젝트의 S3 버킷(`wmtp`) 구조를 정의합니다.

## Bucket Overview

```
s3://wmtp/
├── models/                    # Pre-trained and fine-tuned models
├── datasets/                   # Training and evaluation datasets
├── checkpoints/               # Training checkpoints
├── mlflow/                    # MLflow tracking data
└── mlflow-artifacts/          # MLflow artifacts
```

## Directory Structure

### Models (`models/`)

```
models/
├── 7b_1t_4/                   # Base MTP model
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer/
├── Llama_3_8B_RM/             # Reward model for Critic
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer/
└── CodeLlama-7B-Python/       # Reference model for Rho-1
    ├── config.json
    ├── pytorch_model.bin
    └── tokenizer/
```

### Datasets (`datasets/`)

```
datasets/
├── mbpp/                      # MBPP dataset
│   ├── train.jsonl
│   ├── validation.jsonl
│   └── test.jsonl
└── contest/                   # Code Contest dataset
    ├── train/
    ├── validation/
    └── test/
```

### Checkpoints (`checkpoints/`)

```
checkpoints/
├── rho1-wmtp/
│   ├── step_1000/
│   ├── step_2000/
│   └── final.pt
└── critic-wmtp/
    ├── step_1000/
    ├── step_2000/
    └── final.pt
```

### MLflow (`mlflow/` and `mlflow-artifacts/`)

```
mlflow/
├── 0/                         # Default experiment
└── 1/                         # mtp/wmtp experiment
    ├── meta.yaml
    └── runs/
        └── <run_id>/
            ├── meta.yaml
            ├── metrics/
            ├── params/
            └── tags/

mlflow-artifacts/
└── 1/
    └── <run_id>/
        ├── model/
        ├── predictions/
        ├── weight_stats/
        └── reports/
```

## Upload Instructions

### 1. Upload Models from HuggingFace

```bash
# Download from HuggingFace (if needed)
huggingface-cli download facebook/multi-token-prediction --local-dir ./models/7b_1t_4
huggingface-cli download codellama/CodeLlama-7b-Python-hf --local-dir ./models/CodeLlama-7B-Python

# Upload to S3
aws s3 sync ./models/7b_1t_4 s3://wmtp/models/7b_1t_4/
aws s3 sync ./models/CodeLlama-7B-Python s3://wmtp/models/CodeLlama-7B-Python/
```

### 2. Upload Datasets

```bash
# MBPP dataset
aws s3 sync ./dataset/mbpp s3://wmtp/datasets/mbpp/

# Contest dataset
aws s3 sync ./dataset/contest s3://wmtp/datasets/contest/
```

### 3. Verify Upload

```bash
# List uploaded files
aws s3 ls s3://wmtp/ --recursive

# Check file sizes
aws s3 ls s3://wmtp/models/ --recursive --summarize --human-readable
```

## Access Configuration

### Local Development (.env)

```bash
# .env file
HF_TOKEN=hf_xxx
AWS_ACCESS_KEY_ID=xxx
AWS_SECRET_ACCESS_KEY=xxx
S3_BUCKET_NAME=wmtp
```

### VESSL Secrets

VESSL UI에서 설정:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `HF_TOKEN`
- `S3_BUCKET_NAME`

## Data Flow

### Training Flow

1. **Model Loading**:
   - Check local cache (`/tmp/models/`)
   - If not found, download from S3
   - If not in S3, download from HuggingFace
   - Cache locally for session

2. **Dataset Loading**:
   - Check local cache (`/tmp/datasets/`)
   - If not found, download from S3
   - Process and cache

3. **Checkpoint Saving**:
   - Save to local first (`/tmp/checkpoints/`)
   - Upload to S3 (`s3://wmtp/checkpoints/`)
   - Keep last N checkpoints

4. **MLflow Tracking**:
   - Direct write to S3
   - No local MLflow server needed

### Evaluation Flow

1. Load checkpoint from S3
2. Load evaluation datasets from S3
3. Save results to MLflow (S3)

## Cost Optimization

### S3 Storage Classes

- **Standard**: Models and active checkpoints
- **Infrequent Access**: Old checkpoints and experiments
- **Glacier**: Archived experiments

### Lifecycle Policy

```json
{
  "Rules": [
    {
      "Id": "Archive old checkpoints",
      "Status": "Enabled",
      "Transitions": [
        {
          "Days": 30,
          "StorageClass": "STANDARD_IA"
        },
        {
          "Days": 90,
          "StorageClass": "GLACIER"
        }
      ]
    }
  ]
}
```

## Security

### Bucket Policy

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "VESSLAccess",
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::ACCOUNT:user/vessl-user"
      },
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::wmtp/*",
        "arn:aws:s3:::wmtp"
      ]
    }
  ]
}
```

### Encryption

- Server-side encryption enabled (SSE-S3)
- HTTPS only for data transfer

## Monitoring

### CloudWatch Metrics

- Request count
- Data transfer (GB)
- Storage size
- 4xx/5xx errors

### Cost Alerts

- Set billing alert at $100/month
- Monitor data transfer costs
- Review storage usage monthly
