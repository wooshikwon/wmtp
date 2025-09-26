#!/bin/bash
# WMTP VESSL GPU Cluster Deployment Script
# Automates Docker build and VESSL experiment submission

set -e  # Exit on any error

# Configuration
DOCKER_REGISTRY="your-registry"  # Update with your Docker registry
IMAGE_NAME="wmtp"
TAG="latest"
FULL_IMAGE="$DOCKER_REGISTRY/$IMAGE_NAME:$TAG"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 WMTP VESSL Deployment Script${NC}"
echo "======================================"

# Check prerequisites
echo -e "${YELLOW}📋 Checking prerequisites...${NC}"

if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed${NC}"
    exit 1
fi

if ! command -v vessl &> /dev/null; then
    echo -e "${RED}❌ VESSL CLI is not installed${NC}"
    echo "Install with: pip install vessl"
    exit 1
fi

# Check if logged into VESSL
if ! vessl whoami &> /dev/null; then
    echo -e "${RED}❌ Not logged into VESSL${NC}"
    echo "Login with: vessl configure"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites satisfied${NC}"

# Build Docker image
echo -e "${YELLOW}🐳 Building Docker image...${NC}"
docker build -t $FULL_IMAGE .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Docker image built successfully${NC}"
else
    echo -e "${RED}❌ Docker build failed${NC}"
    exit 1
fi

# Push to registry
echo -e "${YELLOW}📤 Pushing image to registry...${NC}"
docker push $FULL_IMAGE

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Image pushed successfully${NC}"
else
    echo -e "${RED}❌ Docker push failed${NC}"
    exit 1
fi

# Determine recipe file
RECIPE_FILE=${1:-"configs/vessl/recipe.cluster_rho1_wmtp.yaml"}
CONFIG_FILE=${2:-"configs/vessl/config.cluster.yaml"}

if [ ! -f "$RECIPE_FILE" ]; then
    echo -e "${RED}❌ Recipe file not found: $RECIPE_FILE${NC}"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}❌ Config file not found: $CONFIG_FILE${NC}"
    exit 1
fi

# Create VESSL experiment
echo -e "${YELLOW}🎯 Creating VESSL experiment...${NC}"

# Extract run name from recipe
RUN_NAME=$(grep "name:" $RECIPE_FILE | head -1 | awk '{print $2}' | tr -d '"')
EXPERIMENT_NAME="wmtp-${RUN_NAME}-$(date +%Y%m%d-%H%M)"

cat > vessl_experiment.yaml << EOF
name: $EXPERIMENT_NAME
description: "WMTP Cluster Training - S3 Checkpoint Auto-Save"
image: $FULL_IMAGE
resource:
  cluster: vessl-gcp-oregon
  preset: gpu-l4-4
  # Override with A100 if available:
  # preset: gpu-a100-4
working_dir: /workspace
environment:
  AWS_DEFAULT_REGION: ap-northeast-2
  AWS_REGION: ap-northeast-2
  PYTHONPATH: /workspace
  CUDA_VISIBLE_DEVICES: "0,1,2,3"
  NCCL_DEBUG: INFO
  # Add your AWS credentials as VESSL secrets:
  # AWS_ACCESS_KEY_ID: \${{ secrets.AWS_ACCESS_KEY_ID }}
  # AWS_SECRET_ACCESS_KEY: \${{ secrets.AWS_SECRET_ACCESS_KEY }}
mount:
  - name: workspace
    path: /workspace
command: |
  echo "🚀 Starting WMTP Cluster Training"
  echo "Config: $CONFIG_FILE"
  echo "Recipe: $RECIPE_FILE"

  # Start distributed training
  python -m src.cli.train \\
    --config $CONFIG_FILE \\
    --recipe $RECIPE_FILE \\
    --run-name $RUN_NAME \\
    --tags cluster,production,s3-checkpoints \\
    --verbose
EOF

# Submit experiment
echo -e "${YELLOW}🎯 Submitting VESSL experiment...${NC}"
vessl experiment create -f vessl_experiment.yaml

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ VESSL experiment submitted successfully${NC}"
    echo -e "${GREEN}📊 Monitor at: https://app.vessl.ai${NC}"
    echo ""
    echo -e "${YELLOW}💡 Key Features Enabled:${NC}"
    echo "   🔄 Auto S3 checkpoint saving every 500 steps"
    echo "   📈 MLflow experiment tracking"
    echo "   🚀 Multi-GPU distributed training"
    echo "   🔧 Phase 1 checkpoint fix applied"
else
    echo -e "${RED}❌ VESSL experiment submission failed${NC}"
    exit 1
fi

# Cleanup
rm -f vessl_experiment.yaml

echo ""
echo -e "${GREEN}🎉 Deployment complete!${NC}"
echo -e "${YELLOW}S3 checkpoints will be saved to: s3://wmtp/checkpoints/$RUN_NAME${NC}"