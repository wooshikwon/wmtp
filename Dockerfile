# WMTP VESSL GPU Cluster Dockerfile
# Optimized for PyTorch 2.4+ with CUDA support and S3 checkpoint saving

FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    vim \
    htop \
    tmux \
    rsync \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package management
RUN pip install uv

# Copy uv configuration files first (for better Docker layer caching)
COPY pyproject.toml uv.lock ./

# Install Python dependencies using uv
RUN uv sync --frozen --no-cache

# Copy source code
COPY . .

# Set environment variables for optimal performance
ENV PYTHONPATH=/workspace
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Configure for distributed training
ENV NCCL_DEBUG=INFO
ENV NCCL_TREE_THRESHOLD=0
ENV NCCL_IB_DISABLE=1
ENV NCCL_SOCKET_IFNAME=eth0

# AWS/S3 environment variables (to be set by VESSL)
ENV AWS_DEFAULT_REGION=ap-northeast-2
ENV AWS_REGION=ap-northeast-2

# MLflow tracking (container-safe defaults)
ENV MLFLOW_TRACKING_URI=file:///workspace/mlruns
ENV MLFLOW_REGISTRY_URI=file:///workspace/mlruns

# Create necessary directories
RUN mkdir -p /workspace/checkpoints /workspace/mlruns /workspace/logs

# Set proper permissions
RUN chmod -R 755 /workspace

# Default command for VESSL
CMD ["python", "-m", "src.cli.train", "--help"]