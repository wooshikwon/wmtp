# WMTP Fine-Tuning Framework - Makefile
# Docker and deployment operations

# Configuration
IMAGE_NAME := wmtp
IMAGE_TAG := latest
REGISTRY := ghcr.io/wooshikwon # GitHub Container Registry
FULL_IMAGE := $(if $(REGISTRY),$(REGISTRY)/$(IMAGE_NAME),$(IMAGE_NAME))

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m # No Color

.PHONY: help
help: ## Show this help message
	@echo "$(GREEN)WMTP Fine-Tuning Framework - Docker Operations$(NC)"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@awk 'BEGIN {FS = ":.*##"; printf "\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  $(YELLOW)%-15s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(GREEN)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Docker Operations

.PHONY: build
build: ## Build Docker image
	@echo "$(GREEN)Building Docker image: $(FULL_IMAGE):$(IMAGE_TAG)$(NC)"
	docker build -f docker/Dockerfile -t $(FULL_IMAGE):$(IMAGE_TAG) .
	@echo "$(GREEN)Build complete!$(NC)"

.PHONY: build-nocache
build-nocache: ## Build Docker image without cache
	@echo "$(GREEN)Building Docker image without cache: $(FULL_IMAGE):$(IMAGE_TAG)$(NC)"
	docker build --no-cache -f docker/Dockerfile -t $(FULL_IMAGE):$(IMAGE_TAG) .
	@echo "$(GREEN)Build complete!$(NC)"

.PHONY: push
push: ## Push Docker image to registry
	@if [ -z "$(REGISTRY)" ]; then \
		echo "$(RED)Error: REGISTRY is not set. Edit Makefile or use: make push REGISTRY=your.registry.com$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)Pushing image: $(FULL_IMAGE):$(IMAGE_TAG)$(NC)"
	docker push $(FULL_IMAGE):$(IMAGE_TAG)
	@echo "$(GREEN)Push complete!$(NC)"

.PHONY: image
image: build ## Alias for build

.PHONY: run-bash
run-bash: ## Run interactive bash in container
	@echo "$(GREEN)Starting interactive bash session...$(NC)"
	docker run -it --rm \
		--gpus all \
		-v $(PWD)/configs:/app/configs \
		-v $(PWD)/models:/app/models \
		-v $(PWD)/datasets:/app/datasets \
		$(FULL_IMAGE):$(IMAGE_TAG) \
		/bin/bash

.PHONY: run-train
run-train: ## Run training with VESSL config
	@echo "$(GREEN)Starting training...$(NC)"
	docker run --rm \
		--gpus all \
		-v $(PWD)/configs:/app/configs \
		-v $(PWD)/models:/app/models \
		-v $(PWD)/datasets:/app/datasets \
		-v $(PWD)/outputs:/app/outputs \
		$(FULL_IMAGE):$(IMAGE_TAG) \
		uv run python -m src.cli.train \
			--config /app/configs/config.vessl.yaml \
			--recipe /app/configs/recipe.rho1.yaml

.PHONY: run-eval
run-eval: ## Run evaluation
	@echo "$(GREEN)Starting evaluation...$(NC)"
	docker run --rm \
		--gpus all \
		-v $(PWD)/configs:/app/configs \
		-v $(PWD)/models:/app/models \
		-v $(PWD)/datasets:/app/datasets \
		-v $(PWD)/outputs:/app/outputs \
		$(FULL_IMAGE):$(IMAGE_TAG) \
		uv run python -m src.cli.eval \
			--config /app/configs/config.vessl.yaml \
			--recipe /app/configs/recipe.rho1.yaml \
			--checkpoint /app/models/checkpoints/final.pt

##@ Development Operations

.PHONY: test
test: ## Run tests in Docker
	@echo "$(GREEN)Running tests in Docker...$(NC)"
	docker run --rm \
		-v $(PWD)/tests:/app/tests \
		$(FULL_IMAGE):$(IMAGE_TAG) \
		uv run pytest -q

.PHONY: lint
lint: ## Run linting in Docker
	@echo "$(GREEN)Running linting in Docker...$(NC)"
	docker run --rm \
		-v $(PWD)/src:/app/src \
		$(FULL_IMAGE):$(IMAGE_TAG) \
		uv run ruff check .

.PHONY: format
format: ## Format code using ruff
	@echo "$(GREEN)Formatting code...$(NC)"
	uv run ruff format .

##@ VESSL Operations

.PHONY: vessl-run
vessl-run: ## Submit job to VESSL
	@echo "$(GREEN)Submitting job to VESSL...$(NC)"
	vessl run -f docker/vessl.yaml

.PHONY: vessl-logs
vessl-logs: ## Show VESSL run logs (requires RUN_ID)
	@if [ -z "$(RUN_ID)" ]; then \
		echo "$(RED)Error: RUN_ID is required. Use: make vessl-logs RUN_ID=your-run-id$(NC)"; \
		exit 1; \
	fi
	vessl logs $(RUN_ID)

##@ Cleanup Operations

.PHONY: clean
clean: ## Clean Docker images and containers
	@echo "$(YELLOW)Cleaning up Docker resources...$(NC)"
	docker system prune -f
	@echo "$(GREEN)Cleanup complete!$(NC)"

.PHONY: clean-all
clean-all: ## Clean all Docker resources (including volumes)
	@echo "$(RED)Warning: This will remove all Docker volumes!$(NC)"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	docker system prune -af --volumes
	@echo "$(GREEN)Deep cleanup complete!$(NC)"

##@ Info Operations

.PHONY: info
info: ## Show Docker and system info
	@echo "$(GREEN)System Information:$(NC)"
	@echo "Image Name: $(FULL_IMAGE):$(IMAGE_TAG)"
	@echo "Registry: $(if $(REGISTRY),$(REGISTRY),Not set)"
	@echo ""
	@echo "$(GREEN)Docker Info:$(NC)"
	@docker --version
	@echo ""
	@echo "$(GREEN)GPU Info:$(NC)"
	@nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || echo "No NVIDIA GPU detected"

.PHONY: check-gpu
check-gpu: ## Check GPU availability in Docker
	@echo "$(GREEN)Checking GPU availability in Docker...$(NC)"
	docker run --rm --gpus all \
		$(FULL_IMAGE):$(IMAGE_TAG) \
		python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
