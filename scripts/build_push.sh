#!/bin/bash
# WMTP Fine-Tuning Framework - Docker Build and Push Script

set -e  # Exit on error

# Configuration
IMAGE_NAME="${IMAGE_NAME:-wmtp}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
REGISTRY="${DOCKER_REGISTRY:-}"  # Set via environment variable

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Functions
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    cat << EOF
WMTP Docker Build and Push Script

Usage: $(basename $0) [OPTIONS] COMMAND

Commands:
    build       Build Docker image
    push        Push image to registry
    all         Build and push image
    test        Test the built image

Options:
    -r, --registry REGISTRY    Set Docker registry URL
    -t, --tag TAG              Set image tag (default: latest)
    -n, --name NAME            Set image name (default: wmtp)
    --no-cache                 Build without cache
    -h, --help                 Show this help message

Environment Variables:
    DOCKER_REGISTRY            Default registry URL
    IMAGE_NAME                 Default image name
    IMAGE_TAG                  Default image tag

Examples:
    $(basename $0) build
    $(basename $0) -r ghcr.io/username -t v1.0.0 push
    $(basename $0) --registry myregistry.com all
EOF
}

# Parse arguments
NO_CACHE=""
COMMAND=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -n|--name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        build|push|all|test)
            COMMAND="$1"
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate command
if [ -z "$COMMAND" ]; then
    print_error "No command specified"
    show_help
    exit 1
fi

# Set full image name
if [ -n "$REGISTRY" ]; then
    FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}"
else
    FULL_IMAGE="${IMAGE_NAME}"
fi

# Build function
build_image() {
    print_info "Building Docker image: ${FULL_IMAGE}:${IMAGE_TAG}"

    # Check if Dockerfile exists
    if [ ! -f "Dockerfile" ]; then
        print_error "Dockerfile not found in current directory"
        exit 1
    fi

    # Build image
    docker build ${NO_CACHE} -t "${FULL_IMAGE}:${IMAGE_TAG}" .

    if [ $? -eq 0 ]; then
        print_info "Build successful!"

        # Also tag as latest if not already
        if [ "${IMAGE_TAG}" != "latest" ]; then
            docker tag "${FULL_IMAGE}:${IMAGE_TAG}" "${FULL_IMAGE}:latest"
            print_info "Also tagged as ${FULL_IMAGE}:latest"
        fi
    else
        print_error "Build failed!"
        exit 1
    fi
}

# Push function
push_image() {
    if [ -z "$REGISTRY" ]; then
        print_error "Registry not specified. Use -r option or set DOCKER_REGISTRY environment variable"
        exit 1
    fi

    print_info "Pushing image: ${FULL_IMAGE}:${IMAGE_TAG}"

    # Check if image exists locally
    if ! docker image inspect "${FULL_IMAGE}:${IMAGE_TAG}" >/dev/null 2>&1; then
        print_error "Image ${FULL_IMAGE}:${IMAGE_TAG} not found locally. Build it first."
        exit 1
    fi

    # Push image
    docker push "${FULL_IMAGE}:${IMAGE_TAG}"

    if [ $? -eq 0 ]; then
        print_info "Push successful!"

        # Also push latest tag if different
        if [ "${IMAGE_TAG}" != "latest" ]; then
            print_info "Pushing latest tag..."
            docker push "${FULL_IMAGE}:latest"
        fi
    else
        print_error "Push failed!"
        exit 1
    fi
}

# Test function
test_image() {
    print_info "Testing Docker image: ${FULL_IMAGE}:${IMAGE_TAG}"

    # Check if image exists
    if ! docker image inspect "${FULL_IMAGE}:${IMAGE_TAG}" >/dev/null 2>&1; then
        print_error "Image ${FULL_IMAGE}:${IMAGE_TAG} not found. Build it first."
        exit 1
    fi

    print_info "Running basic tests..."

    # Test 1: Check Python version
    echo -n "Testing Python version... "
    docker run --rm "${FULL_IMAGE}:${IMAGE_TAG}" python --version

    # Test 2: Check PyTorch installation
    echo -n "Testing PyTorch installation... "
    docker run --rm "${FULL_IMAGE}:${IMAGE_TAG}" python -c "import torch; print(f'PyTorch {torch.__version__}')"

    # Test 3: Check CUDA availability (if GPU is available)
    echo -n "Testing CUDA availability... "
    docker run --rm --gpus all "${FULL_IMAGE}:${IMAGE_TAG}" python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || echo "No GPU available"

    # Test 4: Check CLI help
    echo -n "Testing CLI... "
    docker run --rm "${FULL_IMAGE}:${IMAGE_TAG}" uv run python -m src.cli.train --help > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "OK"
    else
        echo "Failed"
        print_error "CLI test failed"
        exit 1
    fi

    print_info "All tests passed!"
}

# Main execution
case $COMMAND in
    build)
        build_image
        ;;
    push)
        push_image
        ;;
    all)
        build_image
        push_image
        ;;
    test)
        test_image
        ;;
    *)
        print_error "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac

print_info "Operation completed successfully!"
