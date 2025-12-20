#!/bin/bash
# Build script for Docker images

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building Docker images...${NC}"

# Build production image
echo -e "${YELLOW}Building production image...${NC}"
docker build \
    --target production \
    -t news-classifier-api:latest \
    -t news-classifier-api:production \
    -f Dockerfile \
    .

# Build development image
echo -e "${YELLOW}Building development image...${NC}"
docker build \
    -t news-classifier-api:dev \
    -f Dockerfile.dev \
    .

# Build training image
echo -e "${YELLOW}Building training image...${NC}"
docker build \
    --target training \
    -t news-classifier-api:training \
    -f Dockerfile \
    .

echo -e "${GREEN}All images built successfully!${NC}"
echo ""
echo "Available images:"
echo "  - news-classifier-api:latest (production)"
echo "  - news-classifier-api:dev (development)"
echo "  - news-classifier-api:training (training)"
