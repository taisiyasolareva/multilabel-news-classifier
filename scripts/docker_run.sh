#!/bin/bash
# Run script for Docker containers

set -e

# Default values
MODE="dev"
PORT=8000
MODEL_PATH="models/best_model.pt"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Warning: Model file not found at $MODEL_PATH"
    echo "Container will start but model must be loaded via /model/reload endpoint"
fi

case $MODE in
    dev)
        echo "Starting development container..."
        docker-compose up --build
        ;;
    prod)
        echo "Starting production container..."
        docker-compose -f docker-compose.prod.yml up -d
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Available modes: dev, prod"
        exit 1
        ;;
esac
