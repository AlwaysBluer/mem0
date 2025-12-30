#!/bin/bash

# Lindorm Search Integration Test Runner
# Usage: ./run_lindorm_test.sh
# 
# Required environment variables:
#   LINDORM_HOST - Lindorm Search host address
#   LINDORM_PORT - Lindorm Search port (default: 30070)
#   LINDORM_USER - Lindorm Search username (optional)
#   LINDORM_PASSWORD - Lindorm Search password (optional)

# Check required environment variables
if [ -z "$LINDORM_HOST" ]; then
    echo "Error: LINDORM_HOST environment variable is required"
    echo "Usage: LINDORM_HOST=your_host LINDORM_PORT=30070 LINDORM_USER=user LINDORM_PASSWORD=pass ./run_lindorm_test.sh"
    exit 1
fi

# Set defaults
export LINDORM_PORT=${LINDORM_PORT:-30070}
export LINDORM_USER=${LINDORM_USER:-""}
export LINDORM_PASSWORD=${LINDORM_PASSWORD:-""}

# Print configuration
echo "=============================================="
echo "Lindorm Search Integration Test Configuration"
echo "=============================================="
echo "Host:       $LINDORM_HOST"
echo "Port:       $LINDORM_PORT"
echo "User:       $LINDORM_USER"
echo "Collection: test_mem0_<timestamp>"
echo "=============================================="
echo ""

# Run tests
cd "$(dirname "$0")/../.." || exit 1

echo "Running Lindorm Search integration tests..."
python -m pytest tests/vector_stores/test_lindorm_integration.py -v -s "$@"

# Alternative: Run directly with Python
# python tests/vector_stores/test_lindorm_integration.py "$@"
