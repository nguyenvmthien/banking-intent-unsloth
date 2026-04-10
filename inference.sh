#!/bin/bash
# Exit script if any command fails
set -e

echo "=== Entering scripts directory ==="
cd scripts

echo "=== Running Inference Example ==="
python inference.py
