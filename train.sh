#!/bin/bash
# Exit script if any command fails
set -e

echo "=== Entering scripts directory ==="
cd scripts

echo "=== 1. Starting Data Preprocessing ==="
python preprocess_data.py
echo "Data preprocessing successfully completed."
echo ""

echo "=== 2. Starting Fine-Tuning Process ==="
python train.py
echo "Fine-Tuning successfully completed."
echo ""
