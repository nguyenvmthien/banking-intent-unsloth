#!/bin/bash
set -e
cd scripts

echo "=== Zero-Shot ==="
python inference.py ../configs/inference.yaml zero_shot

echo "=== Few-Shot ==="
python inference.py ../configs/inference.yaml few_shot

echo "=== Fine-Tuned ==="
python inference.py ../configs/inference.yaml finetuned
