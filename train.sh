#!/bin/bash
set -e
cd scripts
python preprocess_data.py
python train.py
