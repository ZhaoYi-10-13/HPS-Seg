#!/bin/bash
# Evaluation script for Enhanced CATSeg
#
# Usage:
#   bash eval_enhanced.sh [CONFIG] [WEIGHTS] [NUM_GPUS]

CONFIG=${1:-"configs/vitb_384_enhanced.yaml"}
WEIGHTS=${2:-"output/enhanced_catseg/model_final.pth"}
NUM_GPUS=${3:-1}

echo "=============================================="
echo "Evaluating Enhanced CATSeg"
echo "=============================================="
echo "Config: $CONFIG"
echo "Weights: $WEIGHTS"
echo "Number of GPUs: $NUM_GPUS"
echo "=============================================="

python train_net.py \
    --config-file $CONFIG \
    --eval-only \
    --num-gpus $NUM_GPUS \
    MODEL.WEIGHTS $WEIGHTS
