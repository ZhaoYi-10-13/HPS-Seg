#!/bin/bash
# Training script for Enhanced CATSeg with Hyperspherical Innovations (HPA + AFR)
# 
# Usage:
#   Single GPU:  bash train_enhanced.sh 1
#   Multi GPU:   bash train_enhanced.sh 4
#
# The enhanced model includes:
# - HPA: Hyperspherical Prototype Alignment for better class separation
# - AFR: Adaptive Feature Rectification for transfer learning

NUM_GPUS=${1:-4}
CONFIG=${2:-"configs/vitb_384_enhanced.yaml"}
OUTPUT_DIR=${3:-"output/enhanced_catseg"}

echo "=============================================="
echo "Training Enhanced CATSeg with HPA + AFR"
echo "=============================================="
echo "Number of GPUs: $NUM_GPUS"
echo "Config: $CONFIG"
echo "Output Directory: $OUTPUT_DIR"
echo "=============================================="

if [ $NUM_GPUS -eq 1 ]; then
    # Single GPU training
    python train_net.py \
        --config-file $CONFIG \
        --num-gpus 1 \
        OUTPUT_DIR $OUTPUT_DIR
else
    # Multi-GPU training with DDP
    python train_net.py \
        --config-file $CONFIG \
        --num-gpus $NUM_GPUS \
        OUTPUT_DIR $OUTPUT_DIR
fi
