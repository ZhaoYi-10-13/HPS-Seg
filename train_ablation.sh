#!/bin/bash
# Ablation study script for Hyperspherical Innovations
#
# This script runs ablation experiments to evaluate:
# 1. Baseline (original H-CLIP)
# 2. HPA only
# 3. AFR only
# 4. HPA + AFR (full)

NUM_GPUS=${1:-4}

echo "=============================================="
echo "Running Ablation Study for Hyperspherical Innovations"
echo "=============================================="

# Baseline: Original H-CLIP with OFT
echo "[1/4] Training Baseline (Original H-CLIP)..."
python train_net.py \
    --config-file configs/vitb_384_oft.yaml \
    --num-gpus $NUM_GPUS \
    OUTPUT_DIR output/ablation_baseline

# HPA Only
echo "[2/4] Training with HPA Only..."
python train_net.py \
    --config-file configs/vitb_384_hpa_only.yaml \
    --num-gpus $NUM_GPUS \
    OUTPUT_DIR output/ablation_hpa_only

# AFR Only  
echo "[3/4] Training with AFR Only..."
python train_net.py \
    --config-file configs/vitb_384_afr_only.yaml \
    --num-gpus $NUM_GPUS \
    OUTPUT_DIR output/ablation_afr_only

# Full Model (HPA + AFR)
echo "[4/4] Training Full Model (HPA + AFR)..."
python train_net.py \
    --config-file configs/vitb_384_enhanced.yaml \
    --num-gpus $NUM_GPUS \
    OUTPUT_DIR output/ablation_full

echo "=============================================="
echo "Ablation Study Complete!"
echo "Results saved in output/ablation_*"
echo "=============================================="
