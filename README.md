# HPS-Seg: Hyperspherical Parameter-efficient Segmentation

**Parameter-efficient Fine-tuning in Hyperspherical Space for Open-vocabulary Semantic Segmentation**

---

## Highlights

- **Hyperspherical Prototype Alignment (HPA)**: Class prototype management on unit hypersphere with geodesic distance
- **Adaptive Feature Rectification (AFR)**: Uncertainty-guided gating for transfer learning improvement
- **Parameter-efficient**: Less than 1M additional parameters while improving transfer performance
- **Compatible with OFT**: Designed to preserve orthogonal fine-tuning properties

## Overview

HPS-Seg extends the H-CLIP framework with two novel hyperspherical innovations designed specifically for open-vocabulary semantic segmentation:

| Innovation | Description | Key Benefit |
|------------|-------------|-------------|
| **HPA** | Maintains class prototypes on hypersphere using geodesic distance | Better class boundary separation |
| **AFR** | Channel and spatial gating with uncertainty estimation | Suppresses harmful transfer features |

## Installation

### Requirements

```bash
# Create conda environment
conda create -n hps-seg python=3.8 -y
conda activate hps-seg

# Install PyTorch (CUDA 11.3)
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# Install Detectron2
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

# Install other dependencies
pip install -r requirements.txt
```

### Dataset Preparation

Please refer to [CAT-Seg](https://github.com/cvlab-kaist/CAT-Seg) for dataset preparation:

```
datasets/
├── coco/
│   ├── coco_stuff/
│   │   ├── images/
│   │   │   ├── train2017/
│   │   │   └── val2017/
│   │   └── annotations_detectron2/
│   │       ├── train2017_stuff/
│   │       └── val2017_stuff/
├── ADEChallengeData2016/
│   ├── images/
│   └── annotations_detectron2/
├── VOCdevkit/
│   └── VOC2012/
└── pcontext/
    ├── images/
    └── annotations_detectron2/
```

## Quick Start

### Training

```bash
# Train Enhanced Model (HPA + AFR) - 4 GPUs
bash train_enhanced.sh 4

# Train with specific config
python train_net.py \
    --config-file configs/vitb_384_enhanced.yaml \
    --num-gpus 4 \
    OUTPUT_DIR output/hps_seg
```

### Evaluation

```bash
# Evaluate trained model
bash eval_enhanced.sh configs/vitb_384_enhanced.yaml output/hps_seg/model_final.pth

# Or manually
python train_net.py \
    --config-file configs/vitb_384_enhanced.yaml \
    --eval-only \
    MODEL.WEIGHTS output/hps_seg/model_final.pth
```

### Ablation Study

```bash
# Run all ablation experiments
bash train_ablation.sh 4
```

## Model Configurations

| Config | Model | HPA | AFR | Description |
|--------|-------|-----|-----|-------------|
| `vitb_384_oft.yaml` | H-CLIP | No | No | Baseline (original) |
| `vitb_384_enhanced.yaml` | HPS-Seg | Yes | Yes | Full model |
| `vitb_384_hpa_only.yaml` | HPS-Seg | Yes | No | HPA only |
| `vitb_384_afr_only.yaml` | HPS-Seg | No | Yes | AFR only |

## Configuration Options

```yaml
MODEL:
  META_ARCHITECTURE: "EnhancedCATSeg"
  HYPERSPHERICAL:
    # HPA: Hyperspherical Prototype Alignment
    USE_HPA: True              # Enable HPA
    HPA_MOMENTUM: 0.99         # Prototype momentum update
    HPA_TEMPERATURE: 0.07      # Base temperature
    
    # AFR: Adaptive Feature Rectification
    USE_AFR: True              # Enable AFR
    AFR_REDUCTION: 16          # Channel reduction ratio
    
    # Loss Configuration
    LOSS_WEIGHT: 0.1           # Innovation loss weight

SOLVER:
  BASE_LR: 0.0002
  MAX_ITER: 80000
  IMS_PER_BATCH: 4
```

## Expected Results

Results will be updated after experiments.

| Method | Backbone | A-847 | A-150 | PC-459 | PC-59 | PAS-20 |
|--------|----------|-------|-------|--------|-------|--------|
| H-CLIP | ViT-B/16 | - | - | - | - | - |
| HPS-Seg (HPA) | ViT-B/16 | - | - | - | - | - |
| HPS-Seg (AFR) | ViT-B/16 | - | - | - | - | - |
| HPS-Seg (Full) | ViT-B/16 | - | - | - | - | - |

## Technical Details

### HPA (Hyperspherical Prototype Alignment)

```python
# Core idea: Maintain class prototypes on unit hypersphere
# Use geodesic distance: d(x,y) = arccos(x . y)

# Momentum update on hypersphere
new_proto = momentum * old_proto + (1-momentum) * batch_proto
new_proto = F.normalize(new_proto, dim=-1)

# Geodesic alignment loss
cos_sim = (features * prototypes).sum(dim=-1)
geodesic_dist = torch.acos(cos_sim.clamp(-1+eps, 1-eps))
loss = geodesic_dist.mean()
```

### AFR (Adaptive Feature Rectification)

```python
# Channel gating (SE-style, lightweight)
channel_weights = channel_gate(features)  # [B, C, 1, 1]

# Spatial gating
spatial_weights = spatial_gate(features)  # [B, 1, H, W]

# Uncertainty-guided mixing
uncertainty = uncertainty_estimator(features)
mix = sigmoid(mix_weight) * (1 - uncertainty)

# Apply rectification
rectified = (1 - mix) * features + mix * (features * gate)
```

## Project Structure

```
HPS-Seg/
├── cat_seg/
│   ├── modeling/
│   │   ├── hyperspherical_innovations.py  # HPA and AFR implementations
│   │   └── ...
│   ├── enhanced_cat_seg_model.py          # Enhanced model
│   ├── config.py                          # Config with HYPERSPHERICAL
│   └── ...
├── configs/
│   ├── vitb_384_enhanced.yaml             # Full model config
│   ├── vitb_384_hpa_only.yaml             # HPA ablation
│   ├── vitb_384_afr_only.yaml             # AFR ablation
│   └── ...
├── train_enhanced.sh                       # Training script
├── train_ablation.sh                       # Ablation script
├── eval_enhanced.sh                        # Evaluation script
└── README.md
```

## Acknowledgement

This project is built upon the excellent works:
- [H-CLIP](https://github.com/SJTU-DeepVisionLab/H-CLIP) - Parameter-efficient Fine-tuning in Hyperspherical Space
- [CAT-Seg](https://github.com/cvlab-kaist/CAT-Seg) - Cost Aggregation for Open-Vocabulary Semantic Segmentation

## Citation

```bibtex
@inproceedings{peng2025parameter,
  title={Parameter-efficient Fine-tuning in Hyperspherical Space for Open-vocabulary Semantic Segmentation},
  author={Peng, Zelin and Xu, Zhengqin and Zeng, Zhilin and Huang, Yu and Wang, Yaoming and Shen, Wei},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={15009--15020},
  year={2025}
}
```

## License

This project is released under the MIT License.
