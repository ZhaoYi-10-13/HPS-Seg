# Copyright (c) Facebook, Inc. and its affiliates.
from .backbone.swin import D2SwinTransformer
from .heads.cat_seg_head import CATSegHead

# Hyperspherical Innovations (HPA + AFR)
from .hyperspherical_innovations import (
    HypersphericalPrototypeBank,
    AdaptiveFeatureRectifier,
    HypersphericalInnovations,
    GeodesicCorrelation,
)