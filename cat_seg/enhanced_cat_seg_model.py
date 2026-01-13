# -*- coding: utf-8 -*-
"""
Enhanced CATSeg Model with Hyperspherical Innovations
=====================================================

This model extends H-CLIP's CATSeg with two key innovations:
1. HPA (Hyperspherical Prototype Alignment): Class prototype management on hypersphere
2. AFR (Adaptive Feature Rectification): Gating mechanism for transfer learning

The innovations are designed to improve transfer learning performance while 
preserving H-CLIP's orthogonal fine-tuning (OFT) properties.

Author: HyperSeg Team
"""

from typing import Tuple
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from einops import rearrange

from .modeling.hyperspherical_innovations import (
    HypersphericalPrototypeBank,
    AdaptiveFeatureRectifier,
    GeodesicCorrelation,
)


@META_ARCH_REGISTRY.register()
class EnhancedCATSeg(nn.Module):
    """
    Enhanced CATSeg with Hyperspherical Innovations (HPA + AFR)
    
    This model builds upon H-CLIP's CATSeg and adds:
    - HPA: Maintains class prototypes on hypersphere for better alignment
    - AFR: Adaptively rectifies features to suppress harmful transfer
    
    All innovations are designed to:
    - Preserve OFT's orthogonal properties
    - Work in hyperspherical space
    - Be parameter-efficient
    """
    
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        size_divisibility: int,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        clip_pixel_mean: Tuple[float],
        clip_pixel_std: Tuple[float],
        train_class_json: str,
        test_class_json: str,
        sliding_window: bool,
        clip_finetune: str,
        backbone_multiplier: float,
        clip_pretrained: str,
        # Hyperspherical Innovation configs
        use_hpa: bool = True,
        use_afr: bool = True,
        hpa_momentum: float = 0.99,
        hpa_temperature: float = 0.07,
        afr_reduction: int = 16,
        innovation_loss_weight: float = 0.1,
    ):
        """
        Args:
            sem_seg_head: a module that predicts semantic segmentation
            use_hpa: whether to use Hyperspherical Prototype Alignment
            use_afr: whether to use Adaptive Feature Rectification
            hpa_momentum: momentum for prototype update
            hpa_temperature: base temperature for HPA
            afr_reduction: reduction ratio for AFR
            innovation_loss_weight: weight for auxiliary losses from innovations
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        if size_divisibility < 0:
            size_divisibility = self.backbone.size_divisibility if backbone else 32
        self.size_divisibility = size_divisibility

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_mean", torch.Tensor(clip_pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_std", torch.Tensor(clip_pixel_std).view(-1, 1, 1), False)
        
        self.train_class_json = train_class_json
        self.test_class_json = test_class_json

        self.clip_finetune = clip_finetune
        self._setup_clip_finetuning()
        
        self.sliding_window = sliding_window
        self.clip_resolution = (384, 384) if clip_pretrained == "ViT-B/16" else (336, 336)

        self.proj_dim = 768 if clip_pretrained == "ViT-B/16" else 1024
        self.embed_dim = 512 if clip_pretrained == "ViT-B/16" else 768
        
        # Upsampling layers for multi-scale features
        self.upsample1 = nn.ConvTranspose2d(self.proj_dim, 256, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(self.proj_dim, 128, kernel_size=4, stride=4)

        self.layer_indexes = [3, 7] if clip_pretrained == "ViT-B/16" else [7, 15] 
        self.layers = []
        for l in self.layer_indexes:
            self.sem_seg_head.predictor.clip_model.visual.transformer.resblocks[l].register_forward_hook(
                lambda m, _, o: self.layers.append(o)
            )

        # ========== Hyperspherical Innovations ==========
        self.use_hpa = use_hpa
        self.use_afr = use_afr
        self.innovation_loss_weight = innovation_loss_weight
        
        # 获取类别数量
        import json
        with open(train_class_json, 'r') as f:
            num_classes = len(json.load(f))
        
        # HPA: Hyperspherical Prototype Alignment
        if use_hpa:
            self.hpa = HypersphericalPrototypeBank(
                feature_dim=self.embed_dim,
                num_classes=num_classes,
                momentum=hpa_momentum,
                temperature=hpa_temperature,
            )
        
        # AFR: Adaptive Feature Rectification
        if use_afr:
            self.afr = AdaptiveFeatureRectifier(
                # AFR operates on the encoded image tokens (res3), which use CLIP's embedding dim
                feature_dim=self.embed_dim,
                num_classes=num_classes,
                reduction=afr_reduction,
            )
        
        # Optional: Geodesic correlation (can replace standard correlation)
        self.use_geodesic_corr = False  # Set to True to use geodesic correlation
        if self.use_geodesic_corr:
            self.geodesic_corr = GeodesicCorrelation(temperature=hpa_temperature)
    
    def _setup_clip_finetuning(self):
        """Setup CLIP fine-tuning based on configuration"""
        for name, params in self.sem_seg_head.predictor.clip_model.named_parameters():
            if "transformer" in name:
                if self.clip_finetune == "prompt":
                    params.requires_grad = True if "prompt" in name else False
                elif self.clip_finetune == "attention":
                    if "attn" in name:
                        params.requires_grad = True if "q_proj" in name or "v_proj" in name else False
                    elif "position" in name:
                        params.requires_grad = True
                    else:
                        params.requires_grad = False
                elif self.clip_finetune == "oft":
                    if "visual" in name:
                        if "attn" in name or "position" in name:
                            params.requires_grad = True
                        else:
                            params.requires_grad = False
                    elif "oft" in name:
                        params.requires_grad = True
                    elif "position" in name:
                        params.requires_grad = True
                    else:
                        params.requires_grad = False
                elif self.clip_finetune == "oft_qv":
                    if "visual" in name:
                        if "attn" in name:
                            params.requires_grad = True if "q_proj" in name or "v_proj" in name else False
                        elif "position" in name:
                            params.requires_grad = True
                        else:
                            params.requires_grad = False
                    elif "oft" in name:
                        params.requires_grad = True
                    elif "position" in name:
                        params.requires_grad = True
                    else:
                        params.requires_grad = False
                elif self.clip_finetune == "oft_both":
                    if "oft" in name:
                        params.requires_grad = True
                    elif "position" in name:
                        params.requires_grad = True
                    else:
                        params.requires_grad = False
                elif self.clip_finetune == "full":
                    params.requires_grad = True
                else:
                    params.requires_grad = False
            else:
                params.requires_grad = False

    @classmethod
    def from_config(cls, cfg):
        backbone = None
        sem_seg_head = build_sem_seg_head(cfg, None)
        
        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "clip_pixel_mean": cfg.MODEL.CLIP_PIXEL_MEAN,
            "clip_pixel_std": cfg.MODEL.CLIP_PIXEL_STD,
            "train_class_json": cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON,
            "test_class_json": cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON,
            "sliding_window": cfg.TEST.SLIDING_WINDOW,
            "clip_finetune": cfg.MODEL.SEM_SEG_HEAD.CLIP_FINETUNE,
            "backbone_multiplier": cfg.SOLVER.BACKBONE_MULTIPLIER,
            "clip_pretrained": cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED,
            # Hyperspherical Innovations
            "use_hpa": cfg.MODEL.HYPERSPHERICAL.USE_HPA,
            "use_afr": cfg.MODEL.HYPERSPHERICAL.USE_AFR,
            "hpa_momentum": cfg.MODEL.HYPERSPHERICAL.HPA_MOMENTUM,
            "hpa_temperature": cfg.MODEL.HYPERSPHERICAL.HPA_TEMPERATURE,
            "afr_reduction": cfg.MODEL.HYPERSPHERICAL.AFR_REDUCTION,
            "innovation_loss_weight": cfg.MODEL.HYPERSPHERICAL.LOSS_WEIGHT,
        }

    @property
    def device(self):
        return self.pixel_mean.device
    
    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
        Returns:
            list[dict]: each dict has the results for one image.
        """
        
        images = [x["image"].to(self.device) for x in batched_inputs]
        if not self.training and self.sliding_window:
            return self.inference_sliding_window(batched_inputs)

        clip_images = [(x - self.clip_pixel_mean) / self.clip_pixel_std for x in images]
        clip_images = ImageList.from_tensors(clip_images, self.size_divisibility)

        self.layers = []

        clip_images_resized = F.interpolate(
            clip_images.tensor, 
            size=self.clip_resolution, 
            mode='bilinear', 
            align_corners=False
        )
        clip_features = self.sem_seg_head.predictor.clip_model.encode_image(
            clip_images_resized, dense=True
        )

        image_features = clip_features[:, 1:, :]

        # CLIP ViT features for guidance
        res3 = rearrange(image_features, "B (H W) C -> B C H W", H=24)
        res4 = rearrange(self.layers[0][1:, :, :], "(H W) B C -> B C H W", H=24)
        res5 = rearrange(self.layers[1][1:, :, :], "(H W) B C -> B C H W", H=24)
        
        # ========== AFR: Apply before upsampling ==========
        innovation_loss = 0.0
        
        if self.use_afr:
            res3, afr_loss = self.afr(res3)
            if afr_loss is not None and self.training:
                innovation_loss = innovation_loss + afr_loss
        
        res4 = self.upsample1(res4)
        res5 = self.upsample2(res5)
        features = {'res5': res5, 'res4': res4, 'res3': res3}

        outputs = self.sem_seg_head(clip_features, features)
        
        if self.training:
            targets = torch.stack([x["sem_seg"].to(self.device) for x in batched_inputs], dim=0)
            outputs = F.interpolate(
                outputs, 
                size=(targets.shape[-2], targets.shape[-1]), 
                mode="bilinear", 
                align_corners=False
            )
            
            # ========== HPA: Update prototypes and compute alignment loss ==========
            if self.use_hpa:
                # Project features to embed_dim for HPA
                res3_hpa = F.normalize(res3, dim=1)
                res3_hpa = F.adaptive_avg_pool2d(res3_hpa, (24, 24))
                
                # 使用projection将特征维度转换为embed_dim
                if res3_hpa.shape[1] != self.embed_dim:
                    res3_hpa = F.interpolate(
                        res3_hpa.permute(0, 2, 3, 1),  # B, H, W, C
                        size=(self.embed_dim,),
                        mode='linear',
                        align_corners=False
                    ).permute(0, 3, 1, 2)  # B, C, H, W
                
                _, hpa_loss = self.hpa(
                    res3_hpa,
                    None,  # text_features not needed for loss computation
                    targets
                )
                if hpa_loss is not None:
                    innovation_loss = innovation_loss + 0.1 * hpa_loss
            
            num_classes = outputs.shape[1]
            mask = targets != self.sem_seg_head.ignore_value

            outputs = outputs.permute(0, 2, 3, 1)
            _targets = torch.zeros(outputs.shape, device=self.device)
            _onehot = F.one_hot(targets[mask], num_classes=num_classes).float()
            _targets[mask] = _onehot
            
            # Main segmentation loss
            loss = F.binary_cross_entropy_with_logits(outputs, _targets)
            
            # Add innovation losses
            total_loss = loss + self.innovation_loss_weight * innovation_loss
            
            losses = {"loss_sem_seg": total_loss}
            
            # Log individual losses for debugging
            if innovation_loss > 0:
                losses["loss_innovation"] = self.innovation_loss_weight * innovation_loss
            
            return losses

        else:
            outputs = outputs.sigmoid()
            image_size = clip_images.image_sizes[0]
            height = batched_inputs[0].get("height", image_size[0])
            width = batched_inputs[0].get("width", image_size[1])

            output = sem_seg_postprocess(outputs[0], image_size, height, width)
            processed_results = [{'sem_seg': output}]
            return processed_results

    @torch.no_grad()
    def inference_sliding_window(self, batched_inputs, kernel=384, overlap=0.333, out_res=[640, 640]):
        """Sliding window inference for high-resolution images"""
        images = [x["image"].to(self.device, dtype=torch.float32) for x in batched_inputs]
        stride = int(kernel * (1 - overlap))
        unfold = nn.Unfold(kernel_size=kernel, stride=stride)
        fold = nn.Fold(out_res, kernel_size=kernel, stride=stride)

        # 保持4D输入给unfold，然后通过unsqueeze(0)添加batch维度给rearrange
        image = F.interpolate(images[0].unsqueeze(0), size=out_res, mode='bilinear', align_corners=False)
        image_unfolded = unfold(image)  # [B, C*kernel*kernel, L] 其中B=1
        # rearrange需要去掉B维度，将[1, C*H*W, L]变成[L, C, H, W]
        image = rearrange(image_unfolded.squeeze(0), "(C H W) L-> L C H W", C=3, H=kernel)
        global_image = F.interpolate(
            images[0].unsqueeze(0), size=(kernel, kernel), mode='bilinear', align_corners=False
        )
        image = torch.cat((image, global_image), dim=0)

        images = (image - self.pixel_mean) / self.pixel_std
        clip_images = (image - self.clip_pixel_mean) / self.clip_pixel_std
        clip_images = F.interpolate(
            clip_images, size=self.clip_resolution, mode='bilinear', align_corners=False
        )
        
        self.layers = []
        clip_features = self.sem_seg_head.predictor.clip_model.encode_image(clip_images, dense=True)
        res3 = rearrange(clip_features[:, 1:, :], "B (H W) C -> B C H W", H=24)
        res4 = self.upsample1(rearrange(self.layers[0][1:, :, :], "(H W) B C -> B C H W", H=24))
        res5 = self.upsample2(rearrange(self.layers[1][1:, :, :], "(H W) B C -> B C H W", H=24))

        # Apply AFR during inference as well
        if self.use_afr:
            res3, _ = self.afr(res3)

        features = {'res5': res5, 'res4': res4, 'res3': res3}
        outputs = self.sem_seg_head(clip_features, features)

        outputs = F.interpolate(outputs, size=kernel, mode="bilinear", align_corners=False)
        outputs = outputs.sigmoid()
        
        global_output = outputs[-1:]
        global_output = F.interpolate(
            global_output, size=out_res, mode='bilinear', align_corners=False
        )
        outputs = outputs[:-1]
        # fold需要3D输入[B, C*kernel*kernel, L]，outputs.flatten(1)是[L, C*kernel*kernel]，需要转置并添加batch维度
        outputs_flat = outputs.flatten(1)  # [L, C*kernel*kernel]
        outputs_flat = outputs_flat.T.unsqueeze(0)  # [1, C*kernel*kernel, L]
        # 修复：torch.ones需要4D输入[B, C, H, W]，不能是3D
        outputs = fold(outputs_flat) / fold(unfold(torch.ones([1, 1] + out_res, device=self.device)))
        outputs = (outputs + global_output) / 2.

        height = batched_inputs[0].get("height", out_res[0])
        width = batched_inputs[0].get("width", out_res[1])
        output = sem_seg_postprocess(outputs[0], out_res, height, width)
        return [{'sem_seg': output}]
