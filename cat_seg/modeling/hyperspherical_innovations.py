# -*- coding: utf-8 -*-
"""
Hyperspherical Innovations for H-CLIP
=====================================

Two lightweight, effective innovations optimized for H-CLIP's OFT-based architecture:
1. HPA (Hyperspherical Prototype Alignment): Class prototype management on hypersphere
2. AFR (Adaptive Feature Rectification): Gating mechanism for harmful feature suppression

These innovations are designed to:
- Preserve H-CLIP's orthogonal fine-tuning (OFT) properties
- Work in hyperspherical space (normalized features on unit sphere)
- Be parameter-efficient (<1M additional parameters)
- Improve transfer learning performance

Author: HyperSeg Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math


class HypersphericalPrototypeBank(nn.Module):
    """
    Hyperspherical Prototype Alignment (HPA)
    
    核心思想：
    - 在超球面上维护类别原型，使用测地线距离进行对齐
    - 动量更新保持原型稳定性
    - 类别自适应温度提高区分性
    
    Key Design Choices for H-CLIP:
    1. Geodesic distance for hyperspherical alignment (more natural than Euclidean)
    2. Momentum update compatible with OFT's orthogonal constraints
    3. Class-adaptive temperature for better discrimination
    4. Integration at correlation computation stage
    """
    
    def __init__(
        self,
        feature_dim: int = 512,
        num_classes: int = 171,
        momentum: float = 0.99,
        temperature: float = 0.07,
        use_adaptive_temp: bool = True,
        prototype_init: str = 'zeros',  # 'zeros' or 'random'
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.momentum = momentum
        self.base_temperature = temperature
        self.use_adaptive_temp = use_adaptive_temp
        
        # 类别原型 (在超球面上，需要L2归一化)
        # 使用register_buffer使其不参与梯度计算但能被保存
        if prototype_init == 'zeros':
            self.register_buffer('prototypes', torch.zeros(num_classes, feature_dim))
        else:
            # 随机初始化并归一化到单位球面
            prototypes = torch.randn(num_classes, feature_dim)
            prototypes = F.normalize(prototypes, dim=-1)
            self.register_buffer('prototypes', prototypes)
        
        # 原型是否已初始化的标记
        self.register_buffer('prototype_initialized', torch.zeros(num_classes, dtype=torch.bool))
        
        # 类别自适应温度参数 (可学习)
        if use_adaptive_temp:
            # 使用log-scale确保温度始终为正
            self.log_temperatures = nn.Parameter(torch.zeros(num_classes))
        
        # 原型对齐投影 (轻量级)
        self.alignment_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 4, feature_dim),
        )
        
        # 初始化为接近恒等映射
        self._init_alignment_proj()
    
    def _init_alignment_proj(self):
        """初始化对齐投影层，使其输出接近输入"""
        for m in self.alignment_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def get_temperatures(self):
        """获取类别自适应温度"""
        if self.use_adaptive_temp:
            # 使用softplus确保温度为正且平滑
            return self.base_temperature * F.softplus(self.log_temperatures)
        else:
            return self.base_temperature * torch.ones(
                self.num_classes, device=self.prototypes.device
            )
    
    @torch.no_grad()
    def update_prototypes(self, features: torch.Tensor, labels: torch.Tensor):
        """
        动量更新类别原型
        
        Args:
            features: [N, C] 归一化的特征向量
            labels: [N] 类别标签
        """
        # 确保特征已归一化
        features = F.normalize(features, dim=-1)
        
        # 获取唯一类别
        unique_labels = labels.unique()
        
        for label in unique_labels:
            label = label.item()
            if label < 0 or label >= self.num_classes:
                continue
                
            mask = labels == label
            class_features = features[mask]
            
            if class_features.shape[0] == 0:
                continue
            
            # 计算当前批次的类别均值原型
            new_prototype = class_features.mean(dim=0)
            new_prototype = F.normalize(new_prototype, dim=0)
            
            if not self.prototype_initialized[label]:
                # 首次初始化
                self.prototypes[label] = new_prototype
                self.prototype_initialized[label] = True
            else:
                # 动量更新 (在超球面上使用球面插值)
                old_prototype = self.prototypes[label]
                
                # 简化版球面插值 (SLERP的近似)
                # 对于接近的向量，线性插值后重新归一化效果良好
                updated = self.momentum * old_prototype + (1 - self.momentum) * new_prototype
                self.prototypes[label] = F.normalize(updated, dim=0)
    
    def geodesic_similarity(self, features: torch.Tensor, prototypes: torch.Tensor = None):
        """
        计算测地线相似度 (基于球面距离)
        
        测地线距离: d(x, y) = arccos(x · y)
        相似度: 1 - d(x, y) / π = 1 - arccos(x · y) / π
        
        Args:
            features: [B, N, C] 或 [B, C] 归一化特征
            prototypes: [K, C] 类别原型，如果为None则使用存储的原型
            
        Returns:
            similarity: [B, N, K] 或 [B, K] 测地线相似度
        """
        if prototypes is None:
            prototypes = self.prototypes
        
        # 确保归一化
        features = F.normalize(features, dim=-1)
        prototypes = F.normalize(prototypes, dim=-1)
        
        # 余弦相似度
        if features.dim() == 3:
            # [B, N, C] x [K, C]^T -> [B, N, K]
            cos_sim = torch.einsum('bnc,kc->bnk', features, prototypes)
        else:
            # [B, C] x [K, C]^T -> [B, K]
            cos_sim = torch.einsum('bc,kc->bk', features, prototypes)
        
        # 裁剪避免数值不稳定
        cos_sim = torch.clamp(cos_sim, -1 + 1e-7, 1 - 1e-7)
        
        # 测地线相似度
        geodesic_dist = torch.acos(cos_sim)
        geodesic_sim = 1.0 - geodesic_dist / math.pi
        
        return geodesic_sim
    
    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        targets: torch.Tensor = None,
    ):
        """
        前向传播
        
        Args:
            image_features: [B, C, H, W] 图像特征
            text_features: [B, T, P, C] 文本特征 (T=类别数, P=prompts数)
            targets: [B, H, W] 分割标签 (训练时使用)
            
        Returns:
            enhanced_features: 增强后的图像特征
            hpa_loss: HPA对齐损失 (训练时)
        """
        B, C, H, W = image_features.shape
        
        # 归一化图像特征
        img_feat_norm = F.normalize(image_features, dim=1)
        
        # 对齐投影
        img_feat_flat = rearrange(img_feat_norm, 'b c h w -> (b h w) c')
        aligned_feat = self.alignment_proj(img_feat_flat)
        aligned_feat = F.normalize(aligned_feat, dim=-1)
        aligned_feat = rearrange(aligned_feat, '(b h w) c -> b c h w', b=B, h=H, w=W)
        
        # 残差连接
        enhanced_features = img_feat_norm + 0.1 * aligned_feat
        enhanced_features = F.normalize(enhanced_features, dim=1)
        
        # 计算HPA损失 (训练时)
        hpa_loss = None
        if targets is not None and self.training:
            # 下采样targets到特征分辨率
            targets_downsampled = F.interpolate(
                targets.unsqueeze(1).float(),
                size=(H, W),
                mode='nearest'
            ).squeeze(1).long()
            
            # 展平特征和标签
            features_flat = rearrange(enhanced_features, 'b c h w -> (b h w) c')
            labels_flat = targets_downsampled.flatten()
            
            # 过滤有效标签
            valid_mask = (labels_flat >= 0) & (labels_flat < self.num_classes)
            if valid_mask.sum() > 0:
                valid_features = features_flat[valid_mask]
                valid_labels = labels_flat[valid_mask]
                
                # 更新原型
                self.update_prototypes(valid_features.detach(), valid_labels)
                
                # 计算原型对齐损失
                if self.prototype_initialized.any():
                    initialized_mask = self.prototype_initialized[valid_labels]
                    if initialized_mask.sum() > 0:
                        aligned_features = valid_features[initialized_mask]
                        aligned_labels = valid_labels[initialized_mask]
                        
                        # 获取对应的原型
                        target_prototypes = self.prototypes[aligned_labels]
                        
                        # 测地线距离损失
                        cos_sim = (aligned_features * target_prototypes).sum(dim=-1)
                        cos_sim = torch.clamp(cos_sim, -1 + 1e-7, 1 - 1e-7)
                        geodesic_dist = torch.acos(cos_sim)
                        
                        hpa_loss = geodesic_dist.mean()
        
        return enhanced_features, hpa_loss


class AdaptiveFeatureRectifier(nn.Module):
    """
    Adaptive Feature Rectification (AFR)
    
    核心思想：
    - 学习识别并抑制对迁移学习有害的特征通道
    - 使用不确定性估计指导门控
    - 空间自适应门控关注关键区域
    
    Key Design Choices for H-CLIP:
    1. Channel gating: Suppress task-irrelevant channels from pre-trained CLIP
    2. Spatial gating: Focus on discriminative regions
    3. Uncertainty-guided: Use correlation entropy for adaptive control
    4. Preserve orthogonality: Gating is multiplicative, preserving OFT properties
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        num_classes: int = 171,
        reduction: int = 16,
        use_spatial_gate: bool = True,
        gate_activation: str = 'sigmoid',  # 'sigmoid' or 'softmax'
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.use_spatial_gate = use_spatial_gate
        self.gate_activation = gate_activation
        
        # 通道门控网络 (SE-style但更轻量)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, feature_dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // reduction, feature_dim),
            nn.Sigmoid()
        )
        
        # 空间门控网络
        if use_spatial_gate:
            self.spatial_gate = nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim // reduction, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature_dim // reduction, 1, 1),
                nn.Sigmoid()
            )
        
        # 不确定性估计器 (基于特征统计)
        self.uncertainty_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, feature_dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // reduction, 1),
            nn.Sigmoid()
        )
        
        # 自适应混合权重
        self.mix_weight = nn.Parameter(torch.zeros(1))
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重，使门控初始值接近1（保持原始特征）"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    # 偏置初始化使sigmoid输出接近1
                    nn.init.constant_(m.bias, 2.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 2.0)
    
    def compute_uncertainty(self, features: torch.Tensor):
        """
        基于特征统计估计不确定性
        
        Args:
            features: [B, C, H, W] 特征图
            
        Returns:
            uncertainty: [B, 1] 不确定性分数 (0-1)
        """
        return self.uncertainty_estimator(features)
    
    def forward(
        self,
        features: torch.Tensor,
        correlation: torch.Tensor = None,
    ):
        """
        前向传播
        
        Args:
            features: [B, C, H, W] 图像特征
            correlation: [B, P, T, H, W] 相关图 (可选，用于不确定性估计)
            
        Returns:
            rectified_features: [B, C, H, W] 校正后的特征
            afr_loss: AFR正则化损失
        """
        B, C, H, W = features.shape
        
        # 计算不确定性
        uncertainty = self.compute_uncertainty(features)  # [B, 1]
        
        # 如果提供了相关图，也从中估计不确定性
        if correlation is not None:
            # 相关图的熵作为额外的不确定性信号
            corr_flat = correlation.flatten(start_dim=1)
            corr_probs = F.softmax(corr_flat, dim=1)
            corr_entropy = -(corr_probs * (corr_probs + 1e-10).log()).sum(dim=1, keepdim=True)
            corr_entropy = corr_entropy / math.log(corr_flat.shape[1])  # 归一化
            
            # 融合两种不确定性
            uncertainty = 0.5 * uncertainty + 0.5 * corr_entropy
        
        # 通道门控
        channel_weights = self.channel_gate(features)  # [B, C]
        channel_weights = channel_weights.view(B, C, 1, 1)
        
        # 空间门控
        if self.use_spatial_gate:
            spatial_weights = self.spatial_gate(features)  # [B, 1, H, W]
        else:
            spatial_weights = torch.ones(B, 1, H, W, device=features.device)
        
        # 组合门控
        gate = channel_weights * spatial_weights  # [B, C, H, W]
        
        # 不确定性自适应：高不确定性时减少门控强度（更保守）
        # mix_weight控制原始特征和门控特征的混合
        mix = torch.sigmoid(self.mix_weight) * (1 - uncertainty.view(B, 1, 1, 1))
        
        # 应用门控
        gated_features = features * gate
        
        # 残差混合
        rectified_features = (1 - mix) * features + mix * gated_features
        
        # AFR正则化损失：鼓励稀疏但不极端的门控
        afr_loss = None
        if self.training:
            # 门控稀疏性损失
            gate_sparsity = torch.abs(gate - 0.5).mean()
            
            # 门控平滑性损失 (空间上相邻的门控值应该相似)
            if self.use_spatial_gate:
                gate_smooth = (
                    torch.abs(spatial_weights[:, :, 1:, :] - spatial_weights[:, :, :-1, :]).mean() +
                    torch.abs(spatial_weights[:, :, :, 1:] - spatial_weights[:, :, :, :-1]).mean()
                )
            else:
                gate_smooth = torch.tensor(0.0, device=features.device)
            
            afr_loss = 0.1 * gate_sparsity + 0.01 * gate_smooth
        
        return rectified_features, afr_loss


class HypersphericalInnovations(nn.Module):
    """
    统一的超球面创新模块
    
    整合HPA和AFR，提供简洁的接口
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        embed_dim: int = 512,
        num_classes: int = 171,
        use_hpa: bool = True,
        use_afr: bool = True,
        hpa_momentum: float = 0.99,
        hpa_temperature: float = 0.07,
        afr_reduction: int = 16,
    ):
        super().__init__()
        self.use_hpa = use_hpa
        self.use_afr = use_afr
        
        if use_hpa:
            self.hpa = HypersphericalPrototypeBank(
                feature_dim=embed_dim,
                num_classes=num_classes,
                momentum=hpa_momentum,
                temperature=hpa_temperature,
            )
        
        if use_afr:
            self.afr = AdaptiveFeatureRectifier(
                feature_dim=feature_dim,
                num_classes=num_classes,
                reduction=afr_reduction,
            )
    
    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor = None,
        correlation: torch.Tensor = None,
        targets: torch.Tensor = None,
    ):
        """
        前向传播
        
        Args:
            image_features: [B, C, H, W] 图像特征
            text_features: [B, T, P, C] 文本特征
            correlation: [B, P, T, H, W] 相关图
            targets: [B, H', W'] 分割标签
            
        Returns:
            enhanced_features: 增强后的特征
            total_loss: 总辅助损失
        """
        total_loss = 0.0
        enhanced_features = image_features
        
        # AFR: 特征校正 (在计算相关图之前)
        if self.use_afr:
            enhanced_features, afr_loss = self.afr(enhanced_features, correlation)
            if afr_loss is not None:
                total_loss = total_loss + afr_loss
        
        # HPA: 原型对齐 (在相关图计算阶段)
        if self.use_hpa and text_features is not None:
            # 对于HPA，我们需要embed_dim维度的特征
            # 这里假设image_features已经是正确的维度，或者需要投影
            _, hpa_loss = self.hpa(
                enhanced_features,
                text_features,
                targets,
            )
            if hpa_loss is not None:
                total_loss = total_loss + 0.1 * hpa_loss
        
        return enhanced_features, total_loss


class GeodesicCorrelation(nn.Module):
    """
    测地线相关图计算
    
    替代标准的余弦相似度相关图，使用测地线距离
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ):
        """
        计算测地线相关图
        
        Args:
            image_features: [B, C, H, W] 归一化的图像特征
            text_features: [B, T, P, C] 归一化的文本特征
            
        Returns:
            correlation: [B, P, T, H, W] 测地线相关图
        """
        # 归一化
        img_feat = F.normalize(image_features, dim=1)  # B C H W
        text_feat = F.normalize(text_features, dim=-1)  # B T P C
        
        # 计算余弦相似度
        cos_sim = torch.einsum('bchw, btpc -> bpthw', img_feat, text_feat)
        
        # 转换为测地线相似度
        cos_sim = torch.clamp(cos_sim, -1 + 1e-7, 1 - 1e-7)
        geodesic_dist = torch.acos(cos_sim)
        geodesic_sim = 1.0 - geodesic_dist / math.pi
        
        # 温度缩放
        correlation = geodesic_sim / self.temperature
        
        return correlation
