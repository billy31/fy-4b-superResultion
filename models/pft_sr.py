# -*- coding: utf-8 -*-
"""
PFT-SR: Progressive Feature Transfer for Super-Resolution
基于 https://github.com/CVL-UESTC/PFT-SR 的模型实现

主要组件:
1. Shallow Feature Extraction (浅层特征提取)
2. Progressive Feature Transfer Blocks (渐进特征转移块)
3. High-Resolution Reconstruction (高分辨率重建)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    残差块: 用于特征提取的基本单元
    """
    def __init__(self, channels, kernel_size=3, bias=True):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, 
                               padding=kernel_size//2, bias=bias)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size,
                               padding=kernel_size//2, bias=bias)
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out + identity  # 残差连接
        return out


class ChannelAttention(nn.Module):
    """
    通道注意力模块 (Channel Attention)
    用于特征通道的自适应加权
    """
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out


class SpatialAttention(nn.Module):
    """
    空间注意力模块 (Spatial Attention)
    用于特征空间的自适应加权
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(concat))
        return x * out


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module
    结合通道注意力和空间注意力
    """
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()
        
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ProgressiveFeatureTransferBlock(nn.Module):
    """
    渐进特征转移块 (PFT Block)
    核心模块: 渐进地学习从低分辨率到高分辨率的特征映射
    """
    def __init__(self, channels, upscale_factor=2, num_rb=3, use_attention=True):
        super(ProgressiveFeatureTransferBlock, self).__init__()
        
        self.channels = channels
        self.upscale_factor = upscale_factor
        self.use_attention = use_attention
        
        # 残差块序列 (用于深度特征提取)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(channels) for _ in range(num_rb)
        ])
        
        # 注意力模块
        if use_attention:
            self.attention = CBAM(channels)
        
        # 上采样层 (PixelShuffle)
        if upscale_factor == 2:
            self.upsample = nn.Sequential(
                nn.Conv2d(channels, channels * 4, 3, padding=1),
                nn.PixelShuffle(2)
            )
        elif upscale_factor == 4:
            self.upsample = nn.Sequential(
                nn.Conv2d(channels, channels * 16, 3, padding=1),
                nn.PixelShuffle(4)
            )
        else:
            # upscale_factor == 1, 不进行空间上采样
            self.upsample = nn.Identity()
        
        # 特征融合层
        self.fusion = nn.Conv2d(channels, channels, 3, padding=1)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征 [B, C, H, W]
        
        Returns:
            upsampled: 上采样后的特征 [B, C, H*upscale, W*upscale]
            residual: 残差特征用于后续处理
        """
        identity = x
        
        # 通过残差块序列
        feat = x
        for rb in self.residual_blocks:
            feat = rb(feat)
        
        # 应用注意力
        if self.use_attention:
            feat = self.attention(feat)
        
        # 残差连接
        feat = feat + identity
        
        # 上采样
        upsampled = self.upsample(feat)
        
        # 特征融合
        upsampled = self.fusion(upsampled)
        
        return upsampled, feat


class PFTSR(nn.Module):
    """
    PFT-SR: Progressive Feature Transfer Network for Super-Resolution
    
    适用于FY-4B卫星数据的超分辨率任务
    支持 2km->1km 和 4km->2km 的上采样
    """
    
    def __init__(
        self,
        in_channels=8,      # 输入通道数 (FY-4B中红外通道数)
        out_channels=8,     # 输出通道数
        num_features=64,    # 特征维度
        num_pft_blocks=3,   # PFT块数量
        num_rb_per_block=3, # 每个PFT块中的残差块数量
        upscale_factor=2,   # 上采样因子 (2或4)
        use_attention=True  # 是否使用注意力机制
    ):
        super(PFTSR, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_features = num_features
        self.num_pft_blocks = num_pft_blocks
        self.upscale_factor = upscale_factor
        
        # ===== 1. 浅层特征提取 =====
        self.shallow_feat = nn.Sequential(
            nn.Conv2d(in_channels, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, 3, padding=1)
        )
        
        # ===== 2. 渐进特征转移块序列 =====
        self.pft_blocks = nn.ModuleList([
            ProgressiveFeatureTransferBlock(
                num_features, 
                upscale_factor=upscale_factor if i == num_pft_blocks - 1 else 1,
                num_rb=num_rb_per_block,
                use_attention=use_attention
            )
            for i in range(num_pft_blocks)
        ])
        
        # ===== 3. 跨层特征融合 =====
        self.cross_layer_fusion = nn.ModuleList([
            nn.Conv2d(num_features * 2, num_features, 3, padding=1)
            for _ in range(num_pft_blocks - 1)
        ])
        
        # ===== 4. 全局残差学习 =====
        self.global_residual = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, out_channels, 3, padding=1)
        )
        
        # ===== 5. 初始上采样 (如果使用渐进式上采样) =====
        if upscale_factor == 4:
            # 4倍上采样分两次2倍进行
            self.initial_upsample = nn.Sequential(
                nn.Conv2d(num_features, num_features * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True)
            )
        else:
            self.initial_upsample = None
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 低分辨率输入 [B, C, H, W]
        
        Returns:
            sr_img: 超分辨率输出 [B, C, H*upscale, W*upscale]
        """
        # 1. 浅层特征提取
        shallow_feat = self.shallow_feat(x)
        
        # 2. 渐进特征转移
        feats = []
        feat = shallow_feat
        
        for i, pft_block in enumerate(self.pft_blocks):
            upsampled_feat, residual_feat = pft_block(feat)
            feats.append(upsampled_feat)
            
            # 如果不是最后一个块，更新特征用于下一个块
            if i < self.num_pft_blocks - 1:
                # 跨层特征融合
                if i > 0:
                    concat_feat = torch.cat([upsampled_feat, feats[i-1]], dim=1)
                    feat = self.cross_layer_fusion[i](concat_feat)
                else:
                    feat = upsampled_feat
        
        # 3. 融合所有PFT块的输出
        if len(feats) > 1:
            # 使用最后的特征作为主要输出
            deep_feat = feats[-1]
            # 可以添加其他融合策略
            for f in feats[:-1]:
                if f.shape == deep_feat.shape:
                    deep_feat = deep_feat + f
        else:
            deep_feat = feats[0]
        
        # 4. 全局残差学习
        sr_img = self.global_residual(deep_feat)
        
        # 5. 全局残差连接 (使用双三次插值上采样的输入)
        base = F.interpolate(x, scale_factor=self.upscale_factor, 
                            mode='bilinear', align_corners=False)
        sr_img = sr_img + base
        
        return sr_img


class PFTSR_MultiScale(nn.Module):
    """
    多尺度PFT-SR模型
    同时支持2km->1km和4km->2km的超分辨率
    """
    
    def __init__(
        self,
        in_channels=8,
        out_channels=8,
        num_features=64,
        num_pft_blocks=3,
        num_rb_per_block=3
    ):
        super(PFTSR_MultiScale, self).__init__()
        
        # 共享的浅层特征提取
        self.shallow_feat = nn.Sequential(
            nn.Conv2d(in_channels, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, 3, padding=1)
        )
        
        # 2倍上采样分支 (4km->2km)
        self.scale2_branch = nn.ModuleList([
            ProgressiveFeatureTransferBlock(
                num_features, 
                upscale_factor=2,
                num_rb=num_rb_per_block
            )
            for _ in range(num_pft_blocks)
        ])
        
        # 4倍上采样分支 (2km->1km 或 4km->1km)
        self.scale4_branch = nn.ModuleList([
            ProgressiveFeatureTransferBlock(
                num_features,
                upscale_factor=2,  # 每次2倍
                num_rb=num_rb_per_block
            )
            for _ in range(num_pft_blocks * 2)  # 需要更多块来达到4倍
        ])
        
        # 输出层
        self.head2 = nn.Conv2d(num_features, out_channels, 3, padding=1)
        self.head4 = nn.Conv2d(num_features, out_channels, 3, padding=1)
    
    def forward(self, x, scale=2):
        """
        前向传播
        
        Args:
            x: 输入
            scale: 上采样因子 (2 或 4)
        """
        feat = self.shallow_feat(x)
        
        if scale == 2:
            for block in self.scale2_branch:
                feat, _ = block(feat)
            out = self.head2(feat)
            # 残差连接
            base = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            out = out + base
        else:
            for block in self.scale4_branch:
                feat, _ = block(feat)
            out = self.head4(feat)
            # 残差连接
            base = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
            out = out + base
        
        return out


def test_model():
    """测试模型"""
    print("=" * 60)
    print("测试PFT-SR模型")
    print("=" * 60)
    
    # 测试2倍上采样 (4km->2km)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    model = PFTSR(
        in_channels=8,
        out_channels=8,
        num_features=64,
        num_pft_blocks=3,
        upscale_factor=2
    ).to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # 测试前向传播
    x = torch.randn(2, 8, 64, 64).to(device)  # [B, C, H, W]
    print(f"\n输入形状: {x.shape}")
    
    with torch.no_grad():
        y = model(x)
    print(f"输出形状: {y.shape}")
    
    # 验证输出尺寸
    expected_h = x.shape[2] * 2
    expected_w = x.shape[3] * 2
    assert y.shape[2] == expected_h and y.shape[3] == expected_w, \
        f"输出尺寸错误: 期望 [{expected_h}, {expected_w}], 实际 [{y.shape[2]}, {y.shape[3]}]"
    
    print("\n✓ 2倍上采样测试通过!")
    
    # 测试4倍上采样 (2km->1km)
    model4x = PFTSR(
        in_channels=8,
        out_channels=8,
        num_features=64,
        num_pft_blocks=3,
        upscale_factor=4
    ).to(device)
    
    x4 = torch.randn(2, 8, 32, 32).to(device)
    with torch.no_grad():
        y4 = model4x(x4)
    
    assert y4.shape[2] == x4.shape[2] * 4 and y4.shape[3] == x4.shape[3] * 4
    print(f"\n4倍上采样测试 - 输入: {x4.shape}, 输出: {y4.shape}")
    print("✓ 4倍上采样测试通过!")
    
    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)


if __name__ == '__main__':
    test_model()
