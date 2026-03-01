# FY-4B 超分辨率网络潜在优化建议

> 日期: 2026-03-01  
> 当前状态: CH07通道训练完成，PSNR=34.54dB, SSIM=0.8792

---

## 一、网络架构优化建议

### 1. 增加网络深度/宽度

**建议**: 增加PFT Block数量和特征维度
```python
# 当前配置
num_features=64, num_pft_blocks=3, num_rb_per_block=3

# 建议尝试
num_features=96, num_pft_blocks=4, num_rb_per_block=4  # 增加容量
```

**理由**: 当前模型参数量1.15M，可适当增加到2-3M以提升拟合能力

---

### 2. 添加残差连接改进

**建议**: 在PFT Block内部添加密集连接 (Dense Connection)
```python
# 类似DenseNet的结构
class DensePFTBlock(nn.Module):
    def __init__(self, ...):
        self.blocks = nn.ModuleList([
            ResidualBlock(num_features + i * growth_rate) 
            for i in range(num_rb)
        ])
```

**理由**: 促进特征重用，改善梯度流动

---

### 3. 引入多尺度特征融合

**建议**: 添加FPN (Feature Pyramid Network) 结构
```python
# 在编码器-解码器之间添加多尺度融合
class MultiScaleFusion(nn.Module):
    def __init__(self, channels):
        self.pyramid = nn.ModuleList([
            nn.Conv2d(channels, channels//2, 3, dilation=d, padding=d)
            for d in [1, 2, 4, 8]  # 不同感受野
        ])
```

**理由**: 卫星图像包含多尺度云层结构，多尺度融合可提升细节恢复

---

### 4. 注意力机制增强

**建议**: 替换现有注意力为CBAM或ECA
```python
# CBAM: Channel + Spatial Attention
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()

# 或更轻量的ECA
class ECA(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size//2)
```

**理由**: 更高效的通道注意力，减少参数量同时提升性能

---

## 二、损失函数优化建议

### 1. 感知损失 (Perceptual Loss)

**建议**: 添加VGG-based感知损失
```python
class PerceptualLoss(nn.Module):
    def __init__(self):
        vgg = vgg19(pretrained=True).features
        self.feature_layers = nn.Sequential(*list(vgg)[:16])  # 到relu3_3
        
    def forward(self, sr, hr):
        sr_feat = self.feature_layers(sr)
        hr_feat = self.feature_layers(hr)
        return F.mse_loss(sr_feat, hr_feat)

# 总损失
loss = l1_loss + 0.1 * ssim_loss + 0.01 * freq_loss + 0.001 * perceptual_loss
```

**理由**: 提升视觉质量，使超分结果更符合人眼感知

---

### 2. 对抗损失 (Adversarial Loss)

**建议**: 引入GAN框架
```python
# 添加判别器
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        # PatchGAN结构
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # ... 多层卷积
            nn.Conv2d(512, 1, 4, padding=1),  # Patch输出
        )
```

**理由**: 进一步提升纹理细节，但训练更不稳定，需谨慎使用

---

### 3. 边缘感知损失

**建议**: 添加拉普拉斯边缘损失
```python
class EdgeLoss(nn.Module):
    def __init__(self):
        laplacian_kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('laplacian', laplacian_kernel)
    
    def forward(self, sr, hr):
        sr_edge = F.conv2d(sr, self.laplacian, padding=1)
        hr_edge = F.conv2d(hr, self.laplacian, padding=1)
        return F.l1_loss(sr_edge, hr_edge)
```

**理由**: 显式约束边缘恢复，对云层边界更清晰

---

## 三、训练策略优化建议

### 1. 分阶段训练

**建议**: 两阶段训练策略
```python
# 阶段1: 仅L1损失预训练 (快速收敛)
for epoch in range(50):
    loss = l1_loss(sr, hr)  # 仅L1
    
# 阶段2: 全损失微调
for epoch in range(50, 500):
    loss = combined_loss(sr, hr)  # 全部损失
```

**理由**: L1损失收敛快，作为热启动；后续加入其他损失精细调整

---

### 2. 课程学习 (Curriculum Learning)

**建议**: 逐渐增加难度
```python
# 从简单patch到复杂patch
# Epoch 0-50:  只训练云层边缘清晰的patch
# Epoch 50-100: 加入中等复杂度patch
# Epoch 100+:   全部patch
```

**理由**: 模仿人类学习过程，逐步提升模型能力

---

### 3. 数据增强策略

**建议**: 增加气象数据专用的增强
```python
# 1. 随机噪声 (模拟传感器噪声)
noise = torch.randn_like(lr) * 0.01
lr = lr + noise

# 2. 随机对比度调整
lr = lr * (1 + torch.rand(1).item() * 0.2 - 0.1)

# 3. Cutout/GridMask
# 模拟卫星数据中的缺失区域
```

---

## 四、学习率调度优化

### 1. 循环学习率 (Cyclical LR)

**建议**: 周期性学习率调整
```python
scheduler = CyclicLR(
    optimizer, 
    base_lr=1e-6, 
    max_lr=1e-4,
    step_size_up=2000,  # 每2000 iteration
    mode='triangular2'
)
```

**理由**: 帮助跳出局部最优，可能进一步提升性能

---

### 2. 自适应学习率 (ReduceLROnPlateau)

**建议**: 基于验证指标调整
```python
scheduler = ReduceLROnPlateau(
    optimizer, 
    mode='max', 
    factor=0.5, 
    patience=5,  # 5次验证无提升则降低学习率
    min_lr=1e-7
)
```

**理由**: 更智能的学习率调整，当前Cosine可能过于固定

---

## 五、评估与后处理

### 1. 多模型集成

**建议**: 训练多个模型并集成
```python
# 训练3-5个不同初始化/超参数的模型
# 推理时取平均
sr_ensemble = (model1(lr) + model2(lr) + model3(lr)) / 3
```

**预期提升**: PSNR +0.2~0.5dB

---

### 2. 测试时增强 (TTA)

**建议**: 推理时数据增强
```python
def tta_inference(model, lr):
    # 原始 + 水平翻转 + 垂直翻转 + 旋转
    predictions = []
    for flip in [lambda x: x, lambda x: torch.flip(x, [2]), 
                 lambda x: torch.flip(x, [3])]:
        pred = model(flip(lr))
        predictions.append(pred)
    return torch.stack(predictions).mean(dim=0)
```

**预期提升**: PSNR +0.1~0.3dB，计算量增加4倍

---

## 六、网络轻量化 (若需要部署)

### 1. 知识蒸馏

**建议**: 大模型蒸馏到小模型
```python
# Teacher: 当前模型 (1.15M)
# Student: 轻量化模型 (0.3M)
loss = alpha * hard_loss(student_sr, hr) + (1-alpha) * soft_loss(student_sr, teacher_sr)
```

---

### 2. 网络剪枝

**建议**: 剪枝不重要的通道
```python
# 基于L1范数的通道剪枝
pruning_rates = [0.1, 0.2, 0.3]  # 逐步剪枝30%通道
```

---

## 七、优先尝试建议

根据当前34.54dB的基线，推荐尝试顺序：

| 优先级 | 优化项 | 预期提升 | 实现难度 |
|-------|-------|---------|---------|
| 🔴 高 | 增加网络容量 (96通道, 4 blocks) | +0.5~1.0dB | 低 |
| 🔴 高 | 分阶段训练 (L1预训练) | +0.3~0.5dB | 低 |
| 🟡 中 | 感知损失 (VGG) | +0.2~0.5dB | 中 |
| 🟡 中 | ECA注意力替换 | +0.2~0.4dB | 低 |
| 🟢 低 | 多模型集成 | +0.2~0.5dB | 中 |
| 🟢 低 | TTA | +0.1~0.3dB | 低 |

---

## 八、当前最优配置备份

```python
# 当前已验证的最优配置 (CH07: 34.54dB)
CONFIG_BEST = {
    'loss': {'l1': 1.0, 'ssim': 0.1, 'freq': 0.01, 'grad': 0.05},
    'optimizer': {'lr': 5e-5, 'weight_decay': 1e-4},
    'scheduler': 'WarmupCosine',
    'batch_size': 4,
    'val_interval': 5,
    'model': {'num_features': 64, 'num_pft_blocks': 3}
}
```

---

*建议从"增加网络容量"和"分阶段训练"开始尝试，预期可提升到35.5+dB*
