# FY-4B 超分辨率训练 - 开发日志

> 日期: 2026-03-01  
> 项目名称: FY-4B卫星图像超分辨率 (PFT-SR)  
> 环境: mamba2 (Python 3.10, PyTorch, CUDA)

---

## 今日完成任务

### 1. 代码架构重构

**完成内容**:
- 重构了训练脚本 `train.py`，支持命令行参数指定训练任务
- 实现了单通道独立训练（CH07/CH08分别训练）
- 创建了优化版本 `train_refinev1.py`，解决训练过程中的问题

**关键代码实现**:
```python
# 命令行格式: python train.py <高分辨率> <波段号>
# 示例: python train.py 2000M CH07

# 自动构建数据路径
low_res_dir = f"/root/autodl-tmp/Calibration-FY4B/{low_res}/{band}"
high_res_dir = f"/root/autodl-tmp/Calibration-FY4B/{high_res}/{band}"

# 自动构建输出路径
output_base = f"/root/autodl-tmp/Calibration-FY4B/trained_model/{high_res}_{band}"
```

---

### 2. 关键问题诊断与修复

**发现的问题**:
| 问题 | 现象 | 解决方案 |
|-----|------|---------|
| 损失函数不平衡 | SSIM和Freq损失过大，掩盖L1贡献 | 调整权重: SSIM 0.5→0.1, Freq 0.1→0.01 |
| 学习率调度不当 | 固定学习率导致收敛慢 | 实现Warmup+Cosine Annealing |
| Batch数量不足 | 每epoch仅11个batch | 减小batch_size 8→4，增加到22个batch |
| 验证频率过低 | 每10epoch验证，发现问题晚 | 改为每5epoch验证 |
| 绘图维度错误 | val_loss和epochs维度不匹配 | 添加val_epochs记录验证时间点 |

**修复记录**:
- ✅ 修改 `train_refinev1.py`: 调整损失权重
- ✅ 实现 `WarmupCosineScheduler`: warmup 10epoch + cosine衰减
- ✅ 修改 `utils/visualize.py`: 支持不同长度的验证/训练记录

---

### 3. 当前网络架构

**模型**: PFT-SR (Progressive Feature Transfer for Super-Resolution)

```
输入 [1, 64, 64] (4km分辨率)
    ↓
浅层特征提取 (Conv 3x3)
    ↓
PFT Block 1 (3个Residual Block + Attention)
    ↓
PixelShuffle上采样 2x
    ↓
PFT Block 2
    ↓
PFT Block 3
    ↓
全局残差连接
    ↓
输出 [1, 128, 128] (2km分辨率)
```

**参数量**: 1,147,687 (1.15M)

**网络配置**:
```python
{
    'in_channels': 1,
    'out_channels': 1,
    'num_features': 64,
    'num_pft_blocks': 3,
    'num_rb_per_block': 3,
    'use_attention': True,
    'upscale_factor': 2
}
```

---

### 4. 训练结果

**CH07 通道训练完成 (200 epochs)**

| 指标 | 最终结果 | 备注 |
|-----|---------|------|
| **PSNR** | **34.54 dB** | 优秀 (>30dB) |
| **SSIM** | **0.8792** | 优秀 (>0.80) |
| Val Loss | 0.048843 | 收敛良好 |
| Train Loss | 0.023691 | 无过拟合 |

**训练过程关键节点**:
- Epoch 1-10: PSNR从~25dB提升到~30dB (Warmup阶段)
- Epoch 10-50: PSNR稳步提升到~32dB
- Epoch 50-150: PSNR缓慢提升到~34dB
- Epoch 150-200: PSNR稳定在34.5dB左右

**损失分解 (Final)**:
```
Train Loss: 0.023691
  l1:     0.006549  (主导，正常)
  ssim:   0.043632  (降低权重后正常)
  freq:   1.034087  (大幅降低后正常)
  grad:   0.048758  (正常)
```

---

### 5. 数据处理情况

**数据源**:
- 路径: `/root/autodl-tmp/Calibration-FY4B/`
- 低分辨率: 4000M (2748×2748) 
- 高分辨率: 2000M (5496×5496)

**数据对**: 
- CH07: 89个数据对
- CH08: 89个数据对

**已完成计算的数据**:
- ✅ CH07: 200 epochs训练完成，模型已保存
- ⏳ CH08: 待训练

**数据预处理**:
- NaN值处理: 使用全局均值填充
- 归一化: [150K, 350K] → [-1, 1]
- 数据增强: 随机翻转、旋转

---

### 6. 输出文件结构

```
/root/autodl-tmp/Calibration-FY4B/trained_model_refinev1/
└── 2000M_CH07/
    ├── checkpoints/
    │   ├── config.yaml              # 训练配置
    │   ├── model_best.pth           # 最佳模型 (PSNR=34.54)
    │   ├── model_final.pth          # 最终模型
    │   └── checkpoint_epoch_*.pth   # 中间检查点
    ├── logs/                        # TensorBoard日志
    └── visualizations/
        ├── epoch_50_sample_0.png    # 训练可视化
        ├── epoch_100_sample_0.png
        ├── epoch_150_sample_0.png
        ├── epoch_200_sample_0.png
        └── training_curves.png      # 训练曲线

./tf-logs/
└── FY4B_RefineV1_4000M_to_2000M_CH07/
    └── events.out.tfevents.*        # TensorBoard事件文件
```

---

### 7. 训练配置总结 (最优)

```yaml
# 损失函数
loss:
  lambda_l1: 1.0      # 主导损失
  lambda_ssim: 0.1    # 降低以避免梯度主导
  lambda_freq: 0.01   # 大幅降低
  lambda_grad: 0.05   # 适当降低

# 优化器
optimizer:
  lr: 5e-5            # 降低初始学习率
  weight_decay: 1e-4
  
# 学习率调度
scheduler:
  type: WarmupCosine
  warmup_epochs: 10
  min_lr: 1e-7

# 训练参数
training:
  batch_size: 4       # 增加batch数量
  num_epochs: 200
  val_interval: 5     # 每5epoch验证
  grad_clip: 1.0
  early_stopping:
    patience: 20
    min_delta: 0.01
```

---

### 8. 待完成任务

| 任务 | 状态 | 优先级 |
|-----|------|-------|
| CH08通道训练 | ⏳ 待开始 | 🔴 高 |
| 测试脚本完善 | ⏳ 待完成 | 🟡 中 |
| 推理速度优化 | ⏳ 待评估 | 🟢 低 |
| 1000M分辨率训练 | ⏳ 待数据 | 🟢 低 |

---

### 9. 关键命令记录

```bash
# 启动训练 (CH07已完成)
python train_refinev1.py 2000M CH07 --epochs 200 --patience 20

# 启动CH08训练
python train_refinev1.py 2000M CH08 --epochs 200 --patience 20

# 查看训练日志
tensorboard --logdir=./tf-logs --port=6016

# 从检查点恢复
python train_refinev1.py 2000M CH07 --resume /path/to/checkpoint.pth
```

---

### 10. 技术债务与注意事项

**已修复问题**:
- ✅ 绘图维度不匹配 (train_refinev1.py + visualize.py)
- ✅ 损失函数权重不平衡
- ✅ 学习率调度优化

**需要注意的问题**:
1. **显存占用**: batch_size=4时约占用~4GB显存，可调节
2. **训练时间**: 200 epoch约需4-5小时 (单卡V100)
3. **早停设置**: patience=20较宽松，如需更快停止可调小

---

## 今日心得

1. **损失函数平衡是关键**: 最初SSIM和Freq损失权重过高，导致L1损失被掩盖，模型无法正确学习。大幅降权重后效果显著提升。

2. **Warmup很重要**: 直接高学习率训练导致初期不稳定，加入10epoch warmup后收敛更平滑。

3. **验证频率**: 从10epoch改为5epoch验证，能更快发现问题，也提供了更精细的训练曲线。

4. **Batch数量 > Batch大小**: 减小batch size增加batch数量，使梯度更新更频繁，训练更稳定。

---

*记录人: AI Assistant*  
*时间: 2026-03-01*
