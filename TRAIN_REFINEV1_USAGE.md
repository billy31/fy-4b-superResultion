# FY-4B 超分辨率训练 - 优化版本 (RefineV1) 使用说明

## 优化内容总结

针对之前训练中出现的问题（PSNR低、Loss不平衡、过拟合），本版本进行了以下优化：

### 1. 损失函数权重调整 (关键优化)

| 损失项 | 原权重 | 优化后权重 | 说明 |
|-------|-------|-----------|------|
| L1 | 1.0 | 1.0 | 保持主导 |
| SSIM | 0.5 | **0.1** | 大幅降低，避免梯度主导 |
| Freq | 0.1 | **0.01** | 大幅降低，数值原来过大 |
| Grad | 0.1 | **0.05** | 适当降低 |

**原因**: 原SSIM损失(0.74)和频域损失(6.0)数值过大，掩盖了L1损失的贡献。

### 2. 学习率调度优化

- **原方案**: 固定学习率 1e-4 + MultiStepLR
- **优化方案**: **Warmup(10epoch) + Cosine Annealing**
  - Warmup: 前10个epoch线性增加到初始学习率
  - Cosine: 之后平滑衰减到 1e-7
  - 初始学习率: **5e-5** (降低一半，更稳定)

### 3. Batch Size调整

- **原方案**: batch_size=8，每epoch约11个batch
- **优化方案**: **batch_size=4**，每epoch约22个batch
- **好处**: 增加batch数量，梯度更新更频繁，训练更稳定

### 4. 验证频率增加

- **原方案**: 每10个epoch验证一次
- **优化方案**: **每5个epoch验证一次**
- **好处**: 更快发现问题，及时调整

### 5. 早停策略优化

- **patience**: 10 → **15** (更宽容，避免过早停止)
- **min_delta**: 0.0001 → **0.01** (PSNR需提升0.01dB才算改善)

### 6. 梯度裁剪

- 保持 `clip_grad_norm=1.0`
- 确保梯度稳定性，防止梯度爆炸

---

## 使用方法

### 基本命令

```bash
python train_refinev1.py <高分辨率> <波段号>
```

### 示例

```bash
# 训练 CH07 通道 (4km->2km)
python train_refinev1.py 2000M CH07

# 训练 CH08 通道
python train_refinev1.py 2000M CH08
```

### 可选参数

```bash
# 自定义训练轮数
python train_refinev1.py 2000M CH07 --epochs 300

# 调整batch size (默认4)
python train_refinev1.py 2000M CH07 --batch-size 8

# 调整学习率 (默认5e-5)
python train_refinev1.py 2000M CH07 --lr 1e-4

# 调整早停patience (默认15)
python train_refinev1.py 2000M CH07 --patience 20

# 从检查点恢复
python train_refinev1.py 2000M CH07 --resume /path/to/checkpoint.pth
```

---

## 预期效果

### 训练初期 (Epoch 1-20)

| 指标 | 优化前 | 优化后预期 |
|-----|-------|-----------|
| 验证PSNR | ~16 dB | **>22 dB** |
| 验证SSIM | ~0.18 | **>0.65** |
| Loss下降 | 停滞 | 持续下降 |
| 训练-验证差距 | 很大 | 较小 |

### 训练中期 (Epoch 20-100)

- PSNR应逐步提升到 **28-32 dB**
- SSIM应达到 **0.75-0.85**
- 训练/验证Loss差距应 **< 20%**

### 训练完成 (Epoch 100-300)

- 最终PSNR目标: **>30 dB**
- 最终SSIM目标: **>0.80**

---

## 监控指标

### 正常训练的信号

✅ **良好指标**:
- 验证PSNR在前20个epoch内突破22dB
- 验证SSIM在前20个epoch内突破0.65
- L1 Loss持续下降
- 训练/验证Loss比值在0.8-1.2之间

⚠️ **需要关注的信号**:
- 验证PSNR在20epoch后仍<20dB → 检查数据/模型
- 训练Loss下降但验证Loss上升 → 过拟合，需数据增强
- Loss波动很大 → 降低学习率

---

## TensorBoard 监控

```bash
# 启动 TensorBoard
tensorboard --logdir=./tf-logs --port=6016

# 访问 http://localhost:6016
```

### 关键图表

1. **Metrics/PSNR**: 应稳步上升，最终>30dB
2. **Metrics/SSIM**: 应稳步上升，最终>0.80
3. **Loss/train_total vs Loss/val_total**: 两者应同步下降
4. **Train/lr**: 学习率曲线， warmup后cosine下降

---

## 输出目录

```
/root/autodl-tmp/Calibration-FY4B/trained_model_refinev1/
├── 2000M_CH07/
│   ├── checkpoints/
│   │   ├── config.yaml          # 保存的配置
│   │   ├── model_final.pth      # 最终模型
│   │   ├── model_best.pth       # 最佳模型
│   │   └── checkpoint_epoch_*.pth
│   ├── logs/                    # TensorBoard日志
│   └── visualizations/          # 可视化结果
└── 2000M_CH08/
    └── ...
```

---

## 对比实验建议

如果你想对比优化前后的效果：

```bash
# 窗口1: 原版本训练
python train.py 2000M CH07 --epochs 100

# 窗口2: 优化版本训练
python train_refinev1.py 2000M CH07 --epochs 100
```

然后在TensorBoard中对比两个版本的曲线。

---

## 故障排除

### 如果PSNR仍然很低 (<20dB after 20 epochs)

1. **检查数据归一化**:
   ```python
   # 检查数据范围
   python -c "from data import FY4BDataset; d = FY4BDataset(...); print(d[0][0].min(), d[0][0].max())"
   ```
   - 正常应在 [-1, 1] 或 [0, 1] 范围

2. **检查模型输出**:
   - SR输出值范围是否合理
   - 如果输出都是同一值，说明模型未学习

3. **进一步降低学习率**:
   ```bash
   python train_refinev1.py 2000M CH07 --lr 1e-5
   ```

### 如果出现过拟合

1. 增加数据增强 (修改 `data/fy4b_dataset.py`)
2. 增加weight decay:
   ```bash
   python train_refinev1.py 2000M CH07  # 已增加到1e-4
   ```
3. 使用Dropout (需修改模型)

---

## 推荐启动命令

```bash
# 第一次训练推荐 (保守设置)
python train_refinev1.py 2000M CH07 --epochs 200 --patience 20

# 如果效果良好，进行完整训练
python train_refinev1.py 2000M CH07 --epochs 500 --patience 15
```

祝训练顺利！
