# FY-4B 超分辨率训练优化方案 (RefineV1) 总结

## 问题诊断

原训练出现的问题：
- ❌ 验证PSNR仅16.10dB（正常应>25dB）
- ❌ 验证SSIM仅0.178（正常应>0.7）
- ❌ 训练-验证差距过大（训练SSIM损失0.74 vs 验证SSIM 0.178）
- ❌ Loss下降停滞（Epoch 11-20总Loss在1.0-1.4波动）
- ❌ Batch数量过少（每Epoch只有5个Batch）

## 优化措施

### 1. 损失函数权重调整 (关键)

```python
# 优化前
lambda_l1=1.0, lambda_ssim=0.5, lambda_freq=0.1, lambda_grad=0.1

# 优化后
lambda_l1=1.0, lambda_ssim=0.1, lambda_freq=0.01, lambda_grad=0.05
```

**原因**: 原SSIM损失(0.74)和频域损失(6.0)数值过大，掩盖了L1损失的贡献。

### 2. 学习率调度优化

```python
# 优化前
lr = 1e-4, scheduler = MultiStepLR

# 优化后
lr = 5e-5, scheduler = Warmup(10epoch) + Cosine Annealing
```

**学习率曲线**:
- Epoch 1-10: 5e-6 → 5e-5 (线性warmup)
- Epoch 11-500: 5e-5 → 1e-7 (cosine衰减)

### 3. Batch Size调整

```python
# 优化前
batch_size = 8  # 每epoch约11个batch

# 优化后
batch_size = 4  # 每epoch约22个batch
```

**好处**: 梯度更新更频繁，训练更稳定。

### 4. 验证频率调整

```python
# 优化前
val_interval = 10  # 每10个epoch验证

# 优化后
val_interval = 5   # 每5个epoch验证
```

**好处**: 更快发现问题，及时调整。

### 5. 早停策略调整

```python
# 优化前
patience = 10, min_delta = 0.0001

# 优化后
patience = 15, min_delta = 0.01
```

**好处**: 更宽容，避免过早停止，PSNR需提升0.01dB才算改善。

### 6. 梯度裁剪

```python
grad_clip = 1.0  # 已存在，保持不变
```

## 文件结构

```
/root/codes/sp0301/
├── train.py                    # 原版训练脚本
├── train_refinev1.py           # 优化版训练脚本 (新增)
├── TRAIN_REFINEV1_USAGE.md     # 详细使用说明 (新增)
└── REFINEV1_SUMMARY.md         # 本文件
```

## 启动命令

### 基础训练

```bash
# 进入项目目录
cd /root/codes/sp0301

# 训练 CH07 (推荐首次运行)
python train_refinev1.py 2000M CH07 --epochs 200 --patience 20

# 训练 CH08
python train_refinev1.py 2000M CH08 --epochs 200 --patience 20
```

### 完整训练

```bash
# 如果效果良好，进行完整训练
python train_refinev1.py 2000M CH07 --epochs 500 --patience 15
```

### 监控训练

```bash
# 启动 TensorBoard
tensorboard --logdir=./tf-logs --port=6016

# 浏览器访问
http://localhost:6016
```

## 预期效果

### Epoch 1-20 (初期)

| 指标 | 优化前 | 优化后目标 |
|-----|-------|-----------|
| 验证PSNR | ~16 dB | >22 dB |
| 验证SSIM | ~0.18 | >0.65 |
| Loss趋势 | 停滞 | 持续下降 |

### Epoch 20-100 (中期)

- PSNR: 22 dB → 28-32 dB
- SSIM: 0.65 → 0.75-0.85
- 训练/验证差距: < 20%

### Epoch 100-500 (后期)

- 最终PSNR目标: >30 dB
- 最终SSIM目标: >0.80

## 关键监控指标

### ✅ 正常信号

- 验证PSNR在前20epoch突破22dB
- 验证SSIM在前20epoch突破0.65
- L1 Loss持续下降
- 训练/验证Loss比值在0.8-1.2

### ⚠️ 异常信号

- 20epoch后PSNR仍<20dB → 检查数据/模型
- 训练Loss↓但验证Loss↑ → 过拟合
- Loss波动大 → 降低学习率

## 输出位置

```
/root/autodl-tmp/Calibration-FY4B/trained_model_refinev1/
├── 2000M_CH07/
│   ├── checkpoints/
│   │   ├── model_best.pth       # 最佳模型
│   │   ├── model_final.pth      # 最终模型
│   │   └── config.yaml          # 配置备份
│   └── visualizations/          # 训练曲线
└── 2000M_CH08/
    └── ...
```

## 对比原版

| 特性 | 原版 train.py | 优化版 train_refinev1.py |
|-----|--------------|-------------------------|
| L1权重 | 1.0 | 1.0 |
| SSIM权重 | 0.5 | **0.1** ⬇️ |
| Freq权重 | 0.1 | **0.01** ⬇️ |
| Grad权重 | 0.1 | **0.05** ⬇️ |
| 初始学习率 | 1e-4 | **5e-5** ⬇️ |
| 学习率调度 | MultiStepLR | **Warmup+Cosine** ✨ |
| Batch Size | 8 | **4** ⬇️ |
| 每epoch batches | ~11 | **~22** ⬆️ |
| 验证间隔 | 10 | **5** ⬇️ |
| 早停patience | 10 | **15** ⬆️ |

## 故障排除

### 如果PSNR仍然很低

```bash
# 进一步降低学习率
python train_refinev1.py 2000M CH07 --lr 1e-5

# 检查数据
python -c "from data import FY4BDataset; d = FY4BDataset(...); print(d[0][0].min(), d[0][0].max())"
```

### 如果出现过拟合

- 已增加weight decay (1e-4)
- 可增加数据增强
- 可增加Dropout

## 联系方式

如有问题，请查看详细文档：`TRAIN_REFINEV1_USAGE.md`

---

**祝训练顺利！** 🚀
