# FY-4B 超分辨率训练使用说明

## 训练脚本

### 基本用法

```bash
python train.py <高分辨率> <波段号>
```

### 参数说明

| 参数 | 可选值 | 说明 |
|-----|-------|------|
| `高分辨率` | `2000M` / `1000M` | 目标高分辨率 |
| `波段号` | `CH07` / `CH08` | 训练通道 |

### 训练任务说明

| 高分辨率参数 | 实际任务 | 数据对 |
|------------|---------|-------|
| `2000M` | 4km → 2km 超分辨率 | 4000M → 2000M |
| `1000M` | 2km → 1km 超分辨率 | 2000M → 1000M |

### 示例命令

```bash
# 训练 4km->2km，CH07 通道
python train.py 2000M CH07

# 训练 4km->2km，CH08 通道
python train.py 2000M CH08

# 自定义训练轮数和批次大小
python train.py 2000M CH07 --epochs 300 --batch-size 16

# 自定义学习率
python train.py 2000M CH07 --lr 0.0002

# 从检查点恢复训练
python train.py 2000M CH07 --resume /path/to/checkpoint.pth

# 设置早停容忍轮数（默认10，设为15更宽松）
python train.py 2000M CH07 --patience 15

# 禁用早停（设置一个很大的值）
python train.py 2000M CH07 --patience 1000
```

## 输出目录结构

训练结果保存到 `/root/autodl-tmp/Calibration-FY4B/trained_model/` 下：

```
/root/autodl-tmp/Calibration-FY4B/trained_model/
├── 2000M_CH07/              # 2000M分辨率 CH07通道
│   ├── checkpoints/         # 模型检查点
│   │   ├── config.yaml      # 训练配置
│   │   ├── model_final.pth  # 最终模型
│   │   ├── model_best.pth   # 最佳模型
│   │   └── checkpoint_epoch_*.pth  # 中间检查点
│   ├── logs/                # TensorBoard日志
│   └── visualizations/      # 可视化结果
│       └── training_curves.png
│
├── 2000M_CH08/              # 2000M分辨率 CH08通道
│   ├── checkpoints/
│   ├── logs/
│   └── visualizations/
│
└── 1000M_CH07/              # (未来) 1000M分辨率
    └── ...
```

## 数据路径

训练数据从以下路径读取：

```
/root/autodl-tmp/Calibration-FY4B/
├── 2000M/          # 高分辨率 (2km)
│   ├── CH07/       # Channel07 定标数据
│   └── CH08/       # Channel08 定标数据
│
└── 4000M/          # 低分辨率 (4km)
    ├── CH07/
    └── CH08/
```

## 训练配置

### 默认参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| 批次大小 | 8 | 每批次样本数 |
| 学习率 | 0.0001 | Adam优化器初始学习率 |
| 训练轮数 | 500 | 总训练epoch数 |
| 图像块大小 | 64 | 低分辨率空间裁剪大小 |
| 上采样倍数 | 2 | 固定2x超分辨率 |
| 验证间隔 | 10 | 每10个epoch验证一次 |
| 保存间隔 | 50 | 每50个epoch保存检查点 |
| **早停 patience** | **10** | **连续10次验证PSNR无提升则自动停止** |

### 早停机制

训练过程中自动监控验证集的 **PSNR** 指标：
- 当 PSNR 连续 **10 次验证**（即 10 × 10 = 100 个 epoch）没有提升时
- 自动触发早停，保存最佳模型并结束训练
- 防止过拟合，节省训练时间
- **可通过 `--patience` 参数调整容忍轮数**

早停触发时会显示：
```
============================================================
早停触发! 连续 10 次验证无提升
最佳 PSNR: XX.XXXX dB
============================================================
```

自定义早停参数：
```bash
# 更严格的早停（5次验证无提升就停止）
python train.py 2000M CH07 --patience 5

# 更宽松的早停（20次验证无提升才停止）
python train.py 2000M CH07 --patience 20

# 基本禁用早停（需要非常大的训练轮数才会触发）
python train.py 2000M CH07 --patience 100 --epochs 200
```

### 模型架构

- **网络**: PFT-SR (Progressive Feature Transfer)
- **输入通道**: 1 (单通道)
- **输出通道**: 1 (单通道)
- **特征维度**: 64
- **PFT块数量**: 3
- **参数量**: ~1.15M

### 损失函数

综合损失 = L1 + 0.5×SSIM + 0.1×Frequency + 0.1×Gradient

## 查看训练日志 (TensorBoard)

训练日志默认保存到当前目录的 `./tf-logs/` 下：

```bash
# 启动 TensorBoard (端口6016)
tensorboard --logdir=./tf-logs --port=6016

# 然后在浏览器访问
# http://localhost:6016
```

日志目录结构：
```
./tf-logs/
├── FY4B_PFTSR_4000M_to_2000M_CH07/     # CH07训练日志
└── FY4B_PFTSR_4000M_to_2000M_CH08/     # CH08训练日志
```

## 数据配对逻辑

根据文件名中的时间戳匹配低-高分辨率数据对：
- 文件名格式: `FY4B-*_MULT_CAL_YYYYMMDDHHMMSS_YYYYMMDDHHMMSS_*M_V0001.HDF`
- 匹配关键: `CAL_YYYYMMDDHHMMSS` 时间戳一致

当前数据：
- 每个通道 89 个数据对
- 4000M (2748×2748) ↔ 2000M (5496×5496)
