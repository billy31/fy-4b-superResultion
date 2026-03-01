# FY-4B卫星图像超分辨率 - PFT-SR

基于 [PFT-SR (Progressive Feature Transfer for Super-Resolution)](https://github.com/CVL-UESTC/PFT-SR) 方法，针对FY-4B卫星AGRI仪器数据定制的超分辨率深度学习项目。

## 项目概述

### 任务目标
通过FY-4B卫星原生的多分辨率数据对进行超分辨率训练：
- **4km → 2km 超分辨率**: 使用 4000M-2000M 数据对训练
- **2km → 1km 超分辨率**: 使用 2000M-1000M 数据对训练（预留）

### 核心优势
- 数据对是**原生配对**: 同一观测视角、同一时间，天然对齐
- 定标后数据已转换为亮温值，可直接用于训练

## 项目结构

```
.
├── data/                   # 数据加载模块
│   ├── __init__.py
│   └── fy4b_dataset.py     # FY-4B数据集类
├── models/                 # 模型定义
│   ├── __init__.py
│   ├── pft_sr.py           # PFT-SR网络结构
│   └── loss.py             # 综合损失函数 (L1+SSIM+频域+梯度)
├── utils/                  # 工具函数
│   ├── __init__.py
│   ├── metrics.py          # 评估指标 (PSNR, SSIM)
│   ├── checkpoint.py       # 检查点管理
│   └── visualize.py        # 可视化工具
├── configs/                # 配置文件
│   ├── train_ch07_config.yaml
│   └── train_ch08_config.yaml
├── train.py                # 训练脚本 (主入口)
├── test.py                 # 测试脚本
├── main.py                 # 项目主入口
├── TRAIN_USAGE.md          # 详细使用说明
└── readme.md               # 项目说明
```

## 快速开始

### 1. 环境安装

```bash
# 安装依赖
pip install -r requirements.txt

# 或使用conda
conda install pytorch torchvision numpy scipy matplotlib pyyaml tensorboard h5py
```

### 2. 训练数据

训练数据路径：
```
/root/autodl-tmp/Calibration-FY4B/
├── 2000M/          # 2km分辨率数据 (高分辨率)
│   ├── CH07/       # Channel07 定标数据 (89个文件)
│   └── CH08/       # Channel08 定标数据 (89个文件)
└── 4000M/          # 4km分辨率数据 (低分辨率)
    ├── CH07/
    └── CH08/
```

### 3. 开始训练

#### 基本用法

```bash
python train.py <高分辨率> <波段号>
```

| 高分辨率 | 说明 |
|---------|------|
| `2000M` | 训练 4km → 2km (4000M → 2000M) |
| `1000M` | 训练 2km → 1km (2000M → 1000M) |

| 波段号 | 说明 |
|-------|------|
| `CH07` | 红外波段 3.90μm (亮温) |
| `CH08` | 红外波段 7.00μm (亮温) |

#### 示例

```bash
# 训练 4km->2km，CH07 通道
python train.py 2000M CH07

# 训练 4km->2km，CH08 通道
python train.py 2000M CH08

# 自定义训练参数
python train.py 2000M CH07 --epochs 300 --batch-size 16 --lr 0.0002

# 从检查点恢复训练
python train.py 2000M CH07 --resume /path/to/checkpoint.pth
```

### 4. 训练输出

训练结果保存到 `/root/autodl-tmp/Calibration-FY4B/trained_model/`：

```
/root/autodl-tmp/Calibration-FY4B/trained_model/
├── 2000M_CH07/              # 2000M CH07训练结果
│   ├── checkpoints/         # 模型检查点
│   │   ├── config.yaml      # 训练配置
│   │   ├── model_final.pth  # 最终模型
│   │   ├── model_best.pth   # 最佳模型
│   │   └── checkpoint_epoch_*.pth
│   ├── logs/                # TensorBoard日志
│   └── visualizations/      # 可视化结果
└── 2000M_CH08/              # 2000M CH08训练结果
    └── ...
```

### 5. 查看训练日志

```bash
tensorboard --logdir /root/autodl-tmp/Calibration-FY4B/trained_model/2000M_CH07/logs
```

## 模型架构: PFT-SR

### 核心组件

1. **Shallow Feature Extraction**
   - 浅层卷积提取初始特征

2. **Progressive Feature Transfer Blocks (PFT Blocks)**
   - 渐进式特征转移
   - 每个块包含多个残差块 + 注意力机制
   - 通过PixelShuffle进行上采样

3. **Cross-Layer Feature Fusion**
   - 跨层特征融合，利用多尺度信息

4. **Global Residual Learning**
   - 全局残差连接，加速收敛

### 模型参数

| 参数 | 值 |
|-----|---|
| 输入通道 | 1 (单通道) |
| 输出通道 | 1 (单通道) |
| 特征维度 | 64 |
| PFT块数量 | 3 |
| 上采样倍数 | 2x |
| 总参数量 | ~1.15M |

## 损失函数

综合损失 = L1 + 0.5×SSIM + 0.1×Frequency + 0.1×Gradient

| 损失类型 | 权重 | 作用 |
|---------|------|------|
| L1 Loss | 1.0 | 像素级保真 |
| SSIM Loss | 0.5 | 结构相似性 |
| Frequency Loss | 0.1 | 频域细节保留 |
| Gradient Loss | 0.1 | 边缘信息保持 |

## 训练配置

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| 批次大小 | 8 | 每批次样本数 |
| 学习率 | 0.0001 | Adam优化器初始学习率 |
| 训练轮数 | 500 | 总训练epoch数 |
| 图像块大小 | 64 | 低分辨率空间裁剪大小 (64→128) |
| 验证间隔 | 10 | 每10个epoch验证一次 |
| 保存间隔 | 50 | 每50个epoch保存检查点 |
| **早停 patience** | **10** | **连续10次验证PSNR无提升则自动停止** |

## 评估指标

- **PSNR**: 峰值信噪比 (dB)
- **SSIM**: 结构相似性指数
- **RMSE**: 均方根误差

## 通道信息

| 波段名称 | 数据集名 | 波长 | 物理量 |
|---------|---------|------|-------|
| CH07 | Channel07 | 3.90 μm | 亮温 (K) |
| CH08 | Channel08 | 7.00 μm | 亮温 (K) |

## 数据配对

根据文件名中的时间戳匹配数据对：
- 文件名格式: `FY4B-*_MULT_CAL_YYYYMMDDHHMMSS_YYYYMMDDHHMMSS_*M_V0001.HDF`
- 匹配关键: `CAL_YYYYMMDDHHMMSS` 时间戳一致
- 当前数据：每个通道 89 个数据对

## 参考资料

- [PFT-SR GitHub](https://github.com/CVL-UESTC/PFT-SR)
- [FY-4B卫星数据说明](http://www.nsmc.org.cn/NSMC/Channels/100028.html)

## TensorBoard 查看训练日志

```bash
# 启动 TensorBoard (端口6016)
tensorboard --logdir=./tf-logs --port=6016

# 访问 http://localhost:6016
```

## 详细文档

更多使用说明请查看 [TRAIN_USAGE.md](./TRAIN_USAGE.md)
