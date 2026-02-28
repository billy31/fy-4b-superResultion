# FY-4B卫星图像超分辨率 - PFT-SR

基于 [PFT-SR (Progressive Feature Transfer for Super-Resolution)](https://github.com/CVL-UESTC/PFT-SR) 方法，针对FY-4B卫星AGRI仪器数据定制的超分辨率深度学习项目。

## 项目概述

### 任务目标
通过FY-4B卫星原生的多分辨率数据对进行超分辨率训练：
- **2km-4km数据对**: 训练 4km → 2km 超分辨率
- **1km-2km数据对**: 训练 2km → 1km 超分辨率

### 核心优势
- 数据对是**原生配对**: 同一观测视角、同一时间，天然对齐
- 4km数据由2km通过插值得到，具有确定的超分辨率映射关系

## 项目结构

```
.
├── data/                   # 数据加载模块
│   ├── __init__.py
│   └── fy4b_dataset.py     # FY-4B数据集类 (使用占位符: lowResfile1/highResfile2)
├── models/                 # 模型定义
│   ├── __init__.py
│   ├── pft_sr.py           # PFT-SR网络结构
│   └── loss.py             # 综合损失函数 (L1+SSIM+频域+梯度)
├── utils/                  # 工具函数
│   ├── __init__.py
│   ├── metrics.py          # 评估指标 (PSNR, SSIM)
│   ├── checkpoint.py       # 检查点管理
│   └── visualize.py        # 可视化工具
├── configs/
│   └── train_config.yaml   # 训练配置
├── checkpoints/            # 模型检查点保存目录
├── logs/                   # 训练日志目录
├── visualizations/         # 可视化结果目录
├── train.py                # 训练脚本
├── test.py                 # 测试脚本
├── main.py                 # 项目主入口
└── readme.md               # 项目说明
```

## 快速开始

### 1. 环境安装

```bash
# 安装依赖
pip install -r requirements.txt

# 或使用conda
conda install pytorch torchvision numpy scipy matplotlib pyyaml tensorboard
```

### 2. 数据占位符说明

当前训练使用占位符文件：
- **低分辨率数据**: `lowResfile1` (模拟4km分辨率数据)
- **高分辨率数据**: `highResfile2` (模拟2km分辨率数据)

实际数据准备好后，修改 `configs/train_config.yaml` 中的路径：
```yaml
data:
  low_res_dir: "/root/autodl-tmp/4000M"   # 实际4km数据目录
  high_res_dir: "/root/autodl-tmp/2000M"  # 实际2km数据目录
```

### 3. 运行测试

```bash
# 查看项目信息
python main.py --mode info

# 测试数据集
python main.py --mode data

# 测试模型
python main.py --mode model

# 测试损失函数
python main.py --mode loss
```

### 4. 开始训练

```bash
# 使用默认配置训练
python train.py

# 或
python main.py --mode train

# 从检查点恢复训练
python train.py --resume checkpoints/checkpoint_epoch_100.pth
```

### 5. 模型测试

```bash
# 测试单个文件
python test.py --checkpoint checkpoints/model_best.pth --input lowResfile1

# 或
python main.py --mode test --checkpoint checkpoints/model_best.pth
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

### 关键特性

- **通道注意力 (Channel Attention)**: 自适应加权通道特征
- **空间注意力 (Spatial Attention)**: 自适应加权空间区域
- **多尺度训练**: 支持2x和4x上采样

## 损失函数

综合损失 = λ₁·L1 + λ₂·SSIM + λ₃·Frequency + λ₄·Gradient

| 损失类型 | 权重 | 作用 |
|---------|------|------|
| L1 Loss | 1.0 | 像素级保真 |
| SSIM Loss | 0.5 | 结构相似性 |
| Frequency Loss | 0.1 | 频域细节保留 |
| Gradient Loss | 0.1 | 边缘信息保持 |

## 配置说明

主要配置项在 `configs/train_config.yaml`：

```yaml
# 数据配置
data:
  low_res_file: "lowResfile1"      # 低分辨率占位符
  high_res_file: "highResfile2"    # 高分辨率占位符
  patch_size: 64                   # 图像块大小
  upscale_factor: 2                # 上采样因子
  batch_size: 16

# 模型配置
model:
  num_features: 64                 # 特征维度
  num_pft_blocks: 3                # PFT块数量
  use_attention: true              # 使用注意力

# 训练配置
training:
  num_epochs: 500
  lr: 0.0001
```

## 数据格式

### FY-4B HDF文件名格式
```
FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_YYYYMMDDHHMMSS_YYYYMMDDHHMMSS_2000M_V0001.HDF
```

### 中红外通道列表
| 通道名称 | HDF数据集名 | 波长 |
|---------|------------|------|
| IR05 | NOMChannel05 | 3.5 μm |
| IR06 | NOMChannel06 | 4.05 μm |
| IR07 | NOMChannel07 | 6.95 μm |
| IR08 | NOMChannel08 | 8.55 μm |
| IR09 | NOMChannel09 | 10.8 μm |
| IR10 | NOMChannel10 | 12.0 μm |
| IR11 | NOMChannel11 | 13.5 μm |
| IR12 | NOMChannel12 | 14.5 μm |

## 评估指标

- **PSNR**: 峰值信噪比 (dB)
- **SSIM**: 结构相似性指数
- **RMSE**: 均方根误差
- **MAE**: 平均绝对误差

## 后续计划

1. [ ] 实现实际的HDF数据读取 (使用h5py或pyhdf)
2. [ ] 数据预处理: 云检测、无效值处理
3. [ ] 添加4km分辨率数据目录
4. [ ] 实现数据时间戳匹配逻辑
5. [ ] 多GPU训练支持
6. [ ] 添加更多数据增强策略

## 参考资料

- [PFT-SR GitHub](https://github.com/CVL-UESTC/PFT-SR)
- [FY-4B卫星数据说明](http://www.nsmc.org.cn/NSMC/Channels/100028.html)

## 联系

如有问题或建议，欢迎提出Issue。
