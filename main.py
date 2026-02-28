# -*- coding: utf-8 -*-
"""
FY-4B卫星超分辨率项目主入口
基于PFT-SR (Progressive Feature Transfer for Super-Resolution) 方法

项目结构:
├── data/               # 数据加载模块
│   └── fy4b_dataset.py # FY-4B数据集类
├── models/             # 模型定义模块
│   ├── pft_sr.py       # PFT-SR模型
│   └── loss.py         # 损失函数
├── utils/              # 工具函数模块
│   ├── metrics.py      # 评估指标
│   ├── checkpoint.py   # 检查点管理
│   └── visualize.py    # 可视化工具
├── configs/            # 配置文件
│   └── train_config.yaml
├── train.py            # 训练脚本
├── test.py             # 测试脚本
└── main.py             # 主入口 (本文件)

使用方法:
1. 训练模型:
   python main.py --mode train

2. 测试模型:
   python main.py --mode test --checkpoint checkpoints/model_best.pth

3. 查看数据:
   python main.py --mode data
"""

import os
import sys
import argparse


def print_banner():
    """打印项目横幅"""
    banner = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║     FY-4B 卫星图像超分辨率 - PFT-SR                             ║
    ║                                                                  ║
    ║     Progressive Feature Transfer for Super-Resolution           ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_data_info():
    """打印数据信息"""
    print("\n" + "=" * 60)
    print("数据信息")
    print("=" * 60)
    print("""
    数据源: FY-4B卫星AGRI仪器
    
    训练数据对:
    - 2km-4km 数据对: 用于 4km->2km 超分辨率
    - 1km-2km 数据对: 用于 2km->1km 超分辨率
    
    数据位置:
    - 2km分辨率: /root/autodl-tmp/2000M/
    - 4km分辨率: /root/autodl-tmp/4000M/
    
    当前占位符:
    - 低分辨率数据: lowResfile1
    - 高分辨率数据: highResfile2
    """)


def test_dataset():
    """测试数据集"""
    print("\n" + "=" * 60)
    print("测试数据集")
    print("=" * 60)
    
    from data import FY4BDataset
    
    dataset = FY4BDataset(
        low_res_file="lowResfile1",
        high_res_file="highResfile2",
        patch_size=64,
        upscale_factor=2,
        mode='train'
    )
    
    print(f"\n数据集大小: {len(dataset)}")
    
    # 获取一个样本
    lr, hr, info = dataset[0]
    print(f"低分辨率图像形状: {lr.shape}")
    print(f"高分辨率图像形状: {hr.shape}")
    print(f"信息: {info}")


def test_model():
    """测试模型"""
    print("\n" + "=" * 60)
    print("测试模型")
    print("=" * 60)
    
    from models.pft_sr import test_model as _test_model
    _test_model()


def test_loss():
    """测试损失函数"""
    print("\n" + "=" * 60)
    print("测试损失函数")
    print("=" * 60)
    
    import torch
    from models.loss import SRLoss
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    pred = torch.randn(2, 8, 64, 64).to(device)
    target = torch.randn(2, 8, 64, 64).to(device)
    
    criterion = SRLoss().to(device)
    total, loss_dict = criterion(pred, target)
    
    print(f"\n损失详情:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.6f}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='FY-4B超分辨率项目')
    parser.add_argument('--mode', type=str, default='info',
                        choices=['info', 'data', 'model', 'loss', 'train', 'test'],
                        help='运行模式')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='模型检查点路径 (用于测试模式)')
    
    args = parser.parse_args()
    
    print_banner()
    
    if args.mode == 'info':
        print_data_info()
        print("\n请使用 --mode 参数选择运行模式:")
        print("  --mode data  : 测试数据集")
        print("  --mode model : 测试模型")
        print("  --mode loss  : 测试损失函数")
        print("  --mode train : 开始训练")
        print("  --mode test  : 测试模型")
    
    elif args.mode == 'data':
        test_dataset()
    
    elif args.mode == 'model':
        test_model()
    
    elif args.mode == 'loss':
        test_loss()
    
    elif args.mode == 'train':
        print("\n启动训练...")
        os.system("python train.py")
    
    elif args.mode == 'test':
        if args.checkpoint is None:
            print("错误: 测试模式需要提供 --checkpoint 参数")
            return
        print(f"\n启动测试: {args.checkpoint}")
        os.system(f"python test.py --checkpoint {args.checkpoint}")


if __name__ == '__main__':
    main()
