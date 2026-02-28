# -*- coding: utf-8 -*-
"""
FY-4B卫星超分辨率测试/推理脚本

使用方法:
1. 测试单个样本:
   python test.py --checkpoint checkpoints/model_best.pth --input lowResfile1

2. 批量测试:
   python test.py --checkpoint checkpoints/model_best.pth --input-dir /path/to/low/res/data

3. 使用配置文件:
   python test.py --config configs/train_config.yaml --checkpoint checkpoints/model_best.pth
"""

import os
import sys
import argparse
import numpy as np

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))

from models import PFTSR
from utils import save_image, visualize_results, calculate_psnr, calculate_ssim


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='FY-4B超分辨率测试')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--input', type=str, default='lowResfile1',
                        help='输入的低分辨率文件')
    parser.add_argument('--input-dir', type=str, default=None,
                        help='输入的低分辨率数据目录')
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='输出结果保存目录')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备 (cuda/cpu)')
    parser.add_argument('--upscale-factor', type=int, default=2,
                        help='上采样因子')
    
    return parser.parse_args()


def load_model(checkpoint_path, config, device):
    """加载训练好的模型"""
    # 创建模型
    model = PFTSR(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        num_features=config['model']['num_features'],
        num_pft_blocks=config['model']['num_pft_blocks'],
        num_rb_per_block=config['model']['num_rb_per_block'],
        use_attention=config['model']['use_attention'],
        upscale_factor=config['data']['upscale_factor']
    )
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


def load_data(filepath, channels):
    """
    加载HDF数据
    目前使用随机数据模拟
    """
    # TODO: 实现实际的HDF数据加载
    # 使用模拟数据进行测试
    size = 64  # 低分辨率尺寸
    
    data = []
    for ch in channels:
        np.random.seed(hash(filepath + ch) % 2**32)
        ch_data = np.random.randn(size, size).astype(np.float32) * 50 + 300
        data.append(ch_data)
    
    data = np.stack(data, axis=0)  # [C, H, W]
    
    # 归一化到[-1, 1]
    min_val = 150.0
    max_val = 350.0
    data = (data - min_val) / (max_val - min_val)
    data = data * 2 - 1
    
    return torch.from_numpy(data).unsqueeze(0)  # [1, C, H, W]


def inference(model, lr_data, device):
    """
    执行超分辨率推理
    
    Args:
        model: 超分辨率模型
        lr_data: 低分辨率输入 [1, C, H, W]
        device: 计算设备
    
    Returns:
        sr_data: 超分辨率输出 [1, C, H*upscale, W*upscale]
    """
    lr_data = lr_data.to(device)
    
    with torch.no_grad():
        sr_data = model(lr_data)
    
    return sr_data


def save_results(sr_data, output_dir, filename, channels):
    """保存超分辨率结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 反归一化
    min_val = 150.0
    max_val = 350.0
    sr_data = (sr_data + 1) / 2.0  # [-1, 1] -> [0, 1]
    sr_data = sr_data * (max_val - min_val) + min_val  # [0, 1] -> [min_val, max_val]
    
    sr_np = sr_data.squeeze(0).cpu().numpy()  # [C, H, W]
    
    # 保存每个通道的图像
    for i, ch_name in enumerate(channels):
        if i >= sr_np.shape[0]:
            break
        
        ch_img = sr_np[i]
        
        # 保存为图像
        img_path = os.path.join(output_dir, f'{filename}_{ch_name}.png')
        save_image(ch_img, img_path, title=f'{ch_name} - SR Result', cmap='jet')
        
        # 保存为numpy数组
        npy_path = os.path.join(output_dir, f'{filename}_{ch_name}.npy')
        np.save(npy_path, ch_img)
    
    # 保存完整的超分辨率结果
    full_path = os.path.join(output_dir, f'{filename}_sr.npy')
    np.save(full_path, sr_np)
    print(f"保存完整结果: {full_path}")


def bicubic_upsample(lr_data, upscale_factor):
    """双三次插值上采样 (作为基准)"""
    return F.interpolate(
        lr_data, 
        scale_factor=upscale_factor,
        mode='bicubic',
        align_corners=False
    )


def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"\n加载模型: {args.checkpoint}")
    model = load_model(args.checkpoint, config, device)
    print("模型加载完成")
    
    # 获取通道列表
    channels = config['data']['channels']
    
    # 处理单个文件
    if args.input:
        print(f"\n处理输入: {args.input}")
        
        # 加载数据
        lr_data = load_data(args.input, channels)
        print(f"输入形状: {lr_data.shape}")
        
        # 执行推理
        sr_data = inference(model, lr_data, device)
        print(f"输出形状: {sr_data.shape}")
        
        # 双三次插值基准
        bicubic_sr = bicubic_upsample(lr_data, config['data']['upscale_factor'])
        
        # 保存结果
        filename = os.path.splitext(os.path.basename(args.input))[0]
        save_results(sr_data, args.output_dir, filename, channels)
        
        # 可视化对比
        print("\n生成可视化对比...")
        from utils.visualize import plot_comparison
        
        comparison_path = os.path.join(args.output_dir, f'{filename}_comparison.png')
        plot_comparison(sr_data[0], bicubic_sr[0], sr_data[0], comparison_path, channel_idx=0)
        
        print(f"\n结果已保存到: {args.output_dir}")
    
    # 处理整个目录
    if args.input_dir:
        print(f"\n批量处理目录: {args.input_dir}")
        # TODO: 实现批量处理逻辑
        pass
    
    print("\n测试完成!")


if __name__ == '__main__':
    main()
