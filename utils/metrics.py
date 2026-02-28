# -*- coding: utf-8 -*-
"""
评估指标模块
包含PSNR、SSIM等超分辨率评估指标
"""

import torch
import torch.nn.functional as F
import numpy as np
from math import log10


def calculate_psnr(img1, img2, max_val=1.0):
    """
    计算PSNR (Peak Signal-to-Noise Ratio)
    
    Args:
        img1: 预测图像
        img2: 目标图像
        max_val: 像素最大值 (归一化图像通常为1.0)
    
    Returns:
        psnr: PSNR值 (dB)
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    psnr = 20 * log10(max_val) - 10 * log10(mse.item())
    return psnr


def calculate_ssim(img1, img2, window_size=11, size_average=True):
    """
    计算SSIM (Structural Similarity Index)
    
    Args:
        img1: 预测图像 [B, C, H, W]
        img2: 目标图像 [B, C, H, W]
        window_size: 滑动窗口大小
        size_average: 是否对所有通道求平均
    
    Returns:
        ssim: SSIM值
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # 创建高斯窗口
    sigma = 1.5
    gauss = torch.Tensor([
        np.exp(-(x - window_size//2)**2 / float(2*sigma**2))
        for x in range(window_size)
    ])
    gauss = gauss / gauss.sum()
    
    window_1d = gauss.unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
    
    window = window_2d.expand(img1.size(1), 1, window_size, window_size).contiguous()
    window = window.to(img1.device).type_as(img1)
    
    # 计算均值
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=img1.size(1))
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=img2.size(1))
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # 计算方差和协方差
    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=img2.size(1)) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=img1.size(1)) - mu1_mu2
    
    # 计算SSIM
    ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def evaluate_model(model, dataloader, device):
    """
    评估模型性能
    
    Args:
        model: 超分辨率模型
        dataloader: 数据加载器
        device: 计算设备
    
    Returns:
        metrics: 包含PSNR和SSIM平均值的字典
    """
    model.eval()
    
    total_psnr = 0.0
    total_ssim = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for batch_idx, (lr, hr, _) in enumerate(dataloader):
            lr = lr.to(device)
            hr = hr.to(device)
            
            # 前向传播
            sr = model(lr)
            
            # 计算指标
            batch_psnr = 0.0
            batch_ssim = 0.0
            
            for i in range(sr.size(0)):
                batch_psnr += calculate_psnr(sr[i:i+1], hr[i:i+1])
                batch_ssim += calculate_ssim(sr[i:i+1], hr[i:i+1])
            
            total_psnr += batch_psnr
            total_ssim += batch_ssim
            num_samples += sr.size(0)
    
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    
    return {
        'psnr': avg_psnr,
        'ssim': avg_ssim
    }


def calculate_channel_metrics(pred, target, channel_names):
    """
    计算每个通道的评估指标
    
    Args:
        pred: 预测图像 [B, C, H, W]
        target: 目标图像 [B, C, H, W]
        channel_names: 通道名称列表
    
    Returns:
        channel_metrics: 每个通道的指标字典
    """
    channel_metrics = {}
    
    for i, name in enumerate(channel_names):
        ch_pred = pred[:, i:i+1]
        ch_target = target[:, i:i+1]
        
        psnr = calculate_psnr(ch_pred, ch_target)
        ssim = calculate_ssim(ch_pred, ch_target)
        
        channel_metrics[name] = {
            'psnr': psnr,
            'ssim': ssim
        }
    
    return channel_metrics


def calculate_rmse(pred, target):
    """
    计算RMSE (Root Mean Square Error)
    """
    mse = torch.mean((pred - target) ** 2)
    rmse = torch.sqrt(mse)
    return rmse.item()


def calculate_mae(pred, target):
    """
    计算MAE (Mean Absolute Error)
    """
    mae = torch.mean(torch.abs(pred - target))
    return mae.item()


if __name__ == '__main__':
    # 测试指标计算
    print("=" * 50)
    print("测试评估指标")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据
    img1 = torch.randn(2, 8, 64, 64).to(device)
    img2 = img1 + torch.randn(2, 8, 64, 64).to(device) * 0.1
    
    # 计算PSNR
    psnr = calculate_psnr(img1, img2)
    print(f"PSNR: {psnr:.2f} dB")
    
    # 计算SSIM
    ssim = calculate_ssim(img1, img2)
    print(f"SSIM: {ssim:.4f}")
    
    # 计算RMSE和MAE
    rmse = calculate_rmse(img1, img2)
    mae = calculate_mae(img1, img2)
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    print("\n✓ 指标计算测试通过!")
