# -*- coding: utf-8 -*-
"""
可视化工具模块
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt


def denormalize(tensor, min_val=150.0, max_val=350.0):
    """
    反归一化: 将[-1, 1]范围的数据恢复到原始范围
    
    Args:
        tensor: 归一化后的张量
        min_val: 原始最小值
        max_val: 原始最大值
    
    Returns:
        denormalized: 反归一化后的数据
    """
    tensor = (tensor + 1) / 2.0  # [-1, 1] -> [0, 1]
    tensor = tensor * (max_val - min_val) + min_val  # [0, 1] -> [min_val, max_val]
    return tensor


def save_image(tensor, save_path, title=None, cmap='jet', vmin=None, vmax=None):
    """
    保存张量为图像
    
    Args:
        tensor: 输入张量 [H, W] 或 [C, H, W] 或 [B, C, H, W]
        save_path: 保存路径
        title: 图像标题
        cmap: 颜色映射
        vmin, vmax: 颜色范围
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 转换为numpy
    if isinstance(tensor, np.ndarray):
        img = tensor
    else:
        img = tensor.detach().cpu().numpy()
    
    # 处理不同维度
    if img.ndim == 4:  # [B, C, H, W]
        img = img[0]  # 取第一个batch
    if img.ndim == 3:  # [C, H, W]
        img = img[0] if img.shape[0] in [1, 3] else img  # 取第一个通道或RGB
    
    # 确保是2D
    img = img.squeeze()
    
    # 绘制图像
    plt.figure(figsize=(8, 6))
    plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()
    if title:
        plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"保存图像: {save_path}")


def visualize_results(lr, sr, hr, save_dir, epoch, idx=0, channel_names=None):
    """
    可视化超分辨率结果
    
    Args:
        lr: 低分辨率图像 [B, C, H, W]
        sr: 超分辨率图像 [B, C, H, W]
        hr: 高分辨率图像 [B, C, H, W]
        save_dir: 保存目录
        epoch: 当前epoch
        idx: 样本索引
        channel_names: 通道名称列表
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if channel_names is None:
        channel_names = [f'Ch{i}' for i in range(lr.size(1))]
    
    # 转换为numpy并反归一化
    lr_np = denormalize(lr[idx].detach().cpu()).numpy()
    sr_np = denormalize(sr[idx].detach().cpu()).numpy()
    hr_np = denormalize(hr[idx].detach().cpu()).numpy()
    
    # 为每个通道创建可视化
    num_channels = lr.size(1)
    num_display = min(num_channels, 8)  # 最多显示8个通道
    
    fig, axes = plt.subplots(num_display, 3, figsize=(12, 4*num_display))
    
    for ch in range(num_display):
        # 计算显示范围
        vmin = min(lr_np[ch].min(), sr_np[ch].min(), hr_np[ch].min())
        vmax = max(lr_np[ch].max(), sr_np[ch].max(), hr_np[ch].max())
        
        if num_display == 1:
            ax_row = [axes[0], axes[1], axes[2]]
        else:
            ax_row = axes[ch]
        
        # 低分辨率 (双三次插值上采样以便比较)
        from scipy.ndimage import zoom
        lr_upscaled = zoom(lr_np[ch], sr_np.shape[-1] / lr_np.shape[-1], order=1)
        ax_row[0].imshow(lr_upscaled, cmap='jet', vmin=vmin, vmax=vmax)
        ax_row[0].set_title(f'{channel_names[ch]} - LR (Bicubic)')
        ax_row[0].axis('off')
        
        # 超分辨率结果
        ax_row[1].imshow(sr_np[ch], cmap='jet', vmin=vmin, vmax=vmax)
        ax_row[1].set_title(f'{channel_names[ch]} - SR (Ours)')
        ax_row[1].axis('off')
        
        # 高分辨率真值
        ax_row[2].imshow(hr_np[ch], cmap='jet', vmin=vmin, vmax=vmax)
        ax_row[2].set_title(f'{channel_names[ch]} - HR (GT)')
        ax_row[2].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'epoch_{epoch}_sample_{idx}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"保存可视化结果: {save_path}")


def plot_training_curves(history, save_path):
    """
    绘制训练曲线
    
    Args:
        history: 训练历史字典，包含loss、psnr、ssim等
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs = range(1, len(history['loss']) + 1)
    
    # Loss曲线
    axes[0].plot(epochs, history['loss'], 'b-', label='Train Loss')
    if 'val_loss' in history:
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # PSNR曲线
    if 'val_psnr' in history:
        axes[1].plot(epochs, history['val_psnr'], 'g-', label='PSNR')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('PSNR (dB)')
        axes[1].set_title('Validation PSNR')
        axes[1].legend()
        axes[1].grid(True)
    
    # SSIM曲线
    if 'val_ssim' in history:
        axes[2].plot(epochs, history['val_ssim'], 'm-', label='SSIM')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('SSIM')
        axes[2].set_title('Validation SSIM')
        axes[2].legend()
        axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"保存训练曲线: {save_path}")


def plot_comparison(sr_img, bicubic_img, hr_img, save_path, channel_idx=0):
    """
    绘制SR、Bicubic和HR的对比图
    
    Args:
        sr_img: 超分辨率结果
        bicubic_img: 双三次插值结果
        hr_img: 高分辨率真值
        save_path: 保存路径
        channel_idx: 通道索引
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 反归一化
    sr_np = denormalize(sr_img[channel_idx].detach().cpu()).numpy()
    bicubic_np = denormalize(bicubic_img[channel_idx].detach().cpu()).numpy()
    hr_np = denormalize(hr_img[channel_idx].detach().cpu()).numpy()
    
    vmin = min(sr_np.min(), bicubic_np.min(), hr_np.min())
    vmax = max(sr_np.max(), bicubic_np.max(), hr_np.max())
    
    axes[0].imshow(bicubic_np, cmap='jet', vmin=vmin, vmax=vmax)
    axes[0].set_title('Bicubic Interpolation')
    axes[0].axis('off')
    
    axes[1].imshow(sr_np, cmap='jet', vmin=vmin, vmax=vmax)
    axes[1].set_title('PFT-SR (Ours)')
    axes[1].axis('off')
    
    axes[2].imshow(hr_np, cmap='jet', vmin=vmin, vmax=vmax)
    axes[2].set_title('Ground Truth')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"保存对比图: {save_path}")
