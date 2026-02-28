# -*- coding: utf-8 -*-
"""
模型检查点管理模块
"""

import os
import torch
import glob


def save_checkpoint(state, save_dir, filename='checkpoint.pth', is_best=False):
    """
    保存模型检查点
    
    Args:
        state: 包含模型状态、优化器状态等的字典
        save_dir: 保存目录
        filename: 文件名
        is_best: 是否为最佳模型
    """
    os.makedirs(save_dir, exist_ok=True)
    
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    
    if is_best:
        best_path = os.path.join(save_dir, 'model_best.pth')
        torch.save(state, best_path)
        print(f"保存最佳模型到: {best_path}")
    
    print(f"保存检查点: {filepath}")


def load_checkpoint(checkpoint_path, model, optimizer=None, device='cuda'):
    """
    加载模型检查点
    
    Args:
        checkpoint_path: 检查点文件路径
        model: 模型实例
        optimizer: 优化器实例 (可选)
        device: 设备
    
    Returns:
        start_epoch: 开始的epoch
        best_psnr: 最佳PSNR值
    """
    if not os.path.exists(checkpoint_path):
        print(f"警告: 检查点文件不存在: {checkpoint_path}")
        return 0, 0.0
    
    print(f"加载检查点: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 加载模型权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # 加载优化器状态
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint.get('epoch', 0)
    best_psnr = checkpoint.get('best_psnr', 0.0)
    
    print(f"从epoch {start_epoch}恢复训练, 最佳PSNR: {best_psnr:.2f}")
    
    return start_epoch, best_psnr


def find_last_checkpoint(save_dir, pattern='checkpoint_epoch_*.pth'):
    """
    查找最新的检查点文件
    
    Args:
        save_dir: 保存目录
        pattern: 文件名模式
    
    Returns:
        last_checkpoint: 最新检查点的路径，如果没有则返回None
    """
    checkpoint_files = glob.glob(os.path.join(save_dir, pattern))
    
    if not checkpoint_files:
        return None
    
    # 按修改时间排序
    checkpoint_files.sort(key=os.path.getmtime)
    
    return checkpoint_files[-1]


def cleanup_old_checkpoints(save_dir, keep_last=5):
    """
    清理旧的检查点文件，只保留最近N个
    
    Args:
        save_dir: 保存目录
        keep_last: 保留的检查点数量
    """
    checkpoint_files = glob.glob(os.path.join(save_dir, 'checkpoint_epoch_*.pth'))
    
    if len(checkpoint_files) <= keep_last:
        return
    
    # 按修改时间排序
    checkpoint_files.sort(key=os.path.getmtime)
    
    # 删除旧的检查点
    for old_file in checkpoint_files[:-keep_last]:
        try:
            os.remove(old_file)
            print(f"删除旧检查点: {old_file}")
        except Exception as e:
            print(f"删除检查点失败 {old_file}: {e}")
