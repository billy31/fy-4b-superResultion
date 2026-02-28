# -*- coding: utf-8 -*-
"""
FY-4B卫星超分辨率训练脚本
基于PFT-SR方法

使用方法:
1. 使用默认配置训练:
   python train.py

2. 使用自定义配置:
   python train.py --config configs/train_config.yaml

3. 从检查点恢复训练:
   python train.py --resume checkpoints/checkpoint_epoch_100.pth
"""

import os
import sys
import time
import argparse
import random
import numpy as np
import yaml
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))

from models import PFTSR, SRLoss
from data import create_dataloaders
from utils import (
    evaluate_model, 
    save_checkpoint, load_checkpoint, cleanup_old_checkpoints,
    visualize_results, plot_training_curves
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='FY-4B超分辨率训练')
    
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    parser.add_argument('--low-res-file', type=str, default='lowResfile1',
                        help='低分辨率数据文件占位符')
    parser.add_argument('--high-res-file', type=str, default='highResfile2',
                        help='高分辨率数据文件占位符')
    
    return parser.parse_args()


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(config):
    """创建模型"""
    model_cfg = config['model']
    
    model = PFTSR(
        in_channels=model_cfg['in_channels'],
        out_channels=model_cfg['out_channels'],
        num_features=model_cfg['num_features'],
        num_pft_blocks=model_cfg['num_pft_blocks'],
        num_rb_per_block=model_cfg['num_rb_per_block'],
        use_attention=model_cfg['use_attention'],
        upscale_factor=config['data']['upscale_factor']
    )
    
    return model


def create_optimizer(model, config):
    """创建优化器"""
    opt_cfg = config['optimizer']
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=opt_cfg['lr'],
        betas=tuple(opt_cfg['betas']),
        weight_decay=opt_cfg['weight_decay']
    )
    
    return optimizer


def create_scheduler(optimizer, config):
    """创建学习率调度器"""
    sched_cfg = config['scheduler']
    
    if sched_cfg['name'] == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=sched_cfg['milestones'],
            gamma=sched_cfg['gamma']
        )
    elif sched_cfg['name'] == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['num_epochs']
        )
    else:
        scheduler = None
    
    return scheduler


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, config, writer=None):
    """训练一个epoch"""
    model.train()
    
    total_loss = 0.0
    loss_dict_sum = {}
    num_batches = len(train_loader)
    
    start_time = time.time()
    
    for batch_idx, (lr, hr, info) in enumerate(train_loader):
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)
        
        # 前向传播
        optimizer.zero_grad()
        sr = model(lr)
        
        # 计算损失
        loss, loss_dict = criterion(sr, hr)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        if config['training']['grad_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
        
        optimizer.step()
        
        # 记录损失
        total_loss += loss.item()
        for k, v in loss_dict.items():
            loss_dict_sum[k] = loss_dict_sum.get(k, 0.0) + v
        
        # 打印进度
        if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
            elapsed = time.time() - start_time
            print(f"  Batch [{batch_idx+1}/{num_batches}] "
                  f"Loss: {loss.item():.6f} "
                  f"Time: {elapsed:.2f}s")
        
        # TensorBoard记录
        global_step = epoch * num_batches + batch_idx
        if writer is not None and (batch_idx + 1) % 50 == 0:
            for k, v in loss_dict.items():
                writer.add_scalar(f'Loss/batch_{k}', v, global_step)
    
    # 计算平均损失
    avg_loss = total_loss / num_batches
    avg_loss_dict = {k: v / num_batches for k, v in loss_dict_sum.items()}
    
    return avg_loss, avg_loss_dict


def validate(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    
    total_loss = 0.0
    loss_dict_sum = {}
    
    with torch.no_grad():
        for lr, hr, info in val_loader:
            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)
            
            # 前向传播
            sr = model(lr)
            
            # 计算损失
            loss, loss_dict = criterion(sr, hr)
            
            total_loss += loss.item()
            for k, v in loss_dict.items():
                loss_dict_sum[k] = loss_dict_sum.get(k, 0.0) + v
    
    # 计算平均损失
    num_batches = len(val_loader)
    avg_loss = total_loss / num_batches
    avg_loss_dict = {k: v / num_batches for k, v in loss_dict_sum.items()}
    
    # 计算评估指标
    metrics = evaluate_model(model, val_loader, device)
    
    return avg_loss, avg_loss_dict, metrics


def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 更新配置中的文件占位符
    config['data']['low_res_file'] = args.low_res_file
    config['data']['high_res_file'] = args.high_res_file
    
    # 设置随机种子
    set_seed(config.get('seed', 42))
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and config['device']['use_cuda'] else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(config['output']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['output']['log_dir'], exist_ok=True)
    os.makedirs(config['output']['viz_dir'], exist_ok=True)
    
    # 创建TensorBoard writer
    writer = SummaryWriter(log_dir=config['output']['log_dir'])
    
    # 创建数据加载器
    print("\n正在创建数据加载器...")
    train_loader, val_loader = create_dataloaders(
        low_res_file=config['data']['low_res_file'],
        high_res_file=config['data']['high_res_file'],
        low_res_dir=config['data'].get('low_res_dir'),
        high_res_dir=config['data'].get('high_res_dir'),
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        patch_size=config['data']['patch_size'],
        upscale_factor=config['data']['upscale_factor'],
        channels=config['data']['channels']
    )
    print(f"训练样本数: {len(train_loader.dataset)}")
    print(f"验证样本数: {len(val_loader.dataset)}")
    
    # 创建模型
    print("\n正在创建模型...")
    model = create_model(config)
    model = model.to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"可训练参数量: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # 创建损失函数
    criterion = SRLoss(
        lambda_l1=config['loss']['lambda_l1'],
        lambda_ssim=config['loss']['lambda_ssim'],
        lambda_freq=config['loss']['lambda_freq'],
        lambda_grad=config['loss']['lambda_grad']
    ).to(device)
    
    # 创建优化器
    optimizer = create_optimizer(model, config)
    
    # 创建学习率调度器
    scheduler = create_scheduler(optimizer, config)
    
    # 恢复训练
    start_epoch = 0
    best_psnr = 0.0
    history = {
        'loss': [],
        'val_loss': [],
        'val_psnr': [],
        'val_ssim': []
    }
    
    if args.resume:
        start_epoch, best_psnr = load_checkpoint(
            args.resume, model, optimizer, device
        )
        start_epoch += 1
    
    print("\n" + "=" * 60)
    print(f"开始训练: {config['experiment_name']}")
    print(f"训练epoch: {start_epoch} -> {config['training']['num_epochs']}")
    print(f"批次大小: {config['data']['batch_size']}")
    print(f"初始学习率: {config['optimizer']['lr']}")
    print("=" * 60 + "\n")
    
    # 训练循环
    for epoch in range(start_epoch, config['training']['num_epochs']):
        epoch_start_time = time.time()
        
        print(f"Epoch [{epoch+1}/{config['training']['num_epochs']}]")
        
        # 训练
        train_loss, train_loss_dict = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config, writer
        )
        
        # 更新学习率
        if scheduler is not None:
            scheduler.step()
        
        # 记录训练指标
        history['loss'].append(train_loss)
        print(f"  Train Loss: {train_loss:.6f}")
        for k, v in train_loss_dict.items():
            if k != 'total':
                print(f"    {k}: {v:.6f}")
            writer.add_scalar(f'Loss/train_{k}', v, epoch)
        
        # 验证
        if (epoch + 1) % config['training']['val_interval'] == 0:
            val_loss, val_loss_dict, metrics = validate(model, val_loader, criterion, device)
            
            history['val_loss'].append(val_loss)
            history['val_psnr'].append(metrics['psnr'])
            history['val_ssim'].append(metrics['ssim'])
            
            print(f"  Val Loss: {val_loss:.6f}")
            print(f"  Val PSNR: {metrics['psnr']:.2f} dB")
            print(f"  Val SSIM: {metrics['ssim']:.4f}")
            
            writer.add_scalar('Loss/val_total', val_loss, epoch)
            writer.add_scalar('Metrics/PSNR', metrics['psnr'], epoch)
            writer.add_scalar('Metrics/SSIM', metrics['ssim'], epoch)
            
            # 保存最佳模型
            if metrics['psnr'] > best_psnr:
                best_psnr = metrics['psnr']
                is_best = True
                print(f"  *** 新的最佳PSNR: {best_psnr:.2f} dB ***")
            else:
                is_best = False
            
            # 可视化结果
            if (epoch + 1) % config['training']['viz_interval'] == 0:
                model.eval()
                with torch.no_grad():
                    lr, hr, info = next(iter(val_loader))
                    lr = lr.to(device)
                    hr = hr.to(device)
                    sr = model(lr)
                    
                    visualize_results(
                        lr, sr, hr,
                        save_dir=config['output']['viz_dir'],
                        epoch=epoch + 1,
                        idx=0,
                        channel_names=config['data']['channels']
                    )
        else:
            is_best = False
        
        # 保存检查点
        if (epoch + 1) % config['training']['save_interval'] == 0 or is_best:
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_psnr': best_psnr,
                'config': config
            }
            
            filename = f'checkpoint_epoch_{epoch+1}.pth'
            save_checkpoint(
                state,
                save_dir=config['output']['checkpoint_dir'],
                filename=filename,
                is_best=is_best
            )
            
            # 清理旧检查点
            cleanup_old_checkpoints(
                config['output']['checkpoint_dir'],
                keep_last=config['training']['keep_last_n']
            )
        
        # 打印epoch时间
        epoch_time = time.time() - epoch_start_time
        print(f"  Epoch time: {epoch_time:.2f}s\n")
    
    # 保存最终模型
    final_path = os.path.join(config['output']['checkpoint_dir'], 'model_final.pth')
    torch.save(model.state_dict(), final_path)
    print(f"保存最终模型: {final_path}")
    
    # 绘制训练曲线
    plot_path = os.path.join(config['output']['viz_dir'], 'training_curves.png')
    plot_training_curves(history, plot_path)
    
    writer.close()
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最佳PSNR: {best_psnr:.2f} dB")
    print("=" * 60)


if __name__ == '__main__':
    main()
