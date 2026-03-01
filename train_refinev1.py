# -*- coding: utf-8 -*-
"""
FY-4B卫星超分辨率训练脚本 - 优化版本 (RefineV1)
基于PFT-SR方法

优化内容:
1. 调整损失函数权重 (降低SSIM和FREQ权重，避免梯度主导)
2. Warmup + Cosine Annealing学习率调度
3. 减小batch size增加batch数量，提高训练稳定性
4. 增加验证频率 (每5个epoch)，更快发现问题
5. 优化的数据增强策略
6. 调整学习率 (降低初始学习率)

使用方法:
    python train_refinev1.py <高分辨率> <波段号>
    
示例:
    python train_refinev1.py 2000M CH07    # 训练 4km->2km，CH07通道
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
    visualize_results
)
from utils.visualize import plot_training_curves


# 优化后的基础配置
BASE_CONFIG = {
    'experiment_name': 'FY4B_PFTSR_RefineV1',
    'data': {
        'base_dir': '/root/autodl-tmp/Calibration-FY4B',
        'patch_size': 64,        # 低分辨率空间裁剪大小
        'upscale_factor': 2,     # 上采样因子 (固定2x)
        'batch_size': 4,         # 减小batch size，增加batch数量
        'num_workers': 4,
        'pin_memory': True,
    },
    'model': {
        'name': 'PFTSR',
        'in_channels': 1,        # 单通道输入
        'out_channels': 1,       # 单通道输出
        'num_features': 64,
        'num_pft_blocks': 3,
        'num_rb_per_block': 3,
        'use_attention': True,
    },
    # 优化: 调整损失函数权重
    'loss': {
        'lambda_l1': 1.0,        # L1保持主导
        'lambda_ssim': 0.1,      # 降低SSIM权重 (原为0.5)
        'lambda_freq': 0.01,     # 大幅降低频域损失权重 (原为0.1)
        'lambda_grad': 0.05,     # 适当降低梯度损失 (原为0.1)
    },
    'optimizer': {
        'name': 'Adam',
        'lr': 5e-5,              # 降低初始学习率 (原为1e-4)
        'betas': [0.9, 0.999],
        'weight_decay': 1e-4,    # 适当降低weight decay
    },
    # 优化: 使用Warmup + Cosine Annealing
    'scheduler': {
        'name': 'CosineWarmup',  # Warmup + Cosine Annealing
        'warmup_epochs': 10,     # 前10个epoch warmup
        'min_lr': 1e-7,          # 最小学习率
    },
    'training': {
        'num_epochs': 500,
        'grad_clip': 1.0,        # 梯度裁剪阈值
        'val_interval': 5,       # 优化: 每5个epoch验证一次 (原为10)
        'viz_interval': 50,      # 每50个epoch可视化一次
        'save_interval': 50,     # 每50个epoch保存一次
        'keep_last_n': 5,
        'early_stopping': {
            'enabled': True,
            'patience': 15,        # 增加到15次验证 (更宽容)
            'metric': 'psnr',
            'mode': 'max',
            'min_delta': 0.01,     # 最小提升阈值0.01dB
        },
    },
    'evaluation': {
        'metrics': ['psnr', 'ssim', 'rmse'],
        'save_predictions': True,
    },
    'device': {
        'use_cuda': True,
        'gpu_ids': [0],
    },
    'seed': 42,
}


class WarmupCosineScheduler:
    """
    Warmup + Cosine Annealing 学习率调度器
    
    前warmup_epochs个epoch线性增加学习率从0到初始值
    之后使用Cosine Annealing衰减到min_lr
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-7, base_lr=5e-5):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = base_lr
        self.current_epoch = 0
        
    def step(self):
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            # Warmup阶段: 线性增加
            warmup_factor = self.current_epoch / self.warmup_epochs
            lr = self.base_lr * warmup_factor
        else:
            # Cosine Annealing阶段
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_last_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='FY-4B超分辨率训练 (优化版本 RefineV1)',
        usage='python train_refinev1.py <高分辨率> <波段号> [选项]'
    )
    
    # 位置参数
    parser.add_argument('high_res', type=str, choices=['2000M', '1000M'],
                        help='目标高分辨率 (2000M 或 1000M)')
    parser.add_argument('band', type=str, choices=['CH07', 'CH08'],
                        help='通道号 (CH07 或 CH08)')
    
    # 可选参数
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练epoch数 (覆盖默认值)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='批次大小 (默认4，增加batch数量)')
    parser.add_argument('--lr', type=float, default=None,
                        help='学习率 (默认5e-5)')
    parser.add_argument('--patience', type=int, default=15,
                        help='早停容忍轮数 (默认15次验证)')
    
    return parser.parse_args()


def build_config(args):
    """根据命令行参数构建配置"""
    config = BASE_CONFIG.copy()
    
    # 解析分辨率
    high_res = args.high_res
    if high_res == '2000M':
        low_res = '4000M'     # 4km -> 2km
    elif high_res == '1000M':
        low_res = '2000M'     # 2km -> 1km
    else:
        raise ValueError(f"不支持的分辨率: {high_res}")
    
    # 解析波段
    band = args.band
    channel_name = band.replace('CH', 'Channel')
    
    # 更新配置
    config['experiment_name'] = f'FY4B_RefineV1_{low_res}_to_{high_res}_{band}'
    config['data']['high_res'] = high_res
    config['data']['low_res'] = low_res
    config['data']['band'] = band
    config['data']['channel'] = channel_name
    config['data']['low_res_dir'] = f"{config['data']['base_dir']}/{low_res}/{band}"
    config['data']['high_res_dir'] = f"{config['data']['base_dir']}/{high_res}/{band}"
    
    # 输出目录
    output_base = f"{config['data']['base_dir']}/trained_model_refinev1/{high_res}_{band}"
    config['output'] = {
        'checkpoint_dir': f'{output_base}/checkpoints',
        'log_dir': f'{output_base}/logs',
        'viz_dir': f'{output_base}/visualizations',
    }
    
    # 覆盖参数
    if args.epochs is not None:
        config['training']['num_epochs'] = args.epochs
        config['scheduler']['total_epochs'] = args.epochs
    if args.batch_size is not None:
        config['data']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['optimizer']['lr'] = args.lr
        config['scheduler']['base_lr'] = args.lr
    if args.patience is not None:
        config['training']['early_stopping']['patience'] = args.patience
    
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
    """创建学习率调度器 (Warmup + Cosine Annealing)"""
    sched_cfg = config['scheduler']
    
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_epochs=sched_cfg['warmup_epochs'],
        total_epochs=config['training']['num_epochs'],
        min_lr=sched_cfg['min_lr'],
        base_lr=sched_cfg.get('base_lr', config['optimizer']['lr'])
    )
    
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
        
        # 梯度裁剪 (确保梯度稳定性)
        if config['training']['grad_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config['training']['grad_clip']
            )
        
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


class EarlyStopping:
    """早停模块"""
    
    def __init__(self, patience=15, metric='psnr', mode='max', min_delta=0.01):
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.min_delta = min_delta
        
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        
        if mode == 'max':
            self.is_better = lambda new, best: new > best + min_delta
            self.best_value = float('-inf')
        else:
            self.is_better = lambda new, best: new < best - min_delta
            self.best_value = float('inf')
    
    def __call__(self, metrics):
        if self.metric not in metrics:
            return False, False
        
        current_value = metrics[self.metric]
        
        if self.is_better(current_value, self.best_value):
            self.best_value = current_value
            self.counter = 0
            return False, True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True, False
            return False, False
    
    def get_status(self):
        return {
            'counter': self.counter,
            'patience': self.patience,
            'best_value': self.best_value,
            'early_stop': self.early_stop,
        }


def main():
    """主函数"""
    args = parse_args()
    
    # 构建配置
    config = build_config(args)
    
    # 设置随机种子
    set_seed(config.get('seed', 42))
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and config['device']['use_cuda'] else 'cpu')
    
    # 打印训练信息
    print("=" * 75)
    print(f"FY-4B 超分辨率训练 - 优化版本 (RefineV1)")
    print("=" * 75)
    print(f"任务: {config['data']['low_res']} -> {config['data']['high_res']} ({args.high_res})")
    print(f"波段: {args.band} ({config['data']['channel']})")
    print(f"上采样倍数: {config['data']['upscale_factor']}x")
    print(f"低分辨率数据: {config['data']['low_res_dir']}")
    print(f"高分辨率数据: {config['data']['high_res_dir']}")
    print(f"输出目录: {config['output']['checkpoint_dir']}")
    print(f"设备: {device}")
    print("-" * 75)
    print("【优化配置】")
    print(f"  • 损失权重: L1={config['loss']['lambda_l1']}, "
          f"SSIM={config['loss']['lambda_ssim']}, "
          f"FREQ={config['loss']['lambda_freq']}, "
          f"GRAD={config['loss']['lambda_grad']}")
    print(f"  • 学习率: {config['optimizer']['lr']} (Warmup {config['scheduler']['warmup_epochs']}epoch + Cosine)")
    print(f"  • Batch Size: {config['data']['batch_size']} (增加batch数量)")
    print(f"  • 验证间隔: 每{config['training']['val_interval']}个epoch")
    print(f"  • 梯度裁剪: {config['training']['grad_clip']}")
    print("=" * 75)
    
    # 创建输出目录
    os.makedirs(config['output']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['output']['log_dir'], exist_ok=True)
    os.makedirs(config['output']['viz_dir'], exist_ok=True)
    
    # 保存配置
    config_save_path = os.path.join(config['output']['checkpoint_dir'], 'config.yaml')
    with open(config_save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"\n配置已保存: {config_save_path}")
    
    # 创建TensorBoard writer
    tb_log_dir = f"./tf-logs/{config['experiment_name']}"
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_dir)
    print(f"TensorBoard日志: {tb_log_dir}")
    
    # 创建数据加载器
    print("\n正在创建数据加载器...")
    train_loader, val_loader = create_dataloaders(
        low_res_dir=config['data']['low_res_dir'],
        high_res_dir=config['data']['high_res_dir'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        patch_size=config['data']['patch_size'],
        upscale_factor=config['data']['upscale_factor'],
        channel=config['data']['channel']
    )
    print(f"训练样本数: {len(train_loader.dataset)}")
    print(f"验证样本数: {len(val_loader.dataset)}")
    print(f"每epoch batch数: {len(train_loader)} (训练), {len(val_loader)} (验证)")
    
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
        # 恢复scheduler状态
        for _ in range(start_epoch):
            scheduler.step()
        print(f"\n从检查点恢复: {args.resume}")
        print(f"起始epoch: {start_epoch}, 当前最佳PSNR: {best_psnr:.2f} dB")
    
    # 初始化早停模块
    early_stopping_cfg = config['training'].get('early_stopping', {})
    early_stopping = None
    if early_stopping_cfg.get('enabled', False):
        early_stopping = EarlyStopping(
            patience=early_stopping_cfg.get('patience', 15),
            metric=early_stopping_cfg.get('metric', 'psnr'),
            mode=early_stopping_cfg.get('mode', 'max'),
            min_delta=early_stopping_cfg.get('min_delta', 0.01)
        )
        print(f"早停已启用: patience={early_stopping.patience}, "
              f"metric={early_stopping.metric}, min_delta={early_stopping.min_delta}")
    
    print("\n" + "=" * 75)
    print(f"开始训练: {config['experiment_name']}")
    print(f"训练epoch: {start_epoch} -> {config['training']['num_epochs']}")
    print(f"批次大小: {config['data']['batch_size']} (每epoch {len(train_loader)} batches)")
    print(f"初始学习率: {config['optimizer']['lr']}")
    print(f"早停patience: {config['training']['early_stopping']['patience']} 次验证")
    print("=" * 75)
    print(f"\n查看TensorBoard: tensorboard --logdir=./tf-logs --port=6016")
    print("=" * 75 + "\n")
    
    # 训练循环
    for epoch in range(start_epoch, config['training']['num_epochs']):
        epoch_start_time = time.time()
        
        # 更新学习率
        current_lr = scheduler.step()
        
        print(f"Epoch [{epoch+1}/{config['training']['num_epochs']}] "
              f"LR: {current_lr:.2e}")
        
        # 训练
        train_loss, train_loss_dict = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config, writer
        )
        
        # 记录训练指标
        history['loss'].append(train_loss)
        print(f"  Train Loss: {train_loss:.6f}")
        for k, v in train_loss_dict.items():
            if k != 'total':
                print(f"    {k}: {v:.6f}")
            writer.add_scalar(f'Loss/train_{k}', v, epoch)
        writer.add_scalar('Train/lr', current_lr, epoch)
        
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
            
            # 检查早停条件
            if early_stopping is not None:
                should_stop, is_best_es = early_stopping(metrics)
                status = early_stopping.get_status()
                print(f"  EarlyStopping: {status['counter']}/{status['patience']}")
                
                if should_stop:
                    print(f"\n{'=' * 75}")
                    print(f"早停触发! 连续 {early_stopping.patience} 次验证无提升")
                    print(f"最佳 {early_stopping.metric.upper()}: {status['best_value']:.4f}")
                    print(f"{'=' * 75}\n")
                    break
            
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
                        channel_names=[config['data']['channel']]
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
    
    # 判断是否因为早停而结束
    stopped_early = early_stopping is not None and early_stopping.early_stop
    
    print("\n" + "=" * 75)
    if stopped_early:
        print("训练提前结束 (早停触发)")
    else:
        print("训练完成!")
    print(f"最佳PSNR: {best_psnr:.2f} dB")
    print(f"模型保存: {config['output']['checkpoint_dir']}")
    print(f"\n查看训练日志: tensorboard --logdir=./tf-logs --port=6016")
    print("=" * 75)


if __name__ == '__main__':
    main()
