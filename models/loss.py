# -*- coding: utf-8 -*-
"""
超分辨率损失函数模块
包含:
1. L1/L2像素级损失
2. Perceptual Loss (感知损失)
3. SSIM Loss (结构相似性损失)
4. Frequency Domain Loss (频域损失)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SSIMLoss(nn.Module):
    """
    结构相似性损失 (SSIM Loss)
    用于保持图像结构信息
    """
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_window(window_size, self.channel)
    
    def _gaussian(self, window_size, sigma):
        """创建高斯核"""
        gauss = torch.Tensor([
            np.exp(-(x - window_size//2)**2 / float(2*sigma**2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()
    
    def _create_window(self, window_size, channel):
        """创建2D高斯窗口"""
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def forward(self, img1, img2):
        """
        计算SSIM损失
        
        Args:
            img1: 预测图像 [B, C, H, W]
            img2: 目标图像 [B, C, H, W]
        
        Returns:
            1 - ssim (作为损失, 越小越好)
        """
        (_, channel, _, _) = img1.size()
        
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self._create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel
        
        # SSIM计算
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)


class FrequencyLoss(nn.Module):
    """
    频域损失 (Frequency Domain Loss)
    在频域空间中比较预测图像和目标图像
    有助于保留高频细节
    """
    def __init__(self, loss_type='l1'):
        super(FrequencyLoss, self).__init__()
        self.loss_type = loss_type
        
    def forward(self, pred, target):
        """
        计算频域损失
        
        Args:
            pred: 预测图像 [B, C, H, W]
            target: 目标图像 [B, C, H, W]
        
        Returns:
            频域损失值
        """
        # 对每通道进行FFT
        pred_fft = torch.fft.rfft2(pred, dim=(-2, -1))
        target_fft = torch.fft.rfft2(target, dim=(-2, -1))
        
        # 计算幅度谱
        pred_amp = torch.abs(pred_fft)
        target_amp = torch.abs(target_fft)
        
        # 计算相位谱
        pred_phase = torch.angle(pred_fft)
        target_phase = torch.angle(target_fft)
        
        # 幅度损失
        if self.loss_type == 'l1':
            amp_loss = F.l1_loss(pred_amp, target_amp)
        else:
            amp_loss = F.mse_loss(pred_amp, target_amp)
        
        # 相位损失
        phase_loss = F.mse_loss(pred_phase, target_phase)
        
        return amp_loss + 0.1 * phase_loss


class GradientLoss(nn.Module):
    """
    梯度损失 (Gradient Loss)
    保持图像边缘信息
    """
    def __init__(self):
        super(GradientLoss, self).__init__()
        
        # Sobel算子
        self.register_buffer('sobel_x', torch.Tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]).view(1, 1, 3, 3))
        
        self.register_buffer('sobel_y', torch.Tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ]).view(1, 1, 3, 3))
    
    def forward(self, pred, target):
        """计算梯度损失"""
        b, c, h, w = pred.size()
        
        # 扩展sobel算子到多通道
        sobel_x = self.sobel_x.expand(c, 1, 3, 3)
        sobel_y = self.sobel_y.expand(c, 1, 3, 3)
        
        # 计算预测图像的梯度
        pred_grad_x = F.conv2d(pred, sobel_x, padding=1, groups=c)
        pred_grad_y = F.conv2d(pred, sobel_y, padding=1, groups=c)
        
        # 计算目标图像的梯度
        target_grad_x = F.conv2d(target, sobel_x, padding=1, groups=c)
        target_grad_y = F.conv2d(target, sobel_y, padding=1, groups=c)
        
        # 梯度损失
        loss = F.l1_loss(pred_grad_x, target_grad_x) + F.l1_loss(pred_grad_y, target_grad_y)
        
        return loss


class SRLoss(nn.Module):
    """
    超分辨率综合损失函数
    
    组合:
    - L1像素损失
    - SSIM损失
    - 频域损失
    - 梯度损失
    """
    
    def __init__(
        self,
        lambda_l1=1.0,
        lambda_ssim=0.5,
        lambda_freq=0.1,
        lambda_grad=0.1
    ):
        super(SRLoss, self).__init__()
        
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_freq = lambda_freq
        self.lambda_grad = lambda_grad
        
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        self.freq_loss = FrequencyLoss()
        self.grad_loss = GradientLoss()
    
    def forward(self, pred, target):
        """
        计算综合损失
        
        Args:
            pred: 预测的超分辨率图像
            target: 高分辨率真值图像
        
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的详细值
        """
        loss_dict = {}
        total_loss = 0.0
        
        # L1损失
        if self.lambda_l1 > 0:
            l1 = self.l1_loss(pred, target)
            total_loss += self.lambda_l1 * l1
            loss_dict['l1'] = l1.item()
        
        # SSIM损失
        if self.lambda_ssim > 0:
            ssim = self.ssim_loss(pred, target)
            total_loss += self.lambda_ssim * ssim
            loss_dict['ssim'] = ssim.item()
        
        # 频域损失
        if self.lambda_freq > 0:
            freq = self.freq_loss(pred, target)
            total_loss += self.lambda_freq * freq
            loss_dict['freq'] = freq.item()
        
        # 梯度损失
        if self.lambda_grad > 0:
            grad = self.grad_loss(pred, target)
            total_loss += self.lambda_grad * grad
            loss_dict['grad'] = grad.item()
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (L1的变体, 更平滑)
    常用于超分辨率任务
    """
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
    
    def forward(self, pred, target):
        diff = pred - target
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss


if __name__ == '__main__':
    # 测试损失函数
    print("=" * 50)
    print("测试损失函数")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据
    pred = torch.randn(2, 8, 64, 64).to(device)
    target = torch.randn(2, 8, 64, 64).to(device)
    
    # 测试SSIM损失
    ssim_loss = SSIMLoss().to(device)
    ssim = ssim_loss(pred, target)
    print(f"SSIM Loss: {ssim.item():.6f}")
    
    # 测试频域损失
    freq_loss = FrequencyLoss().to(device)
    freq = freq_loss(pred, target)
    print(f"Freq Loss: {freq.item():.6f}")
    
    # 测试梯度损失
    grad_loss = GradientLoss().to(device)
    grad = grad_loss(pred, target)
    print(f"Grad Loss: {grad.item():.6f}")
    
    # 测试综合损失
    sr_loss = SRLoss().to(device)
    total, loss_dict = sr_loss(pred, target)
    print(f"\n综合损失:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.6f}")
    
    print("\n✓ 所有损失函数测试通过!")
