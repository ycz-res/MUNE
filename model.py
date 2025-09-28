"""
深度学习模型定义模块
支持多种模型架构，易于扩展
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class LinearModel(nn.Module):
    """
    线性模型：用于MU数量回归和阈值预测
    
    输入: (batch_size, 500) - 500个纵坐标点
    输出: 
        - count_pred: (batch_size, 1) - MU数量回归预测
        - thresholds: (batch_size, max_thresholds) - 阈值预测
    """
    def __init__(self, d_model=64, max_thresholds=1):
        super().__init__()
        self.encoder = nn.Linear(500, d_model)  # 输入500个点
        self.fc = nn.Linear(d_model, d_model)
        
        # head1: 预测 count (回归)
        self.count_head = nn.Linear(d_model, 1)  # 回归预测MU数量
        
        # head2: 预测阈值 (使用max_thresholds参数)
        self.threshold_head = nn.Linear(d_model, max_thresholds)  # 预测阈值
    
    def forward(self, x):
        # x: [B, 500] - 500个纵坐标点
        feat = self.encoder(x)   # (B, d_model)，全局特征
        feat = self.fc(feat)
        
        nus = self.count_head(feat)        # (B, 1) - 回归预测MU数量
        thresholds = self.threshold_head(feat)    # (B, max_thresholds) - 预测阈值
        return nus, thresholds