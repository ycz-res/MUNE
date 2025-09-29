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
    线性模型：用于阈值位置预测 - 优化版本
    
    输入: (batch_size, 500) - 500个CMAP纵坐标点
    输出: (batch_size, 500) - 500维阈值位置预测
    """
    def __init__(self, d_model=64):
        super().__init__()
        self.encoder = nn.Linear(500, d_model)  # 输入500个点
        self.fc = nn.Linear(d_model, d_model)
        
        # 预测500维阈值位置
        self.threshold_head = nn.Linear(d_model, 500)  # 输出500维阈值数据
        
        # 使用Sigmoid + 阈值化来确保稀疏性
        self.sigmoid = nn.Sigmoid()
        self.threshold = 0.1  # 降低阈值，让模型有机会输出非零值
    
    def forward(self, x):
        # x: [batch_size, 500] - 500个CMAP纵坐标点
        feat = self.encoder(x)   # (batch_size, d_model)，全局特征
        feat = self.fc(feat)     # 进一步特征提取
        
        # 使用Sigmoid输出概率，然后阈值化
        prob = self.sigmoid(self.threshold_head(feat))  # (batch_size, 500) - 概率输出
        thresholds = prob * (prob > self.threshold).float()  # 阈值化，只保留高概率位置
        
        return thresholds