#!/usr/bin/env python3
"""
测试数据集和collate_fn的实现
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import SimDataset
from config import get_config

def test_dataset():
    """测试数据集加载和collate_fn"""
    print("=== 测试数据集实现 ===")
    
    # 获取配置
    config = get_config()
    
    # 测试训练数据集
    print("\n1. 测试训练数据集:")
    try:
        train_dataset = SimDataset(config['SimDataset.train'], 'sim')
        print(f"   数据集大小: {len(train_dataset)}")
        print(f"   数据形状: {train_dataset.data.shape}")
        print(f"   标签形状: {train_dataset.labels.shape if train_dataset.labels is not None else 'None'}")
        
        # 测试单个样本
        sample_data, sample_label = train_dataset[0]
        print(f"   单个样本数据形状: {sample_data.shape}")
        print(f"   单个样本标签: {sample_label}")
        
        # 测试collate_fn
        batch = [train_dataset[i] for i in range(4)]  # 取4个样本
        batch_data, batch_labels = SimDataset.collate_fn(batch)
        print(f"   批次数据形状: {batch_data.shape}")
        print(f"   批次标签形状: {batch_labels.shape}")
        print(f"   批次数据范围: [{batch_data.min():.3f}, {batch_data.max():.3f}]")
        print(f"   批次标签范围: [{batch_labels.min():.3f}, {batch_labels.max():.3f}]")
        
    except Exception as e:
        print(f"   ❌ 训练数据集测试失败: {e}")
    
    # 测试验证数据集
    print("\n2. 测试验证数据集:")
    try:
        val_dataset = SimDataset(config['SimDataset.val'], 'sim')
        print(f"   数据集大小: {len(val_dataset)}")
        print(f"   数据形状: {val_dataset.data.shape}")
        
        # 测试单个样本
        sample_data, sample_label = val_dataset[0]
        print(f"   单个样本数据形状: {sample_data.shape}")
        print(f"   单个样本标签: {sample_label}")
        
    except Exception as e:
        print(f"   ❌ 验证数据集测试失败: {e}")
    
    # 测试真实数据集
    print("\n3. 测试真实数据集:")
    try:
        control_dataset = SimDataset(config['SimDataset.control'], 'real')
        print(f"   健康人数据集大小: {len(control_dataset)}")
        print(f"   数据形状: {control_dataset.data.shape}")
        
        # 测试单个样本
        sample_data, sample_label = control_dataset[0]
        print(f"   单个样本数据形状: {sample_data.shape}")
        print(f"   单个样本标签: {sample_label}")
        
    except Exception as e:
        print(f"   ❌ 真实数据集测试失败: {e}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_dataset()
