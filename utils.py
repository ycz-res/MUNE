"""
工具函数模块
包含各种辅助功能
"""

import torch
import os
import random
from typing import Dict, Tuple
import numpy as np
from sklearn.model_selection import StratifiedKFold

def set_seed(seed: int) -> None:
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

def stratified_shuffle_data(data_dict: Dict, random_state: int = 57) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用StratifiedKFold对数据进行分层洗牌，保持标签分布
    
    Args:
        data_dict: 包含'data', 'label_num', 'muThr'的数据字典
        random_state: 随机种子，确保结果可重现
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (shuffled_indices, stratified_labels)
            - shuffled_indices: 洗牌后的索引数组
            - stratified_labels: 对应的标签数组（用于验证分层效果）
    """
    # 提取标签数据
    if 'mus' in data_dict:
        labels = np.array(data_dict['mus']).flatten()
    else:
        raise ValueError("数据字典中缺少'mus'键")
    
    # 获取总样本数
    total_samples = len(labels)
    
    # 使用StratifiedKFold进行分层洗牌
    # 这里我们使用一个大的fold数来近似随机洗牌，同时保持分层
    n_folds = min(10, total_samples // 2)  # 确保每个fold至少有2个样本
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # 生成分层索引
    shuffled_indices = []
    for train_idx, val_idx in skf.split(np.arange(total_samples), labels):
        shuffled_indices.extend(train_idx.tolist())
        shuffled_indices.extend(val_idx.tolist())
    
    # 转换为numpy数组
    shuffled_indices = np.array(shuffled_indices)
    
    # 验证分层效果
    stratified_labels = labels[shuffled_indices]
    
    print(f"🔄 StratifiedKFold洗牌完成:")
    print(f"  📊 总样本数: {total_samples}")
    print(f"  🎯 使用fold数: {n_folds}")
    print(f"  📈 原始标签分布: {np.bincount(labels.astype(int))}")
    print(f"  📈 洗牌后标签分布: {np.bincount(stratified_labels.astype(int))}")
    
    return shuffled_indices, stratified_labels

# 全局缓存变量
_data_cache = {}

def load_data(file_path: str, start_ratio: float = 0.0, end_ratio: float = 1.0, 
              shuffle: bool = True, random_state: int = 57):
    """
    加载预处理后的 .npz 文件，支持数据范围选择和StratifiedKFold洗牌

    Args:
        file_path (str): .npz 文件路径
        start_ratio (float): 起始位置比例，范围 [0, 1)，默认 0.0 表示从头开始
        end_ratio (float): 结束位置比例，范围 (0, 1]，默认 1.0 表示到末尾
        shuffle (bool): 是否使用StratifiedKFold进行分层洗牌，默认True
        random_state (int): 随机种子，确保结果可重现，默认57

    Returns:
        dict[str, np.ndarray]: 包含 'cmap', 'mus', 'thresholds' 的数据字典
    """
    if not (0.0 <= start_ratio < 1.0):
        raise ValueError("start_ratio 必须在 [0, 1) 范围内")
    if not (0.0 < end_ratio <= 1.0):
        raise ValueError("end_ratio 必须在 (0, 1] 范围内")
    if start_ratio >= end_ratio:
        raise ValueError("start_ratio 必须小于 end_ratio")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到文件: {file_path}")
    
    # 使用缓存避免重复加载
    global _data_cache
    cache_key = file_path
    
    if cache_key not in _data_cache:
        print(f"📦 加载预处理文件: {file_path}")
        
        # 加载npz文件
        npz = np.load(file_path, allow_pickle=True)
        
        # 提取所需的数据
        cmap = np.array(npz["cmap"]).astype(np.float32)
        mus = np.array(npz["mus"]).astype(np.float32)
        thresholds = np.array(npz["thresholds"]).astype(np.float32)
        
        # 缓存数据
        _data_cache[cache_key] = {
            'cmap': cmap,
            'mus': mus,
            'thresholds': thresholds
        }
        
        print(f"✅ 原始数据加载完成: cmap={cmap.shape}, mus={mus.shape}, thresholds={thresholds.shape}")
    else:
        print(f"📦 使用缓存数据: {file_path}")
        cmap = _data_cache[cache_key]['cmap']
        mus = _data_cache[cache_key]['mus']
        thresholds = _data_cache[cache_key]['thresholds']
    
    # 计算数据范围
    total_samples = cmap.shape[0]
    start_idx = int(total_samples * start_ratio)
    end_idx = int(total_samples * end_ratio)
    
    if start_ratio > 0 or end_ratio < 1.0:
        ratio = end_ratio - start_ratio
        print(f"📊 数据范围: 从 {start_idx:,} 到 {end_idx:,}（共 {end_idx - start_idx:,} 个样本，占总数 {total_samples:,} 的 {ratio*100:.1f}%）")
        
        # 切片数据
        cmap = cmap[start_idx:end_idx]
        mus = mus[start_idx:end_idx]
        thresholds = thresholds[start_idx:end_idx]
    
    # 构建数据字典
    data_dict = {
        "cmap": cmap,
        "mus": mus,
        "thresholds": thresholds
    }
    
    # 应用StratifiedKFold洗牌
    if shuffle:
        print(f"🔄 开始StratifiedKFold洗牌 (random_state={random_state})...")
        
        # 使用mus作为标签进行分层洗牌
        shuffled_indices, _ = stratified_shuffle_data(data_dict, random_state)
        
        # 使用洗牌后的索引重新排列数据
        data_dict["cmap"] = data_dict["cmap"][shuffled_indices]
        data_dict["mus"] = data_dict["mus"][shuffled_indices]
        data_dict["thresholds"] = data_dict["thresholds"][shuffled_indices]
        
        print(f"✅ 数据洗牌完成，保持标签分布")
    
    print(f"✅ 最终数据形状: cmap={data_dict['cmap'].shape}, mus={data_dict['mus'].shape}, thresholds={data_dict['thresholds'].shape}")
    return data_dict

