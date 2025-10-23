"""
工具函数模块
包含各种辅助功能
"""

import torch
import os
import random
from typing import Dict
import numpy as np
import scipy.io
import h5py

def set_seed(seed: int) -> None:
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

def load_mat_data(file_path: str, lazy: bool = True, start_ratio: float = 0.0, end_ratio: float = 1.0):
    """
    安全加载大型 .mat 文件（支持 v7.3）

    Args:
        file_path (str): .mat 文件路径
        lazy (bool): 是否采用懒加载（True 推荐，用于大文件）
        start_ratio (float): 起始位置比例，范围 [0, 1)，默认 0.0 表示从头开始
        end_ratio (float): 结束位置比例，范围 (0, 1]，默认 1.0 表示到末尾

    Returns:
        dict[str, np.ndarray or h5py.Dataset]: 键名→数据
    """
    if not (0.0 <= start_ratio < 1.0):
        raise ValueError("start_ratio 必须在 [0, 1) 范围内")
    if not (0.0 < end_ratio <= 1.0):
        raise ValueError("end_ratio 必须在 (0, 1] 范围内")
    if start_ratio >= end_ratio:
        raise ValueError("start_ratio 必须小于 end_ratio")
    
    try:
        # ① 优先尝试常规 .mat（适合 <2GB）
        mat_data = scipy.io.loadmat(file_path)
        filtered_data = {
            k: v for k, v in mat_data.items()
            if not k.startswith('__') and isinstance(v, np.ndarray)
        }
        if filtered_data:
            print(f"✅ 使用 scipy.io.loadmat() 成功加载: {file_path}")
            return filtered_data

    except Exception as e:
        # ② 若文件为 v7.3（基于 HDF5），改用 h5py
        if "HDF" in str(e) or "mat file appears to be HDF5" in str(e):
            print(f"🔍 检测到大型 v7.3 文件，使用 h5py 加载: {file_path}")
        else:
            print(f"⚠️ loadmat 失败，自动尝试 h5py: {e}")

    # === 使用 h5py 读取 v7.3 格式 ===
    data_dict = {}
    f = h5py.File(file_path, 'r')  # 仅打开，不读入全部内存

    for key in f.keys():
        try:
            # 获取原始shape
            original_shape = f[key].shape
            
            # 判断样本数在哪个维度（通常是最大的那个维度）
            if key in ['data', 'label_num', 'muThr']:
                # 对于这些关键数据，样本数通常在最后一维
                total_samples = original_shape[-1]
                start_idx = int(total_samples * start_ratio)
                end_idx = int(total_samples * end_ratio)
                
                if start_ratio > 0 or end_ratio < 1.0:
                    # 只加载指定范围的数据
                    if key == 'data':  # 只在第一次打印
                        ratio = end_ratio - start_ratio
                        print(f"  📊 数据范围: 从 {start_idx:,} 到 {end_idx:,}（共 {end_idx - start_idx:,} 个样本，占总数 {total_samples:,} 的 {ratio*100:.1f}%）")
                    
                    if lazy:
                        # 懒加载：h5py支持切片，按最后一维切片
                        data_dict[key] = f[key][..., start_idx:end_idx]
                        print(f"  🔹 懒加载变量: {key}, shape={data_dict[key].shape}")
                    else:
                        # 全量加载指定范围：直接读为 numpy 数组，按最后一维切片
                        data_dict[key] = np.array(f[key][..., start_idx:end_idx])
                        print(f"  ✅ 已加载变量: {key}, shape={data_dict[key].shape}")
                else:
                    # 加载全部数据
                    if lazy:
                        data_dict[key] = f[key]
                        print(f"  🔹 懒加载变量: {key}, shape={f[key].shape}")
                    else:
                        data_dict[key] = np.array(f[key])
                        print(f"  ✅ 已加载变量: {key}, shape={data_dict[key].shape}")
            else:
                # 其他变量直接加载
                if lazy:
                    data_dict[key] = f[key]
                    print(f"  🔹 懒加载变量: {key}, shape={f[key].shape}")
                else:
                    data_dict[key] = np.array(f[key])
                    print(f"  ✅ 已加载变量: {key}, shape={data_dict[key].shape}")
        except Exception as e2:
            print(f"  ⚠️ 无法加载 {key}: {e2}")

    if not data_dict:
        raise ValueError(f"❌ 未能在 {file_path} 中加载有效变量")

    # 处理数据翻转：将data的最后一维从[x,y]翻转为[y,x]
    if 'data' in data_dict:
        if lazy and isinstance(data_dict['data'], h5py.Dataset):
            # 懒加载模式下，需要先转换为numpy数组才能翻转
            data_dict['data'] = np.array(data_dict['data'])
        data_dict['data'] = np.flip(data_dict['data'], axis=1)
        print(f"  🔄 已翻转data数据的最后一维")
    
    print(f"✅ 成功加载 {len(data_dict)} 个变量（{'lazy' if lazy else 'eager'} 模式）")
    return data_dict

