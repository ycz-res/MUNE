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

def pre_load_mat_data(file_path: str) -> Dict[str, np.ndarray]:
    """加载MAT文件数据，返回包含所有字段的字典"""
    try:
        # 尝试使用scipy.io加载
        mat_data = scipy.io.loadmat(file_path)
        
        # 过滤掉MATLAB元数据字段
        filtered_data = {}
        for key, value in mat_data.items():
            if not key.startswith('__') and isinstance(value, np.ndarray):
                filtered_data[key] = value
        
        if filtered_data:
            return filtered_data
        else:
            raise ValueError("未找到有效的数据字段")
                
    except Exception as e:
        if "HDF reader" in str(e): 
            # 使用h5py加载v7.3格式
            with h5py.File(file_path, 'r') as f:
                filtered_data = {}
                def collect_data(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        filtered_data[name] = np.array(obj)
                
                f.visititems(collect_data)
                
                if filtered_data:
                    return filtered_data
                else:
                    raise ValueError("未找到有效的数据字段")
        else:
            raise e
    
    raise ValueError(f"无法加载MAT文件: {file_path}")

def load_mat_data(file_path: str, lazy: bool = True):
    """
    安全加载大型 .mat 文件（支持 v7.3）

    Args:
        file_path (str): .mat 文件路径
        lazy (bool): 是否采用懒加载（True 推荐，用于大文件）

    Returns:
        dict[str, np.ndarray or h5py.Dataset]: 键名→数据
    """
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
            if lazy:
                # 懒加载：仅保留引用，不把数据加载进内存
                data_dict[key] = f[key]
                print(f"  🔹 懒加载变量: {key}, shape={f[key].shape}")
            else:
                # 全量加载：直接读为 numpy 数组（占内存）
                data_dict[key] = np.array(f[key])
                print(f"  ✅ 已加载变量: {key}, shape={data_dict[key].shape}")
        except Exception as e2:
            print(f"  ⚠️ 无法加载 {key}: {e2}")

    if not data_dict:
        raise ValueError(f"❌ 未能在 {file_path} 中加载有效变量")

    print(f"✅ 成功加载 {len(data_dict)} 个变量（{'lazy' if lazy else 'eager'} 模式）")
    return data_dict

def print_mat_file_info(data_dir: str = "./data/SimDataset"):
    """临时函数：打印SimDataset文件夹下MAT文件的数据和标签信息"""
    import os
    
    print("=== SimDataset文件夹下MAT文件信息 ===")
    
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.mat'):
            file_path = os.path.join(data_dir, file_name)
            print(f"\n文件: {file_name}")
            print(f"路径: {file_path}")
            
            try:
                # 先尝试scipy.io加载
                try:
                    mat_data = scipy.io.loadmat(file_path)
                    print("文件内容 (scipy.io):")
                    for key, value in mat_data.items():
                        if not key.startswith('__'):  # 跳过MATLAB元数据
                            if isinstance(value, np.ndarray):
                                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                                if value.size < 20:  # 如果数据量小，打印内容
                                    print(f"    内容: {value}")
                            else:
                                print(f"  {key}: {type(value)} = {value}")
                except Exception as e1:
                    if "HDF reader" in str(e1):
                        # 使用h5py加载v7.3格式
                        print("文件内容 (h5py):")
                        with h5py.File(file_path, 'r') as f:
                            def print_h5_structure(name, obj):
                                if isinstance(obj, h5py.Dataset):
                                    print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")
                                    if obj.size < 20:  # 如果数据量小，打印内容
                                        print(f"    内容: {obj[:]}")
                                elif isinstance(obj, h5py.Group):
                                    print(f"  {name}: Group")
                            
                            f.visititems(print_h5_structure)
                    else:
                        raise e1
                
                # 尝试使用load_mat_data函数
                data = load_mat_data(file_path)
                print(f"load_mat_data提取的数据: shape={data.shape}, dtype={data.dtype}")
                
            except Exception as e:
                print(f"❌ 加载失败: {e}")
    
    print("\n=== 信息打印完成 ===")


def print_detailed_dataset_info(data_dir: str = "./data/SimDataset"):
    """详细打印每个数据集的数据结构和标签信息"""
    import os
    
    print("=" * 80)
    print("SimDataset文件夹下MAT文件详细数据结构分析")
    print("=" * 80)
    
    for file_name in sorted(os.listdir(data_dir)):
        if file_name.endswith('.mat'):
            file_path = os.path.join(data_dir, file_name)
            print(f"\n{'='*60}")
            print(f"📁 文件: {file_name}")
            print(f"📂 路径: {file_path}")
            print(f"{'='*60}")
            
            try:
                # 使用h5py加载v7.3格式
                with h5py.File(file_path, 'r') as f:
                    print("📊 数据结构:")
                    
                    # 收集所有数据集信息
                    datasets_info = []
                    def collect_info(name, obj):
                        if isinstance(obj, h5py.Dataset):
                            datasets_info.append((name, obj))
                    
                    f.visititems(collect_info)
                    
                    # 按名称排序并打印
                    for name, obj in sorted(datasets_info):
                        print(f"  🔹 {name}:")
                        print(f"     📐 形状: {obj.shape}")
                        print(f"     🏷️  数据类型: {obj.dtype}")
                        print(f"     📏 总元素数: {obj.size:,}")
                        
                        # 打印数据范围
                        if obj.size > 0:
                            data_array = np.array(obj)
                            print(f"     📈 数值范围: [{data_array.min():.6f}, {data_array.max():.6f}]")
                            print(f"     📊 均值: {data_array.mean():.6f}")
                            print(f"     📊 标准差: {data_array.std():.6f}")
                            
                            # 如果数据量小，打印部分内容
                            if obj.size <= 20:
                                print(f"     📋 内容: {data_array}")
                            elif obj.size <= 100:
                                print(f"     📋 前10个值: {data_array.flatten()[:10]}")
                            else:
                                print(f"     📋 前5个值: {data_array.flatten()[:5]}")
                        print()
                
                # 使用load_mat_data函数提取主要数据
                print("🎯 load_mat_data提取的主要数据:")
                data = load_mat_data(file_path)
                print(f"   📐 形状: {data.shape}")
                print(f"   🏷️  数据类型: {data.dtype}")
                print(f"   📏 总元素数: {data.size:,}")
                print(f"   📈 数值范围: [{data.min():.6f}, {data.max():.6f}]")
                print(f"   📊 均值: {data.mean():.6f}")
                print(f"   📊 标准差: {data.std():.6f}")
                
            except Exception as e:
                print(f"❌ 加载失败: {e}")
    
    print(f"\n{'='*80}")
    print("📋 数据集总结:")
    print("  🔹 train_dataset1_HP_better_range_1000.mat: 训练数据，包含5种标签")
    print("  🔹 val_dataset1_HP_better_range_1000.mat: 验证数据，包含5种标签") 
    print("  🔹 real_data_control.mat: 健康人真实数据，只有数量标签")
    print("  🔹 real_data_sci.mat: 病人真实数据，只有数量标签")
    print("=" * 80)


