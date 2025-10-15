"""
工具函数模块
包含各种辅助功能
"""

import numpy as np
import scipy.io
import h5py

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
