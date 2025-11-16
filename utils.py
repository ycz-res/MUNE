"""
工具函数模块
包含各种辅助功能
"""

import torch
import os
import numpy as np

def set_seed(seed: int) -> None:
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

# 全局缓存变量：存储原始数据和洗牌后的索引
_data_cache = {}
_shuffle_cache = {}

def _stratified_shuffle_indices(labels: np.ndarray, random_state: int = 57, block: int = 1) -> np.ndarray:
    import numpy as np
    labels = np.asarray(labels)
    if labels.dtype.kind in "fc":
        labels = labels.astype(np.int64)

    rng = np.random.default_rng(random_state)
    n = labels.shape[0]

    uniq, inv = np.unique(labels, return_inverse=True)
    K = uniq.size

    per_cls = [np.where(inv == c)[0] for c in range(K)]
    for arr in per_cls:
        rng.shuffle(arr)

    ptr = [arr.size for arr in per_cls]
    out = np.empty(n, dtype=np.int64)
    w = 0
    alive = [i for i in range(K) if ptr[i] > 0]

    while w < n and alive:
        order = alive[:]
        rng.shuffle(order)
        for c in order:
            remain = ptr[c]
            if remain <= 0:
                continue
            take = block if remain >= block else remain
            start = remain - take
            out[w:w+take] = per_cls[c][start:remain]
            ptr[c] = start
            w += take
            if w >= n:
                break
        alive = [i for i in range(K) if ptr[i] > 0]

    return out


def load_data(file_path: str, start_ratio: float = 0.0, end_ratio: float = 1.0, 
              shuffle: bool = True, random_state: int = 57):
    """
    高效加载预处理后的 .npz 文件，使用分层洗牌保证科研质量
    
    特性：
    - 原始数据缓存：避免重复加载文件
    - 分层洗牌索引缓存：同一随机种子只洗牌一次
    - 分层洗牌：保持各类别比例，适合科研论文
    - 切片后洗牌：仅对需要的数据进行洗牌，提高效率

    Args:
        file_path (str): .npz 文件路径
        start_ratio (float): 起始位置比例，范围 [0, 1)，默认 0.0
        end_ratio (float): 结束位置比例，范围 (0, 1]，默认 1.0
        shuffle (bool): 是否使用分层洗牌，默认 True
        random_state (int): 随机种子，默认 57

    Returns:
        dict[str, np.ndarray]: 包含 'cmap', 'mus', 'thresholds' 的数据字典
    """
    # 参数验证
    if not (0.0 <= start_ratio < 1.0):
        raise ValueError("start_ratio 必须在 [0, 1) 范围内")
    if not (0.0 < end_ratio <= 1.0):
        raise ValueError("end_ratio 必须在 (0, 1] 范围内")
    if start_ratio >= end_ratio:
        raise ValueError("start_ratio 必须小于 end_ratio")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到文件: {file_path}")
    
    # 使用全局缓存避免重复加载
    global _data_cache, _shuffle_cache
    
    # 第一步：加载或使用缓存的原始数据
    if file_path not in _data_cache:
        print(f"📦 加载预处理文件: {file_path}")
        npz = np.load(file_path, allow_pickle=True)
        
        _data_cache[file_path] = {
            'cmap': np.array(npz["cmap"]).astype(np.float32),
            'mus': np.array(npz["mus"]).astype(np.float32),
            'thresholds': np.array(npz["thresholds"]).astype(np.float32)
        }
        print(f"✅ 数据加载完成: {_data_cache[file_path]['cmap'].shape[0]:,} 个样本")
    
    # 第二步：计算数据范围并切片
    cmap = _data_cache[file_path]['cmap']
    mus = _data_cache[file_path]['mus']
    thresholds = _data_cache[file_path]['thresholds']
    
    total_samples = cmap.shape[0]
    start_idx = int(total_samples * start_ratio)
    end_idx = int(total_samples * end_ratio)
    
    # 切片数据
    cmap = cmap[start_idx:end_idx]
    mus = mus[start_idx:end_idx]
    thresholds = thresholds[start_idx:end_idx]
    num_samples = end_idx - start_idx
    
    print(f"📊 使用数据范围: [{start_ratio:.1%} - {end_ratio:.1%}] = {num_samples:,} 个样本")
    
    # 第三步：应用分层洗牌（如果需要）
    if shuffle:
        shuffle_key = f"{num_samples}_{random_state}"
        
        if shuffle_key not in _shuffle_cache:
            print(f"🔄 生成分层洗牌索引 (seed={random_state})...")
            _shuffle_cache[shuffle_key] = _stratified_shuffle_indices(mus, random_state)
        
        indices = _shuffle_cache[shuffle_key]
        cmap = cmap[indices]
        mus = mus[indices]
        thresholds = thresholds[indices]
    
    return {
        "cmap": cmap,
        "mus": mus,
        "thresholds": thresholds
    }

