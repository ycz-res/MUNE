"""
简单的MU数据筛选脚本
从预处理后的npz文件中筛选MU为5到55的数据
"""

import os
import argparse
import numpy as np
import random


def filter_mu_data(input_npz_path: str, output_npz_path: str, 
                  min_mu: int = 5, max_mu: int = 55, 
                  sample_ratio: float = 1.0) -> str:
    """
    从npz文件中筛选指定MU范围的数据
    
    Args:
        input_npz_path: 输入的npz文件路径
        output_npz_path: 输出的npz文件路径
        min_mu: 最小MU数量
        max_mu: 最大MU数量
        sample_ratio: 每个MU类别的采样比例 (0.0-1.0)
    
    Returns:
        输出文件路径
    """
    print("=" * 50)
    print("🔍 MU数据筛选")
    print("=" * 50)
    print(f"📁 输入: {input_npz_path}")
    print(f"📁 输出: {output_npz_path}")
    print(f"🎯 MU范围: [{min_mu}, {max_mu}]")
    print(f"📊 采样比例: {sample_ratio:.1%}")
    print()
    
    # 加载数据
    print("📦 加载数据...")
    if not os.path.exists(input_npz_path):
        raise FileNotFoundError(f"输入文件不存在: {input_npz_path}")
    
    data = np.load(input_npz_path)
    cmap = data['cmap']
    mus = data['mus']
    thresholds = data['thresholds']
    
    print(f"✅ 原始数据: {len(mus)} 个样本")
    print(f"   MU范围: {mus.min():.0f} - {mus.max():.0f}")
    print()
    
    # 筛选
    print("🎯 筛选数据...")
    mask = (mus >= min_mu) & (mus <= max_mu)
    valid_indices = np.where(mask)[0]
    
    print(f"✅ MU范围筛选: {len(valid_indices)} 个样本")
    
    if len(valid_indices) == 0:
        raise ValueError(f"没有找到MU在[{min_mu}, {max_mu}]范围内的样本！")
    
    # 按MU类别采样
    if sample_ratio < 1.0:
        print(f"📊 按MU类别采样 {sample_ratio:.1%}...")
        # 设置随机种子确保可重复性
        np.random.seed(57)
        random.seed(57)
        filtered_indices = []
        
        # 获取所有MU值
        valid_mus = mus[valid_indices]
        unique_mus = np.unique(valid_mus)
        
        print(f"   MU类别数: {len(unique_mus)}")
        
        for mu_val in unique_mus:
            # 找到当前MU值的所有样本索引
            mu_mask = valid_mus == mu_val
            mu_indices = valid_indices[mu_mask]
            
            # 随机采样
            n_samples = int(len(mu_indices) * sample_ratio)
            if n_samples > 0:
                sampled_indices = np.random.choice(mu_indices, n_samples, replace=False)
                filtered_indices.extend(sampled_indices)
        
        filtered_indices = np.array(filtered_indices)
        print(f"✅ 采样后: {len(filtered_indices)} 个样本")
    else:
        filtered_indices = valid_indices
        print(f"✅ 使用全部样本: {len(filtered_indices)} 个样本")
    
    # 保存
    print("💾 保存数据...")
    filtered_cmap = cmap[filtered_indices]
    filtered_mus = mus[filtered_indices]
    filtered_thresholds = thresholds[filtered_indices]
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_npz_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    np.savez_compressed(
        output_npz_path,
        cmap=filtered_cmap.astype(np.float32),
        mus=filtered_mus.astype(np.float32),
        thresholds=filtered_thresholds.astype(np.float32),
    )
    
    print("=" * 50)
    print("✅ 筛选完成!")
    print("=" * 50)
    print(f"📁 输出文件: {output_npz_path}")
    print(f"📊 数据形状: {filtered_cmap.shape}")
    print(f"🎯 MU范围: {filtered_mus.min():.0f} - {filtered_mus.max():.0f}")
    print()
    
    return output_npz_path


def main():
    parser = argparse.ArgumentParser(description="筛选MU为5-55的数据")
    parser.add_argument("--input", default="./data/SimDataset/data_all.npz", 
                       help="输入npz文件（默认: ./data/SimDataset/data_all.npz）")
    parser.add_argument("--output", default="./data/SimDataset/data.npz", 
                       help="输出npz文件（默认: ./data/SimDataset/data.npz）")
    parser.add_argument("--min_mu", type=int, default=5, 
                       help="最小MU数量（默认: 5）")
    parser.add_argument("--max_mu", type=int, default=55, 
                       help="最大MU数量（默认: 55）")
    parser.add_argument("--sample_ratio", type=float, default=1.0, 
                       help="每个MU类别的采样比例 (0.0-1.0，默认: 1.0)")
    
    args = parser.parse_args()
    
    if args.min_mu > args.max_mu:
        raise ValueError("min_mu 必须小于等于 max_mu")
    
    if not (0.0 <= args.sample_ratio <= 1.0):
        raise ValueError("sample_ratio 必须在 [0.0, 1.0] 范围内")
    
    saved = filter_mu_data(args.input, args.output, args.min_mu, args.max_mu, args.sample_ratio)
    print(f"🎉 完成！数据已保存到: {saved}")


if __name__ == "__main__":
    main()
