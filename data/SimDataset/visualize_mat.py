#!/usr/bin/env python3
"""
MAT文件数据可视化脚本
输入一个或多个样本索引，可视化原始MAT文件中的CMAP数据、标注MU数量、标记阈值位置
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from utils import load_mat_data

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def visualize_mat_samples(mat_path: str, indices: list, save_dir: str = None):
    """
    可视化指定索引的原始MAT样本数据
    
    Args:
        mat_path: MAT文件路径
        indices: 要可视化的样本索引列表
        save_dir: 保存图片的目录，如果为None则显示图片
    """
    # 加载数据
    print(f"📊 加载MAT数据: {mat_path}")
    mat = load_mat_data(mat_path, lazy=False)
    
    print(f"✅ 数据加载成功:")
    print(f"   - mat['data'] shape: {mat['data'].shape}")
    print(f"   - mat['label_num'] shape: {mat['label_num'].shape}")
    print(f"   - mat['muThr'] shape: {mat['muThr'].shape}")
    
    # 处理维度
    data = np.array(mat["data"])  # 可能是 (2,500,N) 或 (N,500,2)
    label_num = np.array(mat["label_num"]).squeeze()
    mu_thr = np.array(mat["muThr"])  # 可能是 (M,N) 或 (N,M)
    
    # 处理维度转换
    if data.shape[0] < data.shape[-1]:
        data = np.transpose(data, (2, 1, 0))  # (2,500,N) -> (N,500,2)
    
    if mu_thr.shape[0] < mu_thr.shape[-1]:
        mu_thr = mu_thr.T
    
    if label_num.ndim == 2 and label_num.shape[0] < label_num.shape[1]:
        label_num = label_num.T.squeeze()
    
    print(f"处理后维度:")
    print(f"   - data shape: {data.shape}")
    print(f"   - label_num shape: {label_num.shape}")
    print(f"   - muThr shape: {mu_thr.shape}")
    
    # 创建保存目录
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print(f"📁 图片将保存到: {save_dir}")
    
    # 可视化每个样本
    for idx in indices:
        if idx >= data.shape[0]:
            print(f"⚠️  索引 {idx} 超出范围 (最大: {data.shape[0]-1})")
            continue
            
        print(f"🎨 可视化样本 {idx}...")
        
        # 创建图形
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # 获取该样本的数据
        cmap_data = data[idx]  # (500, 2)
        x_values = cmap_data[:, 0]  # 电流值
        y_values = cmap_data[:, 1]  # CMAP值
        
        # 绘制CMAP数据
        ax.plot(x_values, y_values, 'bo', markersize=2, label='CMAP Data')
        
        # 处理阈值标记
        mu_thr_sample = mu_thr[idx]  # 该样本的阈值
        valid_thresholds = mu_thr_sample[mu_thr_sample > 0]  # 有效阈值
        
        # 标记每个有效阈值的位置
        for threshold in valid_thresholds:
            # 直接用阈值的真实值标记
            ax.axvline(x=threshold, color='green', linestyle='--', alpha=0.7, linewidth=1)
            # 在竖线上标记阈值真实值
            ax.text(threshold, ax.get_ylim()[1] * 0.95, f'{threshold:.1f}', 
                   rotation=90, verticalalignment='top', horizontalalignment='right',
                   fontsize=8, color='green', alpha=0.8)
        
        # 设置图形属性
        ax.set_xlabel('Current (mA)', fontsize=12)
        ax.set_ylabel('CMAP Value', fontsize=12)
        ax.set_title(f'Sample {idx} - MU Count: {label_num[idx]:.1f}, Valid Thresholds: {len(valid_thresholds)}', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 设置坐标轴范围
        ax.set_xlim(x_values.min() - 0.5, x_values.max() + 0.5)
        ax.set_ylim(y_values.min() - 0.1, y_values.max() + 0.1)
        
        # 添加统计信息文本
        stats_text = f'MU Count: {label_num[idx]:.1f}\nValid Thresholds: {len(valid_thresholds)}\nX Range: [{x_values.min():.1f}, {x_values.max():.1f}]\nY Range: [{y_values.min():.2f}, {y_values.max():.2f}]'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 保存或显示图片
        if save_dir:
            save_path = os.path.join(save_dir, f'mat_sample_{idx}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   💾 已保存: {save_path}")
            plt.close()
        else:
            plt.show()
    
    print(f"✅ 可视化完成！")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="可视化原始MAT文件中的样本数据")
    parser.add_argument("--mat", default="./data.mat", help="MAT文件路径")
    parser.add_argument("--indices", nargs='+', type=int, required=True, 
                       help="要可视化的样本索引（可指定多个）")
    parser.add_argument("--save", default="mat_plots", help="保存图片的目录（默认: mat_plots）")
    args = parser.parse_args()
    
    if not os.path.exists(args.mat):
        print(f"❌ 文件不存在: {args.mat}")
        return
    
    # 执行可视化
    visualize_mat_samples(args.mat, args.indices, args.save)

if __name__ == "__main__":
    main()
