#!/usr/bin/env python3
"""
NPZ数据可视化脚本
输入一个或多个样本索引，可视化CMAP数据、标注MU数量、标记阈值位置
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def visualize_samples(npz_path: str, indices: list, save_dir: str = None):
    """
    可视化指定索引的样本数据
    
    Args:
        npz_path: NPZ文件路径
        indices: 要可视化的样本索引列表
        save_dir: 保存图片的目录，如果为None则显示图片
    """
    # 加载数据
    print(f"📊 加载数据: {npz_path}")
    data = np.load(npz_path)
    cmap = data['cmap']  # (N, 500) 归一化的CMAP数据
    mus = data['mus']    # (N,) MU数量
    thresholds = data['thresholds']  # (N, 500) 阈值映射
    
    print(f"✅ 数据加载成功:")
    print(f"   - CMAP shape: {cmap.shape}")
    print(f"   - MU counts shape: {mus.shape}")
    print(f"   - Thresholds shape: {thresholds.shape}")
    
    # 创建保存目录
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print(f"📁 图片将保存到: {save_dir}")
    
    # 可视化每个样本
    for idx in indices:
        if idx >= cmap.shape[0]:
            print(f"⚠️  索引 {idx} 超出范围 (最大: {cmap.shape[0]-1})")
            continue
            
        print(f"🎨 可视化样本 {idx}...")
        
        # 创建图形
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # 绘制CMAP数据
        x_positions = np.arange(500)  # x轴位置 (0-499)
        y_values = cmap[idx]  # y轴值
        
        ax.plot(x_positions, y_values, 'bo', markersize=2, label='CMAP Data')
        
        # 处理阈值标记
        threshold_positions = np.where(thresholds[idx] > 0)[0]  # 找到非零位置
        threshold_values = thresholds[idx][threshold_positions]  # 对应的值
        
        # 标记阈值为1的位置（竖线）
        ones_positions = threshold_positions[threshold_values == 1.0]
        for pos in ones_positions:
            ax.axvline(x=pos, color='green', linestyle='--', alpha=0.7, linewidth=1)
        
        # 标记阈值>1的位置（红线）
        greater_ones_positions = threshold_positions[threshold_values > 1.0]
        for pos in greater_ones_positions:
            ax.axvline(x=pos, color='red', linestyle='-', alpha=0.8, linewidth=2)
        
        # 设置图形属性
        ax.set_xlabel('Position Index', fontsize=12)
        ax.set_ylabel('Normalized CMAP Value', fontsize=12)
        ax.set_title(f'Sample {idx} - MU Count: {mus[idx]:.1f}, Threshold Count: {len(threshold_positions)}', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 设置x轴范围
        ax.set_xlim(0, 499)
        ax.set_ylim(0, 1.1)
        
        # 添加统计信息文本
        stats_text = f'MU Count: {mus[idx]:.1f}\nThreshold Positions: {len(threshold_positions)}\nValue=1 Positions: {len(ones_positions)}\nValue>1 Positions: {len(greater_ones_positions)}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 保存或显示图片
        if save_dir:
            save_path = os.path.join(save_dir, f'sample_{idx}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   💾 已保存: {save_path}")
            plt.close()
        else:
            plt.show()
    
    print(f"✅ 可视化完成！")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="可视化NPZ文件中的样本数据")
    parser.add_argument("--npz", default="./data.npz", help="NPZ文件路径")
    parser.add_argument("--indices", nargs='+', type=int, required=True, 
                       help="要可视化的样本索引（可指定多个）")
    parser.add_argument("--save", default="plots", help="保存图片的目录（默认: plots）")
    args = parser.parse_args()
    
    if not os.path.exists(args.npz):
        print(f"❌ 文件不存在: {args.npz}")
        return
    
    # 执行可视化
    visualize_samples(args.npz, args.indices, args.save)

if __name__ == "__main__":
    main()
