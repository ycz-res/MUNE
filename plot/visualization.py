"""
MU阈值预测任务可视化模块
支持从JSON数据生成可视化图表
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
from typing import Dict, List, Optional
import warnings

# 屏蔽中文字体警告
warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*missing from font.*')

# 设置中文字体
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def load_train_data(train_json_path: str) -> Dict:
    """加载训练数据"""
    with open(train_json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_test_data(test_json_path: str) -> Dict:
    """加载测试数据"""
    with open(test_json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_loss_curves(train_data: Dict, save_dir: str):
    """绘制训练和验证损失曲线"""
    history = train_data['training_history']
    epochs = [h['epoch'] for h in history]
    train_losses = [h['train_loss'] for h in history]
    val_losses = [h['val_loss'] for h in history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-o', label='Train Loss', markersize=4, linewidth=2)
    plt.plot(epochs, val_losses, 'r-s', label='Val Loss', markersize=4, linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    out_path = os.path.join(save_dir, 'loss_curves.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 保存损失曲线: {out_path}")


def plot_val_metrics_curves(train_data: Dict, save_dir: str):
    """绘制验证指标曲线"""
    history = train_data['training_history']
    epochs = [h['epoch'] for h in history]
    
    metrics_names = ['Precision', 'Recall', 'F1', 'IoU', 'Score']
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#ff7f0e']
    markers = ['o', 's', '^', 'D', 'v']
    
    plt.figure(figsize=(12, 6))
    
    for metric, color, marker in zip(metrics_names, colors, markers):
        values = [h['val_metrics'].get(metric, 0.0) for h in history]
        plt.plot(epochs, values, color=color, marker=marker, label=metric, 
                markersize=4, linewidth=2, alpha=0.8)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    plt.title('Validation Metrics Over Epochs', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, loc='best')
    plt.tight_layout()
    
    out_path = os.path.join(save_dir, 'val_metrics_curves.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 保存验证指标曲线: {out_path}")


def plot_test_metrics_summary(test_data: Dict, save_dir: str):
    """绘制测试指标柱状图"""
    test_loss = test_data['test_loss']
    test_metrics = test_data['test_metrics']
    
    labels = ['Loss', 'Precision', 'Recall', 'F1', 'IoU', 'EMD', 'Score']
    values = [
        test_loss,
        test_metrics['Precision'],
        test_metrics['Recall'],
        test_metrics['F1'],
        test_metrics['IoU'],
        test_metrics.get('EMD', 0.0),
        test_metrics['Score']
    ]
    
    colors = ['#4e79a7', '#59a14f', '#f28e2b', '#e15759', '#76b7b2', '#edc948', '#9c755f']
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(labels))
    bars = plt.bar(x, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    plt.xticks(x, labels, fontsize=11)
    plt.ylabel('Value', fontsize=12)
    plt.title('Test Metrics Summary', fontsize=14, fontweight='bold')
    plt.grid(True, axis='y', alpha=0.3)
    
    # 在柱状图上添加数值
    for i, (bar, val) in enumerate(zip(bars, values)):
        plt.text(bar.get_x() + bar.get_width()/2, val, f'{val:.4f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    out_path = os.path.join(save_dir, 'test_metrics_summary.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 保存测试指标汇总: {out_path}")


def plot_random_test_samples(test_data: Dict, save_dir: str, max_samples: int = 20):
    """绘制随机测试样本的阈值预测对比图"""
    if 'sample_data' not in test_data or not test_data['sample_data']:
        print("⚠️  没有样本数据，跳过样本可视化")
        return
    
    sample_data = test_data['sample_data']
    indices = sample_data['indices']
    cmap_data = np.array(sample_data['cmap'])
    thr_true = np.array(sample_data['thresholds_true'])
    thr_pred = np.array(sample_data['thresholds_pred'])
    mus_true = np.array(sample_data['mus_true'])
    
    # 限制样本数量
    num_samples = min(len(indices), max_samples)
    
    # 计算子图布局
    cols = 5
    rows = int(np.ceil(num_samples / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle('Random Test Samples: True vs Predicted Thresholds', 
                 fontsize=16, fontweight='bold')
    
    x_positions = np.arange(500)
    
    for i in range(num_samples):
        row = i // cols
        col = i % cols
        ax = axes[row, col]
        
        # 获取数据
        idx = indices[i]
        cmap_sample = cmap_data[i]
        thr_true_sample = thr_true[i]
        thr_pred_sample = thr_pred[i]
        mu_true = mus_true[i]
        
        # 绘制 CMAP (散点图)
        ax.scatter(x_positions, cmap_sample, c='blue', s=1, alpha=0.3, label='CMAP')
        
        # 计算位置
        true_pos = set(np.where(thr_true_sample > 0)[0].tolist())
        pred_pos = set(np.where(thr_pred_sample > 0)[0].tolist())

            match_pos = sorted(true_pos & pred_pos)
            true_only_pos = sorted(true_pos - pred_pos)
            pred_only_pos = sorted(pred_pos - true_pos)

        # 绘制匹配的阈值 (绿色)
            for p in match_pos:
            ax.axvline(x=p, color='green', linestyle='-', linewidth=1.5, alpha=0.9)

        # 绘制真实但未预测的 (蓝色虚线)
            for p in true_only_pos:
                ax.axvline(x=p, color='#1f77b4', linestyle='--', linewidth=1.2, alpha=0.9)

        # 绘制预测但不真实的 (橙色)
            for p in pred_only_pos:
                ax.axvline(x=p, color='orange', linestyle='-', linewidth=1.2, alpha=0.9)
        
        # 计算预测的 MU 数量
        pred_mu = int(np.sum(thr_pred_sample > 0))
        
        ax.set_title(f'Sample {idx} | True MU: {mu_true} | Pred MU: {pred_mu}', 
                    fontsize=9, fontweight='bold')
            ax.set_xlim(0, 499)
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.2)
        ax.set_xlabel('Position', fontsize=8)
        ax.set_ylabel('Amplitude', fontsize=8)
    
    # 隐藏多余的子图
    total_plots = rows * cols
    for k in range(num_samples, total_plots):
        row = k // cols
        col = k % cols
        axes[row, col].axis('off')
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Match (True & Pred)'),
        Patch(facecolor='#1f77b4', label='True Only (Miss)'),
        Patch(facecolor='orange', label='Pred Only (False Alarm)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', 
              bbox_to_anchor=(0.5, -0.01), ncol=3, fontsize=11, frameon=True)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    out_path = os.path.join(save_dir, 'random_test_samples.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 保存测试样本可视化: {out_path}")


def plot_comprehensive_summary(train_data: Dict, test_data: Dict, save_dir: str):
    """绘制综合汇总图"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. 损失曲线
    ax1 = fig.add_subplot(gs[0, :2])
    history = train_data['training_history']
    epochs = [h['epoch'] for h in history]
    train_losses = [h['train_loss'] for h in history]
    val_losses = [h['val_loss'] for h in history]
    
    ax1.plot(epochs, train_losses, 'b-o', label='Train Loss', markersize=3, linewidth=2)
    ax1.plot(epochs, val_losses, 'r-s', label='Val Loss', markersize=3, linewidth=2)
    if test_data:
        ax1.axhline(y=test_data['test_loss'], color='green', linestyle='--', 
                   linewidth=2, label=f"Test Loss: {test_data['test_loss']:.4f}")
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Progress: Loss Curves', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 训练信息文本
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    info_text = f"""训练信息
    
时间戳: {train_data['timestamp']}
总轮数: {train_data['total_epochs']}
最佳模型: {os.path.basename(train_data['best_model_path'])}

最终损失:
• Train: {train_losses[-1]:.6f}
• Val: {val_losses[-1]:.6f}

最佳损失:
• Train: {min(train_losses):.6f}
• Val: {min(val_losses):.6f}
"""
    ax2.text(0.05, 0.95, info_text, transform=ax2.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # 3. 验证指标曲线
    ax3 = fig.add_subplot(gs[1, :])
    metrics_names = ['Precision', 'Recall', 'F1', 'IoU', 'Score']
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#ff7f0e']
    
    for metric, color in zip(metrics_names, colors):
        values = [h['val_metrics'].get(metric, 0.0) for h in history]
        ax3.plot(epochs, values, color=color, marker='o', label=metric, 
                markersize=3, linewidth=2)
    
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Metric Value')
    ax3.set_title('Validation Metrics Evolution', fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    
    # 4. 测试指标柱状图
    if test_data:
        ax4 = fig.add_subplot(gs[2, :2])
        test_metrics = test_data['test_metrics']
        labels = ['Precision', 'Recall', 'F1', 'IoU', 'EMD', 'Score']
        values = [
            test_metrics['Precision'],
            test_metrics['Recall'],
            test_metrics['F1'],
            test_metrics['IoU'],
            test_metrics.get('EMD', 0.0),
            test_metrics['Score']
        ]
        colors_bar = ['#59a14f', '#f28e2b', '#e15759', '#76b7b2', '#edc948', '#9c755f']
        
        x = np.arange(len(labels))
        bars = ax4.bar(x, values, color=colors_bar, alpha=0.8, edgecolor='black')
        ax4.set_xticks(x)
        ax4.set_xticklabels(labels)
        ax4.set_ylabel('Value')
        ax4.set_title('Test Metrics Summary', fontweight='bold')
        ax4.grid(True, axis='y', alpha=0.3)
        
        for bar, val in zip(bars, values):
            ax4.text(bar.get_x() + bar.get_width()/2, val, f'{val:.3f}', 
                    ha='center', va='bottom', fontsize=8)
        
        # 5. 测试信息文本
        ax5 = fig.add_subplot(gs[2, 2])
        ax5.axis('off')
        test_info = f"""测试结果

测试损失: {test_data['test_loss']:.6f}

性能指标:
• Precision: {test_metrics['Precision']:.4f}
• Recall: {test_metrics['Recall']:.4f}
• F1 Score: {test_metrics['F1']:.4f}
• IoU: {test_metrics['IoU']:.4f}
• Score: {test_metrics['Score']:.4f}

测试样本数: {test_data.get('num_samples', 'N/A')}
"""
        ax5.text(0.05, 0.95, test_info, transform=ax5.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    fig.suptitle('Comprehensive Training & Testing Report', 
                fontsize=16, fontweight='bold')
    
    out_path = os.path.join(save_dir, 'comprehensive_report.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 保存综合报告: {out_path}")


def generate_all_plots(timestamp: str = None, train_json: str = None, test_json: Optional[str] = None, 
                      save_dir: str = './visual_res', max_samples: int = 20):
    """
    从JSON数据生成所有可视化图表
        
        Args:
        timestamp: 时间戳，自动查找对应的训练和测试数据
        train_json: 训练数据JSON路径（如果提供timestamp则忽略）
        test_json: 测试数据JSON路径（可选，如果提供timestamp则忽略）
        save_dir: 保存目录
        max_samples: 最大样本可视化数量
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 60)
    print("📊 生成可视化图表")
    print("=" * 60)
    
    # 如果提供了时间戳，自动组装路径
    if timestamp:
        curve_data_dir = 'plot/curve_data'
        train_json = os.path.join(curve_data_dir, f'train_{timestamp}.json')
        test_json = os.path.join(curve_data_dir, f'test_{timestamp}.json')
        print(f"\n🕐 使用时间戳: {timestamp}")
        
        # 检查文件是否存在
        if not os.path.exists(train_json):
            raise FileNotFoundError(f"未找到训练数据: {train_json}")
        if not os.path.exists(test_json):
            print(f"⚠️  未找到测试数据: {test_json}，仅使用训练数据生成可视化")
            test_json = None
    
    # 加载训练数据
    print(f"\n📂 加载训练数据: {train_json}")
    train_data = load_train_data(train_json)
    
    # 加载测试数据（如果提供）
    test_data = None
    if test_json:
        print(f"📂 加载测试数据: {test_json}")
        test_data = load_test_data(test_json)
    
    print(f"\n💾 保存目录: {save_dir}\n")
    
    # 生成各种图表
    plot_loss_curves(train_data, save_dir)
    plot_val_metrics_curves(train_data, save_dir)
    
    if test_data:
        plot_test_metrics_summary(test_data, save_dir)
        plot_random_test_samples(test_data, save_dir, max_samples)
    
    print("\n" + "=" * 60)
    print("✅ 所有可视化图表生成完成！")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser('Generate visualizations from JSON data')
    parser.add_argument('--timestamp', type=str, default=None, help='Model timestamp (e.g., 20251023_123456). Auto-find train and test JSON files.')
    parser.add_argument('--train_json', type=str, default=None, help='Path to train JSON file (ignored if timestamp is provided)')
    parser.add_argument('--test_json', type=str, default=None, help='Path to test JSON file (optional, ignored if timestamp is provided)')
    parser.add_argument('--save_dir', type=str, default='plot/visual_res', help='Directory to save visualizations')
    parser.add_argument('--max_samples', type=int, default=20, help='Maximum number of test samples to visualize')
    
    args = parser.parse_args()
    
    # 检查参数
    if args.timestamp is None and args.train_json is None:
        parser.error("必须提供 --timestamp 或 --train_json 参数之一")
    
    generate_all_plots(args.timestamp, args.train_json, args.test_json, args.save_dir, args.max_samples)


if __name__ == '__main__':
    main()
