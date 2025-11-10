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
import torch

# 屏蔽中文字体警告
warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*missing from font.*')

# 设置中文字体和科研风格全局参数
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
# 科研风格全局设置
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['axes.titlesize'] = 13
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['figure.titlesize'] = 14
matplotlib.rcParams['axes.linewidth'] = 1.0
matplotlib.rcParams['grid.linewidth'] = 0.5
matplotlib.rcParams['grid.alpha'] = 0.3

# 科研配色方案（专业、清晰）
RESEARCH_COLORS = {
    'blue': '#2E5C8A',      # 深蓝色
    'light_blue': '#5B9BD5', # 浅蓝色
    'green': '#70AD47',     # 绿色
    'light_green': '#92D050', # 浅绿色
    'orange': '#ED7D31',    # 橙色
    'light_orange': '#FFC000', # 浅橙色
    'red': '#C55A11',       # 深红色
    'light_red': '#E74C3C', # 浅红色
    'purple': '#7030A0',    # 紫色
    'light_purple': '#9B59B6', # 浅紫色
    'gray': '#808080',      # 灰色
    'dark_gray': '#505050', # 深灰色
}

# Hatch模式（用于条形图，带颜色的斜线填充）
HATCH_PATTERNS = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']


def load_train_data(train_json_path: str) -> Dict:
    """加载训练数据"""
    with open(train_json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_test_data(test_json_path: str) -> Dict:
    """加载测试数据"""
    with open(test_json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_single_sample(src, thresholds_pred, thresholds_true, save_path, epoch=None, sample_idx=0, threshold=None):
    """
    绘制单个训练样本的可视化图
    
    Args:
        src: 输入数据字典，包含 'cmap' 键
        thresholds_pred: 预测的阈值 logits (tensor, shape: [batch_size, seq_len])
        thresholds_true: 真实的阈值 (tensor, shape: [batch_size, seq_len])
        save_path: 保存路径
        epoch: epoch编号（可选，用于标题）
        sample_idx: 要可视化的样本索引（默认0，即第一个样本）
        threshold: 用于二值化的阈值（默认0.5），将 logits 通过 sigmoid 后与此阈值比较
    """
    # 提取第一个样本
    cmap = src["cmap"][sample_idx].detach().cpu().numpy() if torch.is_tensor(src["cmap"]) else src["cmap"][sample_idx]
    thresholds_pred_sample = thresholds_pred[sample_idx].detach().cpu() if torch.is_tensor(thresholds_pred) else thresholds_pred[sample_idx]
    thresholds_true_sample = thresholds_true[sample_idx].detach().cpu() if torch.is_tensor(thresholds_true) else thresholds_true[sample_idx]
    
    # 转换为numpy数组
    if torch.is_tensor(thresholds_pred_sample):
        thresholds_pred_sample = thresholds_pred_sample.numpy()
    if torch.is_tensor(thresholds_true_sample):
        thresholds_true_sample = thresholds_true_sample.numpy()
    
    # 使用阈值（如果未提供，使用默认值0.5）
    if threshold is None:
        threshold = 0.5
    
    # 计算预测概率（只计算一次）
    if isinstance(thresholds_pred_sample, np.ndarray):
        pred_probs = torch.sigmoid(torch.from_numpy(thresholds_pred_sample)).numpy()
    else:
        pred_probs = torch.sigmoid(thresholds_pred_sample).numpy() if torch.is_tensor(thresholds_pred_sample) else thresholds_pred_sample
    
    # 使用threshold阈值进行二值化
    pred_binary = pred_probs > threshold
    
    # 计算MU数量（使用threshold阈值）
    mu_true = int((thresholds_true_sample > 0).sum())
    mu_pred = int(pred_binary.sum())
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_positions = np.arange(len(cmap))
    
    # 绘制 CMAP (散点图)
    ax.scatter(x_positions, cmap, c=RESEARCH_COLORS['light_blue'], s=1, alpha=0.3, label='CMAP')
    
    # 计算位置（使用threshold阈值）
    true_pos = set(np.where(thresholds_true_sample > 0)[0].tolist())
    pred_pos = set(np.where(pred_binary)[0].tolist())
    
    match_pos = sorted(true_pos & pred_pos)
    true_only_pos = sorted(true_pos - pred_pos)
    pred_only_pos = sorted(pred_pos - true_pos)
    
    # 绘制匹配的阈值 (绿色)
    for p in match_pos:
        ax.axvline(x=p, color=RESEARCH_COLORS['green'], linestyle='-', linewidth=1.5, alpha=0.9)
    
    # 绘制真实但未预测的 (蓝色虚线)
    for p in true_only_pos:
        ax.axvline(x=p, color=RESEARCH_COLORS['blue'], linestyle='--', linewidth=1.2, alpha=0.9)
    
    # 绘制预测但不真实的 (橙色)
    for p in pred_only_pos:
        ax.axvline(x=p, color=RESEARCH_COLORS['orange'], linestyle='-', linewidth=1.2, alpha=0.9)
    
    # 设置标题（显示阈值信息）
    title = f'Epoch {epoch} Training Sample | True MU: {mu_true} | Pred MU({threshold}): {mu_pred}' if epoch else f'Sample {sample_idx} | True MU: {mu_true} | Pred MU({threshold}): {mu_pred}'
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    ax.set_xlim(0, len(cmap) - 1)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_xlabel('Position', fontsize=11)
    ax.set_ylabel('Amplitude', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=RESEARCH_COLORS['green'], label='Match (True & Pred)'),
        Patch(facecolor=RESEARCH_COLORS['blue'], label='True Only (Miss)'),
        Patch(facecolor=RESEARCH_COLORS['orange'], label='Pred Only (False Alarm)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9, edgecolor='black')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_loss_curves(train_data: Dict, save_dir: str):
    """绘制训练和验证损失曲线（支持多个损失分量）"""
    history = train_data['training_history']
    epochs = [h['epoch'] for h in history]
    
    # 提取总损失（支持新旧格式）
    if 'train_loss_result' in history[0]:
        train_losses = [h['train_loss_result']['total'] for h in history]
        val_losses = [h['val_loss_result']['total'] for h in history]
        # 检查是否有多个损失分量
        first_train_losses = history[0]['train_loss_result'].get('losses', {})
        first_val_losses = history[0]['val_loss_result'].get('losses', {})
    else:
        # 旧格式兼容
        train_losses = [h['train_loss'] for h in history]
        val_losses = [h['val_loss'] for h in history]
        first_train_losses = {}
        first_val_losses = {}
    
    has_individual_losses = bool(first_train_losses or first_val_losses)
    
    if has_individual_losses:
        # 如果有多个损失分量，创建子图（共享x轴，优化间距和比例）
        fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
        fig.subplots_adjust(hspace=0.25, top=0.96, bottom=0.08, left=0.1, right=0.95)
        
        # 上图：总损失
        ax1 = axes[0]
        # 使用科研配色
        ax1.plot(epochs, train_losses, color=RESEARCH_COLORS['blue'], marker='o', label='Train Total Loss', 
                markersize=4, linewidth=2.5, alpha=0.85, markevery=max(1, len(epochs)//25), 
                markerfacecolor='white', markeredgewidth=1.2, markeredgecolor=RESEARCH_COLORS['blue'])
        ax1.plot(epochs, val_losses, color=RESEARCH_COLORS['red'], marker='s', label='Val Total Loss', 
                markersize=4, linewidth=2.5, alpha=0.85, markevery=max(1, len(epochs)//25),
                markerfacecolor='white', markeredgewidth=1.2, markeredgecolor=RESEARCH_COLORS['red'])
        ax1.set_ylabel('Total Loss', fontsize=12, fontweight='bold')
        ax1.set_title('Total Loss: Training vs Validation', fontsize=13, fontweight='bold', pad=10)
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color=RESEARCH_COLORS['gray'])
        ax1.legend(fontsize=10, loc='upper right', framealpha=0.9, edgecolor='black', 
                  fancybox=False, shadow=False, frameon=True)
        ax1.tick_params(axis='both', labelsize=10)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_color(RESEARCH_COLORS['dark_gray'])
        ax1.spines['bottom'].set_color(RESEARCH_COLORS['dark_gray'])
        
        # 下图：各个损失分量
        ax2 = axes[1]
        # 收集所有损失名称（从train和val中取并集）
        all_loss_names = set()
        for h in history:
            if 'train_loss_result' in h:
                all_loss_names.update(h['train_loss_result'].get('losses', {}).keys())
                all_loss_names.update(h['val_loss_result'].get('losses', {}).keys())
        all_loss_names = sorted(list(all_loss_names))
        
        # 为每个损失分量分配颜色（使用科研配色方案）
        color_map = {
            'ce': RESEARCH_COLORS['blue'],
            'dice': RESEARCH_COLORS['orange'],
            'iou': RESEARCH_COLORS['green'],
            'focal': RESEARCH_COLORS['red'],
            'bce': RESEARCH_COLORS['purple'],
            'weighted_bce': RESEARCH_COLORS['light_orange'],
        }
        # 如果损失名称不在映射中，使用科研风格的渐变色
        default_colors = [RESEARCH_COLORS['light_blue'], RESEARCH_COLORS['light_green'], 
                         RESEARCH_COLORS['light_purple'], RESEARCH_COLORS['gray']]
        colors = [color_map.get(name, default_colors[i % len(default_colors)]) 
                 for i, name in enumerate(all_loss_names)]
        
        # 绘制训练损失分量（实线，圆形标记）
        for loss_name, color in zip(all_loss_names, colors):
            train_loss_values = []
            for h in history:
                if 'train_loss_result' in h:
                    train_loss_values.append(h['train_loss_result'].get('losses', {}).get(loss_name, None))
                else:
                    train_loss_values.append(None)
            # 过滤掉None值
            valid_indices = [i for i, v in enumerate(train_loss_values) if v is not None]
            if valid_indices:
                valid_epochs = [epochs[i] for i in valid_indices]
                valid_values = [train_loss_values[i] for i in valid_indices]
                ax2.plot(valid_epochs, valid_values, color=color, marker='o', 
                        label=f'Train {loss_name.upper()}', markersize=3.5, linewidth=2.0, 
                        linestyle='-', alpha=0.85, markevery=max(1, len(valid_epochs)//20),
                        markerfacecolor='white', markeredgewidth=1.0, markeredgecolor=color)
        
        # 绘制验证损失分量（虚线，方形标记）
        for loss_name, color in zip(all_loss_names, colors):
            val_loss_values = []
            for h in history:
                if 'val_loss_result' in h:
                    val_loss_values.append(h['val_loss_result'].get('losses', {}).get(loss_name, None))
                else:
                    val_loss_values.append(None)
            # 过滤掉None值
            valid_indices = [i for i, v in enumerate(val_loss_values) if v is not None]
            if valid_indices:
                valid_epochs = [epochs[i] for i in valid_indices]
                valid_values = [val_loss_values[i] for i in valid_indices]
                ax2.plot(valid_epochs, valid_values, color=color, marker='s', 
                        label=f'Val {loss_name.upper()}', markersize=3.5, linewidth=2.0, 
                        linestyle='--', alpha=0.85, markevery=max(1, len(valid_epochs)//20),
                        markerfacecolor='white', markeredgewidth=1.0, markeredgecolor=color)
        
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Individual Loss', fontsize=12, fontweight='bold')
        ax2.set_title('Individual Loss Components: Training vs Validation', fontsize=13, fontweight='bold', pad=10)
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color=RESEARCH_COLORS['gray'])
        ax2.legend(fontsize=9, loc='upper right', ncol=2, framealpha=0.9, 
                  edgecolor='black', fancybox=False, shadow=False, frameon=True, columnspacing=0.8)
        ax2.tick_params(axis='both', labelsize=10)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_color(RESEARCH_COLORS['dark_gray'])
        ax2.spines['bottom'].set_color(RESEARCH_COLORS['dark_gray'])
        
        plt.tight_layout()
        out_path = os.path.join(save_dir, 'loss_curves.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        # 如果没有多个损失分量，只绘制总损失
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, train_losses, color=RESEARCH_COLORS['blue'], marker='o', 
               label='Train Loss', markersize=4, linewidth=2.5, alpha=0.85,
               markerfacecolor='white', markeredgewidth=1.2, markeredgecolor=RESEARCH_COLORS['blue'])
        ax.plot(epochs, val_losses, color=RESEARCH_COLORS['red'], marker='s', 
               label='Val Loss', markersize=4, linewidth=2.5, alpha=0.85,
               markerfacecolor='white', markeredgewidth=1.2, markeredgecolor=RESEARCH_COLORS['red'])
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax.set_title('Training vs Validation Loss', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color=RESEARCH_COLORS['gray'])
        ax.legend(fontsize=10, framealpha=0.9, edgecolor='black')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(RESEARCH_COLORS['dark_gray'])
        ax.spines['bottom'].set_color(RESEARCH_COLORS['dark_gray'])
        plt.tight_layout()
        
        out_path = os.path.join(save_dir, 'loss_curves.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✅ 保存损失曲线: {os.path.join(save_dir, 'loss_curves.png')}")


def plot_val_metrics_curves(train_data: Dict, save_dir: str):
    """绘制验证指标曲线"""
    history = train_data['training_history']
    epochs = [h['epoch'] for h in history]
    
    metrics_names = ['Precision', 'Recall', 'F1', 'IoU', 'EMD', 'Score']
    colors = [RESEARCH_COLORS['blue'], RESEARCH_COLORS['green'], RESEARCH_COLORS['red'], 
              RESEARCH_COLORS['purple'], RESEARCH_COLORS['orange'], RESEARCH_COLORS['light_orange']]
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for metric, color, marker in zip(metrics_names, colors, markers):
        values = [h['val_metrics'].get(metric, 0.0) for h in history]
        ax.plot(epochs, values, color=color, marker=marker, label=metric, 
                markersize=4, linewidth=2.0, alpha=0.85, markevery=max(1, len(epochs)//30),
                markerfacecolor='white', markeredgewidth=1.0, markeredgecolor=color)
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
    ax.set_title('Validation Metrics Over Epochs', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color=RESEARCH_COLORS['gray'])
    ax.legend(fontsize=10, loc='best', framealpha=0.9, edgecolor='black', ncol=3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(RESEARCH_COLORS['dark_gray'])
    ax.spines['bottom'].set_color(RESEARCH_COLORS['dark_gray'])
    plt.tight_layout()
    
    out_path = os.path.join(save_dir, 'val_metrics_curves.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 保存验证指标曲线: {out_path}")


def plot_test_metrics_summary(test_data: Dict, save_dir: str):
    """绘制测试指标柱状图（支持多个损失分量）"""
    # 支持新的损失格式
    test_loss_result = test_data.get('test_loss_result', {})
    if test_loss_result:
        test_loss = test_loss_result.get('total', test_data.get('test_loss', 0.0))
        test_losses = test_loss_result.get('losses', {})
    else:
        test_loss = test_data.get('test_loss', 0.0)
        test_losses = {}
    
    test_metrics = test_data['test_metrics']
    
    # 检查是否有多个损失分量
    has_individual_losses = bool(test_losses)
    
    if has_individual_losses:
        # 如果有多个损失分量，创建上下子图（损失在上，指标在下）
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # 上图：损失（总损失和各个分量）
        ax1 = axes[0]
        loss_labels = ['Total Loss'] + [f'{k} Loss' for k in sorted(test_losses.keys())]
        loss_values = [test_loss] + [test_losses[k] for k in sorted(test_losses.keys())]
        loss_colors = [RESEARCH_COLORS['blue']] + [
            RESEARCH_COLORS['green'], RESEARCH_COLORS['orange'], RESEARCH_COLORS['red'],
            RESEARCH_COLORS['purple'], RESEARCH_COLORS['light_orange'], RESEARCH_COLORS['gray']
        ][:len(test_losses)]
        
        x1 = np.arange(len(loss_labels))
        bars1 = ax1.bar(x1, loss_values, color=loss_colors, alpha=0.7, edgecolor='black', 
                       linewidth=1.2, hatch=[HATCH_PATTERNS[i % len(HATCH_PATTERNS)] for i in range(len(loss_labels))])
        ax1.set_xticks(x1)
        ax1.set_xticklabels(loss_labels, fontsize=11, rotation=15, ha='right')
        ax1.set_ylabel('Loss Value', fontsize=12, fontweight='bold')
        ax1.set_title('Test Loss Components', fontsize=13, fontweight='bold')
        ax1.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.5, color=RESEARCH_COLORS['gray'])
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_color(RESEARCH_COLORS['dark_gray'])
        ax1.spines['bottom'].set_color(RESEARCH_COLORS['dark_gray'])
        
        # 在柱状图上添加数值
        for bar, val in zip(bars1, loss_values):
            ax1.text(bar.get_x() + bar.get_width()/2, val, f'{val:.4f}', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 下图：其他指标
        ax2 = axes[1]
        metric_labels = ['Precision', 'Recall', 'F1', 'IoU', 'EMD', 'Score']
        metric_values = [
            test_metrics['Precision'],
            test_metrics['Recall'],
            test_metrics['F1'],
            test_metrics['IoU'],
            test_metrics.get('EMD', 0.0),
            test_metrics['Score']
        ]
        metric_colors = [RESEARCH_COLORS['green'], RESEARCH_COLORS['orange'], RESEARCH_COLORS['red'],
                        RESEARCH_COLORS['purple'], RESEARCH_COLORS['light_orange'], RESEARCH_COLORS['blue']]
        
        x2 = np.arange(len(metric_labels))
        bars2 = ax2.bar(x2, metric_values, color=metric_colors, alpha=0.7, edgecolor='black', 
                       linewidth=1.2, hatch=[HATCH_PATTERNS[i % len(HATCH_PATTERNS)] for i in range(len(metric_labels))])
        ax2.set_xticks(x2)
        ax2.set_xticklabels(metric_labels, fontsize=11, rotation=15, ha='right')
        ax2.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax2.set_title('Test Metrics', fontsize=13, fontweight='bold')
        ax2.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.5, color=RESEARCH_COLORS['gray'])
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_color(RESEARCH_COLORS['dark_gray'])
        ax2.spines['bottom'].set_color(RESEARCH_COLORS['dark_gray'])
        
        # 在柱状图上添加数值
        for bar, val in zip(bars2, metric_values):
            ax2.text(bar.get_x() + bar.get_width()/2, val, f'{val:.4f}', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        fig.suptitle('Test Metrics Summary', fontsize=14, fontweight='bold')
        plt.tight_layout()
    else:
        # 如果没有多个损失分量，只绘制总损失和其他指标
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
        
        colors = [RESEARCH_COLORS['blue'], RESEARCH_COLORS['green'], RESEARCH_COLORS['orange'],
                 RESEARCH_COLORS['red'], RESEARCH_COLORS['purple'], RESEARCH_COLORS['light_orange'], RESEARCH_COLORS['gray']]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(labels))
        bars = ax.bar(x, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.2,
                     hatch=[HATCH_PATTERNS[i % len(HATCH_PATTERNS)] for i in range(len(labels))])
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11, rotation=15, ha='right')
        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax.set_title('Test Metrics Summary', fontsize=13, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.5, color=RESEARCH_COLORS['gray'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(RESEARCH_COLORS['dark_gray'])
        ax.spines['bottom'].set_color(RESEARCH_COLORS['dark_gray'])
        
        # 在柱状图上添加数值
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(bar.get_x() + bar.get_width()/2, val, f'{val:.4f}', 
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
    
    # 从测试数据配置中获取阈值（如果存在）
    threshold = 0.5  # 默认阈值
    if 'config_args' in test_data and test_data['config_args']:
        threshold = test_data['config_args'].get('metrics_threshold', 0.5)
    
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
    
    fig.suptitle(f'Random Test Samples: True vs Predicted Thresholds (threshold={threshold})', 
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
        ax.scatter(x_positions, cmap_sample, c=RESEARCH_COLORS['light_blue'], s=1, alpha=0.3, label='CMAP')
        
        # 将预测的 logits 转换为概率并二值化
        pred_probs = torch.sigmoid(torch.from_numpy(thr_pred_sample)).numpy()
        pred_binary = pred_probs > threshold
        
        # 计算位置
        true_pos = set(np.where(thr_true_sample > 0)[0].tolist())
        pred_pos = set(np.where(pred_binary)[0].tolist())
        
        match_pos = sorted(true_pos & pred_pos)
        true_only_pos = sorted(true_pos - pred_pos)
        pred_only_pos = sorted(pred_pos - true_pos)
        
        # 绘制匹配的阈值 (绿色)
        for p in match_pos:
            ax.axvline(x=p, color=RESEARCH_COLORS['green'], linestyle='-', linewidth=1.5, alpha=0.9)
        
        # 绘制真实但未预测的 (蓝色虚线)
        for p in true_only_pos:
            ax.axvline(x=p, color=RESEARCH_COLORS['blue'], linestyle='--', linewidth=1.2, alpha=0.9)
        
        # 绘制预测但不真实的 (橙色)
        for p in pred_only_pos:
            ax.axvline(x=p, color=RESEARCH_COLORS['orange'], linestyle='-', linewidth=1.2, alpha=0.9)
        
        # 计算预测的 MU 数量（使用二值化后的结果）
        pred_mu = int(pred_binary.sum())
        
        ax.set_title(f'Sample {idx} | True MU: {mu_true} | Pred MU({threshold}): {pred_mu}', 
                    fontsize=9, fontweight='bold')
        ax.set_xlim(0, 499)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_xlabel('Position', fontsize=9)
        ax.set_ylabel('Amplitude', fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # 隐藏多余的子图
    total_plots = rows * cols
    for k in range(num_samples, total_plots):
        row = k // cols
        col = k % cols
        axes[row, col].axis('off')
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=RESEARCH_COLORS['green'], label='Match (True & Pred)'),
        Patch(facecolor=RESEARCH_COLORS['blue'], label='True Only (Miss)'),
        Patch(facecolor=RESEARCH_COLORS['orange'], label='Pred Only (False Alarm)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', 
              bbox_to_anchor=(0.5, -0.01), ncol=3, fontsize=10, frameon=True, 
              framealpha=0.9, edgecolor='black')
    
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
    # 支持新旧格式
    if 'train_loss_result' in history[0]:
        train_losses = [h['train_loss_result']['total'] for h in history]
        val_losses = [h['val_loss_result']['total'] for h in history]
    else:
        train_losses = [h['train_loss'] for h in history]
        val_losses = [h['val_loss'] for h in history]
    
    ax1.plot(epochs, train_losses, color=RESEARCH_COLORS['blue'], marker='o', 
            label='Train Total Loss', markersize=3, linewidth=2.0, alpha=0.85)
    ax1.plot(epochs, val_losses, color=RESEARCH_COLORS['red'], marker='s', 
            label='Val Total Loss', markersize=3, linewidth=2.0, alpha=0.85)
    if test_data:
        test_loss = test_data.get('test_loss_result', {}).get('total', test_data.get('test_loss', None))
        if test_loss is not None:
            ax1.axhline(y=test_loss, color=RESEARCH_COLORS['green'], linestyle='--', 
                       linewidth=2, alpha=0.85, label=f"Test Loss: {test_loss:.4f}")
    ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Total Loss', fontsize=11, fontweight='bold')
    ax1.set_title('Training Progress: Total Loss Curves', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, framealpha=0.9, edgecolor='black')
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color=RESEARCH_COLORS['gray'])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color(RESEARCH_COLORS['dark_gray'])
    ax1.spines['bottom'].set_color(RESEARCH_COLORS['dark_gray'])
    
    # 2. 训练信息文本
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    # 检查是否有多个损失分量（支持新旧格式）
    loss_info = ""
    if 'train_loss_result' in history[0]:
        first_train_losses = history[0]['train_loss_result'].get('losses', {})
        has_individual_losses = bool(first_train_losses)
        if has_individual_losses:
            # 获取最终损失分量
            final_train_losses = history[-1]['train_loss_result'].get('losses', {})
            final_val_losses = history[-1]['val_loss_result'].get('losses', {})
            loss_info = "\n损失分量:\n"
            for loss_name in sorted(set(list(final_train_losses.keys()) + list(final_val_losses.keys()))):
                train_val = final_train_losses.get(loss_name, 'N/A')
                val_val = final_val_losses.get(loss_name, 'N/A')
                if isinstance(train_val, (int, float)) and isinstance(val_val, (int, float)):
                    loss_info += f"• {loss_name}: T={train_val:.4f}, V={val_val:.4f}\n"
    
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
{loss_info}
"""
    ax2.text(0.05, 0.95, info_text, transform=ax2.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # 3. 验证指标曲线
    ax3 = fig.add_subplot(gs[1, :])
    metrics_names = ['Precision', 'Recall', 'F1', 'IoU', 'EMD', 'Score']
    colors = [RESEARCH_COLORS['blue'], RESEARCH_COLORS['green'], RESEARCH_COLORS['red'],
              RESEARCH_COLORS['purple'], RESEARCH_COLORS['orange'], RESEARCH_COLORS['light_orange']]
    
    for metric, color in zip(metrics_names, colors):
        values = [h['val_metrics'].get(metric, 0.0) for h in history]
        ax3.plot(epochs, values, color=color, marker='o', label=metric, 
                markersize=3, linewidth=2.0, alpha=0.85)
    
    ax3.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Metric Value', fontsize=11, fontweight='bold')
    ax3.set_title('Validation Metrics Evolution', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=9, framealpha=0.9, edgecolor='black', ncol=3)
    ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color=RESEARCH_COLORS['gray'])
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_color(RESEARCH_COLORS['dark_gray'])
    ax3.spines['bottom'].set_color(RESEARCH_COLORS['dark_gray'])
    
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
        colors_bar = [RESEARCH_COLORS['green'], RESEARCH_COLORS['orange'], RESEARCH_COLORS['red'],
                     RESEARCH_COLORS['purple'], RESEARCH_COLORS['light_orange'], RESEARCH_COLORS['blue']]
        
        x = np.arange(len(labels))
        bars = ax4.bar(x, values, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1.2,
                      hatch=[HATCH_PATTERNS[i % len(HATCH_PATTERNS)] for i in range(len(labels))])
        ax4.set_xticks(x)
        ax4.set_xticklabels(labels, fontsize=10, rotation=15, ha='right')
        ax4.set_ylabel('Value', fontsize=11, fontweight='bold')
        ax4.set_title('Test Metrics Summary', fontsize=12, fontweight='bold')
        ax4.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.5, color=RESEARCH_COLORS['gray'])
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.spines['left'].set_color(RESEARCH_COLORS['dark_gray'])
        ax4.spines['bottom'].set_color(RESEARCH_COLORS['dark_gray'])
        
        for bar, val in zip(bars, values):
            ax4.text(bar.get_x() + bar.get_width()/2, val, f'{val:.3f}', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 5. 测试信息文本
        ax5 = fig.add_subplot(gs[2, 2])
        ax5.axis('off')
        # 支持新的损失格式
        test_loss_result = test_data.get('test_loss_result', {})
        if test_loss_result:
            test_loss = test_loss_result.get('total', test_data.get('test_loss', 0.0))
            test_losses = test_loss_result.get('losses', {})
        else:
            test_loss = test_data.get('test_loss', 0.0)
            test_losses = {}
        
        loss_info = ""
        if test_losses:
            loss_info = "\n损失分量:\n"
            for loss_name, loss_value in sorted(test_losses.items()):
                loss_info += f"• {loss_name}: {loss_value:.4f}\n"
        
        test_info = f"""测试结果

测试损失: {test_loss:.6f}{loss_info}

性能指标:
• Precision: {test_metrics['Precision']:.4f}
• Recall: {test_metrics['Recall']:.4f}
• F1 Score: {test_metrics['F1']:.4f}
• IoU: {test_metrics['IoU']:.4f}
• EMD: {test_metrics.get('EMD', 0.0):.4f}
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


def generate_all_plots(timestamp: str, result_dir: str = 'result', max_samples: int = 20):
    """
    从JSON数据生成所有可视化图表
        
        Args:
        timestamp: 时间戳，自动查找对应的训练和测试数据
        result_dir: result根目录，保存目录自动设置为result/{timestamp}/visual/
        max_samples: 最大样本可视化数量
    """
    print("=" * 60)
    print("📊 生成可视化图表")
    print("=" * 60)
    
    # 组装路径
    result_timestamp_dir = os.path.join(result_dir, timestamp)
    train_json = os.path.join(result_timestamp_dir, f'train_{timestamp}.json')
    test_json = os.path.join(result_timestamp_dir, f'test_{timestamp}.json')
    save_dir = os.path.join(result_timestamp_dir, 'visual')
    
    print(f"\n🕐 使用时间戳: {timestamp}")
    
    # 检查文件是否存在
    if not os.path.exists(train_json):
        raise FileNotFoundError(f"未找到训练数据: {train_json}")
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载训练数据
    print(f"\n📂 加载训练数据: {train_json}")
    train_data = load_train_data(train_json)
    
    # 加载测试数据（如果存在）
    test_data = None
    if os.path.exists(test_json):
        print(f"📂 加载测试数据: {test_json}")
        test_data = load_test_data(test_json)
    else:
        print(f"⚠️  未找到测试数据: {test_json}，仅使用训练数据生成可视化")
    
    print(f"\n💾 保存目录: {save_dir}\n")
    
    # 生成各种图表
    plot_loss_curves(train_data, save_dir)
    plot_val_metrics_curves(train_data, save_dir)
    
    if test_data:
        plot_test_metrics_summary(test_data, save_dir)
        plot_random_test_samples(test_data, save_dir, max_samples)
    
    print("\n" + "=" * 60)
    print("✅ 所有可视化图表生成完成！")
    print(f"📁 保存目录: {save_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser('Generate visualizations from JSON data')
    parser.add_argument('--timestamp', type=str, required=True, help='Model timestamp (e.g., 20251023_123456). Auto-find train and test JSON files from result/{timestamp}/')
    parser.add_argument('--result_dir', type=str, default='result', help='Root directory containing experiment results (save dir auto-set to result/{timestamp}/visual/)')
    parser.add_argument('--max_samples', type=int, default=20, help='Maximum number of test samples to visualize')
    
    args = parser.parse_args()
    
    generate_all_plots(args.timestamp, args.result_dir, args.max_samples)


if __name__ == '__main__':
    main()
