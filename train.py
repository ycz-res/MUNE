"""
训练模块
包含训练循环、验证、模型保存等功能
"""

from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
import os
from datetime import datetime
import pandas as pd
import numpy as np

from dataset import SimDataset
from config import get_config
from model import LinearModel
from loss import MuThresholdLoss
from utils import set_seed


def get_args_parser():
    a_parser = argparse.ArgumentParser('VLP scripts', add_help=False)
    a_parser.add_argument('--batch_size', default=1, type=int)
    a_parser.add_argument('--epochs', default=20, type=int)
    a_parser.add_argument('--shuffle', default=True, type=bool)
    a_parser.add_argument('--num_workers', default=0, type=int)
    a_parser.add_argument('--pin_memory', default=False, type=bool)
    a_parser.add_argument('--device', default='cpu', type=str)

    return a_parser

def main(args):
    # 设置随机种子
    set_seed(57)
    
    config = get_config()
    
    # 创建训练曲线保存文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    curves_dir = f"training_curves_{timestamp}"
    os.makedirs(curves_dir, exist_ok=True)
    print(f"训练曲线将保存到: {curves_dir}")
    
    # 初始化训练历史记录
    train_losses = []
    val_losses = []
    epochs = []
    
    print("=== 加载数据集===")
    # 使用同一个数据文件的不同百分比范围，并指定阶段标签
    train_dataset = SimDataset(config['SimDataset.data'], 'sim', start_percent=0.0, end_percent=0.05, stage='train')  # 前70%
    val_dataset = SimDataset(config['SimDataset.data'], 'sim', start_percent=0.05, end_percent=0.1, stage='val')  # 70%-85%
    test_dataset = SimDataset(config['SimDataset.data'], 'sim', start_percent=0.1, end_percent=0.15, stage='test')  # 85%-100%
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             collate_fn=SimDataset.collate_fn, num_workers=args.num_workers, 
                             pin_memory=args.pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                           collate_fn=SimDataset.collate_fn, num_workers=args.num_workers, 
                           pin_memory=args.pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                            collate_fn=SimDataset.collate_fn, num_workers=args.num_workers, 
                            pin_memory=args.pin_memory)

    
    model = LinearModel(d_model=64, max_thresholds=160)
    model.to(args.device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.95), weight_decay=1e-3)
    
    # 使用MU数量和阈值的加权损失函数
    criterion = MuThresholdLoss
    
    # 训练模型
    print(f"开始训练，共 {args.epochs} 个epoch")
    
    for epoch in range(args.epochs):
        # 训练一个epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, args.device, epoch, args.epochs)
        
        # 验证一个epoch
        val_loss = validate_one_epoch(model, val_loader, criterion, args.device, epoch, args.epochs)
        
        # 记录训练历史
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        epochs.append(epoch + 1)
        
        print("-" * 50)
    
    print("训练完成！")
    
    # 测试
    print("\n=== 测试阶段 ===")
    avg_test_loss = test_one_epoch(model, test_loader, criterion, args.device)
    print(f"测试完成, 平均损失: {avg_test_loss:.6f}")
    
    # 生成综合训练过程可视化
    generate_comprehensive_training_visualization(epochs, train_losses, val_losses, curves_dir, args)
    print(f"训练曲线已保存到: {curves_dir}")


def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    batch_count = 0
    
    for batch_idx, batch in enumerate(train_loader):
        src, tgt = batch
        src = src.to(device)
        tgt = tgt.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        nus, thresholds = model(src)
        
        # 计算损失
        # 这里需要根据tgt的格式来计算损失
        # tgt格式: [nus, threshold1, threshold2, ...]
        nus_target = tgt[:, 0:1]  # 取第一个元素作为MU数量目标
        thresholds_target = tgt[:, 1:]  # 取剩余元素作为阈值目标
        loss = criterion(nus, nus_target, thresholds, thresholds_target)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        batch_count += 1
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}/{total_epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}")
    
    avg_loss = total_loss / batch_count
    print(f"Epoch {epoch+1}/{total_epochs} 训练完成, 平均损失: {avg_loss:.6f}")
    return avg_loss


def validate_one_epoch(model, val_loader, criterion, device, epoch, total_epochs):
    model.eval()
    val_loss = 0.0
    val_batch_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            src, tgt = batch
            src = src.to(device)
            tgt = tgt.to(device)
            
            nus, thresholds = model(src)
            nus_target = tgt[:, 0:1]
            thresholds_target = tgt[:, 1:]
            loss = criterion(nus, nus_target, thresholds, thresholds_target)
            
            val_loss += loss.item()
            val_batch_count += 1
    
    avg_val_loss = val_loss / val_batch_count
    print(f"Epoch {epoch+1}/{total_epochs} 验证完成, 平均损失: {avg_val_loss:.6f}")
    return avg_val_loss


def test_one_epoch(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    test_batch_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            src, tgt = batch
            src = src.to(device)
            tgt = tgt.to(device)
            
            nus, thresholds = model(src)
            nus_target = tgt[:, 0:1]
            thresholds_target = tgt[:, 1:]
            loss = criterion(nus, nus_target, thresholds, thresholds_target)
            
            test_loss += loss.item()
            test_batch_count += 1
            
            # 显示前几个预测结果
            if batch_idx < 3:
                print(f"测试批次 {batch_idx+1}:")
                print(f"  真实MU数量: {nus_target.squeeze().tolist()}")
                print(f"  预测MU数量: {nus.squeeze().tolist()}")
                print(f"  损失: {loss.item():.6f}")
    
    avg_test_loss = test_loss / test_batch_count
    return avg_test_loss


def generate_comprehensive_training_visualization(epochs, train_losses, val_losses, save_dir, args):
    """
    生成综合训练过程可视化
    
    Args:
        epochs: epoch列表
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        save_dir: 保存目录
        args: 训练参数
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建大型综合图表
    fig = plt.figure(figsize=(20, 16))
    
    # 1. 主训练曲线 (左上)
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2.5, marker='o', markersize=3)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2.5, marker='s', markersize=3)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss Curves', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#f8f9fa')
    
    # 2. 损失趋势分析 (中上)
    ax2 = plt.subplot(3, 3, 2)
    train_diff = [train_losses[i] - train_losses[i-1] for i in range(1, len(train_losses))]
    val_diff = [val_losses[i] - val_losses[i-1] for i in range(1, len(val_losses))]
    ax2.plot(epochs[1:], train_diff, 'b-', label='Training Loss Change', linewidth=2)
    ax2.plot(epochs[1:], val_diff, 'r-', label='Validation Loss Change', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss Change', fontsize=12)
    ax2.set_title('Loss Change Trend Analysis', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_facecolor('#f8f9fa')
    
    # 3. 训练稳定性分析 (右上)
    ax3 = plt.subplot(3, 3, 3)
    train_ma = [np.mean(train_losses[max(0, i-4):i+1]) for i in range(len(train_losses))]
    val_ma = [np.mean(val_losses[max(0, i-4):i+1]) for i in range(len(val_losses))]
    ax3.plot(epochs, train_ma, 'b-', label='Training MA(5)', linewidth=2)
    ax3.plot(epochs, val_ma, 'r-', label='Validation MA(5)', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Moving Average Loss', fontsize=12)
    ax3.set_title('Training Stability (5-Epoch Moving Average)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_facecolor('#f8f9fa')
    
    # 4. 最终损失对比 (左中)
    ax4 = plt.subplot(3, 3, 4)
    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1]
    bars = ax4.bar(['Training', 'Validation'], [final_train_loss, final_val_loss], 
                   color=['#3498db', '#e74c3c'], alpha=0.8, edgecolor='black', linewidth=1)
    ax4.set_ylabel('Final Loss', fontsize=12)
    ax4.set_title('Final Loss Comparison', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_facecolor('#f8f9fa')
    
    # 添加数值标签
    for bar, value in zip(bars, [final_train_loss, final_val_loss]):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 5. 过拟合分析 (中中)
    ax5 = plt.subplot(3, 3, 5)
    overfitting_gap = [val_losses[i] - train_losses[i] for i in range(len(epochs))]
    ax5.plot(epochs, overfitting_gap, 'g-', linewidth=2.5, marker='o', markersize=3)
    ax5.set_xlabel('Epoch', fontsize=12)
    ax5.set_ylabel('Overfitting Gap', fontsize=12)
    ax5.set_title('Overfitting Analysis', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax5.set_facecolor('#f8f9fa')
    
    # 6. 损失分布直方图 (右中)
    ax6 = plt.subplot(3, 3, 6)
    ax6.hist(train_losses, bins=15, alpha=0.7, color='blue', label='Training', edgecolor='black')
    ax6.hist(val_losses, bins=15, alpha=0.7, color='red', label='Validation', edgecolor='black')
    ax6.set_xlabel('Loss Value', fontsize=12)
    ax6.set_ylabel('Frequency', fontsize=12)
    ax6.set_title('Loss Distribution', fontsize=14, fontweight='bold')
    ax6.legend(fontsize=11)
    ax6.grid(True, alpha=0.3)
    ax6.set_facecolor('#f8f9fa')
    
    # 7. 训练统计信息 (左下)
    ax7 = plt.subplot(3, 3, 7)
    ax7.axis('off')
    
    # 计算统计信息
    best_train_loss = min(train_losses)
    best_val_loss = min(val_losses)
    train_range = max(train_losses) - min(train_losses)
    val_range = max(val_losses) - min(val_losses)
    final_overfitting = final_val_loss - final_train_loss
    
    stats_text = f"""
    📊 训练统计信息
    
    🎯 训练配置:
    • 总Epoch数: {len(epochs)}
    • 批次大小: {args.batch_size}
    • 学习率: 1e-5
    • 优化器: AdamW
    
    📈 损失统计:
    • 最终训练损失: {final_train_loss:.6f}
    • 最终验证损失: {final_val_loss:.6f}
    • 最佳训练损失: {best_train_loss:.6f}
    • 最佳验证损失: {best_val_loss:.6f}
    
    📊 变化范围:
    • 训练损失范围: {train_range:.6f}
    • 验证损失范围: {val_range:.6f}
    • 最终过拟合程度: {final_overfitting:.6f}
    
    🎯 训练质量评估:
    • 收敛性: {'良好' if train_range < 1.0 else '需改进'}
    • 过拟合: {'轻微' if abs(final_overfitting) < 0.5 else '明显'}
           • 稳定性: {'稳定' if len(train_diff) > 0 and max(abs(x) for x in train_diff[-5:]) < 0.1 else '波动'}
    """
    
    ax7.text(0.05, 0.95, stats_text, transform=ax7.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecf0f1', alpha=0.8))
    
    # 8. 训练阶段分析 (中下)
    ax8 = plt.subplot(3, 3, 8)
    # 将训练分为三个阶段
    total_epochs = len(epochs)
    early_stage = total_epochs // 3
    mid_stage = 2 * total_epochs // 3
    
    early_train = np.mean(train_losses[:early_stage])
    mid_train = np.mean(train_losses[early_stage:mid_stage])
    late_train = np.mean(train_losses[mid_stage:])
    
    early_val = np.mean(val_losses[:early_stage])
    mid_val = np.mean(val_losses[early_stage:mid_stage])
    late_val = np.mean(val_losses[mid_stage:])
    
    stages = ['Early\n(1-33%)', 'Middle\n(34-66%)', 'Late\n(67-100%)']
    train_means = [early_train, mid_train, late_train]
    val_means = [early_val, mid_val, late_val]
    
    x = np.arange(len(stages))
    width = 0.35
    
    ax8.bar(x - width/2, train_means, width, label='Training', color='#3498db', alpha=0.8)
    ax8.bar(x + width/2, val_means, width, label='Validation', color='#e74c3c', alpha=0.8)
    
    ax8.set_xlabel('Training Stage', fontsize=12)
    ax8.set_ylabel('Average Loss', fontsize=12)
    ax8.set_title('Loss by Training Stage', fontsize=14, fontweight='bold')
    ax8.set_xticks(x)
    ax8.set_xticklabels(stages)
    ax8.legend(fontsize=11)
    ax8.grid(True, alpha=0.3, axis='y')
    ax8.set_facecolor('#f8f9fa')
    
    # 9. 训练建议 (右下)
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # 基于分析给出建议
    recommendations = []
    if final_overfitting > 1.0:
        recommendations.append("🔧 建议增加正则化或减少模型复杂度")
    if train_range > 2.0:
        recommendations.append("📉 建议降低学习率或增加批次大小")
    if len(train_diff) > 0 and max(abs(x) for x in train_diff[-5:]) > 0.2:
        recommendations.append("⚡ 建议使用学习率调度器")
    if best_val_loss > best_train_loss * 1.5:
        recommendations.append("🎯 建议增加训练数据或数据增强")
    
    if not recommendations:
        recommendations.append("✅ 训练效果良好，建议继续当前配置")
    
    rec_text = "💡 训练建议:\n\n" + "\n".join(recommendations)
    
    ax9.text(0.05, 0.95, rec_text, transform=ax9.transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#d5f4e6', alpha=0.8))
    
    # 设置整体标题
    fig.suptitle('MUNE深度学习训练过程综合分析报告', fontsize=18, fontweight='bold', y=0.98)
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.94, hspace=0.3, wspace=0.3)
    
    # 保存综合图表
    plt.savefig(os.path.join(save_dir, 'comprehensive_training_analysis.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 保存训练数据到CSV
    import pandas as pd
    training_data = pd.DataFrame({
        'Epoch': epochs,
        'Training_Loss': train_losses,
        'Validation_Loss': val_losses,
        'Loss_Difference': [val_losses[i] - train_losses[i] for i in range(len(epochs))],
        'Training_Loss_Change': [0] + train_diff,
        'Validation_Loss_Change': [0] + val_diff
    })
    training_data.to_csv(os.path.join(save_dir, 'detailed_training_data.csv'), index=False)
    
    print(f"📊 综合训练分析报告已保存到: {os.path.join(save_dir, 'comprehensive_training_analysis.png')}")
    print(f"📈 详细训练数据已保存到: {os.path.join(save_dir, 'detailed_training_data.csv')}")


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)

