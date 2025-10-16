"""
MU阈值预测任务可视化模块 - 简化版本
专注于训练损失曲线和验证集阈值分布分析
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
import warnings

# 屏蔽中文字体警告
warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*missing from font.*')

# 设置中文字体和样式
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.style.use('default')


class MUThresholdVisualizer:
    """MU阈值预测任务可视化器 - 简化版本"""
    
    def __init__(self, save_dir: str):
        """
        初始化可视化器
        
        Args:
            save_dir: 保存图表的目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 训练历史数据
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        self.epochs = []
        
        # 验证集指标历史
        self.val_metrics = {
            'precision': [],
            'recall': [],
            'f1': [],
            'iou': [],
            'score': []
        }
        
        # 验证集预测数据（用于阈值分布分析）
        self.val_predictions = []  # 存储每个epoch的验证集预测
        self.val_targets = []      # 存储每个epoch的验证集真实标签
        
    def update_epoch(self, epoch: int, train_loss: float, val_loss: float, 
                    test_loss: Optional[float] = None, val_metrics: Optional[Dict] = None,
                    val_pred: Optional[torch.Tensor] = None, val_target: Optional[torch.Tensor] = None):
        """
        更新一个epoch的训练数据
        
        Args:
            epoch: 当前epoch
            train_loss: 训练损失
            val_loss: 验证损失
            test_loss: 测试损失（可选）
            val_metrics: 验证指标（可选）
            val_pred: 验证集预测结果（可选）
            val_target: 验证集真实标签（可选）
        """
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        if test_loss is not None:
            self.test_losses.append(test_loss)
        
        # 更新验证指标
        if val_metrics:
            for key in self.val_metrics:
                if key in val_metrics:
                    self.val_metrics[key].append(val_metrics[key])
                else:
                    self.val_metrics[key].append(0.0)
        
        # 存储验证集预测数据（用于阈值分布分析）
        if val_pred is not None and val_target is not None:
            self.val_predictions.append(val_pred.detach().cpu().numpy())
            self.val_targets.append(val_target.detach().cpu().numpy())
    
    def plot_loss_curves(self):
        """绘制训练、验证、测试损失曲线"""
        plt.figure(figsize=(12, 8))
        
        # 损失曲线
        plt.subplot(2, 2, 1)
        plt.plot(self.epochs, self.train_losses, 'b-', label='训练损失', linewidth=2, marker='o', markersize=4)
        plt.plot(self.epochs, self.val_losses, 'r-', label='验证损失', linewidth=2, marker='s', markersize=4)
        if self.test_losses:
            plt.plot(self.epochs, self.test_losses, 'g-', label='测试损失', linewidth=2, marker='^', markersize=4)
        
        plt.xlabel('Epoch')
        plt.ylabel('损失值')
        plt.title('训练过程损失曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 验证指标曲线
        plt.subplot(2, 2, 2)
        if self.val_metrics['score']:
            plt.plot(self.epochs, self.val_metrics['score'], 'purple', label='综合分数', linewidth=2, marker='o')
            plt.plot(self.epochs, self.val_metrics['f1'], 'orange', label='F1分数', linewidth=2, marker='s')
            plt.plot(self.epochs, self.val_metrics['iou'], 'green', label='IoU', linewidth=2, marker='^')
            plt.plot(self.epochs, self.val_metrics['precision'], 'blue', label='精确率', linewidth=2, marker='d')
            plt.plot(self.epochs, self.val_metrics['recall'], 'red', label='召回率', linewidth=2, marker='v')
        
        plt.xlabel('Epoch')
        plt.ylabel('指标值')
        plt.title('验证集指标变化')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 损失统计信息
        plt.subplot(2, 2, 3)
        final_train = self.train_losses[-1] if self.train_losses else 0
        final_val = self.val_losses[-1] if self.val_losses else 0
        best_train = min(self.train_losses) if self.train_losses else 0
        best_val = min(self.val_losses) if self.val_losses else 0
        
        stats_text = f"""训练统计信息
        
损失统计:
• 最终训练损失: {final_train:.6f}
• 最终验证损失: {final_val:.6f}
• 最佳训练损失: {best_train:.6f}
• 最佳验证损失: {best_val:.6f}

训练轮数: {len(self.epochs)}
        """
        
        if self.test_losses:
            final_test = self.test_losses[-1]
            stats_text += f"• 最终测试损失: {final_test:.6f}"
        
        plt.text(0.05, 0.5, stats_text, fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        plt.axis('off')
        plt.title('训练统计')
        
        # 最佳指标
        plt.subplot(2, 2, 4)
        if self.val_metrics['score']:
            best_score = max(self.val_metrics['score'])
            best_f1 = max(self.val_metrics['f1'])
            best_iou = max(self.val_metrics['iou'])
            best_precision = max(self.val_metrics['precision'])
            best_recall = max(self.val_metrics['recall'])
            
            metrics_text = f"""最佳验证指标
            
• 最佳综合分数: {best_score:.4f}
• 最佳F1分数: {best_f1:.4f}
• 最佳IoU: {best_iou:.4f}
• 最佳精确率: {best_precision:.4f}
• 最佳召回率: {best_recall:.4f}
            """
            
            plt.text(0.05, 0.5, metrics_text, fontsize=11, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        
        plt.axis('off')
        plt.title('最佳指标')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_threshold_distributions(self, threshold: float = 0.1):
        """
        绘制验证集激活阈值分布
        
        Args:
            threshold: 用于二值化的阈值
        """
        if not self.val_predictions or not self.val_targets:
            print("⚠️ 没有验证集预测数据，跳过阈值分布分析")
            return
        
        # 选择几个关键epoch进行展示
        key_epochs = [0, len(self.epochs)//4, len(self.epochs)//2, len(self.epochs)-1]
        key_epochs = [e for e in key_epochs if e < len(self.val_predictions)]
        
        fig, axes = plt.subplots(2, len(key_epochs), figsize=(5*len(key_epochs), 10))
        if len(key_epochs) == 1:
            axes = axes.reshape(2, 1)
        
        fig.suptitle('验证集激活阈值分布分析', fontsize=16, fontweight='bold')
        
        for i, epoch_idx in enumerate(key_epochs):
            epoch = self.epochs[epoch_idx]
            pred = self.val_predictions[epoch_idx]  # (N, 500)
            target = self.val_targets[epoch_idx]     # (N, 500)
            
            # 将预测转换为概率并二值化
            prob = torch.sigmoid(torch.tensor(pred)).numpy()
            pred_binary = (prob >= threshold).astype(float)
            
            # 上排：预测概率分布
            axes[0, i].hist(prob.flatten(), bins=50, alpha=0.7, color='blue', density=True)
            axes[0, i].axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'阈值={threshold}')
            axes[0, i].set_xlabel('预测概率')
            axes[0, i].set_ylabel('密度')
            axes[0, i].set_title(f'Epoch {epoch}: 预测概率分布')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)
            
            # 下排：激活位置分布
            pred_positions = np.sum(pred_binary, axis=1)  # 每个样本的激活位置数
            target_positions = np.sum(target, axis=1)     # 每个样本的真实位置数
            
            axes[1, i].scatter(target_positions, pred_positions, alpha=0.6, s=30)
            min_val = min(min(target_positions), min(pred_positions))
            max_val = max(max(target_positions), max(pred_positions))
            axes[1, i].plot([min_val, max_val], [min_val, max_val], 'r--', label='完美预测线')
            
            # 计算相关系数
            corr = np.corrcoef(target_positions, pred_positions)[0, 1]
            axes[1, i].text(0.05, 0.95, f'相关系数: {corr:.3f}', transform=axes[1, i].transAxes,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            axes[1, i].set_xlabel('真实激活位置数')
            axes[1, i].set_ylabel('预测激活位置数')
            axes[1, i].set_title(f'Epoch {epoch}: 激活位置对比')
            axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'threshold_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_training_data(self, test_loss: float = None):
        """保存训练数据到CSV"""
        training_data = {
            'epoch': self.epochs,
            'train_loss': self.train_losses,
            'val_loss': self.val_losses
        }
        
        if test_loss is not None:
            training_data['test_loss'] = [test_loss] * len(self.epochs)
        
        # 添加验证指标
        for key, values in self.val_metrics.items():
            if values:
                training_data[f'val_{key}'] = values
        
        df = pd.DataFrame(training_data)
        df.to_csv(os.path.join(self.save_dir, 'training_data.csv'), index=False)
    
    def generate_comprehensive_report(self, test_loss: float = None, 
                                    model_info: Dict[str, Any] = None,
                                    dataset_info: Dict[str, Any] = None,
                                    threshold: float = 0.1):
        """
        生成综合训练报告
        
        Args:
            test_loss: 测试损失
            model_info: 模型信息
            dataset_info: 数据集信息
            threshold: 阈值分布分析的阈值
        """
        # 生成损失曲线
        self.plot_loss_curves()
        
        # 生成阈值分布分析
        self.plot_threshold_distributions(threshold)
        
        # 保存训练数据
        self.save_training_data(test_loss)
        
        print(f"📊 训练曲线已保存到: {self.save_dir}/training_curves.png")
        print(f"📈 阈值分布分析已保存到: {self.save_dir}/threshold_distributions.png")
        print(f"📋 训练数据已保存到: {self.save_dir}/training_data.csv")
