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
        
        # 样本级统计数据
        self.mu_count_stats = {
            'true_counts': [],      # 每轮的真实MU数量
            'pred_counts': [],      # 每轮的预测MU数量
            'epochs': []            # 对应的epoch
        }
        
        self.threshold_stats = {
            'true_positions': [],   # 每轮的真实阈值位置
            'pred_positions': [],   # 每轮的预测阈值位置
            'epochs': []            # 对应的epoch
        }
        
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
            pred_np = val_pred.detach().cpu().numpy()
            target_np = val_target.detach().cpu().numpy()
            
            self.val_predictions.append(pred_np)
            self.val_targets.append(target_np)
            
            # 计算样本级统计
            self._update_sample_stats(epoch, pred_np, target_np)
    
    def _update_sample_stats(self, epoch: int, pred: np.ndarray, target: np.ndarray, threshold: float = 0.1):
        """
        更新样本级统计数据
        
        Args:
            epoch: 当前epoch
            pred: 预测结果 (N, 500)
            target: 真实标签 (N, 500)
            threshold: 二值化阈值
        """
        # 将预测转换为概率并二值化
        prob = torch.sigmoid(torch.tensor(pred)).numpy()
        pred_binary = (prob >= threshold).astype(float)
        
        # 计算每个样本的MU数量（激活位置数）
        true_mu_counts = np.sum(target, axis=1)  # (N,)
        pred_mu_counts = np.sum(pred_binary, axis=1)  # (N,)
        
        # 存储MU数量统计
        self.mu_count_stats['true_counts'].append(true_mu_counts)
        self.mu_count_stats['pred_counts'].append(pred_mu_counts)
        self.mu_count_stats['epochs'].append(epoch)
        
        # 计算阈值位置统计（每个样本的阈值位置）
        true_positions = []
        pred_positions = []
        
        for i in range(len(target)):
            # 真实阈值位置
            true_pos = np.where(target[i] > 0)[0].tolist()
            true_positions.append(true_pos)
            
            # 预测阈值位置
            pred_pos = np.where(pred_binary[i] > 0)[0].tolist()
            pred_positions.append(pred_pos)
        
        # 存储阈值位置统计
        self.threshold_stats['true_positions'].append(true_positions)
        self.threshold_stats['pred_positions'].append(pred_positions)
        self.threshold_stats['epochs'].append(epoch)
    
    def plot_loss_curves(self):
        """绘制训练、验证、测试损失曲线"""
        plt.figure(figsize=(16, 12))
        
        # 1. 损失曲线
        plt.subplot(3, 3, 1)
        plt.plot(self.epochs, self.train_losses, 'b-', label='训练损失', linewidth=2, marker='o', markersize=4)
        plt.plot(self.epochs, self.val_losses, 'r-', label='验证损失', linewidth=2, marker='s', markersize=4)
        if self.test_losses:
            plt.plot(self.epochs, self.test_losses, 'g-', label='测试损失', linewidth=2, marker='^', markersize=4)
        
        plt.xlabel('Epoch')
        plt.ylabel('损失值')
        plt.title('训练过程损失曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. 验证指标曲线
        plt.subplot(3, 3, 2)
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
        
        # 3. 损失变化率（梯度）
        plt.subplot(3, 3, 3)
        if len(self.train_losses) > 1:
            train_grad = np.gradient(self.train_losses)
            val_grad = np.gradient(self.val_losses)
            plt.plot(self.epochs[1:], train_grad[1:], 'b-', label='训练损失梯度', linewidth=2)
            plt.plot(self.epochs[1:], val_grad[1:], 'r-', label='验证损失梯度', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('损失变化率')
            plt.title('损失变化趋势')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 4. 训练/验证损失比值
        plt.subplot(3, 3, 4)
        if len(self.train_losses) > 0 and len(self.val_losses) > 0:
            loss_ratio = np.array(self.val_losses) / (np.array(self.train_losses) + 1e-8)
            plt.plot(self.epochs, loss_ratio, 'purple', linewidth=2, marker='o', markersize=4)
            plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='理想比值=1')
            plt.xlabel('Epoch')
            plt.ylabel('验证损失/训练损失')
            plt.title('过拟合监控')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 5. 指标改善趋势
        plt.subplot(3, 3, 5)
        if self.val_metrics['score'] and len(self.val_metrics['score']) > 1:
            score_improvement = np.diff(self.val_metrics['score'])
            plt.plot(self.epochs[1:], score_improvement, 'green', linewidth=2, marker='o', markersize=4)
            plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            plt.xlabel('Epoch')
            plt.ylabel('Score改善量')
            plt.title('综合分数改善趋势')
            plt.grid(True, alpha=0.3)
        
        # 6. 精确率vs召回率权衡
        plt.subplot(3, 3, 6)
        if self.val_metrics['precision'] and self.val_metrics['recall']:
            plt.plot(self.val_metrics['precision'], self.val_metrics['recall'], 'purple', 
                    linewidth=2, marker='o', markersize=6)
            plt.scatter(self.val_metrics['precision'][0], self.val_metrics['recall'][0], 
                       color='green', s=100, label='开始', zorder=5)
            plt.scatter(self.val_metrics['precision'][-1], self.val_metrics['recall'][-1], 
                       color='red', s=100, label='结束', zorder=5)
            plt.xlabel('精确率')
            plt.ylabel('召回率')
            plt.title('P-R权衡曲线')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 7. 损失统计信息
        plt.subplot(3, 3, 7)
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
        
        plt.text(0.05, 0.5, stats_text, fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        plt.axis('off')
        plt.title('训练统计')
        
        # 8. 最佳指标
        plt.subplot(3, 3, 8)
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
            
            plt.text(0.05, 0.5, metrics_text, fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        
        plt.axis('off')
        plt.title('最佳指标')
        
        # 9. 训练稳定性分析
        plt.subplot(3, 3, 9)
        if len(self.train_losses) > 2:
            # 计算损失的标准差（稳定性指标）
            train_std = np.std(self.train_losses)
            val_std = np.std(self.val_losses)
            
            stability_text = f"""训练稳定性分析
            
• 训练损失标准差: {train_std:.6f}
• 验证损失标准差: {val_std:.6f}
• 训练稳定性: {'稳定' if train_std < 0.01 else '不稳定'}
• 验证稳定性: {'稳定' if val_std < 0.01 else '不稳定'}
            """
            
            plt.text(0.05, 0.5, stability_text, fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
        
        plt.axis('off')
        plt.title('稳定性分析')
        
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
    
    def plot_sample_comparison(self, threshold: float = 0.1):
        """
        绘制样本级MU数量和阈值位置对比图
        
        Args:
            threshold: 用于二值化的阈值
        """
        if not self.mu_count_stats['epochs']:
            print("⚠️ 没有样本统计数据，跳过样本对比分析")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('样本级MU数量和阈值位置对比分析', fontsize=16, fontweight='bold')
        
        epochs = self.mu_count_stats['epochs']
        
        # 1. MU数量预测准确性随时间变化
        axes[0, 0].set_title('MU数量预测准确性')
        for i, epoch in enumerate(epochs):
            true_counts = self.mu_count_stats['true_counts'][i]
            pred_counts = self.mu_count_stats['pred_counts'][i]
            
            # 计算相关系数
            corr = np.corrcoef(true_counts, pred_counts)[0, 1]
            
            # 绘制散点图
            axes[0, 0].scatter(true_counts, pred_counts, alpha=0.6, s=20, 
                             label=f'Epoch {epoch} (r={corr:.3f})')
        
        # 添加完美预测线
        all_true = np.concatenate(self.mu_count_stats['true_counts'])
        all_pred = np.concatenate(self.mu_count_stats['pred_counts'])
        min_val = min(min(all_true), min(all_pred))
        max_val = max(max(all_true), max(all_pred))
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美预测线')
        
        axes[0, 0].set_xlabel('真实MU数量')
        axes[0, 0].set_ylabel('预测MU数量')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. MU数量预测误差分布
        axes[0, 1].set_title('MU数量预测误差分布')
        for i, epoch in enumerate(epochs):
            true_counts = self.mu_count_stats['true_counts'][i]
            pred_counts = self.mu_count_stats['pred_counts'][i]
            errors = pred_counts - true_counts
            
            axes[0, 1].hist(errors, bins=20, alpha=0.5, label=f'Epoch {epoch}', density=True)
        
        axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='零误差线')
        axes[0, 1].set_xlabel('预测误差 (预测 - 真实)')
        axes[0, 1].set_ylabel('密度')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. MU数量预测准确性趋势
        axes[0, 2].set_title('MU数量预测准确性趋势')
        correlations = []
        mae_scores = []
        
        for i, epoch in enumerate(epochs):
            true_counts = self.mu_count_stats['true_counts'][i]
            pred_counts = self.mu_count_stats['pred_counts'][i]
            
            corr = np.corrcoef(true_counts, pred_counts)[0, 1]
            mae = np.mean(np.abs(pred_counts - true_counts))
            
            correlations.append(corr)
            mae_scores.append(mae)
        
        ax2 = axes[0, 2].twinx()
        line1 = axes[0, 2].plot(epochs, correlations, 'b-o', linewidth=2, label='相关系数')
        line2 = ax2.plot(epochs, mae_scores, 'r-s', linewidth=2, label='MAE')
        
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('相关系数', color='b')
        ax2.set_ylabel('MAE', color='r')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        axes[0, 2].legend(lines, labels, loc='upper right')
        
        # 4. 阈值位置重叠分析
        axes[1, 0].set_title('阈值位置重叠分析')
        overlap_ratios = []
        
        for i, epoch in enumerate(epochs):
            true_positions = self.threshold_stats['true_positions'][i]
            pred_positions = self.threshold_stats['pred_positions'][i]
            
            overlaps = []
            for j in range(len(true_positions)):
                true_set = set(true_positions[j])
                pred_set = set(pred_positions[j])
                
                if len(true_set) > 0:
                    overlap_ratio = len(true_set & pred_set) / len(true_set)
                    overlaps.append(overlap_ratio)
                else:
                    overlaps.append(1.0 if len(pred_set) == 0 else 0.0)
            
            overlap_ratios.append(np.mean(overlaps))
        
        axes[1, 0].plot(epochs, overlap_ratios, 'g-o', linewidth=2, markersize=6)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('平均重叠比例')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 样本预测示例（选择几个代表性样本）
        axes[1, 1].set_title('样本预测示例')
        if len(epochs) >= 2:
            # 选择第一个和最后一个epoch
            first_epoch_idx = 0
            last_epoch_idx = -1
            
            # 选择几个代表性样本
            sample_indices = [0, len(self.mu_count_stats['true_counts'][0])//2, -1]
            
            for sample_idx in sample_indices:
                if sample_idx < len(self.mu_count_stats['true_counts'][0]):
                    true_count = self.mu_count_stats['true_counts'][first_epoch_idx][sample_idx]
                    pred_count_first = self.mu_count_stats['pred_counts'][first_epoch_idx][sample_idx]
                    pred_count_last = self.mu_count_stats['pred_counts'][last_epoch_idx][sample_idx]
                    
                    axes[1, 1].bar([f'样本{sample_idx+1}\n(Epoch {epochs[first_epoch_idx]})', 
                                   f'样本{sample_idx+1}\n(Epoch {epochs[last_epoch_idx]})'], 
                                  [true_count, true_count], alpha=0.7, label=f'真实 (样本{sample_idx+1})')
                    axes[1, 1].bar([f'样本{sample_idx+1}\n(Epoch {epochs[first_epoch_idx]})', 
                                   f'样本{sample_idx+1}\n(Epoch {epochs[last_epoch_idx]})'], 
                                  [pred_count_first, pred_count_last], alpha=0.5, 
                                  label=f'预测 (样本{sample_idx+1})')
        
        axes[1, 1].set_ylabel('MU数量')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 统计摘要
        axes[1, 2].set_title('统计摘要')
        final_corr = correlations[-1] if correlations else 0
        final_mae = mae_scores[-1] if mae_scores else 0
        final_overlap = overlap_ratios[-1] if overlap_ratios else 0
        
        summary_text = f"""样本级预测统计摘要
        
最终性能:
• MU数量相关系数: {final_corr:.4f}
• MU数量MAE: {final_mae:.2f}
• 阈值位置重叠率: {final_overlap:.4f}

训练趋势:
• 相关系数改善: {correlations[-1] - correlations[0]:.4f}
• MAE改善: {mae_scores[0] - mae_scores[-1]:.2f}
• 重叠率改善: {overlap_ratios[-1] - overlap_ratios[0]:.4f}

样本数量: {len(self.mu_count_stats['true_counts'][0])}
训练轮数: {len(epochs)}
        """
        
        axes[1, 2].text(0.05, 0.5, summary_text, fontsize=10, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.7))
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'sample_comparison.png'), dpi=300, bbox_inches='tight')
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
        
        # 生成样本级对比分析
        self.plot_sample_comparison(threshold)
        
        # 保存训练数据
        self.save_training_data(test_loss)
        
        print(f"📊 训练曲线已保存到: {self.save_dir}/training_curves.png")
        print(f"📈 阈值分布分析已保存到: {self.save_dir}/threshold_distributions.png")
        print(f"🎯 样本对比分析已保存到: {self.save_dir}/sample_comparison.png")
        print(f"📋 训练数据已保存到: {self.save_dir}/training_data.csv")
