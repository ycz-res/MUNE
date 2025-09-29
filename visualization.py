"""
MU阈值预测任务可视化模块 - 优化版本
专门针对运动单位(MU)阈值预测任务的训练过程可视化
包含更多分析功能和改进的图表设计
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch

# 尝试导入可选依赖
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from sklearn.metrics import confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# 设置中文字体和样式
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置样式
if HAS_SEABORN:
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
else:
    plt.style.use('default')


class MUThresholdVisualizer:
    """MU阈值预测任务可视化器 - 优化版本"""
    
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
        
        # MU阈值预测相关数据
        self.predicted_mu_counts = []
        self.true_mu_counts = []
        self.predicted_threshold_counts = []
        self.true_threshold_counts = []
        
        # 新增：详细指标历史
        self.metrics_history = {
            'count_accuracy': [],
            'pos_iou': [],
            'pos_f1': [],
            'val_mae': [],
            'composite_score': []
        }
        
        # 新增：预测样本存储
        self.sample_predictions = []
        self.sample_targets = []
        
    def update_epoch(self, epoch: int, train_loss: float, val_loss: float, 
                    test_loss: Optional[float] = None, metrics: Optional[Dict] = None):
        """
        更新一个epoch的训练数据
        
        Args:
            epoch: 当前epoch
            train_loss: 训练损失
            val_loss: 验证损失
            test_loss: 测试损失（可选）
            metrics: 验证指标（可选）
        """
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        if test_loss is not None:
            self.test_losses.append(test_loss)
        
        # 更新指标历史
        if metrics:
            for key in self.metrics_history:
                if key in metrics:
                    self.metrics_history[key].append(metrics[key])
                else:
                    self.metrics_history[key].append(0.0)
    
    def update_prediction_stats(self, predicted_mu_counts: List[int], 
                              true_mu_counts: List[int],
                              predicted_threshold_counts: List[int], 
                              true_threshold_counts: List[int],
                              sample_predictions: Optional[List] = None,
                              sample_targets: Optional[List] = None):
        """
        更新预测统计数据
        
        Args:
            predicted_mu_counts: 预测的MU数量列表
            true_mu_counts: 真实的MU数量列表
            predicted_threshold_counts: 预测的阈值数量列表
            true_threshold_counts: 真实的阈值数量列表
            sample_predictions: 样本预测结果（可选）
            sample_targets: 样本真实标签（可选）
        """
        self.predicted_mu_counts.extend(predicted_mu_counts)
        self.true_mu_counts.extend(true_mu_counts)
        self.predicted_threshold_counts.extend(predicted_threshold_counts)
        self.true_threshold_counts.extend(true_threshold_counts)
        
        # 存储样本数据用于详细分析
        if sample_predictions is not None:
            self.sample_predictions.extend(sample_predictions)
        if sample_targets is not None:
            self.sample_targets.extend(sample_targets)
    
    def plot_mu_threshold_analysis(self):
        """绘制MU阈值预测分析图表 - 优化版本"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('MU阈值预测任务分析报告 - 优化版本', fontsize=18, fontweight='bold')
        
        # 1. 训练和验证损失曲线
        axes[0, 0].plot(self.epochs, self.train_losses, 'b-', label='训练损失', linewidth=2, marker='o', markersize=4)
        axes[0, 0].plot(self.epochs, self.val_losses, 'r-', label='验证损失', linewidth=2, marker='s', markersize=4)
        if self.test_losses:
            axes[0, 0].plot(self.epochs, self.test_losses, 'g-', label='测试损失', linewidth=2, marker='^', markersize=4)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('损失值')
        axes[0, 0].set_title('训练过程损失曲线')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 指标变化曲线
        if self.metrics_history['composite_score']:
            axes[0, 1].plot(self.epochs, self.metrics_history['composite_score'], 'purple', label='综合分数', linewidth=2, marker='o')
            axes[0, 1].plot(self.epochs, self.metrics_history['count_accuracy'], 'orange', label='数量准确率', linewidth=2, marker='s')
            axes[0, 1].plot(self.epochs, self.metrics_history['pos_iou'], 'green', label='位置IoU', linewidth=2, marker='^')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('指标值')
            axes[0, 1].set_title('验证指标变化')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. MU数量预测准确性
        if self.predicted_mu_counts and self.true_mu_counts:
            axes[0, 2].scatter(self.true_mu_counts, self.predicted_mu_counts, alpha=0.6, s=50, c='blue')
            min_val = min(min(self.true_mu_counts), min(self.predicted_mu_counts))
            max_val = max(max(self.true_mu_counts), max(self.predicted_mu_counts))
            axes[0, 2].plot([min_val, max_val], [min_val, max_val], 'r--', label='完美预测线', linewidth=2)
            
            # 添加相关系数
            corr = np.corrcoef(self.true_mu_counts, self.predicted_mu_counts)[0, 1]
            axes[0, 2].text(0.05, 0.95, f'相关系数: {corr:.3f}', transform=axes[0, 2].transAxes, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            axes[0, 2].set_xlabel('真实MU数量')
            axes[0, 2].set_ylabel('预测MU数量')
            axes[0, 2].set_title('MU数量预测准确性')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 阈值数量预测准确性
        if self.predicted_threshold_counts and self.true_threshold_counts:
            axes[1, 0].scatter(self.true_threshold_counts, self.predicted_threshold_counts, alpha=0.6, s=50, c='green')
            min_val = min(min(self.true_threshold_counts), min(self.predicted_threshold_counts))
            max_val = max(max(self.true_threshold_counts), max(self.predicted_threshold_counts))
            axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', label='完美预测线', linewidth=2)
            
            # 添加相关系数
            corr = np.corrcoef(self.true_threshold_counts, self.predicted_threshold_counts)[0, 1]
            axes[1, 0].text(0.05, 0.95, f'相关系数: {corr:.3f}', transform=axes[1, 0].transAxes, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            axes[1, 0].set_xlabel('真实阈值数量')
            axes[1, 0].set_ylabel('预测阈值数量')
            axes[1, 0].set_title('阈值数量预测准确性')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 预测误差分布
        if self.predicted_mu_counts and self.true_mu_counts:
            mu_errors = np.array(self.predicted_mu_counts) - np.array(self.true_mu_counts)
            axes[1, 1].hist(mu_errors, bins=20, alpha=0.7, color='blue', edgecolor='black', density=True)
            axes[1, 1].set_xlabel('MU数量预测误差')
            axes[1, 1].set_ylabel('密度')
            axes[1, 1].set_title('MU数量预测误差分布')
            axes[1, 1].axvline(x=0, color='red', linestyle='--', label='零误差线')
            axes[1, 1].axvline(x=np.mean(mu_errors), color='orange', linestyle='-', label=f'均值: {np.mean(mu_errors):.2f}')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 阈值预测误差分布
        if self.predicted_threshold_counts and self.true_threshold_counts:
            threshold_errors = np.array(self.predicted_threshold_counts) - np.array(self.true_threshold_counts)
            axes[1, 2].hist(threshold_errors, bins=20, alpha=0.7, color='green', edgecolor='black', density=True)
            axes[1, 2].set_xlabel('阈值数量预测误差')
            axes[1, 2].set_ylabel('密度')
            axes[1, 2].set_title('阈值数量预测误差分布')
            axes[1, 2].axvline(x=0, color='red', linestyle='--', label='零误差线')
            axes[1, 2].axvline(x=np.mean(threshold_errors), color='orange', linestyle='-', label=f'均值: {np.mean(threshold_errors):.2f}')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        # 7. 混淆矩阵（如果数据足够）
        if self.predicted_mu_counts and self.true_mu_counts and HAS_SKLEARN:
            # 创建简化的混淆矩阵
            true_bins = np.digitize(self.true_mu_counts, bins=[0, 5, 10, 20, 50, 100, 200])
            pred_bins = np.digitize(self.predicted_mu_counts, bins=[0, 5, 10, 20, 50, 100, 200])
            cm = confusion_matrix(true_bins, pred_bins)
            
            if HAS_SEABORN:
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2, 0])
            else:
                im = axes[2, 0].imshow(cm, cmap='Blues')
                axes[2, 0].set_xticks(range(len(cm)))
                axes[2, 0].set_yticks(range(len(cm)))
                for i in range(len(cm)):
                    for j in range(len(cm)):
                        axes[2, 0].text(j, i, str(cm[i, j]), ha='center', va='center')
                plt.colorbar(im, ax=axes[2, 0])
            
            axes[2, 0].set_title('MU数量预测混淆矩阵')
            axes[2, 0].set_xlabel('预测类别')
            axes[2, 0].set_ylabel('真实类别')
        else:
            axes[2, 0].axis('off')
            axes[2, 0].text(0.5, 0.5, '混淆矩阵\n(需要sklearn)', ha='center', va='center', transform=axes[2, 0].transAxes)
        
        # 8. 训练统计信息
        axes[2, 1].axis('off')
        final_train = self.train_losses[-1] if self.train_losses else 0
        final_val = self.val_losses[-1] if self.val_losses else 0
        best_train = min(self.train_losses) if self.train_losses else 0
        best_val = min(self.val_losses) if self.val_losses else 0
        
        # 计算预测准确性指标
        mu_mae = np.mean(np.abs(np.array(self.predicted_mu_counts) - np.array(self.true_mu_counts))) if self.predicted_mu_counts else 0
        threshold_mae = np.mean(np.abs(np.array(self.predicted_threshold_counts) - np.array(self.true_threshold_counts))) if self.predicted_threshold_counts else 0
        
        stats_text = f"""训练统计信息
        
损失统计:
• 最终训练损失: {final_train:.6f}
• 最终验证损失: {final_val:.6f}
• 最佳训练损失: {best_train:.6f}
• 最佳验证损失: {best_val:.6f}

预测准确性:
• MU数量MAE: {mu_mae:.2f}
• 阈值数量MAE: {threshold_mae:.2f}
• 总训练轮数: {len(self.epochs)}
• 测试样本数: {len(self.predicted_mu_counts)}
        """
        axes[2, 1].text(0.05, 0.5, stats_text, fontsize=11, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        # 9. 样本预测示例
        axes[2, 2].axis('off')
        if self.sample_predictions and self.sample_targets:
            # 显示前几个样本的预测结果
            sample_text = "样本预测示例:\n\n"
            for i in range(min(3, len(self.sample_predictions))):
                pred = self.sample_predictions[i]
                target = self.sample_targets[i]
                sample_text += f"样本 {i+1}:\n"
                sample_text += f"  真实: {target}\n"
                sample_text += f"  预测: {pred}\n\n"
            
            axes[2, 2].text(0.05, 0.5, sample_text, fontsize=10, verticalalignment='center',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'mu_threshold_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_training_data(self, test_loss: float = None):
        """
        保存详细训练数据到CSV
        
        Args:
            test_loss: 测试损失（可选）
        """
        # 训练数据
        training_data = {
            'epoch': self.epochs,
            'train_loss': self.train_losses,
            'val_loss': self.val_losses
        }
        
        if test_loss is not None:
            training_data['test_loss'] = [test_loss] * len(self.epochs)
        
        df_training = pd.DataFrame(training_data)
        df_training.to_csv(os.path.join(self.save_dir, 'training_data.csv'), index=False)
        
        # 预测数据
        if self.predicted_mu_counts:
            prediction_data = {
                'true_mu_count': self.true_mu_counts,
                'predicted_mu_count': self.predicted_mu_counts,
                'true_threshold_count': self.true_threshold_counts,
                'predicted_threshold_count': self.predicted_threshold_counts,
                'mu_count_error': np.array(self.predicted_mu_counts) - np.array(self.true_mu_counts),
                'threshold_count_error': np.array(self.predicted_threshold_counts) - np.array(self.true_threshold_counts)
            }
            df_prediction = pd.DataFrame(prediction_data)
            df_prediction.to_csv(os.path.join(self.save_dir, 'prediction_results.csv'), index=False)
    
    def generate_comprehensive_report(self, test_loss: float = None, 
                                    model_info: Dict[str, Any] = None,
                                    dataset_info: Dict[str, Any] = None):
        """
        生成MU阈值预测任务综合报告
        
        Args:
            test_loss: 测试损失
            model_info: 模型信息
            dataset_info: 数据集信息
        """
        # 生成MU阈值预测分析图表
        self.plot_mu_threshold_analysis()
        
        # 生成综合报告
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('MU阈值预测任务综合报告', fontsize=16, fontweight='bold')
        
        # 1. 训练和验证损失曲线
        axes[0, 0].plot(self.epochs, self.train_losses, 'b-', label='训练损失', linewidth=2)
        axes[0, 0].plot(self.epochs, self.val_losses, 'r-', label='验证损失', linewidth=2)
        if test_loss is not None:
            axes[0, 0].axhline(y=test_loss, color='g', linestyle='--', label=f'测试损失: {test_loss:.6f}')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('损失值')
        axes[0, 0].set_title('训练过程损失曲线')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. MU数量预测准确性
        if self.predicted_mu_counts and self.true_mu_counts:
            axes[0, 1].scatter(self.true_mu_counts, self.predicted_mu_counts, alpha=0.6, s=50)
            min_val = min(min(self.true_mu_counts), min(self.predicted_mu_counts))
            max_val = max(max(self.true_mu_counts), max(self.predicted_mu_counts))
            axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', label='完美预测线')
            axes[0, 1].set_xlabel('真实MU数量')
            axes[0, 1].set_ylabel('预测MU数量')
            axes[0, 1].set_title('MU数量预测准确性')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 阈值数量预测准确性
        if self.predicted_threshold_counts and self.true_threshold_counts:
            axes[0, 2].scatter(self.true_threshold_counts, self.predicted_threshold_counts, alpha=0.6, s=50)
            min_val = min(min(self.true_threshold_counts), min(self.predicted_threshold_counts))
            max_val = max(max(self.true_threshold_counts), max(self.predicted_threshold_counts))
            axes[0, 2].plot([min_val, max_val], [min_val, max_val], 'r--', label='完美预测线')
            axes[0, 2].set_xlabel('真实阈值数量')
            axes[0, 2].set_ylabel('预测阈值数量')
            axes[0, 2].set_title('阈值数量预测准确性')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 训练统计信息
        final_train = self.train_losses[-1] if self.train_losses else 0
        final_val = self.val_losses[-1] if self.val_losses else 0
        best_train = min(self.train_losses) if self.train_losses else 0
        best_val = min(self.val_losses) if self.val_losses else 0
        
        # 计算预测准确性指标
        mu_mae = np.mean(np.abs(np.array(self.predicted_mu_counts) - np.array(self.true_mu_counts))) if self.predicted_mu_counts else 0
        threshold_mae = np.mean(np.abs(np.array(self.predicted_threshold_counts) - np.array(self.true_threshold_counts))) if self.predicted_threshold_counts else 0
        
        stats_text = f"""训练统计信息
        
损失统计:
• 最终训练损失: {final_train:.6f}
• 最终验证损失: {final_val:.6f}
• 最佳训练损失: {best_train:.6f}
• 最佳验证损失: {best_val:.6f}"""
        
        if test_loss is not None:
            stats_text += f"\n• 测试损失: {test_loss:.6f}"
            
        stats_text += f"""

预测准确性:
• MU数量MAE: {mu_mae:.2f}
• 阈值数量MAE: {threshold_mae:.2f}
• 总训练轮数: {len(self.epochs)}
• 测试样本数: {len(self.predicted_mu_counts)}
        """
        
        axes[1, 0].text(0.05, 0.5, stats_text, fontsize=11, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        axes[1, 0].set_title('训练统计')
        axes[1, 0].axis('off')
        
        # 5. 模型信息
        if model_info:
            model_text = "模型信息\n\n"
            for key, value in model_info.items():
                model_text += f"{key}: {value}\n"
        else:
            model_text = "模型信息\n\n未提供模型信息"
            
        axes[1, 1].text(0.05, 0.5, model_text, fontsize=11, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        axes[1, 1].set_title('模型配置')
        axes[1, 1].axis('off')
        
        # 6. 数据集信息
        if dataset_info:
            dataset_text = "数据集信息\n\n"
            for key, value in dataset_info.items():
                dataset_text += f"{key}: {value}\n"
        else:
            dataset_text = "数据集信息\n\n未提供数据集信息"
            
        axes[1, 2].text(0.05, 0.5, dataset_text, fontsize=11, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
        axes[1, 2].set_title('数据概况')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'comprehensive_training_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存详细数据
        self.save_training_data(test_loss)
        
        print(f"📊 MU阈值预测分析报告已保存到: {self.save_dir}/mu_threshold_analysis.png")
        print(f"📈 综合训练分析报告已保存到: {self.save_dir}/comprehensive_training_analysis.png")
        print(f"📋 详细训练数据已保存到: {self.save_dir}/training_data.csv")
        if self.predicted_mu_counts:
            print(f"🎯 预测结果数据已保存到: {self.save_dir}/prediction_results.csv")
