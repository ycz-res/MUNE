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
import warnings

# 屏蔽中文字体警告
warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*missing from font.*')

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
        
        # 新增：验证指标历史（与metrics.py返回一致的键）
        self.val_metrics_history = {
            'Precision': [],
            'Recall': [],
            'F1': [],
            'IoU': [],
            'EMD': [],
            'Score': []
        }
        
        # 新增：测试汇总
        self.test_loss = None
        self.test_metrics = None
        
        # 新增：用于绘制随机测试样本
        self.sample_indices = None           # List[int]
        self.sample_cmap = None             # np.ndarray, shape (M, 500)
        self.sample_thr_true = None         # np.ndarray, shape (M, 500)
        self.sample_thr_pred = None         # np.ndarray, shape (M, 500)
        self.sample_mus_true = None         # np.ndarray, shape (M,)
        
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
            # 记录与metrics.py一致的验证指标
            for k in self.val_metrics_history.keys():
                self.val_metrics_history[k].append(float(metrics.get(k, 0.0)))

    # ==== 新增：设置测试结果 ====
    def set_test_results(self, test_loss: float, test_metrics: Dict[str, float]):
        self.test_loss = float(test_loss)
        # 只保留关心的键
        self.test_metrics = {
            'Precision': float(test_metrics.get('Precision', 0.0)),
            'Recall': float(test_metrics.get('Recall', 0.0)),
            'F1': float(test_metrics.get('F1', 0.0)),
            'IoU': float(test_metrics.get('IoU', 0.0)),
            'EMD': float(test_metrics.get('EMD', 0.0)),
            'Score': float(test_metrics.get('Score', 0.0)),
        }

    # ==== 新增：设置用于随机样本可视化的数据 ====
    def set_sample_data(
        self,
        sample_indices: List[int],
        cmap: np.ndarray,
        thresholds_true: np.ndarray,
        thresholds_pred: np.ndarray,
        mus_true: np.ndarray,
    ):
        self.sample_indices = list(sample_indices)
        self.sample_cmap = np.asarray(cmap)
        self.sample_thr_true = np.asarray(thresholds_true)
        self.sample_thr_pred = np.asarray(thresholds_pred)
        self.sample_mus_true = np.asarray(mus_true)

    # ==== 新增：图1 训练/验证损失曲线 ====
    def plot_loss_curves(self):
        plt.figure(figsize=(8, 5))
        plt.plot(self.epochs, self.train_losses, 'b-o', label='Train Loss', markersize=3)
        plt.plot(self.epochs, self.val_losses, 'r-s', label='Val Loss', markersize=3)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training vs Validation Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        out = os.path.join(self.save_dir, 'loss_curves.png')
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()

    # ==== 新增：图2 验证指标曲线 ====
    def plot_val_metrics_curves(self):
        plt.figure(figsize=(10, 6))
        # 主要指标（左y轴）
        for key, style in zip(['Precision', 'Recall', 'F1', 'IoU', 'Score'], ['b-', 'g-', 'r-', 'm-', 'k-']):
            vals = self.val_metrics_history.get(key, [])
            if vals:
                plt.plot(self.epochs, vals, style, label=key, linewidth=2)
        
        # EMD指标（右y轴，越小越好）
        emd_vals = self.val_metrics_history.get('EMD', [])
        if emd_vals:
            ax2 = plt.gca().twinx()
            ax2.plot(self.epochs, emd_vals, 'orange', linestyle='--', label='EMD', linewidth=2)
            ax2.set_ylabel('EMD (越大越好)', color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')
        
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Validation Metrics Over Epochs')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left')
        out = os.path.join(self.save_dir, 'val_metrics_curves.png')
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()

    # ==== 新增：图3 测试指标柱状图 ====
    def plot_test_metrics_hist(self):
        if self.test_loss is None or self.test_metrics is None:
            return
        labels = ['Loss', 'Precision', 'Recall', 'F1', 'IoU', 'EMD', 'Score']
        values = [self.test_loss] + [float(self.test_metrics.get(k, 0.0)) for k in labels[1:]]
        plt.figure(figsize=(10, 5))
        x = np.arange(len(labels))
        colors = ['#4e79a7', '#59a14f', '#f28e2b', '#e15759', '#76b7b2', '#ff7f0e', '#9c755f']
        plt.bar(x, values, color=colors)
        plt.xticks(x, labels)
        for xi, v in zip(x, values):
            plt.text(xi, v, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        plt.ylabel('Value')
        plt.title('Test Metrics Summary')
        plt.grid(True, axis='y', alpha=0.2)
        out = os.path.join(self.save_dir, 'test_metrics_summary.png')
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()

    # ==== 新增：图4 随机抽取测试样本进行阈值对比 ====
    def plot_random_test_samples(self):
        if self.sample_indices is None or self.sample_cmap is None or self.sample_thr_true is None or self.sample_thr_pred is None:
            return
        indices = self.sample_indices
        cmap = self.sample_cmap
        thr_true = self.sample_thr_true
        thr_pred = self.sample_thr_pred
        mus_true = self.sample_mus_true if self.sample_mus_true is not None else None
        num = len(indices)
        rows, cols = 4, 5
        rows = max(1, min(rows, int(np.ceil(num / cols))))
        cols = min(cols, num) if num < cols else cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), squeeze=False)
        fig.suptitle('Random Test Samples: True vs Predicted Thresholds', fontsize=14)
        x_positions = np.arange(500)
        for i, idx in enumerate(indices):
            r, c = divmod(i, cols)
            ax = axes[r][c]
            y_vals = cmap[i]
            ax.plot(x_positions, y_vals, 'bo', markersize=2, label='CMAP')
            # Compute positions
            true_pos = set(np.where(thr_true[i] > 0)[0].tolist())
            pred_pos = set(np.where(thr_pred[i] > 0)[0].tolist())

            match_pos = sorted(true_pos & pred_pos)
            true_only_pos = sorted(true_pos - pred_pos)
            pred_only_pos = sorted(pred_pos - true_pos)

            # Draw matches in green
            for p in match_pos:
                ax.axvline(x=p, color='green', linestyle='-', linewidth=1.4, alpha=0.9)

            # Draw true-only in blue
            for p in true_only_pos:
                ax.axvline(x=p, color='#1f77b4', linestyle='--', linewidth=1.2, alpha=0.9)

            # Draw pred-only in orange
            for p in pred_only_pos:
                ax.axvline(x=p, color='orange', linestyle='-', linewidth=1.2, alpha=0.9)
            true_mu = int(mus_true[i]) if mus_true is not None else len(true_pos)
            pred_mu = int(np.sum(thr_pred[i] == 1.0))
            ax.set_title(f'ID {idx} | True MU: {true_mu} | Pred MU: {pred_mu}', fontsize=9)
            ax.set_xlim(0, 499)
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.2)
        # 隐藏未使用子图
        total_ax = rows * cols
        for k in range(num, total_ax):
            r, c = divmod(k, cols)
            axes[r][c].axis('off')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out = os.path.join(self.save_dir, 'random_test_samples.png')
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()

    # ==== 新增：一键生成四张图 ====
    def generate_four_figs(self):
        self.plot_loss_curves()
        self.plot_val_metrics_curves()
        self.plot_test_metrics_hist()
        self.plot_random_test_samples()
        print(f"📊 Saved: {os.path.join(self.save_dir, 'loss_curves.png')}")
        print(f"📊 Saved: {os.path.join(self.save_dir, 'val_metrics_curves.png')}")
        if self.test_loss is not None and self.test_metrics is not None:
            print(f"📊 Saved: {os.path.join(self.save_dir, 'test_metrics_summary.png')}")
        print(f"📊 Saved: {os.path.join(self.save_dir, 'random_test_samples.png')}")
    
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
