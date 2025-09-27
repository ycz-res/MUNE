"""
评估指标模块
支持多种评估指标，易于扩展
"""

import torch
import numpy as np
from typing import Dict, Any, List, Tuple
from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


class BaseMetric(ABC):
    """评估指标基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.reset()
    
    @abstractmethod
    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """更新指标状态"""
        pass
    
    @abstractmethod
    def compute(self) -> Dict[str, float]:
        """计算最终指标值"""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """重置指标状态"""
        pass


class Accuracy(BaseMetric):
    """准确率指标"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.top_k = config['top_k']
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        if self.top_k == 1:
            pred_labels = torch.argmax(predictions, dim=1)
            correct = (pred_labels == targets).float()
        else:
            _, pred_labels = torch.topk(predictions, self.top_k, dim=1)
            correct = torch.any(pred_labels == targets.unsqueeze(1), dim=1).float()
        
        self.correct_count += correct.sum().item()
        self.total_count += targets.size(0)
    
    def compute(self) -> Dict[str, float]:
        if self.total_count == 0:
            return {f'top_{self.top_k}_accuracy': 0.0}
        
        accuracy = self.correct_count / self.total_count
        return {f'top_{self.top_k}_accuracy': accuracy}
    
    def reset(self) -> None:
        self.correct_count = 0
        self.total_count = 0


class PrecisionRecallF1(BaseMetric):
    """精确率、召回率、F1分数"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.average = config['average']
        self.num_classes = config['num_classes']
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        pred_labels = torch.argmax(predictions, dim=1)
        
        self.predictions.extend(pred_labels.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
    
    def compute(self) -> Dict[str, float]:
        if len(self.predictions) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.targets, self.predictions, average=self.average, zero_division=0
        )
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
    
    def reset(self) -> None:
        self.predictions = []
        self.targets = []


class ConfusionMatrix(BaseMetric):
    """混淆矩阵"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_classes = config['num_classes']
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        pred_labels = torch.argmax(predictions, dim=1)
        
        for pred, target in zip(pred_labels.cpu().numpy(), targets.cpu().numpy()):
            self.matrix[target, pred] += 1
    
    def compute(self) -> Dict[str, Any]:
        return {'confusion_matrix': self.matrix.tolist()}
    
    def reset(self) -> None:
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)


class AUROC(BaseMetric):
    """AUC-ROC指标"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.average = config['average']
        self.multi_class = config['multi_class']
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        # 将预测转换为概率
        probs = torch.softmax(predictions, dim=1)
        
        self.predictions.extend(probs.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
    
    def compute(self) -> Dict[str, float]:
        if len(self.predictions) == 0:
            return {'auroc': 0.0}
        
        try:
            auroc = roc_auc_score(
                self.targets, self.predictions, 
                multi_class=self.multi_class, 
                average=self.average
            )
            return {'auroc': float(auroc)}
        except ValueError:
            return {'auroc': 0.0}
    
    def reset(self) -> None:
        self.predictions = []
        self.targets = []


class LossMetric(BaseMetric):
    """损失指标"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        # 这里需要传入loss值，而不是predictions和targets
        # 为了简化，我们假设predictions就是loss值
        if predictions.dim() == 0:  # 标量loss
            self.loss_sum += predictions.item()
            self.count += 1
        else:
            self.loss_sum += predictions.sum().item()
            self.count += predictions.numel()
    
    def compute(self) -> Dict[str, float]:
        if self.count == 0:
            return {'loss': 0.0}
        
        avg_loss = self.loss_sum / self.count
        return {'loss': avg_loss}
    
    def reset(self) -> None:
        self.loss_sum = 0.0
        self.count = 0


class MetricManager:
    """指标管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics = {}
        self._setup_metrics()
    
    def _setup_metrics(self):
        """设置指标"""
        metric_configs = self.config['metrics']
        
        for metric_config in metric_configs:
            metric_type = metric_config['type']
            metric_name = metric_config['name']
            config = metric_config['config']
            
            metric = create_metric(metric_type, config)
            self.metrics[metric_name] = metric
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, loss: torch.Tensor = None):
        """更新所有指标"""
        for metric in self.metrics.values():
            if isinstance(metric, LossMetric) and loss is not None:
                metric.update(loss, targets)
            else:
                metric.update(predictions, targets)
    
    def compute(self) -> Dict[str, float]:
        """计算所有指标"""
        results = {}
        for name, metric in self.metrics.items():
            results.update(metric.compute())
        return results
    
    def reset(self):
        """重置所有指标"""
        for metric in self.metrics.values():
            metric.reset()
    
    def get_metric(self, name: str) -> BaseMetric:
        """获取指定指标"""
        return self.metrics[name]


def create_metric(metric_name: str, config: Dict[str, Any]) -> BaseMetric:
    """指标工厂函数"""
    metrics = {
        'accuracy': Accuracy,
        'precision_recall_f1': PrecisionRecallF1,
        'confusion_matrix': ConfusionMatrix,
        'auroc': AUROC,
        'loss': LossMetric,
    }
    
    if metric_name not in metrics:
        raise ValueError(f"Unknown metric: {metric_name}. Available: {list(metrics.keys())}")
    
    return metrics[metric_name](config)
