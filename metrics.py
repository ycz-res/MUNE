import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


def get_nonzero_indices(x):
    """
    获取非零索引集合
    Args:
        x: (500,) 向量
    Returns:
        set of indices
    """
    return set((x != 0).nonzero(as_tuple=True)[0].tolist())


def mu_threshold_metrics(pred, gt, eps=1e-8, alpha=(1.0, 1.0, 1.0, 0.5)):
    """
    评价指标函数：同时评估 MU 数量、位置和数值精度，并给出综合分数
    Args:
        pred: (B, 500) 模型预测 (0 or 阈值)
        gt:   (B, 500) 真实标签 (0 or 阈值)
        alpha: 权重 (count, pos_iou, pos_f1, mae)
    Returns:
        metrics: dict, 包含四个指标 + 综合分数
    """
    B = pred.size(0)
    acc_count, iou_pos, f1_pos, mae_val = [], [], [], []

    for b in range(B):
        p, g = pred[b], gt[b]

        # --- 1. 非零索引集合 ---
        set_p = get_nonzero_indices(p)
        set_g = get_nonzero_indices(g)

        # --- 2. MU数量准确率 ---
        acc_count.append(1.0 if len(set_p) == len(set_g) else 0.0)

        # --- 3. 位置 IoU & F1 ---
        inter, union = len(set_p & set_g), len(set_p | set_g)
        iou = inter / (union + eps)
        precision = inter / (len(set_p) + eps)
        recall = inter / (len(set_g) + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        iou_pos.append(iou)
        f1_pos.append(f1)

        # --- 4. 数值 MAE (仅在位置相同处比较) ---
        common_idx = list(set_p & set_g)
        if len(common_idx) > 0:
            mae = torch.mean(torch.abs(p[common_idx] - g[common_idx])).item()
            mae_val.append(mae)

    # 取平均
    acc_count = sum(acc_count) / B
    iou_pos = sum(iou_pos) / B
    f1_pos = sum(f1_pos) / B
    mae_val = sum(mae_val) / max(len(mae_val), 1)

    # 综合分数
    score = (alpha[0] * acc_count +
             alpha[1] * iou_pos +
             alpha[2] * f1_pos -
             alpha[3] * mae_val)

    metrics = {
        "count_acc": acc_count,
        "pos_iou": iou_pos,
        "pos_f1": f1_pos,
        "val_mae": mae_val,
        "score": score  # 综合指标，用来挑选最优模型
    }
    return metrics


def comprehensive_metrics(pred, gt, eps=1e-8):
    """
    综合评价指标函数 - 优化版本
    包含更多评价指标，批量计算提高效率
    
    Args:
        pred: (B, 500) 模型预测
        gt:   (B, 500) 真实标签
        eps: 数值稳定性参数
    Returns:
        metrics: dict, 包含各种评价指标
    """
    B = pred.size(0)
    
    # 转换为二进制掩码
    pred_mask = (pred != 0).float()
    gt_mask = (gt != 0).float()
    
    # 1. 数量相关指标
    pred_counts = pred_mask.sum(dim=1)
    gt_counts = gt_mask.sum(dim=1)
    
    count_mae = torch.mean(torch.abs(pred_counts - gt_counts)).item()
    count_rmse = torch.sqrt(torch.mean((pred_counts - gt_counts) ** 2)).item()
    count_accuracy = torch.mean((pred_counts == gt_counts).float()).item()
    
    # 2. 位置相关指标
    intersection = torch.sum(pred_mask * gt_mask, dim=1)
    union = torch.sum(pred_mask + gt_mask - pred_mask * gt_mask, dim=1)
    iou = intersection / (union + eps)
    mean_iou = torch.mean(iou).item()
    
    # Precision, Recall, F1
    precision = intersection / (pred_mask.sum(dim=1) + eps)
    recall = intersection / (gt_mask.sum(dim=1) + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    
    mean_precision = torch.mean(precision).item()
    mean_recall = torch.mean(recall).item()
    mean_f1 = torch.mean(f1).item()
    
    # 3. 数值相关指标
    common_mask = pred_mask * gt_mask
    if torch.sum(common_mask) > 0:
        pred_values = pred[common_mask.bool()]
        gt_values = gt[common_mask.bool()]
        value_mae = torch.mean(torch.abs(pred_values - gt_values)).item()
        value_rmse = torch.sqrt(torch.mean((pred_values - gt_values) ** 2)).item()
    else:
        value_mae = 0.0
        value_rmse = 0.0
    
    # 4. 稀疏性指标
    sparsity_pred = torch.mean(pred_mask).item()
    sparsity_gt = torch.mean(gt_mask).item()
    sparsity_error = abs(sparsity_pred - sparsity_gt)
    
    # 5. 综合分数
    # 权重可以根据任务需求调整
    composite_score = (
        0.3 * count_accuracy +
        0.3 * mean_iou +
        0.2 * mean_f1 +
        0.1 * (1 - min(count_mae / 10, 1)) +  # 归一化数量误差
        0.1 * (1 - min(value_mae, 1))  # 归一化数值误差
    )
    
    metrics = {
        # 数量指标
        "count_mae": count_mae,
        "count_rmse": count_rmse,
        "count_accuracy": count_accuracy,
        
        # 位置指标
        "pos_iou": mean_iou,
        "pos_precision": mean_precision,
        "pos_recall": mean_recall,
        "pos_f1": mean_f1,
        
        # 数值指标
        "val_mae": value_mae,
        "val_rmse": value_rmse,
        
        # 稀疏性指标
        "sparsity_pred": sparsity_pred,
        "sparsity_gt": sparsity_gt,
        "sparsity_error": sparsity_error,
        
        # 综合指标
        "composite_score": composite_score,
        
        # 兼容性指标（保持与原有代码兼容）
        "count_acc": count_accuracy,
        "pos_iou": mean_iou,
        "pos_f1": mean_f1,
        "val_mae": value_mae,
        "score": composite_score
    }
    
    return metrics


def batch_metrics(pred, gt, eps=1e-8):
    """
    批量计算指标 - 高效版本
    适用于大批量数据
    
    Args:
        pred: (B, 500) 模型预测
        gt:   (B, 500) 真实标签
        eps: 数值稳定性参数
    Returns:
        metrics: dict, 包含各种评价指标
    """
    return comprehensive_metrics(pred, gt, eps)


def print_metrics_summary(metrics: Dict[str, float], prefix: str = ""):
    """
    打印指标摘要
    
    Args:
        metrics: 指标字典
        prefix: 前缀字符串
    """
    print(f"\n{prefix}评价指标摘要:")
    print("=" * 50)
    
    # 数量指标
    print("📊 数量预测指标:")
    print(f"  MAE: {metrics.get('count_mae', 0):.4f}")
    print(f"  RMSE: {metrics.get('count_rmse', 0):.4f}")
    print(f"  准确率: {metrics.get('count_accuracy', 0):.4f}")
    
    # 位置指标
    print("\n🎯 位置预测指标:")
    print(f"  IoU: {metrics.get('pos_iou', 0):.4f}")
    print(f"  Precision: {metrics.get('pos_precision', 0):.4f}")
    print(f"  Recall: {metrics.get('pos_recall', 0):.4f}")
    print(f"  F1: {metrics.get('pos_f1', 0):.4f}")
    
    # 数值指标
    print("\n📈 数值预测指标:")
    print(f"  MAE: {metrics.get('val_mae', 0):.4f}")
    print(f"  RMSE: {metrics.get('val_rmse', 0):.4f}")
    
    # 综合指标
    print("\n🏆 综合指标:")
    print(f"  综合分数: {metrics.get('composite_score', 0):.4f}")
    print(f"  稀疏性误差: {metrics.get('sparsity_error', 0):.4f}")
    
    print("=" * 50)
