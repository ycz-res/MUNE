"""
metrics.py
模型评估指标模块
支持 binary（二值阈值检测）与 value（真实阈值回归）两种模式
"""

import torch
import torch.nn.functional as F


def b_v_metrics(pred_logits: torch.Tensor,
                    target: torch.Tensor,
                    mode: str = "binary",
                    threshold: float = 0.5) -> dict:
    """
    计算模型评估指标（F1、IoU、MAE 等）

    Args:
        pred_logits (torch.Tensor): 模型输出的 logits，形状 (B, 500)
        target (torch.Tensor): 真实标签，形状 (B, 500)
        mode (str): 模式类型
            - "binary"：0/1 标签（仅关心位置）
            - "value"：真实值标签（同时关心数值）
        threshold (float): Sigmoid 激活后的阈值，用于二值化预测（默认 0.5）

    Returns:
        dict: 包含各项指标的字典
    """

    # ============================================================
    # 1️⃣ 根据模式激活输出
    # ============================================================
    if mode == "binary":
        # 概率模式：Sigmoid + 阈值化
        prob = torch.sigmoid(pred_logits)
        pred = (prob > threshold).float()
    elif mode == "value":
        # 回归模式：ReLU 保证非负输出
        pred = F.relu(pred_logits)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # ============================================================
    # 2️⃣ 计算基础指标（按样本平均）
    # ============================================================

    metrics = {}

    # -------- Binary 模式：计算 F1、IoU --------
    if mode == "binary":
        tp = (pred * target).sum(dim=1)
        fp = (pred * (1 - target)).sum(dim=1)
        fn = ((1 - pred) * target).sum(dim=1)

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        intersection = (pred * target).sum(dim=1)
        union = ((pred + target) > 0).float().sum(dim=1)
        iou = intersection / (union + 1e-8)

        metrics["Precision"] = precision.mean().item()
        metrics["Recall"] = recall.mean().item()
        metrics["F1"] = f1.mean().item()
        metrics["IoU"] = iou.mean().item()

    # -------- Value 模式：F1 + IoU + MAE --------
    elif mode == "value":
        # 定义“激活点”位置（非零即视为阈值点）
        pred_mask = (pred > 0).float()
        target_mask = (target > 0).float()

        # 位置指标
        tp = (pred_mask * target_mask).sum(dim=1)
        fp = (pred_mask * (1 - target_mask)).sum(dim=1)
        fn = ((1 - pred_mask) * target_mask).sum(dim=1)

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        intersection = (pred_mask * target_mask).sum(dim=1)
        union = ((pred_mask + target_mask) > 0).float().sum(dim=1)
        iou = intersection / (union + 1e-8)

        # 数值误差（仅在真实点上计算 MAE）
        mae = torch.abs(pred - target)
        mae_masked = mae[target > 0]
        mae_value = mae_masked.mean().item() if mae_masked.numel() > 0 else 0.0

        metrics["Precision"] = precision.mean().item()
        metrics["Recall"] = recall.mean().item()
        metrics["F1"] = f1.mean().item()
        metrics["IoU"] = iou.mean().item()
        metrics["MAE"] = mae_value

    # ============================================================
    # 3️⃣ 综合指标（可用于 early stopping 选择最佳 epoch）
    # ============================================================
    if mode == "binary":
        # 对于稀疏数据，使用更平衡的评分策略
        # 考虑Precision和Recall的平衡，以及IoU的重要性
        if metrics["F1"] > 0:
            metrics["Score"] = 0.4 * metrics["F1"] + 0.3 * metrics["IoU"] + 0.2 * metrics["Precision"] + 0.1 * metrics["Recall"]
        else:
            # 如果F1为0，使用Precision和Recall的调和平均
            if metrics["Precision"] > 0 or metrics["Recall"] > 0:
                metrics["Score"] = 0.3 * metrics["Precision"] + 0.3 * metrics["Recall"] + 0.4 * metrics["IoU"]
            else:
                metrics["Score"] = 0.0
    else:
        # 位置与数值共同评分（MAE 越小越好，取反）
        metrics["Score"] = 0.4 * metrics["F1"] + 0.4 * metrics["IoU"] + 0.2 * (1 - metrics["MAE"])

    return metrics
