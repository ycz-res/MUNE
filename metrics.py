"""
metrics.py
模型评估指标模块
支持 binary（二值阈值检测）与 value（真实阈值回归）两种模式
"""

import torch
import torch.nn.functional as F

def emd_binary_norm(pred_bin: torch.Tensor, target_bin: torch.Tensor) -> torch.Tensor:
    """
    一维二值序列的归一化 EMD（推土机距离），越小越好，范围 [0,1]。
    计算方式：前缀差累计 δ（截图中的做法）
        δ_0 = 0, δ_{i+1} = δ_i + P_i - Q_i
        EMD_raw = sum_i |δ_i|
        EMD_norm = EMD_raw / ((L-1) * max(#1s(P), #1s(Q), 1))

    Args:
        pred_bin   (B, L): 已二值化 0/1 预测序列（如 [0,1,0,1,1,1,...]）
        target_bin (B, L): 0/1 真值序列

    Returns:
        (B,) 逐样本归一化 EMD（torch.Tensor）
    """
    if pred_bin.shape != target_bin.shape:
        raise ValueError(f"Shape mismatch: {pred_bin.shape} vs {target_bin.shape}")

    B, L = target_bin.shape
    # 前缀差：cumsum(P - Q) = [δ_1, δ_2, ..., δ_L]
    delta = torch.cumsum(pred_bin - target_bin, dim=1)    # (B, L)
    emd_raw = torch.sum(torch.abs(delta), dim=1)          # (B,)

    ones_pred = pred_bin.sum(dim=1)                       # (B,)
    ones_tgt  = target_bin.sum(dim=1)                     # (B,)
    mass_ref  = torch.maximum(ones_pred, ones_tgt).clamp(min=1)
    denom     = mass_ref * max(L - 1, 1)                  # (B,)
    emd_norm  = (emd_raw / denom).clamp(0.0, 1.0)         # (B,)

    return emd_norm


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
        pred = (prob >= threshold).float()
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
        # 计算 TP、FP、FN
        tp = (pred * target).sum(dim=1)
        fp = (pred * (1 - target)).sum(dim=1)
        fn = ((1 - pred) * target).sum(dim=1)

        # 计算 Precision、Recall、F1
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        # 计算 IoU
        intersection = (pred * target).sum(dim=1)
        union = ((pred + target) > 0).float().sum(dim=1)
        iou = intersection / (union + 1e-8)

        # 计算归一化 EMD
        emd_vec = emd_binary_norm(pred, target)  # (B,)

        metrics["Precision"] = precision.mean().item()
        metrics["Recall"] = recall.mean().item()
        metrics["F1"] = f1.mean().item()
        metrics["IoU"] = iou.mean().item()
        metrics["EMD"] = 1.0 - emd_vec.mean().item()  # 转换为越大越好

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
        # 综合评分策略：F1、IoU和EMD（EMD已转换为越大越好）
        metrics["Score"] = 0.4 * metrics["F1"] + 0.4 * metrics["IoU"] + 0.2 * metrics["EMD"]


    return metrics
