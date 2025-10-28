import torch
import torch.nn.functional as F
from typing import Literal

def thr(thresholds_pred, thresholds_target, 
             lambda_count=20.0, lambda_pos=10.0, lambda_val=1.0, lambda_sparse=0.5):
    """
    MU 阈值损失函数 - 优化版本
    同时考虑 MU 数量、阈值位置、数值精度和稀疏性

    Args:
        thresholds_pred:   (B, 500)  模型预测 (0 or 阈值)
        thresholds_target: (B, 500)  真实标签 (0 or 阈值)
        lambda_count: 数量loss权重 (增加权重)
        lambda_pos:   位置loss权重
        lambda_val:   数值loss权重
        lambda_sparse: 稀疏性loss权重
    Returns:
        total_loss: 标量损失
    """
    eps = 1e-8
    B = thresholds_pred.size(0)
    
    # 批量计算，避免循环
    # 1. 数量损失 (增加权重，因为这是主要问题)
    pred_counts = (thresholds_pred != 0).sum(dim=1).float()
    true_counts = (thresholds_target != 0).sum(dim=1).float()
    loss_count = torch.mean(torch.abs(pred_counts - true_counts))
    
    # 2. 位置损失 (IoU)
    pred_mask = (thresholds_pred != 0).float()
    true_mask = (thresholds_target != 0).float()
    
    intersection = torch.sum(pred_mask * true_mask, dim=1)
    union = torch.sum(pred_mask + true_mask - pred_mask * true_mask, dim=1)
    iou = intersection / (union + eps)
    loss_pos = torch.mean(1 - iou)
    
    # 3. 数值损失 (仅在位置相同处比较)
    common_mask = pred_mask * true_mask
    if torch.sum(common_mask) > 0:
        pred_values = thresholds_pred[common_mask.bool()]
        true_values = thresholds_target[common_mask.bool()]
        loss_val = torch.mean(torch.abs(pred_values - true_values))
    else:
        loss_val = torch.tensor(0.0, device=thresholds_pred.device, requires_grad=True)
    
    # 4. 稀疏性损失 (鼓励预测稀疏)
    # 使用L1正则化鼓励稀疏性
    loss_sparse = torch.mean(torch.abs(thresholds_pred))
    
    # 总损失
    total_loss = (lambda_count * loss_count +
                  lambda_pos * loss_pos +
                  lambda_val * loss_val +
                  lambda_sparse * loss_sparse)
    
    return total_loss

def ce(thresholds_pred, thresholds_target, pos_weight=None):
    """
    交叉熵式阈值损失函数
    对正样本位置(1)最大化预测分数，对负样本位置(0)最小化预测分数
    
    Args:
        thresholds_pred:   (B, 500)  模型预测分数 [0.1, 0.7, 0.8, 0.2, 0.5, 0.3]
        thresholds_target: (B, 500)  真实标签 [0, 1, 1, 0, 1, 0]
        pos_weight:        float     正样本权重，用于处理不平衡数据
    
    Returns:
        total_loss: 标量损失
    """
    # 使用PyTorch内置的BCEWithLogitsLoss，自动处理sigmoid和log
    bce_loss = F.binary_cross_entropy_with_logits(
        thresholds_pred, 
        thresholds_target.float(),
        pos_weight=pos_weight,
        reduction='mean'
    )
    
    return bce_loss

def focal_ce(thresholds_pred, thresholds_target, gamma=2.0, alpha=0.25, reduction='mean'):
    """
    焦点交叉熵损失函数（Focal BCE with logits）
    ---------------------------------------------------
    用于解决类别极度不平衡问题。
    在 BCE 的基础上增加了“聚焦因子 (1 - p_t)^γ” 和“平衡系数 α”。

    Args:
        thresholds_pred:   (B, 500)  模型预测的 logits（未经过 sigmoid）
        thresholds_target: (B, 500)  真实标签（0 或 1）
        gamma:             焦点参数，控制对“难样本”的关注度（默认 2.0）
        alpha:             平衡参数，控制正负样本的相对权重（默认 0.25）
        reduction:         'mean' | 'sum' | 'none'，损失聚合方式

    Returns:
        total_loss: 标量损失（或逐样本损失张量）
    """
    # --- Step 1: 计算标准 BCE loss（每个元素单独计算） ---
    bce_loss = F.binary_cross_entropy_with_logits(
        thresholds_pred,
        thresholds_target.float(),
        reduction='none'  # 暂时不聚合
    )

    # --- Step 2: 计算 p_t（模型对真实类别的预测概率） ---
    # p_t = sigmoid(x)  if y=1
    # p_t = 1 - sigmoid(x)  if y=0
    p_t = torch.exp(-bce_loss)

    # --- Step 3: 计算 Focal 调制因子 ---
    focal_weight = (1 - p_t) ** gamma

    # --- Step 4: 应用 alpha 平衡正负样本 ---
    alpha_t = alpha * thresholds_target + (1 - alpha) * (1 - thresholds_target)

    # --- Step 5: 组合最终损失 ---
    loss = alpha_t * focal_weight * bce_loss

    # --- Step 6: 根据 reduction 参数返回 ---
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss  # 保留逐元素损失（例如用于调试或可视化）

# 语义化枚举
NormalizeT = Literal["none", "per_one", "per_length"]
PairingStrategyT = Literal["truncate", "sink_right", "sink_left"]
AggregationT = Literal["none", "mean", "sum"]
ModeT = Literal["prefix", "pair"]

def emd(
    predictions: torch.Tensor,      # (batch_size, L) 预测 0/1 序列
    targets: torch.Tensor,          # (batch_size, L) 真实 0/1 序列
    *,
    mode: ModeT = "prefix",                 # "prefix" 可导（训练） | "pair" 不可导（评估）
    normalization: NormalizeT = "per_one",  # "none" | "per_one" | "per_length"
    pairing_strategy: PairingStrategyT = "truncate",  # 仅 pair 用
    aggregation: AggregationT = "mean",     # "none" | "mean" | "sum"
) -> torch.Tensor:
    """
    统一版 EMD（A/B 同长，元素为 0/1；L 自适应）：
      - mode="prefix": 前缀流量法（可导，训练主损失推荐）
      - mode="pair"  : 配对法（不可导，评估/日志；支持 truncate/sink_right/sink_left）

    normalization:
      - "none"       ：原始搬运量
      - "per_one"    ：/ max(#1_pred, #1_true) —— "每目标平均位移"
      - "per_length" ：/ L —— "单位长度搬运密度"（L=序列长度）
    """
    # 基本校验
    assert predictions.dim() == 2 and targets.dim() == 2, "predictions/targets 需为 (B, L)"
    assert predictions.shape == targets.shape, "predictions 与 targets 形状需一致"
    B, L = predictions.shape
    device = predictions.device

    # 转 float 运算（输入应为 0/1）
    p = predictions.to(torch.float32)  # (B, L)
    t = targets.to(torch.float32)      # (B, L)

    if mode == "prefix":
        # —— 可导：EMD = Σ |cumsum(p) - cumsum(t)| —— #
        flow = (torch.cumsum(p, dim=1) - torch.cumsum(t, dim=1)).abs()  # (B, L)
        loss_vec = flow.sum(dim=1)                                      # (B,)

        if normalization == "per_one":
            denom = torch.maximum(p.sum(dim=1), t.sum(dim=1)).clamp_min(1e-8)
            loss_vec = loss_vec / denom
        elif normalization == "per_length":
            loss_vec = loss_vec / float(max(L, 1))

    elif mode == "pair":
        # —— 不可导：配对法（批量向量化，无 Python for） —— #
        with torch.no_grad():
            p_bin = predictions.to(torch.long)  # 已是 0/1
            t_bin = targets.to(torch.long)

            n_pred_ones = p_bin.sum(dim=1)                  # (B,)
            n_true_ones = t_bin.sum(dim=1)                  # (B,)
            n_pairs     = torch.minimum(n_pred_ones, n_true_ones)  # (B,)
            K = int(n_pairs.max().item()) if B > 0 else 0

            if K == 0:
                if pairing_strategy == "truncate":
                    loss_vec = torch.zeros(B, device=device, dtype=torch.float32)
                else:
                    idx = torch.arange(L, device=device).view(1, L)     # (1, L)
                    sink = L if pairing_strategy == "sink_right" else -1
                    extra_cost = ((p_bin * (idx - sink).abs()).sum(dim=1) +
                                  (t_bin * (idx - sink).abs()).sum(dim=1)).float()
                    loss_vec = extra_cost
            else:
                # 给每个 1 编秩次（第 k 个 1 的位置）
                order_pred = (torch.cumsum(p_bin, dim=1) * p_bin).to(torch.int64)  # (B, L)
                order_true = (torch.cumsum(t_bin, dim=1) * t_bin).to(torch.int64)  # (B, L)

                idx = torch.arange(L, device=device).view(1, L)                    # (1, L)
                ks  = torch.arange(1, K+1, device=device).view(1, 1, K)            # (1, 1, K)

                mask_pred = (order_pred.unsqueeze(2) == ks)                        # (B, L, K)
                mask_true = (order_true.unsqueeze(2) == ks)                        # (B, L, K)
                pos_predK = (mask_pred * idx.unsqueeze(2)).sum(dim=1)              # (B, K)
                pos_trueK = (mask_true * idx.unsqueeze(2)).sum(dim=1)              # (B, K)

                k_mask = (torch.arange(1, K+1, device=device).view(1, K)
                          <= n_pairs.view(B, 1))                                   # (B, K)

                pair_cost = ((pos_predK - pos_trueK).abs() * k_mask).sum(dim=1).float()  # (B,)

                # 多余 1 的惩罚
                if pairing_strategy == "truncate":
                    extra_cost = torch.zeros(B, device=device, dtype=torch.float32)
                else:
                    sink = L if pairing_strategy == "sink_right" else -1
                    extras_pred = (order_pred > n_pairs.view(B, 1))                # (B, L)
                    extras_true = (order_true > n_pairs.view(B, 1))                # (B, L)
                    extra_cost = ((extras_pred * (idx - sink).abs()).sum(dim=1) +
                                  (extras_true * (idx - sink).abs()).sum(dim=1)).float()

                loss_vec = pair_cost + extra_cost

            # 归一化（pair 用二值计数近似"质量"）
            if normalization == "per_one":
                denom = torch.maximum(n_pred_ones, n_true_ones).to(torch.float32).clamp_min(1.0)
                loss_vec = loss_vec / denom
            elif normalization == "per_length":
                loss_vec = loss_vec / float(max(L, 1))
    else:
        raise ValueError("mode 必须是 'prefix' 或 'pair'")

    # 聚合
    if aggregation == "mean":
        return loss_vec.mean()
    elif aggregation == "sum":
        return loss_vec.sum()
    else:  # "none"
        return loss_vec
