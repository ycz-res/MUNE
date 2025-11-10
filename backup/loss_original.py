import torch
import torch.nn.functional as F
from typing import Literal

def thr(thresholds_pred, thresholds_target, 
             lambda_count=25.0, lambda_pos=40.0, lambda_sparse=0.8,
             threshold=0.5):
    """
    MU 阈值损失函数 - 优化版本
    考虑 MU 数量、阈值位置和稀疏性
    
    支持 logits 输入：自动检测输入类型并转换

    Args:
        thresholds_pred:   (B, 500)  模型预测 (logits 或 0/阈值)
        thresholds_target: (B, 500)  真实标签 (0 or 阈值)
        lambda_count: 数量loss权重 (增加权重)
        lambda_pos:   位置loss权重 (进一步提高，解决位置偏差问题)
        lambda_sparse: 稀疏性loss权重 (鼓励预测稀疏，避免过度预测)
        threshold: 用于二值化的阈值（默认0.5），仅在输入为logits时使用
    Returns:
        total_loss: 标量损失
    """
    eps = 1e-8
    B = thresholds_pred.size(0)
    
    # 判断输入是logits还是阈值值
    # 如果输入范围不在[0,1]或包含负值，可能是logits
    if thresholds_pred.min() < 0 or thresholds_pred.max() > 1:
        # 是logits，转换为概率值
        pred_prob = torch.sigmoid(thresholds_pred)
        # 使用阈值进行软二值化（保留梯度）
        pred_mask = pred_prob  # 使用概率值作为软mask
    else:
        # 已经是阈值值，直接使用
        pred_mask = (thresholds_pred != 0).float()
    
    true_mask = (thresholds_target != 0).float()
    true_counts = true_mask.sum(dim=1).float()
    
    # 批量计算，避免循环
    # 1. 数量损失 (增加权重，因为这是主要问题)
    # 使用软mask的sum作为预测数量（可导）
    pred_counts = pred_mask.sum(dim=1)
    loss_count = torch.mean(torch.abs(pred_counts - true_counts))
    
    # 2. 位置损失 (IoU) - 使用软mask
    intersection = torch.sum(pred_mask * true_mask, dim=1)
    union = torch.sum(pred_mask + true_mask - pred_mask * true_mask, dim=1)
    iou = intersection / (union + eps)
    loss_pos = torch.mean(1 - iou)
    
    # 3. 位置分布损失 - 惩罚预测集中在后半部分
    # 计算预测和真实位置的重心（加权平均位置）
    seq_len = thresholds_pred.size(1)
    positions = torch.arange(seq_len, dtype=torch.float32, device=thresholds_pred.device).unsqueeze(0)  # (1, 500)
    
    # 预测位置重心
    pred_weighted_sum = torch.sum(pred_mask * positions, dim=1)  # (B,)
    pred_total = pred_mask.sum(dim=1) + eps
    pred_center = pred_weighted_sum / pred_total
    
    # 真实位置重心
    true_weighted_sum = torch.sum(true_mask * positions, dim=1)  # (B,)
    true_total = true_mask.sum(dim=1) + eps
    true_center = true_weighted_sum / true_total
    
    # 位置重心差异损失（归一化到[0,1]）
    loss_center = torch.mean(torch.abs(pred_center - true_center) / seq_len)
    
    # 4. 稀疏性损失 (鼓励预测稀疏)
    # 使用L1正则化鼓励稀疏性（对logits或阈值都适用）
    loss_sparse = torch.mean(torch.abs(thresholds_pred))
    
    # 总损失（位置损失和位置分布损失都用于约束位置）
    total_loss = (lambda_count * loss_count +
                  lambda_pos * (loss_pos + loss_center) +
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

def focal(thresholds_pred, thresholds_target, gamma=2.0, alpha=0.25, reduction='mean'):
    """
    焦点交叉熵损失函数（Focal BCE with logits）- 优化版本
    ---------------------------------------------------
    用于解决类别极度不平衡问题。
    在 BCE 的基础上增加了"聚焦因子 (1 - p_t)^γ" 和"平衡系数 α"。

    优化点：
    1. 直接计算 sigmoid，避免通过 exp(-bce_loss) 反推（数值更稳定）
    2. 添加数值稳定性保护（clamp p_t）
    3. 使用 log-sum-exp 技巧提高数值稳定性

    Args:
        thresholds_pred:   (B, 500)  模型预测的 logits（未经过 sigmoid）
        thresholds_target: (B, 500)  真实标签（0 或 1）
        gamma:             焦点参数，控制对"难样本"的关注度（默认 2.0）
        alpha:             平衡参数，控制正负样本的相对权重（默认 0.25）
        reduction:         'mean' | 'sum' | 'none'，损失聚合方式

    Returns:
        total_loss: 标量损失（或逐样本损失张量）
    """
    # 转换为 float 类型
    targets = thresholds_target.float()
    
    # --- Step 1: 直接计算 sigmoid（数值稳定） ---
    # 使用 logits 版本的 sigmoid，避免数值溢出
    p = torch.sigmoid(thresholds_pred)
    
    # --- Step 2: 计算 p_t（模型对真实类别的预测概率） ---
    # p_t = p  if y=1
    # p_t = 1-p  if y=0
    p_t = p * targets + (1 - p) * (1 - targets)
    
    # --- Step 3: 数值稳定性保护 ---
    # 防止 p_t 接近 0 或 1 时出现数值问题
    p_t = torch.clamp(p_t, min=1e-8, max=1.0 - 1e-8)
    
    # --- Step 4: 计算 Focal 调制因子 ---
    focal_weight = (1 - p_t) ** gamma
    
    # --- Step 5: 应用 alpha 平衡正负样本 ---
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    
    # --- Step 6: 计算 BCE loss（使用数值稳定的 log） ---
    # BCE = -[y*log(p) + (1-y)*log(1-p)]
    # 使用 logits 版本更稳定：log(sigmoid(x)) = -log(1 + exp(-x))
    bce_loss = F.binary_cross_entropy_with_logits(
        thresholds_pred,
        targets,
        reduction='none'
    )
    
    # --- Step 7: 组合最终损失 ---
    loss = alpha_t * focal_weight * bce_loss
    
    # --- Step 8: 根据 reduction 参数返回 ---
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss  # 保留逐元素损失（例如用于调试或可视化）

# 语义化枚举
AggregationT = Literal["none", "mean", "sum"]

def emd(
    predictions: torch.Tensor,      # (batch_size, L) 预测 0/1 序列
    targets: torch.Tensor,          # (batch_size, L) 真实 0/1 序列
    *,
    aggregation: AggregationT = "mean",     # "none" | "mean" | "sum"
) -> torch.Tensor:
    """
    前缀流量法 EMD（可导，用于训练）
    归一化：除以目标序列中1的数量（考虑稀疏性，避免鼓励预测过多）
    """
    # 基本校验
    assert predictions.dim() == 2 and targets.dim() == 2, "predictions/targets 需为 (B, L)"
    assert predictions.shape == targets.shape, "predictions 与 targets 形状需一致"
    B, L = predictions.shape

    # 转 float 运算（输入应为 0/1）
    p = predictions.to(torch.float32)  # (B, L)
    t = targets.to(torch.float32)      # (B, L)

    # —— 可导：EMD = Σ |cumsum(p) - cumsum(t)| —— #
    flow = (torch.cumsum(p, dim=1) - torch.cumsum(t, dim=1)).abs()  # (B, L)
    loss_vec = flow.sum(dim=1)                                      # (B,)

    # 归一化：除以目标序列中1的数量（考虑稀疏性，避免鼓励模型预测过多）
    denom = torch.clamp(t.sum(dim=1), min=1e-8)
    loss_vec = loss_vec / denom

    # 聚合
    if aggregation == "mean":
        return loss_vec.mean()
    elif aggregation == "sum":
        return loss_vec.sum()
    else:  # "none"
        return loss_vec


def emd_mae(thresholds_pred, thresholds_target, 
                 lambda_emd=1.0, lambda_mae=0.5, lambda_bce=5.0):
    """
    EMD + MAE + BCE 组合损失函数（可导版本）
    - EMD: 处理位置匹配（哪些位置有阈值）
    - MAE: 处理数值精度（阈值的具体数值）
    - BCE: 鼓励预测非零位置（避免预测全0），使用加权BCE（正样本权重更高）
    
    使用软二值化（直接使用概率值），保证整个损失函数可导。
    
    Args:
        thresholds_pred:   (B, L)  模型预测 (logits或值)
        thresholds_target: (B, L)  真实标签 (0 or 阈值)
        lambda_emd:        float   EMD损失权重（默认1.0）
        lambda_mae:        float   MAE损失权重（默认0.5）
        lambda_bce:        float   BCE损失权重（默认5.0，提高权重以鼓励预测非零）
    
    Returns:
        total_loss: 标量损失（可导）
    """
    # 1. EMD损失：位置匹配（使用软二值化，可导）
    # 判断输入是logits还是概率值
    if thresholds_pred.min() < 0 or thresholds_pred.max() > 1:
        # 可能是logits，需要sigmoid
        pred_prob = torch.sigmoid(thresholds_pred)
        pred_logits = thresholds_pred
    else:
        # 已经是概率值，转换为logits用于BCE
        pred_prob = thresholds_pred
        pred_prob_clamped = pred_prob.clamp(1e-8, 1-1e-8)
        pred_logits = torch.log(pred_prob_clamped / (1 - pred_prob_clamped))
    
    # 可导：直接使用概率值（软二值化）
    # EMD函数本身可以处理连续值，概率值可以视为"软"的0/1值
    pred_bin = pred_prob
    target_bin = (thresholds_target != 0).float()
    
    loss_emd = emd(pred_bin, target_bin, aggregation="mean")
    
    # 2. BCE损失：鼓励预测非零位置（避免预测全0）
    # 计算正样本权重（处理类别不平衡）
    # 正样本（阈值位置）远少于负样本（0位置），需要更高的权重
    num_neg = (target_bin == 0).sum().float()
    num_pos = (target_bin == 1).sum().float().clamp(min=1.0)
    pos_weight_value = (num_neg / num_pos).clamp(min=1.0, max=100.0)
    pos_weight = torch.tensor(pos_weight_value, device=thresholds_pred.device)
    
    loss_bce = F.binary_cross_entropy_with_logits(
        pred_logits,
        target_bin,
        pos_weight=pos_weight,
        reduction='mean'
    )
    
    # 3. MAE损失：数值精度（只在目标序列非零位置计算）
    mask = (thresholds_target != 0).float()
    if mask.sum() > 0:
        # 如果输入是logits，使用ReLU确保非负（或直接使用原始值）
        if thresholds_pred.min() < 0:
            # logits输入，使用ReLU确保非负用于MAE
            pred_values = F.relu(thresholds_pred)
        else:
            pred_values = thresholds_pred
        
        mae = torch.abs(pred_values - thresholds_target) * mask
        loss_mae = mae.sum() / (mask.sum() + 1e-8)
    else:
        # 如果没有非零位置，返回预测的L1范数（鼓励稀疏）
        if thresholds_pred.min() < 0:
            loss_mae = torch.mean(torch.abs(F.relu(thresholds_pred)))
        else:
            loss_mae = torch.mean(torch.abs(thresholds_pred))
    
    # 4. 加权组合
    total_loss = lambda_emd * loss_emd + lambda_mae * loss_mae + lambda_bce * loss_bce
    
    return total_loss
