import torch
import torch.nn.functional as F

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

def ce(thresholds_pred, thresholds_target):
    """
    交叉熵式阈值损失函数
    对正样本位置(1)最大化预测分数，对负样本位置(0)最小化预测分数
    
    Args:
        thresholds_pred:   (B, 500)  模型预测分数 [0.1, 0.7, 0.8, 0.2, 0.5, 0.3]
        thresholds_target: (B, 500)  真实标签 [0, 1, 1, 0, 1, 0]
    
    Returns:
        total_loss: 标量损失
    """
    # 使用PyTorch内置的BCEWithLogitsLoss，自动处理sigmoid和log
    bce_loss = F.binary_cross_entropy_with_logits(
        thresholds_pred, 
        thresholds_target.float(),
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
