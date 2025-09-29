import torch
import torch.nn.functional as F

def thr_loss(thresholds_pred, thresholds_target, 
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


def focal_thr_loss(thresholds_pred, thresholds_target, 
                   alpha=0.25, gamma=2.0, lambda_count=1.0, lambda_pos=2.0, lambda_val=1.0):
    """
    基于Focal Loss的阈值损失函数
    用于处理类别不平衡问题
    
    Args:
        thresholds_pred:   (B, 500)  模型预测
        thresholds_target: (B, 500)  真实标签
        alpha: Focal Loss参数
        gamma: Focal Loss参数
        lambda_count: 数量loss权重
        lambda_pos:   位置loss权重
        lambda_val:   数值loss权重
    Returns:
        total_loss: 标量损失
    """
    eps = 1e-8
    B = thresholds_pred.size(0)
    
    # 将问题转换为二分类问题
    pred_mask = (thresholds_pred != 0).float()
    true_mask = (thresholds_target != 0).float()
    
    # Focal Loss for position prediction
    pt = pred_mask * true_mask + (1 - pred_mask) * (1 - true_mask)
    focal_weight = alpha * (1 - pt) ** gamma
    loss_pos = -torch.mean(focal_weight * torch.log(pt + eps))
    
    # 数量损失
    pred_counts = pred_mask.sum(dim=1)
    true_counts = true_mask.sum(dim=1)
    loss_count = torch.mean(torch.abs(pred_counts - true_counts))
    
    # 数值损失
    common_mask = pred_mask * true_mask
    if torch.sum(common_mask) > 0:
        pred_values = thresholds_pred[common_mask.bool()]
        true_values = thresholds_target[common_mask.bool()]
        loss_val = torch.mean(torch.abs(pred_values - true_values))
    else:
        loss_val = torch.tensor(0.0, device=thresholds_pred.device, requires_grad=True)
    
    total_loss = lambda_count * loss_count + lambda_pos * loss_pos + lambda_val * loss_val
    return total_loss
