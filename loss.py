"""
损失函数模块
MU数量和阈值的加权损失函数
"""

import torch
import torch.nn.functional as F

def MuThresholdLoss(nus_pred, nus_target, thresholds_pred, thresholds_target, lambda_nus=1.0, lambda_thr=1.0):
    """
    MU数量和阈值的加权损失函数
    
    Args:
        nus_pred:           (B, 1)         # 预测的MU数量
        nus_target:         (B, 1)         # 真实的MU数量
        thresholds_pred:    (B, N_max)     # 预测的阈值序列
        thresholds_target:  (B, N_max)     # 真实的阈值序列
        lambda_nus:         float          # MU数量损失权重
        lambda_thr:         float          # 阈值损失权重
    
    Returns:
        total_loss: 加权总损失
    """
    B = nus_pred.shape[0]
    
    # MU数量损失 (回归)
    nus_loss = F.l1_loss(nus_pred, nus_target)
    
    # 阈值损失 (L1损失)
    # 根据预测的MU数量选择对应的阈值进行比较
    thr_loss = 0.0
    for b in range(B):
        # 获取当前样本的预测MU数量
        pred_nus = int(nus_pred[b].item())
        target_nus = int(nus_target[b].item())
        
        # 选择较小的数量进行比较，避免超出范围
        compare_nus = min(pred_nus, target_nus, thresholds_pred.shape[1])
        
        if compare_nus > 0:
            # 比较前compare_nus个阈值
            pred_thr = thresholds_pred[b, :compare_nus]
            target_thr = thresholds_target[b, :compare_nus]
            thr_loss += F.l1_loss(pred_thr, target_thr)
    
    # 平均阈值损失
    thr_loss = thr_loss / B if B > 0 else torch.tensor(0.0, device=nus_pred.device)
    
    total_loss = lambda_nus * nus_loss + lambda_thr * thr_loss
    
    return total_loss