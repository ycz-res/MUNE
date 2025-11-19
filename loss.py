"""
独立损失函数模块
用于500维01序列匹配任务

任务说明：
- 真实值：01序列（0或1）
# 统一配置：混合损失包含'weight'字段，单个损失不包含'weight'
LOSS_CONFIG = {
    'ce': {'weight': 0.5, 'pos_weight': 5.0},
    'count': {'weight': 1.0},
    # 'dice': {'weight': 0.3, 'smooth': 1e-8},
    # 'iou': {'weight': 0.2, 'smooth': 1e-8},
}
- 所有损失函数使用软mask确保可导

损失函数按适合程度排序：
1. 最合适：CE, Weighted BCE, Dice, IoU, F1
2. 较合适：Count, EMD, Hamming
3. 一般合适：Jaccard, Tversky, Focal Tversky, Combo
"""

import torch
import torch.nn.functional as F

def _auto_pos_weight(thresholds_target, clip=(1.0, 20.0)):
    """
    自动计算 pos_weight = 负样本数 / 正样本数，并夹到 clip 区间。
    返回标量 Tensor，可直接喂给 BCEWithLogits。
    """
    pos = thresholds_target.sum().item()
    neg = thresholds_target.numel() - pos
    pw = neg / max(pos, 1.0)
    pw = max(clip[0], min(clip[1], pw))
    return torch.tensor(pw, device=thresholds_target.device)


def ce(thresholds_pred, thresholds_target, config=None):
    """
    交叉熵损失（支持 pos_weight='auto' + pos_weight_clip）
    config 示例：
      {'pos_weight': 'auto', 'pos_weight_clip': (1.0, 20.0)}
      或 {'pos_weight': 5.0}
    """
    if config is None:
        # 允许从全局 LOSS_CONFIG 读取（mixed 会传进来，这里只是兜底）
        config = LOSS_CONFIG.get('ce', {})
    target_binary = thresholds_target.float()

    pos_weight_cfg = config.get('pos_weight', None)
    if isinstance(pos_weight_cfg, str) and pos_weight_cfg.lower() == 'auto':
        clip = config.get('pos_weight_clip', (1.0, 20.0))
        pos_weight = _auto_pos_weight(target_binary, clip=clip)
    elif isinstance(pos_weight_cfg, (int, float)):
        pos_weight = torch.tensor(float(pos_weight_cfg), device=thresholds_pred.device)
    else:
        pos_weight = None

    return F.binary_cross_entropy_with_logits(
        thresholds_pred, target_binary, pos_weight=pos_weight, reduction='mean'
    )


def focal_ce(thresholds_pred, thresholds_target, config=None):
    """
    Focal Cross-Entropy 损失，缓解正负样本不平衡
    config: {'alpha': 0.25, 'gamma': 2.0}
    """
    if config is None:
        config = LOSS_CONFIG.get('focal_ce', {})
    alpha = float(config.get('alpha', 0.25))
    gamma = float(config.get('gamma', 2.0))

    target_binary = thresholds_target.float()
    pred_prob = torch.sigmoid(thresholds_pred)
    ce = F.binary_cross_entropy_with_logits(
        thresholds_pred, target_binary, reduction='none'
    )

    p_t = pred_prob * target_binary + (1.0 - pred_prob) * (1.0 - target_binary)
    alpha_t = alpha * target_binary + (1.0 - alpha) * (1.0 - target_binary)
    focal_loss = alpha_t * ((1.0 - p_t).clamp(min=1e-8) ** gamma) * ce
    return focal_loss.mean()

# ============================================================================
# 1. 最合适：基础二分类损失
# ============================================================================

def ori_ce(thresholds_pred, thresholds_target, config=None):
    """
    交叉熵损失

    Args:
        thresholds_pred:   (B, L)  预测 logits
        thresholds_target: (B, L)  真实标签 01序列
        config:            dict    参数配置，如果为None则从LOSS_CONFIG读取

    Returns:
        ce_loss: 标量损失
    """
    # 从config或LOSS_CONFIG读取参数
    if config is None:
        config = LOSS_CONFIG.get('ce', {})
    pos_weight = config.get('pos_weight', None)
    
    target_binary = thresholds_target.float()
    # 处理pos_weight：如果是数字转换为Tensor
    if pos_weight is not None and isinstance(pos_weight, (int, float)):
        pos_weight = torch.tensor(pos_weight, device=thresholds_pred.device)
    ce_loss = F.binary_cross_entropy_with_logits(
        thresholds_pred,
        target_binary,
        pos_weight=pos_weight,
        reduction='mean'
    )
    return ce_loss


def weighted_bce(thresholds_pred, thresholds_target, config=None):
    """
    加权BCE损失

    Args:
        thresholds_pred:   (B, L)  预测 logits
        thresholds_target: (B, L)  真实标签 01序列
        config:            dict    参数配置，如果为None则从LOSS_CONFIG读取

    Returns:
        weighted_bce_loss: 标量损失
    """
    # 从config或LOSS_CONFIG读取参数
    if config is None:
        config = LOSS_CONFIG.get('weighted_bce', {})
    pos_weight = config.get('pos_weight', 5.0)
    
    target_binary = thresholds_target.float()
    # 处理pos_weight：如果是数字转换为Tensor
    if isinstance(pos_weight, (int, float)):
        pos_weight = torch.tensor(pos_weight, device=thresholds_pred.device)
    weighted_bce_loss = F.binary_cross_entropy_with_logits(
        thresholds_pred,
        target_binary,
        pos_weight=pos_weight,
        reduction='mean'
    )
    return weighted_bce_loss


def dice(thresholds_pred, thresholds_target, config=None):
    """
    Dice损失

    Args:
        thresholds_pred:   (B, L)  预测 logits
        thresholds_target: (B, L)  真实标签 01序列
        config:            dict    参数配置，如果为None则从LOSS_CONFIG读取

    Returns:
        dice_loss: 标量损失
    """
    # 从config或LOSS_CONFIG读取参数
    if config is None:
        config = LOSS_CONFIG.get('dice', {})
    smooth = config.get('smooth', 1e-8)
    
    pred_prob = torch.sigmoid(thresholds_pred)
    pred_mask = pred_prob
    true_mask = thresholds_target.float()

    intersection = torch.sum(pred_mask * true_mask, dim=1)
    union = torch.sum(pred_mask, dim=1) + torch.sum(true_mask, dim=1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    dice_loss = torch.mean(1 - dice)

    return dice_loss


def iou(thresholds_pred, thresholds_target, config=None):
    """
    IoU损失

    Args:
        thresholds_pred:   (B, L)  预测 logits
        thresholds_target: (B, L)  真实标签 01序列
        config:            dict    参数配置，如果为None则从LOSS_CONFIG读取

    Returns:
        iou_loss: 标量损失
    """
    # 从config或LOSS_CONFIG读取参数
    if config is None:
        config = LOSS_CONFIG.get('iou', {})
    smooth = config.get('smooth', 1e-8)
    
    pred_prob = torch.sigmoid(thresholds_pred)
    pred_mask = pred_prob
    true_mask = thresholds_target.float()
    
    intersection = torch.sum(pred_mask * true_mask, dim=1)
    union = torch.sum(pred_mask + true_mask - pred_mask * true_mask, dim=1)
    iou = intersection / (union + smooth)
    iou_loss = torch.mean(1 - iou)

    return iou_loss


def f1(thresholds_pred, thresholds_target, config=None):
    """
    F1损失
    
    Args:
        thresholds_pred:   (B, L)  预测 logits
        thresholds_target: (B, L)  真实标签 01序列
        config:            dict    参数配置，如果为None则从LOSS_CONFIG读取
    
    Returns:
        f1_loss: 标量损失
    """
    # 从config或LOSS_CONFIG读取参数
    if config is None:
        config = LOSS_CONFIG.get('f1', {})
    smooth = config.get('smooth', 1e-8)
    
    pred_prob = torch.sigmoid(thresholds_pred)
    pred_mask = pred_prob
    true_mask = thresholds_target.float()

    tp = torch.sum(pred_mask * true_mask, dim=1)
    fp = torch.sum(pred_mask * (1 - true_mask), dim=1)
    fn = torch.sum((1 - pred_mask) * true_mask, dim=1)

    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    f1 = 2.0 * (precision * recall) / (precision + recall + smooth)
    f1_loss = torch.mean(1 - f1)

    return f1_loss


# ============================================================================
# 2. 较合适：序列匹配专用损失
# ============================================================================

def count(thresholds_pred, thresholds_target, config=None):
    """
    数量损失：计算预测数量和真实数量的差异（归一化版本）

    Args:
        thresholds_pred:   (B, L)  预测 logits
        thresholds_target: (B, L)  真实标签 01序列
        config:            dict    参数配置，如果为None则从LOSS_CONFIG读取

    Returns:
        count_loss: 标量损失（已归一化到[0, 1]范围）
    """
    # count损失不需要参数，config参数保留以保持接口统一
    pred_prob = torch.sigmoid(thresholds_pred)
    pred_counts = pred_prob.sum(dim=1)

    true_mask = thresholds_target.float()
    true_counts = true_mask.sum(dim=1)

    # 归一化：除以序列长度，使损失范围从[0, L]变为[0, 1]
    seq_length = thresholds_pred.shape[1]
    count_loss = torch.mean(torch.abs(pred_counts - true_counts)) / seq_length
    return count_loss


def emd(thresholds_pred, thresholds_target, config=None):
    """
    EMD损失：前缀流量法，衡量分布差异

    Args:
        thresholds_pred:   (B, L)  预测 logits
        thresholds_target: (B, L)  真实标签 01序列
        config:            dict    参数配置，如果为None则从LOSS_CONFIG读取

    Returns:
        emd_loss: 标量损失
    """
    # emd损失不需要参数，config参数保留以保持接口统一
    eps = 1e-8

    pred_prob = torch.sigmoid(thresholds_pred)
    pred_bin = pred_prob
    true_bin = thresholds_target.float()

    flow = (torch.cumsum(pred_bin, dim=1) - torch.cumsum(true_bin, dim=1)).abs()
    loss_vec = flow.sum(dim=1)

    denom = torch.clamp(true_bin.sum(dim=1), min=eps)
    loss_vec = loss_vec / denom
    emd_loss = loss_vec.mean()

    return emd_loss


def hamming(thresholds_pred, thresholds_target, config=None):
    """
    Hamming损失：计算序列差异

    Args:
        thresholds_pred:   (B, L)  预测 logits
        thresholds_target: (B, L)  真实标签 01序列
        config:            dict    参数配置，如果为None则从LOSS_CONFIG读取

    Returns:
        hamming_loss: 标量损失
    """
    # hamming损失不需要参数，config参数保留以保持接口统一
    pred_prob = torch.sigmoid(thresholds_pred)
    pred_mask = pred_prob
    true_mask = thresholds_target.float()

    diff = torch.abs(pred_mask - true_mask)
    hamming_dist = torch.mean(diff, dim=1)
    hamming_loss = torch.mean(hamming_dist)

    return hamming_loss


# ============================================================================
# 3. 一般合适：变体损失
# ============================================================================

def jaccard(thresholds_pred, thresholds_target, config=None):
    """
    Jaccard损失

    Args:
        thresholds_pred:   (B, L)  预测 logits
        thresholds_target: (B, L)  真实标签 01序列
        config:            dict    参数配置，如果为None则从LOSS_CONFIG读取

    Returns:
        jaccard_loss: 标量损失
    """
    # 从config或LOSS_CONFIG读取参数
    if config is None:
        config = LOSS_CONFIG.get('jaccard', {})
    smooth = config.get('smooth', 1e-8)
    
    pred_prob = torch.sigmoid(thresholds_pred)
    pred_mask = pred_prob
    true_mask = thresholds_target.float()

    intersection = torch.sum(pred_mask * true_mask, dim=1)
    union = torch.sum(pred_mask, dim=1) + torch.sum(true_mask, dim=1) - intersection
    jaccard = (intersection + smooth) / (union + smooth)
    jaccard_loss = torch.mean(1 - jaccard)

    return jaccard_loss


def tversky(thresholds_pred, thresholds_target, config=None):
    """
    Tversky损失：可调整FP和FN的权重

    Args:
        thresholds_pred:   (B, L)  预测 logits
        thresholds_target: (B, L)  真实标签 01序列
        config:            dict    参数配置，如果为None则从LOSS_CONFIG读取

    Returns:
        tversky_loss: 标量损失
    """
    # 从config或LOSS_CONFIG读取参数
    if config is None:
        config = LOSS_CONFIG.get('tversky', {})
    alpha = config.get('alpha', 0.5)
    beta = config.get('beta', 0.5)
    smooth = config.get('smooth', 1e-8)
    
    pred_prob = torch.sigmoid(thresholds_pred)
    pred_mask = pred_prob
    true_mask = thresholds_target.float()

    tp = torch.sum(pred_mask * true_mask, dim=1)
    fp = torch.sum(pred_mask * (1 - true_mask), dim=1)
    fn = torch.sum((1 - pred_mask) * true_mask, dim=1)

    tversky = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    tversky_loss = torch.mean(1 - tversky)

    return tversky_loss


def focal_tversky(thresholds_pred, thresholds_target, config=None):
    """
    Focal Tversky损失：关注难样本

    Args:
        thresholds_pred:   (B, L)  预测 logits
        thresholds_target: (B, L)  真实标签 01序列
        config:            dict    参数配置，如果为None则从LOSS_CONFIG读取

    Returns:
        focal_tversky_loss: 标量损失
    """
    # 从config或LOSS_CONFIG读取参数
    if config is None:
        config = LOSS_CONFIG.get('focal_tversky', {})
    alpha = config.get('alpha', 0.5)
    beta = config.get('beta', 0.5)
    gamma = config.get('gamma', 1.0)
    smooth = config.get('smooth', 1e-8)
    
    pred_prob = torch.sigmoid(thresholds_pred)
    pred_mask = pred_prob
    true_mask = thresholds_target.float()

    tp = torch.sum(pred_mask * true_mask, dim=1)
    fp = torch.sum(pred_mask * (1 - true_mask), dim=1)
    fn = torch.sum((1 - pred_mask) * true_mask, dim=1)

    tversky = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    focal_tversky_loss = torch.mean((1 - tversky) ** gamma)

    return focal_tversky_loss


def combo(thresholds_pred, thresholds_target, config=None):
    """
    Combo损失：组合Dice和CE
    
    Args:
        thresholds_pred:   (B, L)  预测 logits
        thresholds_target: (B, L)  真实标签 01序列
        config:            dict    参数配置，如果为None则从LOSS_CONFIG读取
    
    Returns:
        combo_loss: 标量损失
    """
    # 从config或LOSS_CONFIG读取参数
    if config is None:
        config = LOSS_CONFIG.get('combo', {})
    alpha = config.get('alpha', 0.5)
    smooth = config.get('smooth', 1e-8)
    
    pred_prob = torch.sigmoid(thresholds_pred)
    pred_mask = pred_prob
    true_mask = thresholds_target.float()

    intersection = torch.sum(pred_mask * true_mask, dim=1)
    union = torch.sum(pred_mask, dim=1) + torch.sum(true_mask, dim=1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    dice_loss = torch.mean(1 - dice)

    target_binary = thresholds_target.float()
    ce_loss = F.binary_cross_entropy_with_logits(thresholds_pred, target_binary, reduction='mean')

    combo_loss = alpha * dice_loss + (1 - alpha) * ce_loss
    return combo_loss


# ============================================================================
# 混合损失函数
# ============================================================================

def mixed(thresholds_pred, thresholds_target, loss_config=None):
    """
    混合损失函数：支持多个损失函数加权组合，可配置每个损失的参数

    Args:
        thresholds_pred:   (B, L)  预测 logits
        thresholds_target: (B, L)  真实标签 01序列
        loss_config:       dict    损失配置，如果为None则使用LOSS_CONFIG：
                            简单格式: {'loss_name': weight}
                            详细格式: {'loss_name': {'weight': w, 'param': v, ...}}
                            可混合使用

    Returns:
        dict: {
            'total': 加权总损失,
            'losses': {'loss_name': loss_value, ...}
        }

    Example:
        >>> # 使用默认LOSS_CONFIG
        >>> result = mixed(pred, target)
        
        >>> # 自定义配置
        >>> loss_config = {
        ...     'ce': 0.5,
        ...     'dice': {'weight': 0.3, 'smooth': 1e-6},
        ...     'tversky': {'weight': 0.2, 'alpha': 0.7, 'beta': 0.3}
        ... }
        >>> result = mixed(pred, target, loss_config)
    """
    if loss_config is None:
        loss_config = LOSS_CONFIG
    
    if not isinstance(loss_config, dict):
        raise ValueError(f"loss_config 必须是 dict，当前类型: {type(loss_config)}")

    # 获取所有可用的损失函数
    available_losses = {
        'ce': ce,
        'weighted_bce': weighted_bce,
        'focal_ce': focal_ce,
        'dice': dice,
        'iou': iou,
        'f1': f1,
        'count': count,
        'emd': emd,
        'hamming': hamming,
        'jaccard': jaccard,
        'tversky': tversky,
        'focal_tversky': focal_tversky,
        'combo': combo,
    }

    # 计算各个损失
    individual_losses = {}
    total_loss = 0.0

    for loss_name, config in loss_config.items():
        if loss_name not in available_losses:
            raise ValueError(f"未知的损失函数: {loss_name}，可用函数: {list(available_losses.keys())}")

        # 解析配置：支持简单格式（数字）和详细格式（字典）
        if isinstance(config, (int, float)):
            # 简单格式：只有权重
            weight = config
            params = {}
        elif isinstance(config, dict):
            # 详细格式：包含权重和参数
            if 'weight' not in config:
                raise ValueError(f"{loss_name} 配置中缺少 'weight' 字段")
            weight = config['weight']
            params = {k: v for k, v in config.items() if k != 'weight'}
        else:
            raise ValueError(f"{loss_name} 配置格式错误，必须是数字或字典，当前类型: {type(config)}")

        if not isinstance(weight, (int, float)) or weight < 0:
            raise ValueError(f"权重必须是非负数，当前: {loss_name}={weight}")

        # 获取损失函数
        loss_fn = available_losses[loss_name]

        # 计算损失：使用新的config参数接口
        # params已经去掉了weight字段，直接作为config传递
        loss_val = loss_fn(thresholds_pred, thresholds_target, config=params if params else None)

        # 存储单个损失
        individual_losses[loss_name] = loss_val

        # 累加加权损失
        total_loss = total_loss + weight * loss_val

    return {
        'total': total_loss,
        'losses': individual_losses
    }


# ============================================================================
# Loss配置字典
# ============================================================================

# 统一配置：混合损失包含'weight'字段，单个损失不包含'weight'
LOSS_CONFIG = {
    'ce':       {'weight': 1.0, 'pos_weight': 40.0},
    'focal_ce': {'weight': 1.0, 'alpha': 0.25, 'gamma': 2.0},
    'count':    {'weight': 0.05},
}

