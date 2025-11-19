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
    改进版 1D EMD 损失：
    总 loss = w_emd * 形状EMD + w_bce * 带正类权重的BCE

    Args:
        thresholds_pred:   (B, L) logits
        thresholds_target: (B, L) 0/1 或非负
        config:
            - eps: float, 数值稳定项，默认 1e-8
            - reduction: "mean" | "sum" | "none"
            - ignore_empty: bool, 是否忽略全 0 样本（按 target 判断）
            - w_emd: float, EMD 形状项权重（默认 1.0）
            - w_bce: float, BCE 项权重（默认 0.5~1.0 建议先 0.5）
            - pos_weight: float, 正类权重，处理“模型爱全 0”（默认 3.0）
    """
    if config is None:
        config = {}

    eps         = config.get("eps", 1e-8)
    reduction   = config.get("reduction", "mean")
    ignore_empty= config.get("ignore_empty", True)
    w_emd       = config.get("w_emd", 1.0)
    w_bce       = config.get("w_bce", 0.5)
    pos_w_value = config.get("pos_weight", 3.0)   # 正类权重大一点，克制“全 0”

    # ---- 1. logits -> prob, 准备好 true ----
    pred = torch.sigmoid(thresholds_pred)        # (B, L)
    true = thresholds_target.float()             # (B, L)

    B, L = pred.shape

    # ---- 2. 形状 EMD（和你原来的版本保持一致思想）----
    pred_mass = pred.sum(dim=1, keepdim=True)    # (B, 1)
    true_mass = true.sum(dim=1, keepdim=True)    # (B, 1)

    pred_dist = pred / (pred_mass + eps)
    true_dist = true / (true_mass + eps)

    cdf_pred = torch.cumsum(pred_dist, dim=1)
    cdf_true = torch.cumsum(true_dist, dim=1)

    emd_shape = (cdf_pred - cdf_true).abs().mean(dim=1)   # (B,)

    # ---- 3. BCE 项：直接对每个位置做 0/1 分类 ----
    #   使用 pos_weight 提高“把 1 预测成 0”的惩罚，克服“全 0”偏好
    pos_weight = torch.full(
        (L,),
        fill_value=pos_w_value,
        device=thresholds_pred.device,
        dtype=thresholds_pred.dtype,
    )  # 形状 (L,)，会按列广播到 (B, L)

    bce = F.binary_cross_entropy_with_logits(
        thresholds_pred,
        true,
        reduction="none",
        pos_weight=pos_weight,
    )   # (B, L)

    bce_per_sample = bce.mean(dim=1)            # (B,)

    # ---- 4. 按样本组合总 loss ----
    loss_per_sample = w_emd * emd_shape + w_bce * bce_per_sample   # (B,)

    # ---- 5. 处理“target 全 0”的样本（很多时候你不关心这些）----
    if ignore_empty:
        non_empty_mask = (true_mass.squeeze(1) > eps).float()   # (B,)
        loss_per_sample = loss_per_sample * non_empty_mask

        if reduction == "none":
            return loss_per_sample

        total = loss_per_sample.sum()
        denom = non_empty_mask.sum().clamp_min(1.0)
        if reduction == "mean":
            return total / denom
        elif reduction == "sum":
            return total
        else:
            raise ValueError(f"Unsupported reduction: {reduction}")
    else:
        if reduction == "none":
            return loss_per_sample
        elif reduction == "mean":
            return loss_per_sample.mean()
        elif reduction == "sum":
            return loss_per_sample.sum()
        else:
            raise ValueError(f"Unsupported reduction: {reduction}")


def emd_ori(thresholds_pred, thresholds_target, config=None):
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
        # 新增：基于序列整体性的损失
        'binary_numeric': binary_numeric,
        'position_weighted_bce': position_weighted_bce,
        'set_matching': set_matching,
        'ordered_position': ordered_position,
        'combined_sequence': combined_sequence,
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

# ============================================================================
# 基于序列整体性的损失函数（将01序列看作整体）
# ============================================================================

def binary_numeric(thresholds_pred, thresholds_target, config=None):
    """
    二进制数值损失：将01序列转换为归一化的数值，计算数值差异
    
    思路：
    - 将500维01序列看作500位二进制数
    - 使用位置权重：位置i的权重为 w_i = 2^i / (2^500 - 1) 归一化
    - 或者使用线性权重：w_i = i / sum(0..499)
    - 计算加权和作为"数值"，然后计算差异
    
    Args:
        thresholds_pred:   (B, L)  预测 logits
        thresholds_target: (B, L)  真实标签 01序列
        config:            dict    参数配置
            - weight_mode: 'exponential' (2^i) 或 'linear' (i) 或 'uniform' (1)
            - normalize: bool, 是否归一化权重
    
    Returns:
        numeric_loss: 标量损失
    """
    if config is None:
        config = LOSS_CONFIG.get('binary_numeric', {})
    
    weight_mode = config.get('weight_mode', 'linear')  # 'exponential', 'linear', 'uniform'
    normalize = config.get('normalize', True)
    
    B, L = thresholds_pred.shape
    device = thresholds_pred.device
    
    # 创建位置权重
    if weight_mode == 'exponential':
        # 指数权重：2^i，但需要归一化避免数值溢出
        positions = torch.arange(L, dtype=torch.float32, device=device)
        weights = torch.pow(2.0, positions)
        if normalize:
            weights = weights / weights.sum()  # 归一化到[0,1]
    elif weight_mode == 'linear':
        # 线性权重：位置越靠后权重越大
        positions = torch.arange(L, dtype=torch.float32, device=device)
        weights = positions + 1.0  # 从1到L
        if normalize:
            weights = weights / weights.sum()
    else:  # uniform
        # 均匀权重
        weights = torch.ones(L, dtype=torch.float32, device=device) / L
    
    # 将预测转换为概率
    pred_prob = torch.sigmoid(thresholds_pred)
    target_binary = thresholds_target.float()
    
    # 计算加权数值
    pred_numeric = (pred_prob * weights).sum(dim=1)  # (B,)
    target_numeric = (target_binary * weights).sum(dim=1)  # (B,)
    
    # 计算数值差异（L1或L2）
    loss_mode = config.get('loss_mode', 'l1')  # 'l1' 或 'l2'
    if loss_mode == 'l2':
        numeric_loss = torch.mean((pred_numeric - target_numeric) ** 2)
    else:  # l1
        numeric_loss = torch.mean(torch.abs(pred_numeric - target_numeric))
    
    return numeric_loss


def position_weighted_bce(thresholds_pred, thresholds_target, config=None):
    """
    位置加权BCE损失：不同位置有不同的重要性权重
    
    思路：
    - 位置越靠后（阈值越大），权重越大
    - 或者使用指数权重，强调高阈值位置
    
    Args:
        thresholds_pred:   (B, L)  预测 logits
        thresholds_target: (B, L)  真实标签 01序列
        config:            dict    参数配置
            - weight_mode: 'exponential', 'linear', 'quadratic'
            - pos_weight: float, 正样本额外权重
            - normalize: bool, 是否归一化位置权重
    
    Returns:
        weighted_bce_loss: 标量损失
    """
    if config is None:
        config = LOSS_CONFIG.get('position_weighted_bce', {})
    
    weight_mode = config.get('weight_mode', 'linear')
    pos_weight = config.get('pos_weight', None)
    normalize = config.get('normalize', True)
    
    B, L = thresholds_pred.shape
    device = thresholds_pred.device
    
    # 创建位置权重
    positions = torch.arange(L, dtype=torch.float32, device=device)
    if weight_mode == 'exponential':
        weights = torch.exp(positions / L)  # 指数增长
    elif weight_mode == 'quadratic':
        weights = (positions + 1.0) ** 2
    else:  # linear
        weights = positions + 1.0
    
    if normalize:
        weights = weights / weights.mean()  # 归一化，保持均值不变
    
    # 扩展维度以匹配batch
    weights = weights.view(1, L)  # (1, L)
    
    # 计算逐位置的BCE
    target_binary = thresholds_target.float()
    bce_per_pos = F.binary_cross_entropy_with_logits(
        thresholds_pred, target_binary, reduction='none'
    )  # (B, L)
    
    # 应用位置权重
    weighted_bce = (bce_per_pos * weights).mean()
    
    # 如果提供了pos_weight，可以额外加权正样本
    if pos_weight is not None:
        pos_mask = target_binary > 0.5
        pos_weight_tensor = torch.where(pos_mask, pos_weight, 1.0)
        weighted_bce = (bce_per_pos * weights * pos_weight_tensor).mean()
    
    return weighted_bce


def set_matching(thresholds_pred, thresholds_target, config=None):
    """
    集合匹配损失：将01序列看作集合（1的位置），计算集合差异
    
    思路：
    - 提取1的位置作为集合
    - 计算集合的交集、并集、差集
    - 使用Jaccard距离或集合差异
    
    Args:
        thresholds_pred:   (B, L)  预测 logits
        thresholds_target: (B, L)  真实标签 01序列
        config:            dict    参数配置
            - threshold: float, 用于二值化预测的阈值
            - mode: 'jaccard', 'dice', 'symmetric_diff'
    
    Returns:
        set_loss: 标量损失
    """
    if config is None:
        config = LOSS_CONFIG.get('set_matching', {})
    
    threshold = config.get('threshold', 0.5)
    mode = config.get('mode', 'jaccard')
    smooth = config.get('smooth', 1e-8)
    
    # 二值化预测
    pred_prob = torch.sigmoid(thresholds_pred)
    pred_binary = (pred_prob >= threshold).float()
    target_binary = thresholds_target.float()
    
    if mode == 'jaccard':
        # Jaccard距离 = 1 - Jaccard相似度
        intersection = (pred_binary * target_binary).sum(dim=1)  # (B,)
        union = ((pred_binary + target_binary) > 0).float().sum(dim=1)  # (B,)
        jaccard = (intersection + smooth) / (union + smooth)
        set_loss = torch.mean(1.0 - jaccard)
    
    elif mode == 'dice':
        # Dice距离 = 1 - Dice相似度
        intersection = (pred_binary * target_binary).sum(dim=1)
        pred_sum = pred_binary.sum(dim=1)
        target_sum = target_binary.sum(dim=1)
        dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
        set_loss = torch.mean(1.0 - dice)
    
    else:  # symmetric_diff
        # 对称差：|A Δ B| = |A ∪ B| - |A ∩ B|
        intersection = (pred_binary * target_binary).sum(dim=1)
        union = ((pred_binary + target_binary) > 0).float().sum(dim=1)
        sym_diff = union - intersection
        # 归一化：除以序列长度
        set_loss = torch.mean(sym_diff.float() / thresholds_pred.shape[1])
    
    return set_loss


def ordered_position(thresholds_pred, thresholds_target, config=None):
    """
    有序位置损失：考虑1的位置顺序，强调位置顺序的重要性
    
    思路：
    - 提取1的位置索引，按顺序排列
    - 计算位置序列的差异（如L1距离）
    - 或者使用排序损失
    
    Args:
        thresholds_pred:   (B, L)  预测 logits
        thresholds_target: (B, L)  真实标签 01序列
        config:            dict    参数配置
            - threshold: float, 用于二值化预测的阈值
            - mode: 'l1', 'l2', 'rank'
    
    Returns:
        ordered_loss: 标量损失
    """
    if config is None:
        config = LOSS_CONFIG.get('ordered_position', {})
    
    threshold = config.get('threshold', 0.5)
    mode = config.get('mode', 'l1')
    smooth = config.get('smooth', 1e-8)
    
    # 二值化预测
    pred_prob = torch.sigmoid(thresholds_pred)
    pred_binary = (pred_prob >= threshold).float()
    target_binary = thresholds_target.float()
    
    B, L = thresholds_pred.shape
    device = thresholds_pred.device
    
    # 创建位置索引
    positions = torch.arange(L, dtype=torch.float32, device=device).view(1, L)  # (1, L)
    
    if mode == 'rank':
        # 排序损失：计算位置加权差异
        # 对于每个样本，计算预测和真实位置序列的差异
        pred_positions = (pred_binary * positions).sum(dim=1) / (pred_binary.sum(dim=1) + smooth)  # (B,)
        target_positions = (target_binary * positions).sum(dim=1) / (target_binary.sum(dim=1) + smooth)  # (B,)
        ordered_loss = torch.mean(torch.abs(pred_positions - target_positions))
    
    else:
        # L1或L2：直接计算位置加权差异
        pred_weighted = (pred_binary * positions).sum(dim=1)  # (B,)
        target_weighted = (target_binary * positions).sum(dim=1)  # (B,)
        
        if mode == 'l2':
            ordered_loss = torch.mean((pred_weighted - target_weighted) ** 2) / (L ** 2)  # 归一化
        else:  # l1
            ordered_loss = torch.mean(torch.abs(pred_weighted - target_weighted)) / L  # 归一化
    
    return ordered_loss


def combined_sequence(thresholds_pred, thresholds_target, config=None):
    """
    组合序列损失：结合多种序列整体性损失
    
    思路：
    - 结合位置加权、集合匹配、有序位置等多种损失
    - 可以灵活配置权重
    
    Args:
        thresholds_pred:   (B, L)  预测 logits
        thresholds_target: (B, L)  真实标签 01序列
        config:            dict    参数配置
            - weights: dict, 各损失函数的权重
                {'position_weighted': 0.4, 'set_matching': 0.3, 'ordered': 0.3}
    
    Returns:
        dict: {'total': 总损失, 'losses': 各损失值}
    """
    if config is None:
        config = LOSS_CONFIG.get('combined_sequence', {})
    
    weights = config.get('weights', {
        'position_weighted': 0.4,
        'set_matching': 0.3,
        'ordered': 0.3
    })
    
    losses = {}
    total_loss = 0.0
    
    # 位置加权BCE
    if 'position_weighted' in weights:
        pos_config = config.get('position_weighted_config', {'weight_mode': 'linear'})
        loss_val = position_weighted_bce(thresholds_pred, thresholds_target, pos_config)
        losses['position_weighted'] = loss_val
        total_loss += weights['position_weighted'] * loss_val
    
    # 集合匹配
    if 'set_matching' in weights:
        set_config = config.get('set_matching_config', {'mode': 'jaccard'})
        loss_val = set_matching(thresholds_pred, thresholds_target, set_config)
        losses['set_matching'] = loss_val
        total_loss += weights['set_matching'] * loss_val
    
    # 有序位置
    if 'ordered' in weights:
        ord_config = config.get('ordered_config', {'mode': 'l1'})
        loss_val = ordered_position(thresholds_pred, thresholds_target, ord_config)
        losses['ordered'] = loss_val
        total_loss += weights['ordered'] * loss_val
    
    return {
        'total': total_loss,
        'losses': losses
    }


# 统一配置：混合损失包含'weight'字段，单个损失不包含'weight'
LOSS_CONFIG = {
    'ce':       {'weight': 1.0, 'pos_weight': 40.0},
    'focal_ce': {'weight': 1.0, 'alpha': 0.25, 'gamma': 2.0},
    'count':    {'weight': 0.05},
    # 新增：基于序列整体性的损失
    'binary_numeric': {'weight': 0.3, 'weight_mode': 'linear', 'loss_mode': 'l1', 'normalize': True},
    'position_weighted_bce': {'weight': 0.5, 'weight_mode': 'linear', 'pos_weight': 10.0, 'normalize': True},
    'set_matching': {'weight': 0.3, 'threshold': 0.5, 'mode': 'jaccard', 'smooth': 1e-8},
    'ordered_position': {'weight': 0.2, 'threshold': 0.5, 'mode': 'l1'},
    'combined_sequence': {
        'weight': 1.0,
        'weights': {'position_weighted': 0.4, 'set_matching': 0.3, 'ordered': 0.3},
        'position_weighted_config': {'weight_mode': 'linear', 'pos_weight': 10.0},
        'set_matching_config': {'mode': 'jaccard'},
        'ordered_config': {'mode': 'l1'}
    },
}

