"""
测试模块
用于加载已训练的模型并在测试集上评估性能
可以独立运行，也可以被train.py调用
"""

import torch
import argparse
import os
import numpy as np
import warnings

# 忽略 NVML 警告
warnings.filterwarnings('ignore', message='.*NVML.*')

from dataset import Sim
from config import get_config
from model import Linear, CNN, LSTM
from metrics import b_v_metrics
from loss import ce, focal_ce, thr, emd
import json


def get_args_parser():
    """解析命令行参数"""
    parser = argparse.ArgumentParser('MU Threshold Prediction Testing', add_help=False)
    parser.add_argument('--device', default='cuda', type=str, help='Device to use (cpu/cuda)')
    parser.add_argument('--model_type', default='LSTM', choices=['Linear', 'CNN', 'LSTM'], help='Model architecture type')
    parser.add_argument('--hidden_size', default=128, type=int, help='Hidden size for LSTM model (d_model, should match training)')
    parser.add_argument('--threshold_mode', default='binary', choices=['value', 'binary'], help='Threshold output mode')
    parser.add_argument('--dataset_type', default='Sim', choices=['Sim', 'Real'], help='Dataset type')
    parser.add_argument('--metrics_threshold', default=0.5, type=float, help='Threshold for metrics calculation')
    parser.add_argument('--timestamp', required=True, type=str, help='Model timestamp (e.g., 20251023_123456)')
    parser.add_argument('--result_dir', default='result', type=str, help='Root directory containing experiment results')
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size for testing')
    parser.add_argument('--num_collect', default=20, type=int, help='Number of samples to collect and save (0=do not collect)')
    parser.add_argument('--save_samples', default=True, type=bool, help='Save sample data in JSON')
    parser.add_argument('--loss_type', default='emd', choices=['thr', 'focal', 'ce', 'emd'], help='Loss function type')
    parser.add_argument('--use_weighted_loss', default=True, type=bool, help='Use weighted loss')
    parser.add_argument('--pos_weight', default=7.0, type=float, help='Positive class weight')
    
    return parser




def load_best_model(model_type, timestamp, result_dir_path, device, hidden_size=64):
    """
    根据时间戳加载最佳模型
    
    Args:
        model_type: 模型类型字符串
        timestamp: 模型时间戳
        result_dir_path: result目录路径 result/{timestamp}
        device: 设备
        hidden_size: 模型的隐藏层大小 (d_model)
    
    Returns:
        model: 加载权重后的模型
    """
    # 构建checkpoint路径：result/{timestamp}/checkpoints/best_model_{timestamp}.pth
    checkpoint_path = os.path.join(result_dir_path, 'checkpoints', f'best_model_{timestamp}.pth')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"未找到模型文件: {checkpoint_path}")
    
    # 创建模型并加载权重
    try:
        model = eval(model_type)(d_model=hidden_size).to(device)
    except (NameError, AttributeError) as e:
        raise ValueError(f"无法创建模型 '{model_type}': {e}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 打印加载信息
    print(f"✅ 成功加载模型: {checkpoint_path}")
    print(f"   模型类型={model_type}, d_model={hidden_size}")
    
    return model


def test(model, dataset, loss_fn, metrics_fn, device, threshold=0.5,
         show_progress=True, num_collect=None, batch_size=4):
    """
    在数据集上测试模型（随机采样指定数量样本）
    
    Args:
        model: 模型
        dataset: 数据集
        loss_fn: 损失函数
        metrics_fn: 指标计算函数
        device: 设备
        threshold: 阈值（用于损失计算，不影响metrics_fn，metrics_fn内部会根据mode处理）
        show_progress: 是否显示进度
        num_collect: 测试和收集的样本数量，None表示使用全部数据
        batch_size: 批处理大小
    
    Returns:
        avg_loss: 平均损失
        metrics: 指标字典
        sample_data: 样本数据字典
    """
    model.eval()
    total_samples = len(dataset)
    
    # 确定测试样本数量
    if num_collect is None:
        num_test = total_samples
        test_indices = list(range(total_samples))
        if show_progress:
            print(f"  🔍 测试全部数据 ({num_test} 个样本)...")
    else:
        num_test = min(num_collect, total_samples)
        test_indices = torch.randperm(total_samples)[:num_test].tolist()
        if show_progress:
            if num_collect > total_samples:
                print(f"  ⚠️  请求样本数 {num_collect} 超过数据集大小 {total_samples}，调整为 {num_test}")
            print(f"  🎲 随机采样 {num_test} 个样本进行测试...")
    
    # 收集数据
    cmap_list = []
    thresholds_list = []
    mus_list = []
    
    for idx in test_indices:
        cmap_data, mu_count, threshold_data = dataset[idx]
        cmap_list.append(cmap_data)
        thresholds_list.append(threshold_data)
        mus_list.append(mu_count)
    
    # 转换为tensor（CPU端）
    cmap_tensor = torch.stack(cmap_list)
    thresholds_tensor = torch.stack(thresholds_list)
    
    # 分批预测（避免显存溢出）
    total_loss = 0.0
    all_preds = []
    all_targets = []
    num_batches = (num_test + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(0, num_test, batch_size):
            batch_end = min(i + batch_size, num_test)
            batch_cmap = cmap_tensor[i:batch_end].to(device)
            batch_thresholds = thresholds_tensor[i:batch_end].to(device)
            
            # 前向传播
            outputs = model(batch_cmap)
            
            # 计算损失
            loss = loss_fn(outputs, batch_thresholds)
            total_loss += loss.item()
            
            # 根据threshold_mode决定是否二值化
            # 注意：这里preds用于计算指标，metrics_fn内部会根据mode参数处理
            # 对于binary模式，metrics_fn内部会进行二值化；对于value模式，直接使用原始值
            all_preds.append(outputs.cpu())
            all_targets.append(batch_thresholds.cpu())
            
            # 打印进度
            if show_progress and (i // batch_size + 1) % max(1, num_batches // 5) == 0:
                progress = (i + batch_size) / num_test * 100
                print(f"    🔍 测试进度: {min(i + batch_size, num_test)}/{num_test} ({progress:.0f}%)")
    
    if show_progress:
        print(f"  ✅ 测试完成: {num_test} 个样本")
    
    # 拼接所有批次
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # 计算平均损失和指标
    avg_loss = total_loss / num_batches
    metrics = metrics_fn(all_preds, all_targets)
    
    # 组装样本数据
    sample_data = {
        'indices': test_indices,
        'cmap': cmap_tensor.numpy(),
        'thresholds_true': thresholds_tensor.numpy(),
        'thresholds_pred': all_preds.numpy(),
        'mus_true': torch.stack(mus_list).numpy()
    }
    
    return avg_loss, metrics, sample_data


def save_test_data(test_loss, test_metrics, sample_data, timestamp, result_dir=None, save_samples=True, args=None, config=None):
    """保存测试数据"""
    print(f"\n📊 保存测试数据...")
    
    # 组装保存路径（保存到result/{timestamp}/目录）
    if result_dir is None:
        result_dir = os.path.join('result', timestamp)
    os.makedirs(result_dir, exist_ok=True)
    
    # 准备测试数据
    test_data = {
        'timestamp': timestamp,
        'test_loss': float(test_loss),
        'test_metrics': {k: float(v) for k, v in test_metrics.items()},
        'num_samples': len(sample_data['indices'])
    }
    
    # 添加args配置（转换为字典格式）
    if args is not None:
        test_data['config_args'] = vars(args)
    
    # 添加config配置（转换为字典格式）
    if config is not None:
        test_data['config'] = config.to_dict() if hasattr(config, 'to_dict') else config
    
    # 如果需要保存样本数据，添加到 JSON
    if save_samples:
        test_data['sample_data'] = {
            'indices': sample_data['indices'],
            'cmap': sample_data['cmap'].tolist(),
            'thresholds_true': sample_data['thresholds_true'].tolist(),
            'thresholds_pred': sample_data['thresholds_pred'].tolist(),
            'mus_true': sample_data['mus_true'].tolist()
        }
    
    # 保存为 JSON（保存到result/{timestamp}/目录）
    test_data_path = os.path.join(result_dir, f'test_{timestamp}.json')
    with open(test_data_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 测试数据已保存: {test_data_path}")


def main(args):
    """主测试函数"""
    # 加载配置
    config = get_config()
    
    print("=" * 50)
    print("🧪 MU Threshold Prediction - 模型测试")
    print("=" * 50)
    
    # 创建测试数据集
    print("\n📦 加载测试数据集...")
    Dataset = eval(args.dataset_type)
    test_dataset = Dataset(
        config['SimDataset.data'], 
        args.dataset_type, 
        start_percent=0.95, 
        end_percent=1.0,
        stage='test',
        threshold_mode=args.threshold_mode
    )
    
    print(f"✅ 测试集大小: {len(test_dataset)} 个样本")
    
    # 确定时间戳和result目录
    timestamp = args.timestamp
    result_dir_path = os.path.join(args.result_dir, timestamp)
    checkpoints_dir = os.path.join(result_dir_path, 'checkpoints')
    
    # 确保result目录和checkpoints目录存在
    if not os.path.exists(result_dir_path):
        raise FileNotFoundError(f"未找到结果目录: {result_dir_path}，请检查时间戳是否正确")
    if not os.path.exists(checkpoints_dir):
        raise FileNotFoundError(f"未找到checkpoints目录: {checkpoints_dir}")
    
    print(f"\n📥 使用时间戳: {timestamp}")
    print(f"📁 结果目录: {result_dir_path}")
    print(f"📁 Checkpoints目录: {checkpoints_dir}")
    
    # 创建并加载模型
    print(f"\n🔧 创建并加载模型: {args.model_type}")
    model = load_best_model(args.model_type, timestamp, result_dir_path, args.device, args.hidden_size)
    
    # 创建损失函数（支持加权）
    if args.use_weighted_loss and args.loss_type == 'ce':
        pos_weight_tensor = torch.tensor(args.pos_weight, device=args.device)
        def loss_fn(pred, target):
            return ce(pred, target, pos_weight=pos_weight_tensor)
    else:
        loss_fn = eval(args.loss_type)
    
    # 创建指标函数（使用自定义阈值和模式）
    def metrics_fn(pred, target):
        return b_v_metrics(pred, target, mode=args.threshold_mode, threshold=args.metrics_threshold)
    
    # 执行测试
    print("\n🧪 测试阶段")
    test_loss, test_metrics, sample_data = test(
        model, test_dataset, loss_fn, metrics_fn, args.device,
        threshold=args.metrics_threshold,
        show_progress=True, 
        num_collect=args.num_collect if args.num_collect > 0 else None,
        batch_size=args.batch_size
    )
    
    # 打印测试结果
    print("\n" + "=" * 60)
    print("📊 测试结果:")
    print("=" * 60)
    print(f"   Loss: {test_loss:.6f}")
    for metric_name, metric_value in test_metrics.items():
        print(f"   {metric_name}: {metric_value:.4f}")
    print("=" * 60)
    
    # 保存测试数据
    save_test_data(test_loss, test_metrics, sample_data, timestamp, result_dir=result_dir_path, save_samples=args.save_samples, args=args, config=config)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
