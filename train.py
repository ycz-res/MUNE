"""
训练模块
包含训练循环、验证、模型保存等功能
"""

from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import argparse
import os
from datetime import datetime
import numpy as np
import time
import json
import pandas as pd

from dataset import Sim
from config import get_config
from model import Linear, CNN, LSTM
from loss import ce, focal_ce, thr
from visualization import MUThresholdVisualizer
from utils import set_seed
from metrics import b_v_metrics


def get_args_parser():
    a_parser = argparse.ArgumentParser('MU Threshold Prediction Training', add_help=False)
    a_parser.add_argument('--batch_size', default=2, type=int, help='Batch size for training')
    a_parser.add_argument('--epochs', default=10, type=int, help='Number of training epochs')
    a_parser.add_argument('--shuffle', default=True, type=bool, help='Shuffle training data')
    a_parser.add_argument('--num_workers', default=0, type=int, help='Number of data loading workers')
    a_parser.add_argument('--pin_memory', default=False, type=bool, help='Pin memory for data loading')
    a_parser.add_argument('--device', default='cpu', type=str, help='Device to use (cpu/cuda)')
    a_parser.add_argument('--lr', default=1e-5, type=float, help='Learning rate')
    a_parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight decay')
    a_parser.add_argument('--patience', default=5, type=int, help='Early stopping patience')
    a_parser.add_argument('--loss_type', default='ce', choices=['thr', 'focal', 'ce'], help='Loss function type: thr=threshold loss, focal=focal loss, ce=cross entropy')
    a_parser.add_argument('--model_type', default='Linear', choices=['Linear', 'CNN', 'LSTM'], help='Model architecture type')
    a_parser.add_argument('--save_best', default=True, type=bool, help='Save best model')
    a_parser.add_argument('--save_dir', default='checkpoints', type=str, help='Directory to save models')
    a_parser.add_argument('--threshold_mode', default='binary', choices=['value', 'binary'], help='Threshold output mode: binary=0/1 mask, value=actual threshold values')
    a_parser.add_argument('--dataset_type', default='Sim', choices=['Sim'], help='Dataset type')
    a_parser.add_argument('--metrics_threshold', default=0.1, type=float, help='Threshold for metrics calculation (0.1-0.3 recommended for sparse data)')
    a_parser.add_argument('--use_weighted_loss', default=True, type=bool, help='Use weighted loss for imbalanced data')
    a_parser.add_argument('--pos_weight', default=50.0, type=float, help='Positive class weight for weighted loss')
    
    return a_parser

def main(args):
    # 设置随机种子
    set_seed(57)
    
    config = get_config()
    
    # 创建保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    curves_dir = f"plot/training_curves_{timestamp}"
    os.makedirs(curves_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    
    Dataset = eval(args.dataset_type)
    # 数据划分比例：训练集90%，验证集5%，测试集5%
    train_dataset = Dataset(config['SimDataset.data'], 'sim', start_percent=0.0, end_percent=0.9, stage='train', threshold_mode=args.threshold_mode)
    val_dataset = Dataset(config['SimDataset.data'], 'sim', start_percent=0.9, end_percent=0.95, stage='val', threshold_mode=args.threshold_mode)
    test_dataset = Dataset(config['SimDataset.data'], 'sim', start_percent=0.95, end_percent=1.0, stage='test', threshold_mode=args.threshold_mode)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             collate_fn=Dataset.collate_fn, num_workers=args.num_workers, 
                             pin_memory=args.pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                           collate_fn=Dataset.collate_fn, num_workers=args.num_workers, 
                           pin_memory=args.pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                            collate_fn=Dataset.collate_fn, num_workers=args.num_workers, 
                            pin_memory=args.pin_memory)

    # 初始化训练组件
    model = eval(args.model_type)(d_model=64).to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    
    # 创建损失函数（支持加权）
    if args.use_weighted_loss and args.loss_type == 'ce':
        pos_weight_tensor = torch.tensor(args.pos_weight, device=args.device)
        def loss_fn(pred, target):
            return ce(pred, target, pos_weight=pos_weight_tensor)
    else:
        loss_fn = eval(args.loss_type)
    
    # 创建指标函数（使用自定义阈值）
    def metrics_fn(pred, target):
        return b_v_metrics(pred, target, threshold=args.metrics_threshold)
    
    # 训练状态
    best_score = -float('inf')
    best_epoch = 0
    patience_counter = 0
    training_history = []  # 存储训练历史
    
    # 创建日志保存目录
    log_dir = os.path.join(curves_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # 初始化日志数据存储
    prediction_logs = {
        'train': [],
        'val': [],
        'metadata': {
            'timestamp': timestamp,
            'model_type': args.model_type,
            'loss_type': args.loss_type,
            'threshold_mode': args.threshold_mode,
            'metrics_threshold': args.metrics_threshold,
            'batch_size': args.batch_size,
            'lr': args.lr
        }
    }
    
    print(f"🚀 开始训练: {args.model_type} + {args.loss_type} | 数据集: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)} | Epochs: {args.epochs}")
    
    # 训练循环
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        print(f"\n🔄 Epoch {epoch+1}/{args.epochs} 开始训练...")
        
        # 训练和验证
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, args.device, epoch+1, args.epochs, prediction_logs)
        val_loss, val_metrics, val_pred, val_target = validate_epoch(model, val_loader, loss_fn, metrics_fn, args.device, prediction_logs)
        
        epoch_time = time.time() - epoch_start_time
        
        # 打印基础指标
        print(f"⏱️  Epoch {epoch+1} 完成，耗时: {epoch_time:.2f}秒")
        print(f"📊 训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f}")
        if val_metrics:
            print(f"📈 验证指标: {val_metrics}")
        else:
            print("📈 验证指标: None")
        
        # 记录训练历史
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'val_pred': val_pred,
            'val_target': val_target
        })
        
        # 早停和模型保存
        current_score = val_metrics.get('score', 0) if val_metrics else 0
        is_best = current_score > best_score
        
        if is_best:
            best_score = current_score
            best_epoch = epoch + 1
            patience_counter = 0
            save_model(model, optimizer, epoch + 1, best_score, val_metrics, args.save_dir, timestamp)
            print(f"🎯 新最佳模型! Score={best_score:.4f} ⭐ (耐心值重置)")
        else:
            patience_counter += 1
            print(f"⏳ 耐心值: {patience_counter}/{args.patience} (Score={current_score:.4f})")
        
        # 早停检查
        if patience_counter >= args.patience:
            print(f"⏹️ 早停触发! 连续 {args.patience} 个epoch无改善")
            break
    
    print(f"🏆 最佳模型在第 {best_epoch} 个epoch，综合分数: {best_score:.4f}")
    
    # 保存预测日志
    save_prediction_logs(prediction_logs, log_dir, timestamp)
    
    # 测试阶段
    load_best_model(model, args.save_dir, timestamp)
    print("🧪 测试阶段")
    test_loss, test_metrics, _, _ = validate_epoch(model, test_loader, loss_fn, metrics_fn, args.device, None)
    
    # 打印测试指标
    print(f"✅ 测试完成, 平均损失: {test_loss:.6f}")
    if test_metrics:
        print(f"test_metrics: {test_metrics}")
    else:
        print("test_metrics: None")
    
    # 生成可视化报告
    generate_training_report(
        training_history=training_history,
        test_loss=test_loss,
        test_metrics=test_metrics,
        best_epoch=best_epoch,
        best_score=best_score,
        model_info={
            "模型类型": args.model_type,
            "隐藏维度": 64,
            "优化器": "AdamW",
            "学习率": args.lr,
            "权重衰减": args.weight_decay,
            "损失函数": args.loss_type,
            "早停耐心": args.patience
        },
        dataset_info={
            "训练样本数": len(train_dataset),
            "验证样本数": len(val_dataset),
            "测试样本数": len(test_dataset),
            "批次大小": args.batch_size
        },
        save_dir=curves_dir,
        timestamp=timestamp
    )


def train_epoch(model, train_loader, optimizer, loss_fn, device, current_epoch, total_epochs, prediction_logs=None):
    model.train()
    total_loss = 0.0
    batch_count = 0
    total_batches = len(train_loader)
    
    # 每10个batch或每25%进度打印一次
    print_interval = max(1, total_batches // 10)  # 至少每10%打印一次
    if total_batches < 10:
        print_interval = max(1, total_batches // 4)  # 小数据集时更频繁
    
    batch_start_time = time.time()
    
    for batch_idx, batch in enumerate(train_loader):
        src, tgt = batch
        # 移动到设备
        src = {key: value.to(device) for key, value in src.items()}
        tgt = {key: value.to(device) for key, value in tgt.items()}
        
        # 前向传播
        optimizer.zero_grad()
        thresholds_pred = model(src["cmap"])  # 模型只输出阈值预测
        
        # 计算损失
        loss = loss_fn(thresholds_pred, tgt["thresholds"])
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        batch_count += 1
        
        # 定期打印进度和预测对比
        if (batch_idx + 1) % print_interval == 0 or (batch_idx + 1) == total_batches:
            progress = (batch_idx + 1) / total_batches * 100
            batch_time = time.time() - batch_start_time
            avg_batch_time = batch_time / print_interval
            current_avg_loss = total_loss / batch_count
            
            print(f"  📦 Batch {batch_idx+1}/{total_batches} ({progress:.1f}%) | "
                  f"Loss: {current_avg_loss:.4f} | "
                  f"速度: {avg_batch_time:.2f}s/batch")
            
            # 输出当前batch的预测对比
            batch_log = print_batch_predictions(thresholds_pred, tgt["thresholds"], batch_idx+1, current_epoch, "训练", 0.1)
            # 收集日志数据
            if prediction_logs is not None:
                prediction_logs['train'].append(batch_log)
            
            batch_start_time = time.time()
        
    avg_loss = total_loss / batch_count
    return avg_loss


def validate_epoch(model, val_loader, loss_fn, metrics_fn, device, prediction_logs=None):
    model.eval()
    val_loss = 0.0
    val_batch_count = 0
    total_val_batches = len(val_loader)
    
    # 收集所有预测和真实值用于计算指标
    all_predictions = []
    all_targets = []
    
    print(f"  🔍 开始验证 ({total_val_batches} batches)...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            src, tgt = batch
            # 移动到设备
            src = {key: value.to(device) for key, value in src.items()}
            tgt = {key: value.to(device) for key, value in tgt.items()}
            
            thresholds_pred = model(src["cmap"])  # 模型只输出阈值预测
            loss = loss_fn(thresholds_pred, tgt["thresholds"])
            
            val_loss += loss.item()
            val_batch_count += 1
            
            # 收集预测和真实值
            all_predictions.append(thresholds_pred)
            all_targets.append(tgt["thresholds"])
            
            # 验证进度提示和预测对比（只在验证集较大时显示）
            if total_val_batches > 5 and (batch_idx + 1) % max(1, total_val_batches // 5) == 0:
                progress = (batch_idx + 1) / total_val_batches * 100
                print(f"    🔍 验证进度: {batch_idx+1}/{total_val_batches} ({progress:.0f}%)")
                
                # 输出当前batch的预测对比
                batch_log = print_batch_predictions(thresholds_pred, tgt["thresholds"], batch_idx+1, 0, "验证", 0.1)
                # 收集日志数据
                if prediction_logs is not None:
                    prediction_logs['val'].append(batch_log)
    
    # 计算验证指标
    if all_predictions:
        all_pred = torch.cat(all_predictions, dim=0)
        all_true = torch.cat(all_targets, dim=0)
        
        # 使用综合评价指标
        val_metrics = metrics_fn(all_pred, all_true)
        
        avg_val_loss = val_loss / val_batch_count
        print(f"  ✅ 验证完成: {val_batch_count} batches")
        return avg_val_loss, val_metrics, all_pred, all_true
    else:
        raise RuntimeError("验证阶段无法计算指标，请检查数据或指标函数")


def print_batch_predictions(pred: torch.Tensor, target: torch.Tensor, batch_idx: int, epoch: int, stage: str, threshold: float = 0.1):
    """
    输出batch级别的预测对比信息并返回日志数据
    
    Args:
        pred: 预测结果 (batch_size, 500)
        target: 真实标签 (batch_size, 500)
        batch_idx: batch索引
        epoch: epoch索引
        stage: 阶段（训练/验证）
        threshold: 二值化阈值
    
    Returns:
        dict: 包含预测对比数据的字典
    """
    # 将预测转换为概率并二值化
    prob = torch.sigmoid(pred)
    pred_binary = (prob >= threshold).float()
    
    batch_size = pred.shape[0]
    
    print(f"    📊 {stage} Batch {batch_idx} 预测对比:")
    
    # 存储样本级数据
    sample_data = []
    
    # 输出前3个样本的详细对比
    for i in range(min(3, batch_size)):
        true_count = int(target[i].sum().item())
        pred_count = int(pred_binary[i].sum().item())
        
        # 找到真实和预测的阈值位置
        true_positions = torch.where(target[i] > 0)[0].cpu().numpy()
        pred_positions = torch.where(pred_binary[i] > 0)[0].cpu().numpy()
        
        print(f"      样本 {i+1}: 真实MU数量={true_count}, 预测MU数量={pred_count}")
        
        if len(true_positions) > 0:
            print(f"        真实阈值位置: {true_positions[:10]}{'...' if len(true_positions) > 10 else ''}")
        else:
            print(f"        真实阈值位置: 无")
            
        if len(pred_positions) > 0:
            print(f"        预测阈值位置: {pred_positions[:10]}{'...' if len(pred_positions) > 10 else ''}")
        else:
            print(f"        预测阈值位置: 无")
        
        # 计算重叠
        if len(true_positions) > 0:
            overlap = len(set(true_positions) & set(pred_positions))
            overlap_ratio = overlap / len(true_positions)
            print(f"        重叠位置数: {overlap}/{len(true_positions)} (重叠率: {overlap_ratio:.3f})")
        else:
            overlap = 0
            overlap_ratio = 1.0
            print(f"        重叠位置数: 0/0 (重叠率: 1.000)")
        
        # 存储样本数据
        sample_data.append({
            'sample_idx': i,
            'true_mu_count': true_count,
            'pred_mu_count': pred_count,
            'true_positions': true_positions.tolist(),
            'pred_positions': pred_positions.tolist(),
            'overlap_count': overlap,
            'overlap_ratio': overlap_ratio,
            'true_prob_values': prob[i][true_positions].cpu().numpy().tolist() if len(true_positions) > 0 else [],
            'pred_prob_values': prob[i][pred_positions].cpu().numpy().tolist() if len(pred_positions) > 0 else []
        })
    
    if batch_size > 3:
        print(f"      ... 还有 {batch_size - 3} 个样本")
    
    # 计算整体统计
    true_counts = target.sum(dim=1).cpu().numpy()
    pred_counts = pred_binary.sum(dim=1).cpu().numpy()
    
    avg_true = np.mean(true_counts)
    avg_pred = np.mean(pred_counts)
    mae = np.mean(np.abs(pred_counts - true_counts))
    
    print(f"    📈 Batch统计: 平均真实MU={avg_true:.2f}, 平均预测MU={avg_pred:.2f}, MAE={mae:.2f}")
    
    # 返回日志数据
    batch_log = {
        'epoch': epoch,
        'batch_idx': batch_idx,
        'stage': stage,
        'threshold': threshold,
        'batch_size': batch_size,
        'batch_stats': {
            'avg_true_mu': float(avg_true),
            'avg_pred_mu': float(avg_pred),
            'mae': float(mae),
            'true_counts': true_counts.tolist(),
            'pred_counts': pred_counts.tolist()
        },
        'sample_details': sample_data,
        'timestamp': datetime.now().isoformat()
    }
    
    return batch_log


def save_prediction_logs(prediction_logs, log_dir, timestamp):
    """
    保存预测日志数据为多种格式
    
    Args:
        prediction_logs: 预测日志数据
        log_dir: 日志保存目录
        timestamp: 时间戳
    """
    # 1. 保存为JSON格式（完整数据）
    json_path = os.path.join(log_dir, f'prediction_logs_{timestamp}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(prediction_logs, f, ensure_ascii=False, indent=2)
    
    # 2. 保存为CSV格式（便于可视化）
    csv_data = []
    for stage in ['train', 'val']:
        for log_entry in prediction_logs[stage]:
            for sample in log_entry['sample_details']:
                csv_data.append({
                    'epoch': log_entry['epoch'],
                    'batch_idx': log_entry['batch_idx'],
                    'stage': log_entry['stage'],
                    'sample_idx': sample['sample_idx'],
                    'true_mu_count': sample['true_mu_count'],
                    'pred_mu_count': sample['pred_mu_count'],
                    'overlap_count': sample['overlap_count'],
                    'overlap_ratio': sample['overlap_ratio'],
                    'true_positions_count': len(sample['true_positions']),
                    'pred_positions_count': len(sample['pred_positions']),
                    'timestamp': log_entry['timestamp']
                })
    
    if csv_data:
        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(log_dir, f'prediction_summary_{timestamp}.csv')
        df.to_csv(csv_path, index=False)
        
        # 3. 保存batch级统计CSV
        batch_stats_data = []
        for stage in ['train', 'val']:
            for log_entry in prediction_logs[stage]:
                batch_stats_data.append({
                    'epoch': log_entry['epoch'],
                    'batch_idx': log_entry['batch_idx'],
                    'stage': log_entry['stage'],
                    'avg_true_mu': log_entry['batch_stats']['avg_true_mu'],
                    'avg_pred_mu': log_entry['batch_stats']['avg_pred_mu'],
                    'mae': log_entry['batch_stats']['mae'],
                    'batch_size': log_entry['batch_size'],
                    'timestamp': log_entry['timestamp']
                })
        
        df_batch = pd.DataFrame(batch_stats_data)
        batch_csv_path = os.path.join(log_dir, f'batch_stats_{timestamp}.csv')
        df_batch.to_csv(batch_csv_path, index=False)
    
    print(f"📝 预测日志已保存:")
    print(f"   JSON格式: {json_path}")
    if csv_data:
        print(f"   样本级CSV: {csv_path}")
        print(f"   Batch级CSV: {batch_csv_path}")


def generate_training_report(training_history, test_loss, test_metrics, best_epoch, best_score, 
                           model_info, dataset_info, save_dir, timestamp):
    """生成训练报告和可视化图表"""
    visualizer = MUThresholdVisualizer(save_dir)
    
    # 更新训练历史
    for epoch_data in training_history:
        visualizer.update_epoch(
            epoch_data['epoch'], 
            epoch_data['train_loss'], 
            epoch_data['val_loss'],
            test_loss=test_loss,
            val_metrics=epoch_data.get('val_metrics'),
            val_pred=epoch_data.get('val_pred'),
            val_target=epoch_data.get('val_target')
        )
    
    # 生成综合报告
    visualizer.generate_comprehensive_report(
        test_loss=test_loss,
        model_info=model_info,
        dataset_info=dataset_info
    )
    
    print(f"📊 训练报告已生成: {save_dir}")


def save_model(model, optimizer, epoch, best_score, val_metrics, save_dir, timestamp):
    """保存最佳模型"""
    model_path = os.path.join(save_dir, f'best_model_{timestamp}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_score': best_score,
        'val_metrics': val_metrics
    }, model_path)


def load_best_model(model, save_dir, timestamp):
    """加载最佳模型"""
    model_path = os.path.join(save_dir, f'best_model_{timestamp}.pth')
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])



if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)

