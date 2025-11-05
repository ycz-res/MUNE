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
import warnings

# 忽略 NVML 警告
warnings.filterwarnings('ignore', message='.*NVML.*')

from dataset import Sim
from config import get_config
from model import Linear, CNN, LSTM, MUNECNN, Transformer
from loss import ce, focal_ce, thr, emd
from utils import set_seed
from metrics import b_v_metrics
from visualization import plot_single_sample
import json


def get_args_parser():
    a_parser = argparse.ArgumentParser('MU Threshold Prediction Training', add_help=False)
    a_parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
    a_parser.add_argument('--epochs', default=10, type=int, help='Number of training epochs')
    a_parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers')
    a_parser.add_argument('--pin_memory', default=True, type=bool, help='Pin memory for data loading')
    a_parser.add_argument('--device', default='cuda', type=str, help='Device to use (cpu/cuda)')
    a_parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    a_parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight decay (L2 regularization)')
    a_parser.add_argument('--grad_clip', default=1.0, type=float, help='Gradient clipping value (0=disabled)')
    a_parser.add_argument('--patience', default=20, type=int, help='Early stopping patience')
    a_parser.add_argument('--loss_type', default='emd', choices=['thr', 'focal', 'ce', 'emd'], help='Loss function type: thr=threshold loss, focal=focal loss, ce=cross entropy, emd=earth mover\'s distance')
    a_parser.add_argument('--model_type', default='LSTM', choices=['Linear', 'CNN', 'LSTM', 'MUNECNN', 'Transformer'], help='Model architecture type')
    a_parser.add_argument('--save_best', default=True, type=bool, help='Save best model')
    a_parser.add_argument('--result_dir', default='result', type=str, help='Root directory to save experiment results')
    a_parser.add_argument('--timestamp', default=None, type=str, help='Experiment timestamp (e.g., 20251023_123456). If not provided, auto-generate')
    a_parser.add_argument('--threshold_mode', default='binary', choices=['value', 'binary'], help='Threshold output mode: binary=0/1 mask, value=actual threshold values')
    a_parser.add_argument('--dataset_type', default='Sim', choices=['Sim', 'Real'], help='Dataset type')
    a_parser.add_argument('--metrics_threshold', default=0.5, type=float, help='Threshold for metrics calculation (0.5 is standard, consistent with test)')
    a_parser.add_argument('--use_weighted_loss', default=True, type=bool, help='Use weighted loss for imbalanced data (only works with --loss_type ce)')
    a_parser.add_argument('--pos_weight', default=5.0, type=float, help='Positive class weight for CE loss only (ignored for other loss types)')
    a_parser.add_argument('--d_model', default=128, type=int, help='Model hidden dimension (default: 128, larger for better capacity)')
    a_parser.add_argument('--lr_scheduler', default='cosine', choices=['none', 'cosine', 'plateau'], help='Learning rate scheduler type')
    a_parser.add_argument('--warmup_epochs', default=5, type=int, help='Warmup epochs for cosine scheduler')
    a_parser.add_argument('--dropout', default=0.1, type=float, help='Dropout rate for regularization (0.0-1.0)')
    
    return a_parser

def main(args):
    # 设置随机种子
    set_seed(57)
    
    config = get_config()
    
    # 创建保存目录（如果提供了timestamp则使用，否则自动生成）
    if args.timestamp:
        timestamp = args.timestamp
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(args.result_dir, timestamp)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(os.path.join(result_dir, "checkpoints"), exist_ok=True)
    
    Dataset = eval(args.dataset_type)
    # 数据划分比例：训练集90%，验证集5%，测试集5%
    train_dataset = Dataset(config['SimDataset.data'], args.dataset_type, start_percent=0.0, end_percent=0.9, stage='train', threshold_mode=args.threshold_mode)
    val_dataset = Dataset(config['SimDataset.data'], args.dataset_type, start_percent=0.9, end_percent=0.95, stage='val', threshold_mode=args.threshold_mode)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             collate_fn=Dataset.collate_fn, num_workers=args.num_workers, 
                             pin_memory=args.pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                           collate_fn=Dataset.collate_fn, num_workers=args.num_workers, 
                           pin_memory=args.pin_memory)


    # 初始化训练组件
    model = eval(args.model_type)(d_model=args.d_model, dropout=args.dropout).to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    
    # 添加学习率调度器
    if args.lr_scheduler == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=args.lr * 0.01)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[args.warmup_epochs])
        print(f"📈 使用Cosine学习率调度器 (Warmup={args.warmup_epochs} epochs)")
    elif args.lr_scheduler == 'plateau':
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        print(f"📈 使用Plateau学习率调度器 (patience=5)")
    else:
        scheduler = None
        print(f"📈 不使用学习率调度器")
    
    # 创建损失函数
    # 注意：pos_weight 只在 CE 损失时生效
    if args.use_weighted_loss and args.loss_type == 'ce':
        pos_weight_tensor = torch.tensor(args.pos_weight, device=args.device)
        def loss_fn(pred, target):
            return ce(pred, target, pos_weight=pos_weight_tensor)
        print(f"📊 使用加权CE损失，正样本权重: {args.pos_weight}")
    else:
        loss_fn = eval(args.loss_type)
    
    # 创建指标函数（使用自定义阈值和模式）
    def metrics_fn(pred, target):
        return b_v_metrics(pred, target, mode=args.threshold_mode, threshold=args.metrics_threshold)
    
    # 训练状态
    best_score = float('inf')  # 损失越小越好
    best_epoch = 0
    patience_counter = 0
    training_history = []  # 存储训练历史
    
    # 组装保存路径（保存到result/{timestamp}/目录）
    best_model_path = os.path.join(result_dir, "checkpoints", f'best_model_{timestamp}.pth')
    train_data_path = os.path.join(result_dir, f'train_{timestamp}.json')
    
    print(f"🚀 开始训练: {args.model_type} + {args.loss_type} | 数据集: {len(train_dataset)}/{len(val_dataset)} | Epochs: {args.epochs}")
    
    # 训练循环
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        print(f"\n🔄 Epoch {epoch+1}/{args.epochs} 开始训练...")
        
        # 训练和验证
        visual_dir = os.path.join(result_dir, 'train_visual') if result_dir else None
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, args.device, epoch+1, args.grad_clip, visual_dir)
        val_loss, val_metrics = validate_epoch(model, val_loader, loss_fn, metrics_fn, args.device)
        
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
            'val_metrics': val_metrics
        })
        
        # 早停和模型保存（使用验证损失指导）
        current_loss = val_loss
        is_best = current_loss < best_score  # 损失越小越好
        
        if is_best:
            best_score = current_loss
            best_epoch = epoch + 1
            patience_counter = 0
            save_model(model, optimizer, epoch + 1, best_score, val_metrics, best_model_path)
            print(f"🎯 新最佳模型! Val_Loss={best_score:.4f} ⭐ (耐心值重置)")
        else:
            patience_counter += 1
            print(f"⏳ 耐心值: {patience_counter}/{args.patience} (Val_Loss={current_loss:.4f})")
        
        # 更新学习率调度器
        if scheduler is not None:
            if args.lr_scheduler == 'plateau':
                scheduler.step(current_loss)
            else:
                scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"📉 当前学习率: {current_lr:.6f}")
        
        # 早停检查
        if patience_counter >= args.patience:
            print(f"⏹️ 早停触发! 连续 {args.patience} 个epoch无改善")
            break
    
    print(f"🏆 最佳模型在第 {best_epoch} 个epoch，验证损失: {best_score:.4f}")
    
    # 保存训练数据
    print("\n📊 保存训练数据...")
    save_training_data(
        training_history=training_history,
        save_path=train_data_path,
        timestamp=timestamp,
        best_model_path=best_model_path,
        args=args,
        config=config
    )
    
    print(f"\n✅ 训练完成! 最佳模型已保存")
    print(f"   - 模型路径: {best_model_path}")
    print(f"   - 训练数据: {train_data_path}")
    print(f"\n💡 使用以下命令进行测试:")
    print(f"   python3 test.py --checkpoint {best_model_path}")


def train_epoch(model, train_loader, optimizer, loss_fn, device, current_epoch, grad_clip=1.0, visual_dir=None):
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
        
        # 每100个batch画一次图
        if visual_dir and batch_idx % 100 == 0:
            os.makedirs(visual_dir, exist_ok=True)
            try:
                save_path = os.path.join(visual_dir, f'train_{current_epoch}_{batch_idx}.png')
                plot_single_sample(src, thresholds_pred, tgt["thresholds"], save_path, epoch=current_epoch)
            except Exception as e:
                print(f"⚠️  训练样本可视化生成失败: {e}")
        
        # 计算损失
        loss = loss_fn(thresholds_pred, tgt["thresholds"])
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        total_loss += loss.item()
        batch_count += 1
        
        # 定期打印进度
        if (batch_idx + 1) % print_interval == 0 or (batch_idx + 1) == total_batches:
            progress = (batch_idx + 1) / total_batches * 100
            batch_time = time.time() - batch_start_time
            avg_batch_time = batch_time / print_interval
            current_avg_loss = total_loss / batch_count
            
            print(f"  📦 Batch {batch_idx+1}/{total_batches} ({progress:.1f}%) | "
                  f"Loss: {current_avg_loss:.4f} | "
                  f"速度: {avg_batch_time:.2f}s/batch")
            
            batch_start_time = time.time()
        
    avg_loss = total_loss / batch_count
    return avg_loss


def validate_epoch(model, val_loader, loss_fn, metrics_fn, device):
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
            
            # 验证进度提示（只在验证集较大时显示）
            if total_val_batches > 5 and (batch_idx + 1) % max(1, total_val_batches // 5) == 0:
                progress = (batch_idx + 1) / total_val_batches * 100
                print(f"    🔍 验证进度: {batch_idx+1}/{total_val_batches} ({progress:.0f}%)")
    
    # 计算验证指标
    if all_predictions:
        all_pred = torch.cat(all_predictions, dim=0)
        all_true = torch.cat(all_targets, dim=0)
        
        # 使用综合评价指标
        val_metrics = metrics_fn(all_pred, all_true)
        
        avg_val_loss = val_loss / val_batch_count
        print(f"  ✅ 验证完成: {val_batch_count} batches")
        return avg_val_loss, val_metrics
    else:
        raise RuntimeError("验证阶段无法计算指标，请检查数据或指标函数")


def save_training_data(training_history, save_path, timestamp, best_model_path, args=None, config=None):
    """保存训练数据为JSON格式"""
    # 准备保存的数据
    training_data = {
        'timestamp': timestamp,
        'total_epochs': len(training_history),
        'best_model_path': best_model_path,
        'training_history': training_history
    }
    
    # 添加args配置（转换为字典格式）
    if args is not None:
        training_data['config_args'] = vars(args)
    
    # 添加config配置（转换为字典格式）
    if config is not None:
        training_data['config'] = config.to_dict() if hasattr(config, 'to_dict') else config
    
    # 保存为JSON
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 训练数据已保存: {save_path}")


def save_model(model, optimizer, epoch, best_score, val_metrics, save_path):
    """保存最佳模型"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_score': best_score,
        'val_metrics': val_metrics
    }, save_path)





if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)

