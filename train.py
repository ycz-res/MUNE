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
    loss_fn = eval(args.loss_type)
    metrics_fn = b_v_metrics
    
    # 训练状态
    best_score = -float('inf')
    best_epoch = 0
    patience_counter = 0
    training_history = []  # 存储训练历史
    
    print(f"🚀 开始训练: {args.model_type} + {args.loss_type} | 数据集: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)} | Epochs: {args.epochs}")
    
    # 训练循环
    for epoch in range(args.epochs):
        # 训练和验证
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, args.device)
        val_loss, val_metrics = validate_epoch(model, val_loader, loss_fn, metrics_fn, args.device)
        
        # 打印基础指标
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        if val_metrics:
            print(f"val_metrics: {val_metrics}")
        else:
            print("val_metrics: None")
        
        # 记录训练历史
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_metrics': val_metrics
        })
        
        # 早停和模型保存
        current_score = val_metrics.get('score', 0) if val_metrics else 0
        is_best = current_score > best_score
        
        if is_best:
            best_score = current_score
            best_epoch = epoch + 1
            patience_counter = 0
            save_model(model, optimizer, epoch + 1, best_score, val_metrics, args.save_dir, timestamp)
            print(f"📈 Epoch {epoch+1}: Loss={train_loss:.4f}/{val_loss:.4f} | Score={best_score:.4f} ⭐")
        else:
            patience_counter += 1
            print(f"📊 Epoch {epoch+1}: Loss={train_loss:.4f}/{val_loss:.4f} | Score={current_score:.4f}")
        
        # 早停检查
        if patience_counter >= args.patience:
            print(f"⏹️ 早停触发! 连续 {args.patience} 个epoch无改善")
            break
    
    print(f"🏆 最佳模型在第 {best_epoch} 个epoch，综合分数: {best_score:.4f}")
    
    # 测试阶段
    load_best_model(model, args.save_dir, timestamp)
    print("🧪 测试阶段")
    test_loss, test_metrics = validate_epoch(model, test_loader, loss_fn, metrics_fn, args.device)
    
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


def train_epoch(model, train_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    batch_count = 0
    
    for _, batch in enumerate(train_loader):
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
        
    avg_loss = total_loss / batch_count
    return avg_loss


def validate_epoch(model, val_loader, loss_fn, metrics_fn, device):
    model.eval()
    val_loss = 0.0
    val_batch_count = 0
    
    # 收集所有预测和真实值用于计算指标
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for _, batch in enumerate(val_loader):
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
    
    # 计算验证指标
    if all_predictions:
        all_pred = torch.cat(all_predictions, dim=0)
        all_true = torch.cat(all_targets, dim=0)
        
        # 使用综合评价指标
        val_metrics = metrics_fn(all_pred, all_true)
        
        avg_val_loss = val_loss / val_batch_count
        return avg_val_loss, val_metrics
    else:
        raise RuntimeError("验证阶段无法计算指标，请检查数据或指标函数")


def generate_training_report(training_history, test_loss, test_metrics, best_epoch, best_score, 
                           model_info, dataset_info, save_dir, timestamp):
    """生成训练报告和可视化图表"""
    visualizer = MUThresholdVisualizer(save_dir)
    
    # 更新训练历史
    for epoch_data in training_history:
        visualizer.update_epoch(
            epoch_data['epoch'], 
            epoch_data['train_loss'], 
            epoch_data['val_loss']
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

