"""
训练模块
包含训练循环、验证、模型保存等功能
"""

from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import os
from datetime import datetime
import numpy as np

from dataset import SimDataset
from config import get_config
from model import LinearModel
from loss import thr_loss, focal_thr_loss
from visualization import MUThresholdVisualizer
from utils import set_seed
from metrics import mu_threshold_metrics, comprehensive_metrics, print_metrics_summary


def get_args_parser():
    a_parser = argparse.ArgumentParser('MU Threshold Prediction Training', add_help=False)
    a_parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training')
    a_parser.add_argument('--epochs', default=20, type=int, help='Number of training epochs')
    a_parser.add_argument('--shuffle', default=True, type=bool, help='Shuffle training data')
    a_parser.add_argument('--num_workers', default=0, type=int, help='Number of data loading workers')
    a_parser.add_argument('--pin_memory', default=False, type=bool, help='Pin memory for data loading')
    a_parser.add_argument('--device', default='cpu', type=str, help='Device to use (cpu/cuda)')
    
    # 新增参数
    a_parser.add_argument('--lr', default=1e-5, type=float, help='Learning rate')
    a_parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight decay')
    a_parser.add_argument('--patience', default=5, type=int, help='Early stopping patience')
    a_parser.add_argument('--loss_type', default='thr', choices=['thr', 'focal'], help='Loss function type')
    a_parser.add_argument('--save_best', default=True, type=bool, help='Save best model')
    a_parser.add_argument('--save_dir', default='checkpoints', type=str, help='Directory to save models')

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
    print(f"训练曲线将保存到: {curves_dir}")
    print(f"模型将保存到: {args.save_dir}")
    
    print("=== 加载数据集 ===")
    # 使用同一个数据文件的不同百分比范围，并指定阶段标签
    train_dataset = SimDataset(config['SimDataset.data'], 'sim', start_percent=0.0, end_percent=0.05, stage='train')
    val_dataset = SimDataset(config['SimDataset.data'], 'sim', start_percent=0.05, end_percent=0.1, stage='val')
    test_dataset = SimDataset(config['SimDataset.data'], 'sim', start_percent=0.1, end_percent=0.15, stage='test')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             collate_fn=SimDataset.collate_fn, num_workers=args.num_workers, 
                             pin_memory=args.pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                           collate_fn=SimDataset.collate_fn, num_workers=args.num_workers, 
                           pin_memory=args.pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                            collate_fn=SimDataset.collate_fn, num_workers=args.num_workers, 
                            pin_memory=args.pin_memory)

    print("=== 初始化模型 ===")
    model = LinearModel(d_model=64)
    model.to(args.device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    
    # 选择损失函数
    if args.loss_type == 'focal':
        criterion = focal_thr_loss
        print("使用 Focal Loss")
    else:
        criterion = thr_loss
        print("使用标准阈值损失")
    
    # 初始化可视化器
    visualizer = MUThresholdVisualizer(curves_dir)
    
    # 早停机制
    best_score = -float('inf')
    best_epoch = 0
    patience_counter = 0
    
    print(f"=== 开始训练，共 {args.epochs} 个epoch ===")
    
    for epoch in range(args.epochs):
        # 训练一个epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, args.device, epoch, args.epochs)
        
        # 验证一个epoch
        val_result = validate_one_epoch(model, val_loader, criterion, args.device, epoch, args.epochs)
        if isinstance(val_result, tuple):
            val_loss, val_metrics = val_result
        else:
            val_loss = val_result
            val_metrics = None
        
        # 更新可视化器
        visualizer.update_epoch(epoch + 1, train_loss, val_loss)
        
        # 早停和最佳模型保存
        if val_metrics and val_metrics['score'] > best_score:
            best_score = val_metrics['score']
            best_epoch = epoch + 1
            patience_counter = 0
            print(f"  🎯 新的最佳模型! 综合分数: {best_score:.4f}")
            
            # 保存最佳模型
            if args.save_best:
                model_path = os.path.join(args.save_dir, f'best_model_{timestamp}.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_score': best_score,
                    'val_metrics': val_metrics
                }, model_path)
                print(f"  💾 最佳模型已保存到: {model_path}")
        else:
            patience_counter += 1
        
        # 早停检查
        if patience_counter >= args.patience:
            print(f"  ⏹️ 早停触发! 连续 {args.patience} 个epoch无改善")
            break
        
        print("-" * 50)
    
    print(f"\n🏆 最佳模型在第 {best_epoch} 个epoch，综合分数: {best_score:.4f}")
    
    # 加载最佳模型进行测试
    if args.save_best and os.path.exists(os.path.join(args.save_dir, f'best_model_{timestamp}.pth')):
        checkpoint = torch.load(os.path.join(args.save_dir, f'best_model_{timestamp}.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        print("已加载最佳模型进行测试")
    
    print("=== 测试阶段 ===")
    avg_test_loss = test_one_epoch(model, test_loader, criterion, args.device, visualizer)
    print(f"测试完成, 平均损失: {avg_test_loss:.6f}")
    
    # 生成综合训练分析报告
    model_info = {
        "模型类型": "LinearModel",
        "隐藏维度": 64,
        "优化器": "AdamW",
        "学习率": args.lr,
        "权重衰减": args.weight_decay,
        "损失函数": args.loss_type,
        "早停耐心": args.patience
    }
    
    dataset_info = {
        "训练样本数": len(train_dataset),
        "验证样本数": len(val_dataset),
        "测试样本数": len(test_dataset),
        "批次大小": args.batch_size
    }
    
    visualizer.generate_comprehensive_report(
        test_loss=avg_test_loss,
        model_info=model_info,
        dataset_info=dataset_info
    )
    print(f"训练曲线已保存到: {curves_dir}")


def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    batch_count = 0
    
    for batch_idx, batch in enumerate(train_loader):
        src, tgt = batch
        # 移动到设备
        src = {key: value.to(device) for key, value in src.items()}
        tgt = {key: value.to(device) for key, value in tgt.items()}
        
        # 前向传播
        optimizer.zero_grad()
        thresholds_pred = model(src["cmap"])  # 模型只输出阈值预测
        
        # 计算损失
        loss = criterion(thresholds_pred, tgt["thresholds"])
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        batch_count += 1
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}/{total_epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}")
    
    avg_loss = total_loss / batch_count
    print(f"Epoch {epoch+1}/{total_epochs} 训练完成, 平均损失: {avg_loss:.6f}")
    return avg_loss


def validate_one_epoch(model, val_loader, criterion, device, epoch, total_epochs):
    model.eval()
    val_loss = 0.0
    val_batch_count = 0
    
    # 收集所有预测和真实值用于计算指标
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            src, tgt = batch
            # 移动到设备
            src = {key: value.to(device) for key, value in src.items()}
            tgt = {key: value.to(device) for key, value in tgt.items()}
            
            thresholds_pred = model(src["cmap"])  # 模型只输出阈值预测
            loss = criterion(thresholds_pred, tgt["thresholds"])
            
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
        val_metrics = comprehensive_metrics(all_pred, all_true)
        
        avg_val_loss = val_loss / val_batch_count
        print(f"Epoch {epoch+1}/{total_epochs} 验证完成, 平均损失: {avg_val_loss:.6f}")
        print(f"  验证指标 - 数量准确率: {val_metrics['count_accuracy']:.4f}, "
              f"位置IoU: {val_metrics['pos_iou']:.4f}, "
              f"位置F1: {val_metrics['pos_f1']:.4f}, "
              f"数值MAE: {val_metrics['val_mae']:.4f}, "
              f"综合分数: {val_metrics['composite_score']:.4f}")
        return avg_val_loss, val_metrics
    else:
        avg_val_loss = val_loss / val_batch_count
        print(f"Epoch {epoch+1}/{total_epochs} 验证完成, 平均损失: {avg_val_loss:.6f}")
        return avg_val_loss, None


def test_one_epoch(model, test_loader, criterion, device, visualizer):
    model.eval()
    test_loss = 0.0
    test_batch_count = 0
    
    # 收集预测统计数据
    all_predicted_mu_counts = []
    all_true_mu_counts = []
    all_predicted_threshold_counts = []
    all_true_threshold_counts = []
    
    # 收集所有预测和真实值用于计算指标
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            src, tgt = batch
            # 移动到设备
            src = {key: value.to(device) for key, value in src.items()}
            tgt = {key: value.to(device) for key, value in tgt.items()}
            
            thresholds_pred = model(src["cmap"])  # 模型只输出阈值预测
            loss = criterion(thresholds_pred, tgt["thresholds"])
            
            test_loss += loss.item()
            test_batch_count += 1
            
            # 收集预测和真实值用于指标计算
            all_predictions.append(thresholds_pred)
            all_targets.append(tgt["thresholds"])
            
            # 收集预测统计数据
            true_mu_counts = tgt['mus'].cpu().numpy().astype(int).tolist()
            true_threshold_counts = (tgt['thresholds'] != 0).sum(dim=1).cpu().numpy().astype(int).tolist()
            predicted_threshold_counts = (thresholds_pred != 0).sum(dim=1).cpu().numpy().astype(int).tolist()
            
            # 从阈值预测中推断MU数量（简化方法：假设每个阈值对应一个MU）
            predicted_mu_counts = predicted_threshold_counts
            
            all_predicted_mu_counts.extend(predicted_mu_counts)
            all_true_mu_counts.extend(true_mu_counts)
            all_predicted_threshold_counts.extend(predicted_threshold_counts)
            all_true_threshold_counts.extend(true_threshold_counts)
            
            # 显示前几个预测结果
            if batch_idx < 3:
                print(f"测试批次 {batch_idx+1}:")
                print(f"  真实MU数量: {true_mu_counts}")
                print(f"  真实阈值数量: {true_threshold_counts}")
                print(f"  预测阈值数量: {predicted_threshold_counts}")
                print(f"  损失: {loss.item():.6f}")
    
    # 计算测试指标
    if all_predictions:
        all_pred = torch.cat(all_predictions, dim=0)
        all_true = torch.cat(all_targets, dim=0)
        test_metrics = comprehensive_metrics(all_pred, all_true)
        
        # 使用新的指标打印函数
        print_metrics_summary(test_metrics, "测试阶段")
    
    # 更新可视化器的预测统计数据
    visualizer.update_prediction_stats(
        all_predicted_mu_counts, all_true_mu_counts,
        all_predicted_threshold_counts, all_true_threshold_counts
    )
    
    avg_test_loss = test_loss / test_batch_count
    return avg_test_loss



if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)

