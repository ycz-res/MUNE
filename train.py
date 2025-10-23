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

from dataset import Sim
from config import get_config
from model import Linear, CNN, LSTM
from loss import ce, focal_ce, thr
from visualization import MUThresholdVisualizer
from utils import set_seed
from metrics import b_v_metrics


def get_args_parser():
    a_parser = argparse.ArgumentParser('MU Threshold Prediction Training', add_help=False)
    a_parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training')
    a_parser.add_argument('--epochs', default=10, type=int, help='Number of training epochs')
    a_parser.add_argument('--shuffle', default=True, type=bool, help='Shuffle training data')
    a_parser.add_argument('--num_workers', default=0, type=int, help='Number of data loading workers')
    a_parser.add_argument('--pin_memory', default=False, type=bool, help='Pin memory for data loading')
    a_parser.add_argument('--device', default='cpu', type=str, help='Device to use (cpu/cuda)')
    a_parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    a_parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight decay')
    a_parser.add_argument('--patience', default=5, type=int, help='Early stopping patience')
    a_parser.add_argument('--loss_type', default='ce', choices=['thr', 'focal', 'ce'], help='Loss function type: thr=threshold loss, focal=focal loss, ce=cross entropy')
    a_parser.add_argument('--model_type', default='LSTM', choices=['Linear', 'CNN', 'LSTM'], help='Model architecture type')
    a_parser.add_argument('--save_best', default=True, type=bool, help='Save best model')
    a_parser.add_argument('--save_dir', default='checkpoints', type=str, help='Directory to save models')
    a_parser.add_argument('--threshold_mode', default='binary', choices=['value', 'binary'], help='Threshold output mode: binary=0/1 mask, value=actual threshold values')
    a_parser.add_argument('--dataset_type', default='Sim', choices=['Sim'], help='Dataset type')
    a_parser.add_argument('--metrics_threshold', default=0.5, type=float, help='Threshold for metrics calculation (0.1-0.3 recommended for sparse data)')
    a_parser.add_argument('--use_weighted_loss', default=False, type=bool, help='Use weighted loss for imbalanced data')
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
    best_score = float('inf')  # 损失越小越好
    best_epoch = 0
    patience_counter = 0
    training_history = []  # 存储训练历史
    
    print(f"🚀 开始训练: {args.model_type} + {args.loss_type} | 数据集: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)} | Epochs: {args.epochs}")
    
    # 训练循环
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        print(f"\n🔄 Epoch {epoch+1}/{args.epochs} 开始训练...")
        
        # 训练和验证
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, args.device, epoch+1, args.epochs)
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
            save_model(model, optimizer, epoch + 1, best_score, val_metrics, args.save_dir, timestamp)
            print(f"🎯 新最佳模型! Val_Loss={best_score:.4f} ⭐ (耐心值重置)")
        else:
            patience_counter += 1
            print(f"⏳ 耐心值: {patience_counter}/{args.patience} (Val_Loss={current_loss:.4f})")
        
        # 早停检查
        if patience_counter >= args.patience:
            print(f"⏹️ 早停触发! 连续 {args.patience} 个epoch无改善")
            break
    
    print(f"🏆 最佳模型在第 {best_epoch} 个epoch，验证损失: {best_score:.4f}")
    
    # 测试阶段
    load_best_model(model, args.save_dir, timestamp)
    print("🧪 测试阶段")
    test_loss, test_metrics = validate_epoch(model, test_loader, loss_fn, metrics_fn, args.device)
    
    # 收集随机测试样本用于可视化
    print("📊 收集测试样本用于可视化...")
    sample_data = collect_test_samples(model, test_loader, args.device, num_samples=20, threshold=args.metrics_threshold)
    
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
        sample_data=sample_data,
        save_dir=curves_dir
    )


def train_epoch(model, train_loader, optimizer, loss_fn, device, current_epoch, total_epochs):
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


def generate_training_report(training_history, test_loss, test_metrics, sample_data, save_dir):
    """生成训练报告和可视化图表"""
    visualizer = MUThresholdVisualizer(save_dir)
    
    # 更新训练历史
    for epoch_data in training_history:
        visualizer.update_epoch(
            epoch_data['epoch'], 
            epoch_data['train_loss'], 
            epoch_data['val_loss'],
            metrics=epoch_data.get('val_metrics')
        )
    
    # 设置测试结果
    if test_metrics:
        visualizer.set_test_results(test_loss, test_metrics)
    
    # 设置样本数据
    if sample_data:
        visualizer.set_sample_data(
            sample_data['indices'],
            sample_data['cmap'],
            sample_data['thresholds_true'],
            sample_data['thresholds_pred'],
            sample_data['mus_true']
        )
    
    # 生成四张图
    visualizer.generate_four_figs()
    
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


def collect_test_samples(model, test_loader, device, num_samples=20, threshold=0.5):
    """直接通过索引收集随机测试样本"""
    model.eval()
    
    print(f"  🔍 收集 {num_samples} 个随机测试样本...")
    
    # 获取测试集总样本数
    test_dataset = test_loader.dataset
    total_samples = len(test_dataset)
    
    # 调整样本数
    if num_samples > total_samples:
        num_samples = total_samples
        print(f"  ⚠️  请求样本数超过测试集大小，调整为 {num_samples}")
    
    # 生成随机索引
    random_indices = torch.randperm(total_samples)[:num_samples].tolist()
    print(f"  📊 从 {total_samples} 个样本中随机选择了 {num_samples} 个")
    
    # 直接通过索引获取样本
    cmap_list = []
    thresholds_true_list = []
    mus_true_list = []
    
    for idx in random_indices:
        cmap_data, mu_count, threshold_data = test_dataset[idx]
        cmap_list.append(cmap_data)
        thresholds_true_list.append(threshold_data)
        mus_true_list.append(mu_count)
    
    # 转换为tensor并移到设备
    cmap_tensor = torch.stack(cmap_list).to(device)
    
    # 批量预测
    print(f"  🎯 对 {num_samples} 个样本进行批量预测（阈值={threshold}）...")
    with torch.no_grad():
        thresholds_pred_raw = model(cmap_tensor)
        # 二值化预测结果（使用指定阈值）
        thresholds_pred = (torch.sigmoid(thresholds_pred_raw) >= threshold).float()
    
    # 组装结果
    sample_data = {
        'indices': random_indices,
        'cmap': cmap_tensor.cpu().numpy(),
        'thresholds_true': torch.stack(thresholds_true_list).cpu().numpy(),
        'thresholds_pred': thresholds_pred.cpu().numpy(),
        'mus_true': torch.stack(mus_true_list).cpu().numpy()
    }
    
    print(f"  ✅ 样本收集完成: {len(sample_data['indices'])} 个样本")
    return sample_data


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

