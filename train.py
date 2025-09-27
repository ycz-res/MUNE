"""
训练模块
包含训练循环、验证、模型保存等功能
"""

from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import argparse

from dataset import SimDataset
from config import get_config
from model import LinearModel
from loss import MuThresholdLoss
from utils import set_seed


def get_args_parser():
    a_parser = argparse.ArgumentParser('VLP scripts', add_help=False)
    a_parser.add_argument('--batch_size', default=1, type=int)
    a_parser.add_argument('--epochs', default=20, type=int)
    a_parser.add_argument('--shuffle', default=True, type=bool)
    a_parser.add_argument('--num_workers', default=0, type=int)
    a_parser.add_argument('--pin_memory', default=False, type=bool)
    a_parser.add_argument('--device', default='cpu', type=str)

    return a_parser

def main(args):
    # 设置随机种子
    set_seed(57)
    
    config = get_config()
    
    print("=== 加载数据集===")
    # 使用同一个数据文件的不同百分比范围，并指定阶段标签
    train_dataset = SimDataset(config['SimDataset.data'], 'sim', start_percent=0.0, end_percent=0.05, stage='train')  # 前70%
    val_dataset = SimDataset(config['SimDataset.data'], 'sim', start_percent=0.05, end_percent=0.1, stage='val')  # 70%-85%
    test_dataset = SimDataset(config['SimDataset.data'], 'sim', start_percent=0.1, end_percent=0.15, stage='test')  # 85%-100%
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             collate_fn=SimDataset.collate_fn, num_workers=args.num_workers, 
                             pin_memory=args.pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                           collate_fn=SimDataset.collate_fn, num_workers=args.num_workers, 
                           pin_memory=args.pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                            collate_fn=SimDataset.collate_fn, num_workers=args.num_workers, 
                            pin_memory=args.pin_memory)

    
    model = LinearModel(d_model=64, max_thresholds=160)
    model.to(args.device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.95), weight_decay=1e-3)
    
    # 使用MU数量和阈值的加权损失函数
    criterion = MuThresholdLoss
    
    # 训练模型
    print(f"开始训练，共 {args.epochs} 个epoch")
    
    for epoch in range(args.epochs):
        # 训练一个epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, args.device, epoch, args.epochs)
        
        # 验证一个epoch
        val_loss = validate_one_epoch(model, val_loader, criterion, args.device, epoch, args.epochs)
        
        print("-" * 50)
    
    print("训练完成！")
    
    # 测试
    print("\n=== 测试阶段 ===")
    avg_test_loss = test_one_epoch(model, test_loader, criterion, args.device)
    print(f"测试完成, 平均损失: {avg_test_loss:.6f}")


def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    batch_count = 0
    
    for batch_idx, batch in enumerate(train_loader):
        src, tgt = batch
        src = src.to(device)
        tgt = tgt.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        nus, thresholds = model(src)
        
        # 计算损失
        # 这里需要根据tgt的格式来计算损失
        # tgt格式: [nus, threshold1, threshold2, ...]
        nus_target = tgt[:, 0:1]  # 取第一个元素作为MU数量目标
        thresholds_target = tgt[:, 1:]  # 取剩余元素作为阈值目标
        loss = criterion(nus, nus_target, thresholds, thresholds_target)
        
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
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            src, tgt = batch
            src = src.to(device)
            tgt = tgt.to(device)
            
            nus, thresholds = model(src)
            nus_target = tgt[:, 0:1]
            thresholds_target = tgt[:, 1:]
            loss = criterion(nus, nus_target, thresholds, thresholds_target)
            
            val_loss += loss.item()
            val_batch_count += 1
    
    avg_val_loss = val_loss / val_batch_count
    print(f"Epoch {epoch+1}/{total_epochs} 验证完成, 平均损失: {avg_val_loss:.6f}")
    return avg_val_loss


def test_one_epoch(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    test_batch_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            src, tgt = batch
            src = src.to(device)
            tgt = tgt.to(device)
            
            nus, thresholds = model(src)
            nus_target = tgt[:, 0:1]
            thresholds_target = tgt[:, 1:]
            loss = criterion(nus, nus_target, thresholds, thresholds_target)
            
            test_loss += loss.item()
            test_batch_count += 1
            
            # 显示前几个预测结果
            if batch_idx < 3:
                print(f"测试批次 {batch_idx+1}:")
                print(f"  真实MU数量: {nus_target.squeeze().tolist()}")
                print(f"  预测MU数量: {nus.squeeze().tolist()}")
                print(f"  损失: {loss.item():.6f}")
    
    avg_test_loss = test_loss / test_batch_count
    return avg_test_loss


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)

