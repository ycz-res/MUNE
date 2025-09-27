"""
HP数据集训练脚本
专门用于训练你的HP Better数据集
"""

import argparse
import os
import sys
from typing import Dict, Any

from config import Config, create_hp_config
from model import create_model
from train import Trainer
from hp_dataset_manager import HPDatasetManager
from logger import create_logger
from utils import set_seed, get_device


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='HP数据集训练脚本')
    
    parser.add_argument('--data_root', type=str, default='./data',
                       help='数据根目录')
    parser.add_argument('--model', type=str, default='simple_mlp',
                       choices=['simple_mlp', 'simple_cnn'],
                       help='模型名称')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--device', type=str, default='auto',
                       help='设备 (auto, cuda, cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='日志保存目录')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--eval_only', action='store_true',
                       help='仅评估模式')
    parser.add_argument('--test_real_data', action='store_true',
                       help='在真实数据上测试')
    
    return parser.parse_args()


def create_hp_model_config(model_name: str, input_size: int = None) -> Config:
    """创建HP模型的配置"""
    config = create_hp_config()
    
    if model_name == 'simple_mlp':
        config['model.name'] = 'simple_mlp'
        config['model.config.input_size'] = input_size or 1000
        config['model.config.hidden_sizes'] = [512, 256, 128]
        config['model.config.num_classes'] = 2
        config['model.config.dropout_rate'] = 0.5
    elif model_name == 'simple_cnn':
        config['model.name'] = 'simple_cnn'
        config['model.config.input_channels'] = 1
        config['model.config.num_classes'] = 2
        config['model.config.input_size'] = input_size or 1000
    else:
        raise ValueError(f"不支持的模型: {model_name}")
    
    return config


def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建日志器
    logger = create_logger(
        name='HP_Training',
        log_dir=args.log_dir,
        level='INFO',
        console=True,
        file=True
    )
    
    # 创建HP数据集管理器
    dataset_manager = HPDatasetManager(data_root=args.data_root)
    
    # 记录数据集信息
    dataset_info = dataset_manager.get_dataset_info()
    logger.info("数据集信息:")
    for key, value in dataset_info.items():
        logger.info(f"  {key}: {value}")
    
    # 获取设备
    device = get_device(args.device)
    logger.info(f"使用设备: {device}")
    
    try:
        # 创建数据集配置
        dataset_config = create_hp_config()
        dataset_config['dataset.batch_size'] = args.batch_size
        
        # 创建数据集
        logger.info("创建数据集...")
        train_loader, val_loader, real_control_loader, real_sci_loader = dataset_manager.create_dataloaders(dataset_config)
        
        logger.info(f"训练集批次数: {len(train_loader)}")
        logger.info(f"验证集批次数: {len(val_loader)}")
        logger.info(f"真实健康人数据集批次数: {len(real_control_loader)}")
        logger.info(f"真实病人数据集批次数: {len(real_sci_loader)}")
        
        # 获取输入特征数（从第一个批次中获取）
        sample_batch = next(iter(train_loader))
        input_size = sample_batch[0].shape[1]  # (batch_size, features)
        logger.info(f"输入特征数: {input_size}")
        
        # 创建模型配置
        config = create_hp_model_config(args.model, input_size)
        
        # 更新配置
        config['optimizer.lr'] = args.lr
        config['training.num_epochs'] = args.epochs
        config['training.device'] = str(device)
        config['training.save_dir'] = args.save_dir
        config['training.early_stopping_patience'] = 15
        config['training.early_stopping_min_delta'] = 0.001
        
        # 记录配置
        logger.log_config(config.to_dict())
        
        # 创建训练器
        trainer = Trainer(config)
        
        # 记录模型信息
        model_info = trainer.model.get_model_info()
        logger.log_model_info(model_info)
        
        if args.eval_only:
            # 仅评估模式
            if args.resume is None:
                logger.error("评估模式需要指定检查点路径 (--resume)")
                return
            
            logger.info("加载模型进行评估...")
            trainer.load_checkpoint(args.resume)
            
            # 在验证集上评估
            logger.info("在验证集上评估...")
            val_metrics = trainer.evaluate(val_loader)
            logger.log_metrics(val_metrics, epoch=None)
            
            if args.test_real_data:
                # 在真实数据上测试
                logger.info("在真实健康人数据上测试...")
                real_control_metrics = trainer.evaluate(real_control_loader)
                logger.log_metrics(real_control_metrics, epoch=None)
                
                logger.info("在真实病人数据上测试...")
                real_sci_metrics = trainer.evaluate(real_sci_loader)
                logger.log_metrics(real_sci_metrics, epoch=None)
        
        else:
            # 训练模式
            if args.resume:
                logger.info("从检查点恢复训练...")
                trainer.load_checkpoint(args.resume)
            
            # 开始训练
            logger.log_training_start(args.epochs, str(device))
            
            training_results = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=args.epochs,
                save_dir=args.save_dir
            )
            
            logger.log_training_end(
                training_results['training_time'],
                training_results['best_metric']
            )
            
            # 保存训练历史
            metrics_history = {
                'train_losses': training_results['train_losses'],
                'val_losses': training_results['val_losses'],
                'train_metrics': training_results['train_metrics'],
                'val_metrics': training_results['val_metrics']
            }
            logger.save_metrics_history(metrics_history)
            
            # 最终评估
            logger.info("进行最终评估...")
            final_metrics = trainer.evaluate(val_loader)
            logger.log_metrics(final_metrics, epoch=None)
            
            if args.test_real_data:
                # 在真实数据上测试
                logger.info("在真实健康人数据上测试...")
                real_control_metrics = trainer.evaluate(real_control_loader)
                logger.log_metrics(real_control_metrics, epoch=None)
                
                logger.info("在真实病人数据上测试...")
                real_sci_metrics = trainer.evaluate(real_sci_loader)
                logger.log_metrics(real_sci_metrics, epoch=None)
    
    except Exception as e:
        logger.error(f"训练过程中出现错误: {str(e)}")
        raise
    
    logger.info("程序执行完成")


if __name__ == '__main__':
    main()
