"""
主训练脚本
使用示例和完整的训练流程
"""

import argparse
import os
import sys
from typing import Dict, Any

from config import load_config, create_default_config, create_sim_config, Config
from dataset import create_dataset, create_dataloader
from train import Trainer
from logger import create_logger
from utils import set_seed, get_device


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='MUNE深度学习训练框架')
    
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径')
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'cifar10'],
                       help='数据集名称')
    parser.add_argument('--model', type=str, default='simple_mlp',
                       choices=['simple_mlp', 'simple_cnn'],
                       help='模型名称')
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64,
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
    
    return parser.parse_args()


def create_config_from_args(args) -> Config:
    """从命令行参数创建配置"""
    if args.config:
        # 从配置文件加载
        config = load_config(args.config)
    else:
        # 创建默认配置
        config = create_sim_config()
    
    # 覆盖命令行参数
    config['model.name'] = args.model
    config['training.num_epochs'] = args.epochs
    config['dataset.batch_size'] = args.batch_size
    config['optimizer.lr'] = args.lr
    config['training.device'] = args.device
    config['training.save_dir'] = args.save_dir
    
    return config


def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建配置
    config = create_config_from_args(args)
    
    # 创建日志器
    logger = create_logger(
        name='MUNE',
        log_dir=args.log_dir,
        level='INFO',
        console=True,
        file=True
    )
    
    # 记录配置
    logger.log_config(config.to_dict())
    
    # 获取设备
    device = get_device(config['training.device'])
    logger.info(f"使用设备: {device}")
    
    try:
        # 创建数据集
        logger.info("创建数据集...")
        train_dataset = create_dataset(config['dataset.name'], config['dataset'], train=True)
        val_dataset = create_dataset(config['dataset.name'], config['dataset'], train=False)
        
        # 创建数据加载器
        train_loader = create_dataloader(train_dataset, config['dataset'], train=True)
        val_loader = create_dataloader(val_dataset, config['dataset'], train=False)
        
        logger.info(f"训练集大小: {len(train_dataset)}")
        logger.info(f"验证集大小: {len(val_dataset)}")
        logger.info(f"批次大小: {config['dataset.batch_size']}")
        
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
            
            # 评估
            val_metrics = trainer.evaluate(val_loader)
            logger.log_metrics(val_metrics, epoch=None)
            
        else:
            # 训练模式
            if args.resume:
                logger.info("从检查点恢复训练...")
                trainer.load_checkpoint(args.resume)
            
            # 开始训练
            logger.log_training_start(config['training.num_epochs'], str(device))
            
            training_results = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=config['training.num_epochs'],
                save_dir=config['training.save_dir']
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
    
    except Exception as e:
        logger.error(f"训练过程中出现错误: {str(e)}")
        raise
    
    logger.info("程序执行完成")


if __name__ == '__main__':
    main()
