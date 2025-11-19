"""
PyTorch Lightning 训练入口脚本
"""

import argparse
import os
import sys
import json
import warnings
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from framework.Lightning.module import MUNEModule
from framework.Lightning.data_module import MUNEDataModule
from config import get_config
from utils import set_seed

# 忽略 NVML 警告
warnings.filterwarnings('ignore', message='.*NVML.*')


def get_args_parser():
    """解析命令行参数"""
    parser = argparse.ArgumentParser('MU Threshold Prediction Training (Lightning)', add_help=False)
    
    # 数据相关
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers')
    parser.add_argument('--pin_memory', default=True, type=bool, help='Pin memory for data loading')
    parser.add_argument('--dataset_type', default='Sim', choices=['Sim', 'Real'], help='Dataset type')
    parser.add_argument('--threshold_mode', default='binary', choices=['value', 'binary'], 
                       help='Threshold output mode: binary=0/1 mask, value=actual threshold values')
    
    # 训练相关
    parser.add_argument('--epochs', default=10, type=int, help='Number of training epochs')
    parser.add_argument('--device', default='cuda', type=str, help='Device to use (cpu/cuda)')
    parser.add_argument('--lr', default=5e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay (L2 regularization)')
    parser.add_argument('--grad_clip', default=1.0, type=float, help='Gradient clipping value (0=disabled)')
    parser.add_argument('--patience', default=20, type=int, help='Early stopping patience')
    
    # 模型相关
    parser.add_argument('--model_type', default='LSTM', 
                       choices=['Linear', 'CNN', 'LSTM', 'MUNECNN', 'Transformer'], 
                       help='Model architecture type')
    parser.add_argument('--d_model', default=128, type=int, 
                       help='Model hidden dimension (default: 128)')
    parser.add_argument('--dropout', default=0.1, type=float, 
                       help='Dropout rate for regularization (0.0-1.0)')
    
    # 损失函数相关
    parser.add_argument('--loss_type', default='emd', 
                       choices=['ce', 'weighted_bce', 'dice', 'iou', 'f1', 'count', 'emd', 'hamming',
                               'jaccard', 'tversky', 'focal_tversky', 'combo', 'mixed'],
                       help='Loss function type')
    
    # 学习率调度器
    parser.add_argument('--lr_scheduler', default='plateau', 
                       choices=['none', 'cosine', 'plateau'], 
                       help='Learning rate scheduler type')
    parser.add_argument('--warmup_epochs', default=2, type=int, 
                       help='Warmup epochs for cosine scheduler')
    
    # 指标相关
    parser.add_argument('--metrics_threshold', default=0.65, type=float, 
                       help='Threshold for metrics calculation')
    
    # 输出相关
    parser.add_argument('--result_dir', default='result', type=str, 
                       help='Root directory to save experiment results')
    parser.add_argument('--timestamp', default=None, type=str, 
                       help='Experiment timestamp (e.g., 20251023_123456). If not provided, auto-generate')
    parser.add_argument('--save_log', default=False, type=bool, 
                       help='Save console output to log file in result directory')

    # 数据划分
    parser.add_argument('--train_split', default=0.85, type=float,
                       help='Fraction of data used for training (0-1)')
    parser.add_argument('--val_split', default=0.95, type=float,
                       help='Fraction of data used for train+val (train <= val <=1)')
    
    # Lightning相关
    parser.add_argument('--accelerator', default='gpu', type=str, 
                       help='Lightning accelerator (gpu, cpu, etc.)')
    parser.add_argument('--devices', default=1, type=int, 
                       help='Number of devices to use')
    parser.add_argument('--precision', default=32, type=int, 
                       choices=[16, 32], help='Training precision (16 or 32)')
    parser.add_argument('--enable_progress_bar', default=True, type=bool, 
                       help='Enable progress bar')
    
    return parser


def setup_log(result_dir, timestamp, enable=True):
    """
    设置日志重定向：将控制台输出同时保存到文件
    
    Args:
        result_dir: 结果目录路径
        timestamp: 时间戳
        enable: 是否启用重定向
    
    Returns:
        restore_func: 恢复函数，调用后恢复标准输出并关闭日志文件
        log_file_path: 日志文件路径，如果未启用则返回None
    """
    log_file_path = os.path.join(result_dir, f'train_{timestamp}.log')
    
    if not enable:
        return lambda: None, None
    
    log_file_obj = open(log_file_path, 'w', encoding='utf-8')
    
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(sys.stdout, log_file_obj)
    sys.stderr = Tee(sys.stderr, log_file_obj)
    print(f"📝 日志文件: {log_file_path}")
    
    def restore():
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file_obj.close()
    
    return restore, log_file_path


def main(args):
    """主训练函数"""
    # 设置随机种子
    set_seed(57)
    
    # 获取配置
    config = get_config()

    if not (0.0 < args.train_split < args.val_split <= 1.0):
        raise ValueError("train_split must be within (0, val_split) and val_split <= 1.0")
    
    # 创建保存目录
    if args.timestamp:
        timestamp = args.timestamp
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(args.result_dir, timestamp)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(os.path.join(result_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(result_dir, "train_visual"), exist_ok=True)
    
    # 设置日志重定向
    restore, log_file_path = setup_log(result_dir, timestamp, enable=args.save_log)
    
    try:
        print(f"🚀 开始训练: {args.model_type} + {args.loss_type}")
        print(f"📊 数据集: {args.dataset_type} | Epochs: {args.epochs}")
        print(f"💾 结果目录: {result_dir}\n")
        
        # 获取数据路径
        data_path = config['SimDataset.data']
        
        # 创建数据模块
        data_module = MUNEDataModule(
            data_path=data_path,
            dataset_type=args.dataset_type,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            threshold_mode=args.threshold_mode,
            train_split=args.train_split,
            val_split=args.val_split
        )
        
        # 创建模型模块
        model_module = MUNEModule(
            model_type=args.model_type,
            d_model=args.d_model,
            dropout=args.dropout,
            loss_type=args.loss_type,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            lr_scheduler=args.lr_scheduler,
            warmup_epochs=args.warmup_epochs,
            threshold_mode=args.threshold_mode,
            metrics_threshold=args.metrics_threshold,
            visual_dir=os.path.join(result_dir, 'train_visual')
        )
        
        # 配置回调函数
        callbacks = []
        
        # 模型检查点回调
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(result_dir, "checkpoints"),
            filename=f'best_model_{timestamp}',
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            save_last=True,
            verbose=True
        )
        callbacks.append(checkpoint_callback)
        
        # 早停回调
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=args.patience,
            verbose=True
        )
        callbacks.append(early_stop_callback)
        
        # 学习率监控回调
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        callbacks.append(lr_monitor)
        
        # 配置日志记录器
        logger = TensorBoardLogger(
            save_dir=result_dir,
            name='lightning_logs',
            version=timestamp
        )
        
        # 配置训练器
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            accelerator=args.accelerator if args.device == 'cuda' else 'cpu',
            devices=args.devices if args.device == 'cuda' else 1,
            precision=args.precision,
            callbacks=callbacks,
            logger=logger,
            enable_progress_bar=args.enable_progress_bar,
            log_every_n_steps=10,
            check_val_every_n_epoch=1,
            gradient_clip_val=args.grad_clip if args.grad_clip > 0 else None,
        )
        
        # 开始训练
        trainer.fit(model_module, data_module)
        
        # 保存训练配置和结果
        print("\n📊 保存训练数据...")
        save_training_data(
            trainer=trainer,
            model_module=model_module,
            data_module=data_module,
            save_path=os.path.join(result_dir, f'train_{timestamp}.json'),
            timestamp=timestamp,
            best_model_path=checkpoint_callback.best_model_path,
            args=args,
            config=config
        )
        
        print(f"\n✅ 训练完成!")
        print(f"   - 最佳模型: {checkpoint_callback.best_model_path}")
        print(f"   - 训练数据: {os.path.join(result_dir, f'train_{timestamp}.json')}")
        if args.save_log:
            print(f"   - 日志文件: {log_file_path}")
        
    finally:
        # 恢复日志重定向
        restore()


def save_training_data(trainer, model_module, data_module, save_path, timestamp, 
                      best_model_path, args=None, config=None):
    """保存训练数据为JSON格式"""
    # 从 LightningModule 中获取真实的训练历史
    training_history = getattr(model_module, 'training_history', [])
    
    # 准备保存的数据
    training_data = {
        'timestamp': timestamp,
        'total_epochs': trainer.current_epoch + 1,
        'best_model_path': best_model_path,
        'training_history': training_history,
        'framework': 'pytorch_lightning'
    }
    
    # 添加args配置
    if args is not None:
        training_data['config_args'] = vars(args)
    
    # 添加config配置
    if config is not None:
        training_data['config'] = config.to_dict() if hasattr(config, 'to_dict') else config
    
    # 保存为JSON
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 训练数据已保存: {save_path}")


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)

