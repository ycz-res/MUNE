"""
日志系统模块
支持控制台和文件日志输出
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any
import json


class Logger:
    """日志管理器"""
    
    def __init__(self, name: str = 'MUNE', log_dir: str = './logs', 
                 level: str = 'INFO', console: bool = True, file: bool = True):
        self.name = name
        self.log_dir = log_dir
        self.level = getattr(logging, level.upper())
        
        # 创建日志目录
        if file:
            os.makedirs(log_dir, exist_ok=True)
        
        # 创建logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        
        # 清除已有的处理器
        self.logger.handlers.clear()
        
        # 设置格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 控制台处理器
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # 文件处理器
        if file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(self.level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
            self.log_file = log_file
        else:
            self.log_file = None
    
    def info(self, message: str, **kwargs):
        """信息日志"""
        self.logger.info(message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """调试日志"""
        self.logger.debug(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """警告日志"""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """错误日志"""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """严重错误日志"""
        self.logger.critical(message, **kwargs)
    
    def log_metrics(self, metrics: Dict[str, float], epoch: Optional[int] = None):
        """记录指标"""
        if epoch is not None:
            message = f"Epoch {epoch} - "
        else:
            message = "Metrics - "
        
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.info(message + metric_str)
    
    def log_config(self, config: Dict[str, Any]):
        """记录配置"""
        self.info("Configuration:")
        self.info(json.dumps(config, indent=2, ensure_ascii=False))
    
    def log_model_info(self, model_info: Dict[str, Any]):
        """记录模型信息"""
        self.info("Model Information:")
        for key, value in model_info.items():
            self.info(f"  {key}: {value}")
    
    def log_training_start(self, num_epochs: int, device: str):
        """记录训练开始"""
        self.info("=" * 50)
        self.info(f"Training started - Epochs: {num_epochs}, Device: {device}")
        self.info("=" * 50)
    
    def log_training_end(self, training_time: float, best_metric: float):
        """记录训练结束"""
        self.info("=" * 50)
        self.info(f"Training completed - Time: {training_time:.2f}s, Best metric: {best_metric:.4f}")
        self.info("=" * 50)
    
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """记录epoch开始"""
        self.info(f"Epoch {epoch+1}/{total_epochs} started")
    
    def log_epoch_end(self, epoch: int, train_metrics: Dict[str, float], 
                     val_metrics: Dict[str, float]):
        """记录epoch结束"""
        self.info(f"Epoch {epoch+1} completed:")
        self.info(f"  Train: {self._format_metrics(train_metrics)}")
        self.info(f"  Val:   {self._format_metrics(val_metrics)}")
    
    def _format_metrics(self, metrics: Dict[str, float]) -> str:
        """格式化指标"""
        return " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    
    def save_metrics_history(self, metrics_history: Dict[str, list], 
                           filepath: Optional[str] = None):
        """保存指标历史"""
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = os.path.join(self.log_dir, f'metrics_history_{timestamp}.json')
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics_history, f, indent=2, ensure_ascii=False)
        
        self.info(f"Metrics history saved to: {filepath}")


def create_logger(name: str = 'MUNE', log_dir: str = './logs', 
                 level: str = 'INFO', console: bool = True, file: bool = True) -> Logger:
    """创建日志器"""
    return Logger(name, log_dir, level, console, file)


# 全局日志器实例
_global_logger: Optional[Logger] = None


def get_logger() -> Logger:
    """获取全局日志器"""
    global _global_logger
    if _global_logger is None:
        _global_logger = create_logger()
    return _global_logger


def set_logger(logger: Logger) -> None:
    """设置全局日志器"""
    global _global_logger
    _global_logger = logger
