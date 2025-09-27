"""
MUNE - 深度学习训练框架
"""

__version__ = "1.0.0"
__author__ = "MUNE Team"

from .model import BaseModel, create_model
from .dataset import BaseDataset, SimDataset, create_dataset, create_dataloader
from .train import Trainer
from .loss import BaseLoss, create_loss
from .metrics import BaseMetric, MetricManager, create_metric
from .config import Config, ConfigDict, load_config, create_default_config, create_sim_config, create_hp_config
from .utils import *
from .logger import Logger, create_logger

__all__ = [
    # 模型
    'BaseModel', 'create_model',
    # 数据集
    'BaseDataset', 'SimDataset', 'create_dataset', 'create_dataloader',
    # 训练
    'Trainer',
    # 损失函数
    'BaseLoss', 'create_loss',
    # 指标
    'BaseMetric', 'MetricManager', 'create_metric',
    # 配置
    'Config', 'ConfigDict', 'load_config', 'create_default_config', 'create_sim_config', 'create_hp_config',
    # 工具
    'Logger', 'create_logger',
]
