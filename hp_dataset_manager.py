"""
HP数据集管理器
专门处理你的四个训练数据集和真实数据集
"""

import os
from typing import List, Dict, Any, Tuple
from dataset import SimDataset, create_dataloader
from torch.utils.data import ConcatDataset, DataLoader


class HPDatasetManager:
    """HP数据集管理器"""
    
    def __init__(self, data_root: str = './data'):
        self.data_root = data_root
        
        # 定义数据集文件
        self.train_dataset_names = [
            'train_dataset1_HP_better_range_1000.mat',
            # 'train_dataset2_HP_better_range_5000.mat',
            # 'train_dataset3_HP_better_range_5000.mat',
            # 'train_dataset4_HP_better_range_5000.mat'
        ]
        
        self.val_dataset_names = [
            'val_dataset1_HP_better_range_1000.mat',
            # 'val_dataset2_HP_better_range_5000.mat',
            # 'val_dataset3_HP_better_range_5000.mat',
            # 'val_dataset4_HP_better_range_5000.mat'
        ]
        
        self.real_control_dataset_name = 'real_data_control.mat'
        self.real_sci_dataset_name = 'real_data_sci.mat'
    
    def get_dataset_paths(self, dataset_names: List[str]) -> List[str]:
        """获取数据集文件路径"""
        paths = []
        for name in dataset_names:
            path = os.path.join(self.data_root, name)
            if os.path.exists(path):
                paths.append(path)
            else:
                print(f"警告: 文件不存在 {path}")
        return paths
    
    def create_train_dataset(self, config: Dict[str, Any]) -> ConcatDataset:
        """创建训练数据集"""
        train_paths = self.get_dataset_paths(self.train_dataset_names)
        
        if not train_paths:
            raise ValueError("没有找到任何训练数据文件")
        
        datasets = []
        for i, path in enumerate(train_paths):
            dataset_config = {
                'data_path': path,
                'dataset_name': f'train_dataset_{i+1}',
                **config
            }
            dataset = SimDataset(dataset_config)
            datasets.append(dataset)
            print(f"加载训练数据集 {i+1}: {path}, 样本数: {len(dataset)}")
        
        return ConcatDataset(datasets)
    
    def create_val_dataset(self, config: Dict[str, Any]) -> ConcatDataset:
        """创建验证数据集"""
        val_paths = self.get_dataset_paths(self.val_dataset_names)
        
        if not val_paths:
            raise ValueError("没有找到任何验证数据文件")
        
        datasets = []
        for i, path in enumerate(val_paths):
            dataset_config = {
                'data_path': path,
                'dataset_name': f'val_dataset_{i+1}',
                **config
            }
            dataset = SimDataset(dataset_config)
            datasets.append(dataset)
            print(f"加载验证数据集 {i+1}: {path}, 样本数: {len(dataset)}")
        
        return ConcatDataset(datasets)
    
    def create_real_control_dataset(self, config: Dict[str, Any]) -> SimDataset:
        """创建真实健康人数据集"""
        path = os.path.join(self.data_root, self.real_control_dataset_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"真实健康人数据文件不存在: {path}")
        
        dataset_config = {
            'data_path': path,
            'dataset_name': 'real_control',
            **config
        }
        dataset = SimDataset(dataset_config)
        print(f"加载真实健康人数据集: {path}, 样本数: {len(dataset)}")
        return dataset
    
    def create_real_sci_dataset(self, config: Dict[str, Any]) -> SimDataset:
        """创建真实病人数据集"""
        path = os.path.join(self.data_root, self.real_sci_dataset_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"真实病人数据文件不存在: {path}")
        
        dataset_config = {
            'data_path': path,
            'dataset_name': 'real_sci',
            **config
        }
        dataset = SimDataset(dataset_config)
        print(f"加载真实病人数据集: {path}, 样本数: {len(dataset)}")
        return dataset
    
    def create_dataloaders(self, config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
        """创建所有数据加载器"""
        # 训练和验证数据集
        train_dataset = self.create_train_dataset(config)
        val_dataset = self.create_val_dataset(config)
        
        # 真实数据集
        real_control_dataset = self.create_real_control_dataset(config)
        real_sci_dataset = self.create_real_sci_dataset(config)
        
        # 创建数据加载器
        train_loader = create_dataloader(train_dataset, config, train=True)
        val_loader = create_dataloader(val_dataset, config, train=False)
        real_control_loader = create_dataloader(real_control_dataset, config, train=False)
        real_sci_loader = create_dataloader(real_sci_dataset, config, train=False)
        
        return train_loader, val_loader, real_control_loader, real_sci_loader
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """获取数据集信息"""
        info = {
            'train_datasets': len(self.train_dataset_names),
            'val_datasets': len(self.val_dataset_names),
            'real_control_dataset': self.real_control_dataset_name,
            'real_sci_dataset': self.real_sci_dataset_name,
            'data_root': self.data_root
        }
        return info


def create_hp_config() -> Dict[str, Any]:
    """创建HP数据集的默认配置"""
    from config import create_hp_config as create_config
    config = create_config()
    return config['dataset']
