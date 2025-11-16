import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Any
from utils import load_data
import os



class Sim(Dataset):
    """仿真数据集类"""
    
    def __init__(self, data_path: str, data_type: str = 'Sim', start_percent: float = 0.0, 
                 end_percent: float = 1.0, stage: str = 'train', threshold_mode: str = 'binary'):
        self.data_path = data_path
        self.data_type = data_type
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.stage = stage
        self.threshold_mode = threshold_mode
        
        if stage not in ['train', 'val', 'test']:
            raise ValueError("阶段标签必须是 'train', 'val', 或 'test'")
        if not (0.0 <= start_percent <= 1.0 and 0.0 <= end_percent <= 1.0):
            raise ValueError("百分比必须在0.0到1.0之间")
        if start_percent >= end_percent:
            raise ValueError("起始百分比必须小于结束百分比")
        
        # 根据数据类型选择加载方法
        self.data_dict = self.__load_data(self.data_type)
        
        # 提取数据
        self.cmap_amplitudes = self.data_dict['data']       # (N, 500) - CMAP幅值数据
        self.mu_count_labels = self.data_dict['label_num'] # (N,) - 运动单位数量标签
        self.mu_thresholds = self.data_dict['muThr']  # (N, 500) - 运动单位阈值位置
        
        # 计算数据范围
        self.total_samples = self.cmap_amplitudes.shape[0]
        self.start_idx = int(self.total_samples * start_percent)
        self.end_idx = int(self.total_samples * end_percent)
        self.num_samples = self.end_idx - self.start_idx
        
        print(f"数据集信息 - 当前阶段: {self.stage}: 总样本数={self.total_samples}, "
              f"使用范围=[{self.start_idx}:{self.end_idx}], 实际样本数={self.num_samples}")
    
    def __load_data(self, data_type: str):
        if data_type == 'Sim':
            return self.__load_sim_data()
        elif data_type == 'Real':
            return self.__load_real_data()
        else:
            raise ValueError(f"不支持的数据类型: {data_type}")

    def __load_sim_data(self):
        """加载预处理后的数据（.npz）"""
        
        # 直接使用load_data函数加载数据
        data_dict = load_data(
            file_path=self.data_path,
            shuffle=True,
            random_state=57
        )
        
        # 转换为dataset期望的格式
        result = {
            "data": data_dict["cmap"],
            "label_num": data_dict["mus"], 
            "muThr": data_dict["thresholds"]
        }
        return result
    
    

    def __load_real_data(self):
        return {}
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取单个样本数据
        
        Args:
            idx (int): 样本索引（相对于当前数据范围的索引）
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - cmap_data: CMAP幅值数据 (500,)
                - mu_count: 运动单位数量标签 (标量)
                - threshold_data: 运动单位阈值数据 (500,)
        """

        # 验证索引范围
        if not (0 <= idx < self.num_samples):
            raise IndexError(f"索引 {idx} 超出范围 [0, {self.num_samples})")

        # 转换为全局索引
        actual_idx = self.start_idx + idx

        # 直接使用numpy数组数据
        cmap_data = torch.from_numpy(self.cmap_amplitudes[actual_idx, :]).float()
        mu_count = torch.tensor(self.mu_count_labels[actual_idx], dtype=torch.float32)
        threshold_data = torch.from_numpy(self.mu_thresholds[actual_idx, :]).float()

        return cmap_data, mu_count, threshold_data


    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        批处理函数，将单个样本组合成批次数据
        
        Args:
            batch: 包含多个样本的列表，每个样本为(cmap_data, mu_count, threshold_data)
            
        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
                - src: {"cmap": tensor} - CMAP数据，形状(batch_size, 500)
                - tgt: {"mus": tensor, "thresholds": tensor} - MUs数量和阈值数据
        """
        # 解包批次数据
        cmap_data_list, mu_counts_list, threshold_data_list = zip(*batch)
        
        # 构建src: {"cmap": tensor}格式
        src = {"cmap": torch.stack(cmap_data_list, dim=0)}  # (batch_size, 500) - CMAP数据
        
        # 构建tgt: {"mus": tensor, "thresholds": tensor}格式
        tgt = {
            "mus": torch.stack(mu_counts_list, dim=0),           # (batch_size,)
            "thresholds": torch.stack(threshold_data_list, dim=0)  # (batch_size, 500)
        }
        
        return src, tgt
