import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Any
from utils import load_mat_data


class SimDataset(Dataset):
    """仿真数据集类"""
    
    def __init__(self, data_path: str, data_type: str = 'sim', start_percent: float = 0.0, 
                 end_percent: float = 1.0, stage: str = 'train'):
        self.data_path = data_path
        self.data_type = data_type
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.stage = stage
        
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
        if data_type == 'sim':
            return self.__load_sim_data()
        elif data_type == 'real':
            return self.__load_real_data()
        else:
            raise ValueError(f"不支持的数据类型: {data_type}")

    def __load_sim_data(self):
        """加载仿真数据并进行预处理"""
        mat_data = load_mat_data(self.data_path)
        print(f"MAT文件键值: {list(mat_data.keys())}")
        
        # Step1: 归一化CMAP幅值数据 (N,500)
        cmap_normalized = self._normalize_cmap_data(mat_data['data'])
        
        # Step2: 加载运动单位数量标签
        mu_counts = np.array(mat_data['label_num']).squeeze().astype(np.float32)
        
        # Step3: 加载原始运动单位阈值
        mu_thresholds_raw = np.array(mat_data['label_thr']).squeeze()
        
        # Step4: 将阈值映射到x轴对应的位置
        mu_thresholds_aligned = self._map_mu_thresholds(mat_data['data'], mu_thresholds_raw)  # (N,500)

        result = {
            'data': cmap_normalized,        # (N,500) 归一化CMAP幅值
            'label_num': mu_counts,         # (N,) 运动单位数量
            'muThr': mu_thresholds_aligned  # (N,500) 对齐到x轴的阈值位置
        }
        
        print(f"数据预处理完成:")
        print(f"  - 样本数量: {len(cmap_normalized)}")
        print(f"  - CMAP数据形状: {cmap_normalized.shape}")
        print(f"  - MU数量范围: [{mu_counts.min():.1f}, {mu_counts.max():.1f}]")
        
        return result
    
    def _normalize_cmap_data(self, data):
        """
        对CMAP数据进行归一化处理
        
        Args:
            data: 原始CMAP数据，形状为(N, 500, 2)，其中最后一维为[x坐标, y幅值]
            
        Returns:
            Y_norm: 归一化后的y值数据，形状为(N, 500)
            
        处理步骤:
            1. 按x坐标排序 
            2. 对y幅值进行[0,1]归一化
        """
        N, P, _ = data.shape
        Y_norm = np.zeros((N, P), dtype=np.float32)

        for i in range(N):
            x, y = data[i, :, 0], data[i, :, 1]
            # 按x坐标排序
            idx = np.argsort(x)
            y = y[idx]
            # y归一化到[0,1]
            y = (y - y.min()) / (y.max() - y.min() + 1e-8)
            Y_norm[i] = y

        return Y_norm

    def _map_mu_thresholds(self, data, muThr):
        """
        把 muThr 阈值映射到对应的 x 位置。
        规则：x >= 当前最小阈值时标记，并丢弃该阈值
        """
        N, P, _ = data.shape
        thr_matrix = np.zeros((N, P), dtype=np.float32)

        for n in range(N):
            x = data[n, :, 0]  # (500,)
            thr_vector = np.zeros(P, dtype=np.float32)

            mu_vals = muThr[n][muThr[n] > 0]
            mu_vals = list(np.sort(mu_vals))

            for i in range(P):
                if len(mu_vals) == 0:
                    break
                if x[i] >= mu_vals[0]:
                    thr_vector[i] = mu_vals[0]
                    mu_vals.pop(0)

            thr_matrix[n] = thr_vector
        return thr_matrix
    
    def __load_real_data(self):
        return {}
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取单个样本数据
        
        Args:
            idx: 样本索引（相对于当前数据范围的索引）
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - cmap_data: CMAP幅值数据，形状(500,)
                - mu_count: 运动单位数量标签，标量
                - threshold_data: 运动单位阈值数据，形状(500,)
        """
        # 验证索引范围
        if not (0 <= idx < self.num_samples):
            raise IndexError(f"索引 {idx} 超出范围 [0, {self.num_samples})")
        
        # 转换为实际的数据索引
        actual_idx = self.start_idx + idx
        
        # 获取CMAP幅值数据
        cmap_data = torch.from_numpy(self.cmap_amplitudes[actual_idx, :]).float()
        
        # 获取运动单位数量标签
        mu_count = torch.tensor(self.mu_count_labels[actual_idx], dtype=torch.float32)
        
        # 获取运动单位阈值数据
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
