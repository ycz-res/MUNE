import torch
import numpy as np
import h5py
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Any
from utils import load_mat_data


class SimDataset(Dataset):
    """仿真数据集类"""
    
    def __init__(self, data_path: str, data_type: str = 'sim', start_percent: float = 0.0, 
                 end_percent: float = 1.0, stage: str = 'train'):
        """
        Args:
            data_path: 数据文件路径
            data_type: 数据类型 ('sim', 'real')
            start_percent: 起始百分比 (0.0-1.0)，默认0.0
            end_percent: 结束百分比 (0.0-1.0)，默认1.0
            stage: 阶段标签 ('train', 'val', 'test')，默认'train'
        """
        self.data_path = data_path
        self.data_type = data_type
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.stage = stage
        
        # 验证阶段标签
        if stage not in ['train', 'val', 'test']:
            raise ValueError("阶段标签必须是 'train', 'val', 或 'test'")
        
        # 验证百分比范围
        if not (0.0 <= start_percent <= 1.0 and 0.0 <= end_percent <= 1.0):
            raise ValueError("百分比必须在0.0到1.0之间")
        if start_percent >= end_percent:
            raise ValueError("起始百分比必须小于结束百分比")
        
        # 根据数据类型选择加载方法
        self.data_dict = self.__load_data(self.data_type)
        
        # 提取数据
        self.data = self.data_dict['data']  # 主要数据 (N, 500) - N个样本，每个样本500个点的纵坐标
        self.labels = self.data_dict['label_num']  # MU数量标签 (N,)
        self.muThr = self.data_dict.get('muThr', None)  # 阈值数据 (N, 160)
        
        # 计算数据范围
        self.total_samples = self.data.shape[0]
        self.start_idx = int(self.total_samples * start_percent)
        self.end_idx = int(self.total_samples * end_percent)
        self.num_samples = self.end_idx - self.start_idx
        
        print(f"数据集信息 - 当前阶段: {self.stage}: 总样本数={self.total_samples}, 使用范围=[{self.start_idx}:{self.end_idx}], 实际样本数={self.num_samples}")
    
    
    def __load_data(self, data_type: str):
        """加载MAT文件数据"""
        if data_type == 'sim':
            return self.__load_sim_data()
        elif data_type == 'real':
            return self.__load_real_data()
        else:
            raise ValueError(f"不支持的数据类型: {data_type}")

    def __load_sim_data(self):
        """加载仿真数据并进行预处理"""
        # 加载.mat数据文件
        mat_file = h5py.File(self.data_path, 'r')
        print(f"MAT文件键值: {list(mat_file.keys())}")
        
        # 提取主要数据
        X_load, Y_load = mat_file['data'], mat_file['label_num']
        X_load, Y_load = np.array(X_load).transpose(2, 1, 0)[:, :, 1].squeeze(), np.array(Y_load).squeeze()
        
        # 截断：从5个unit开始预测
        index = Y_load.squeeze() >= 5
        X_load, Y_load = X_load[index,], Y_load[index]
        
        print(f"截断后数据形状: X={X_load.shape}, Y={Y_load.shape}")
        
        # 预处理 (排序->归一化)
        X = self._preprocess(X_load, Y_load, np.array([0, 1, 2]), True)
        
        # 提取阈值数据
        thr_load = mat_file['label_thr']
        thr_load = np.array(thr_load).squeeze()[index]
        
        # 获取数据维度信息
        x_len_ = X.shape[-1]  # 500个点
        x_dim_ = 1  # 只有纵坐标，所以是1维
        
        # 数据类型转换
        y1 = Y_load.astype(np.float32)  # MU数量
        y4 = thr_load.astype(np.float32)  # 阈值
        
        # 关闭文件
        mat_file.close()
        
        # 返回处理后的数据
        result = {
            'data': X,  # shape: (N, 500) - 500个点的纵坐标
            'label_num': y1,  # MU数量 (N,)
            'muThr': y4,  # 阈值数据 (N,)
            'x_len': x_len_,  # 输入长度 (500)
            'x_dim': x_dim_,  # 输入维度 (1)
            'num_samples': len(X)  # 样本数量
        }
        
        print(f"数据预处理完成:")
        print(f"  - 样本数量: {result['num_samples']}")
        print(f"  - 输入形状: {X.shape}")
        print(f"  - MU数量范围: [{y1.min():.1f}, {y1.max():.1f}]")
        
        return result
    
    def _preprocess(self, image_data: np.ndarray, labels: np.ndarray, orders: np.ndarray, tmp_flag: bool) -> np.ndarray:
        """
        数据预处理函数（排序+归一化）
        
        Args:
            image_data: 图像数据 (N, 500) - N个样本，每个样本500个点的纵坐标
            labels: 标签数据 (N,)
            orders: 预处理阶数
            tmp_flag: 临时标志
        
        Returns:
            处理后的数据 (N, 500)
        """
        print(f"预处理前数据形状: {image_data.shape}")
        
        # 1. 对每个样本的纵坐标进行排序
        sorted_data = np.zeros_like(image_data)
        
        for i in range(image_data.shape[0]):  # 遍历每个样本
            sample_y = image_data[i, :]  # shape: (500,)
            # 对纵坐标进行排序
            sorted_indices = np.argsort(sample_y)
            sorted_data[i, :] = sample_y[sorted_indices]
        
        print(f"排序后数据形状: {sorted_data.shape}")
        
        # 2. 对每个样本的纵坐标进行归一化
        normalized_data = np.zeros_like(sorted_data)
        
        for i in range(sorted_data.shape[0]):  # 遍历每个样本
            sample_y = sorted_data[i, :]  # shape: (500,)
            
            # 对纵坐标进行Z-score归一化
            if np.std(sample_y) > 0:  # 避免除零
                normalized_y = (sample_y - np.mean(sample_y)) / np.std(sample_y)
            else:
                normalized_y = sample_y - np.mean(sample_y)
            
            normalized_data[i, :] = normalized_y
        
        print(f"归一化后数据形状: {normalized_data.shape}")
        print(f"归一化后纵坐标范围: [{normalized_data.min():.3f}, {normalized_data.max():.3f}]")
        
        return normalized_data
    
    def __load_real_data(self):
        """加载真实数据"""
        # TODO: 加载真实数据
        data_dict = {}
        return data_dict
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取单个样本"""
        # 验证索引范围
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"索引 {idx} 超出范围 [0, {self.num_samples})")
        
        # 转换为实际的数据索引
        actual_idx = self.start_idx + idx
        
        # 获取第actual_idx个样本的数据
        # 数据现在是 shape (N, 500) - 500个点的纵坐标
        sample_data = self.data[actual_idx, :]  # shape: (500,)
        sample_data = torch.from_numpy(sample_data).float()  # shape: (500,)
        
        # 获取标签
        if self.labels is not None:
            # labels 形状是 (N,)，直接获取标量值
            label = self.labels[actual_idx]  # 标量值
            sample_label = torch.tensor(label, dtype=torch.float32)
        else:
            # 如果没有标签，返回-1作为占位符
            sample_label = torch.tensor(-1, dtype=torch.float32)
        
        # 获取阈值数据
        if self.muThr is not None:
            # muThr 形状是 (N,)，获取单个阈值
            muThr = self.muThr[actual_idx]  # 标量值
            sample_muThr = torch.tensor(muThr, dtype=torch.float32)
        else:
            sample_muThr = torch.tensor([-1], dtype=torch.float32)
        
        return sample_data, sample_label, sample_muThr

    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        批处理函数，处理MU数量和阈值输出
        
        数据格式:
            src: (batch_size, 500) - 500个点的纵坐标
            tgt: (batch_size, 2) - [MU数量, 阈值]
        """
        # 分离数据、标签和阈值
        data_list, label_list, muThr_list = zip(*batch)
        batch_size = len(batch)
        
        # src: 数据集合
        src = torch.stack(data_list, dim=0)
        
        # tgt: 简单的双输出 [MU数量, 阈值]
        tgt = torch.stack([
            torch.stack([label, muThr]) for label, muThr in zip(label_list, muThr_list)
        ], dim=0)

        print(f"src: {src.shape}, tgt: {tgt.shape}")
        return src, tgt
