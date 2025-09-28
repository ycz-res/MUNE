import torch
import numpy as np
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
        mat_data = load_mat_data(self.data_path)
        print(f"MAT文件键值: {list(mat_data.keys())}")
        
        # 使用_preprocess进行完整的数据预处理
        processed_data = self.__preprocess(mat_data['data'])
        
        # 提取标签数据
        Y_load = mat_data['label_num']
        Y_load = np.array(Y_load).squeeze()
        
        # 提取阈值数据
        thr_load = mat_data['label_thr']
        thr_load = np.array(thr_load).squeeze()
        
        # 将单个阈值扩展为160维向量（用0填充）
        y4_expanded = np.zeros((len(thr_load), 160), dtype=np.float32)
        y4_expanded[:, 0] = thr_load  # 第一个位置放实际阈值
        
        # 数据类型转换
        y1 = Y_load.astype(np.float32)  # MU数量
        y4 = y4_expanded  # 160维阈值向量
        
        # 构建结果字典
        result = {
            'data': processed_data,  # shape: (N, 500) - 500个点的纵坐标
            'label_num': y1,  # MU数量 (N,)
            'muThr': y4,  # 阈值数据 (N,)
            'x_len': processed_data.shape[-1],  # 输入长度 (500)
            'x_dim': 1,  # 输入维度 (1)
            'num_samples': len(processed_data)  # 样本数量
        }
        
        print(f"数据预处理完成:")
        print(f"  - 样本数量: {result['num_samples']}")
        print(f"  - 输入形状: {processed_data.shape}")
        print(f"  - MU数量范围: [{y1.min():.1f}, {y1.max():.1f}]")
        
        return result
    
    def __preprocess(self, data):
        N, P, _ = data.shape
        Y_norm = np.zeros((N, P), dtype=np.float32)

        for i in range(N):
            x, y = data[i, :, 0], data[i, :, 1]

            # Step 1: 按x排序
            idx = np.argsort(x)
            y = y[idx]

            # Step 2: y归一化 (样本内 [0,1])
            y = (y - y.min()) / (y.max() - y.min() + 1e-8)

            # 保存
            Y_norm[i] = y

        return Y_norm
    
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
            # muThr 形状是 (N, 160)，获取160维阈值向量
            muThr = self.muThr[actual_idx]  # 160维向量
            sample_muThr = torch.from_numpy(muThr).float()
        else:
            sample_muThr = torch.zeros(160, dtype=torch.float32)
        
        return sample_data, sample_label, sample_muThr

    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        批处理函数，处理MU数量和阈值输出
        
        数据格式:
            src: (batch_size, 500) - 500个点的纵坐标
            tgt: (batch_size, 161) - [MU数量(1维), 阈值(160维)]
        """
        # 分离数据、标签和阈值
        data_list, label_list, muThr_list = zip(*batch)
        batch_size = len(batch)
        
        # src: 数据集合
        src = torch.stack(data_list, dim=0)
        
        # tgt: [MU数量, 160维阈值向量]
        tgt = torch.stack([
            torch.cat([label.unsqueeze(0), muThr]) for label, muThr in zip(label_list, muThr_list)
        ], dim=0)

        print(f"src: {src.shape}, tgt: {tgt.shape}")
        return src, tgt
