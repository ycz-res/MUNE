import torch
from torch.utils.data import Dataset
from typing import List, Tuple
from utils import load_mat_data


class SimDataset(Dataset):
    """仿真数据集类"""
    
    def __init__(self, data_path: str, data_type: str, start_percent: float = 0.0, end_percent: float = 1.0, stage: str = 'train'):
        """
            data_path: 数据文件路径
            data_type: 数据类型 ('sim' 或 'real')
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
        
        self.data_dict = self.__load_data(self.data_type)
        self.data = self.data_dict['data']  # 主要数据 (N, 500, 2) - N个样本，每个样本500个(x,y)坐标
        self.labels = self.data_dict['label_num']  # 标签数据 (N, 1)
        self.muThr = self.data_dict['muThr']  # 阈值数据 (N, 160) - 每个样本160个阈值
        
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
        """加载仿真数据"""
        data_dict = load_mat_data(self.data_path)
        return data_dict
    
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
        
        # 获取第actual_idx个样本的数据 (500, 2) - 500个(x,y)坐标点
        sample_data = self.data[actual_idx, :, :]  # shape: (500, 2)
        
        # 转换为tensor，保持 (500, 2) 形状，每行是一个(x,y)坐标
        sample_data = torch.from_numpy(sample_data).float()  # shape: (500, 2)
        
        # 获取标签
        if self.labels is not None:
            # labels 形状是 (N, 1)，所以使用 [actual_idx, 0] 获取标量值
            label = self.labels[actual_idx, 0]  # 标量值
            sample_label = torch.tensor(label, dtype=torch.float32)
        else:
            # 如果没有标签，返回-1作为占位符
            sample_label = torch.tensor(-1, dtype=torch.float32)
        
        if self.muThr is not None:
            # muThr 形状是 (N, 160)，所以获取第actual_idx个样本的所有160个阈值
            muThr = self.muThr[actual_idx, :]  # shape: (160,)
            sample_muThr = torch.from_numpy(muThr).float()  # shape: (160,)
        else:
            sample_muThr = torch.tensor([-1], dtype=torch.float32)
        
        return sample_data, sample_label, sample_muThr

    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            src: (batch_size, 500, 2) - 坐标数据
            tgt: (batch_size, max_length) - [MU数量, 对应数量的阈值]
        """
        # 分离数据、标签和阈值
        data_list, label_list, muThr_list = zip(*batch)
        batch_size = len(batch)
        
        # src: 数据集合 (batch_size, 500, 2)
        src = torch.stack(data_list, dim=0)
        
        # 预分配tgt_list以提高效率
        tgt_list = []
        max_length = 0
        
        # 第一遍：构建tgt并计算最大长度
        for label, muThr in zip(label_list, muThr_list):
            # 获取MU数量
            mu_count = int(label.item())
            
            # 验证并截取阈值
            if 0 < mu_count <= muThr.shape[0]:
                selected_muThr = muThr[:mu_count]
            elif mu_count > muThr.shape[0]:
                # 如果MU数量超过阈值数量，使用所有阈值
                selected_muThr = muThr
                mu_count = muThr.shape[0]
            else:
                # 无效的MU数量，使用空序列
                selected_muThr = torch.empty(0, dtype=torch.float32)
                mu_count = 0
            
            # 组合MU数量和阈值: [mu_count, threshold1, threshold2, ...]
            if mu_count > 0:
                tgt = torch.cat([label.unsqueeze(0), selected_muThr])
            else:
                tgt = torch.tensor([0.0], dtype=torch.float32)  # 至少包含MU数量
            
            tgt_list.append(tgt)
            max_length = max(max_length, tgt.shape[0])
        
        # 第二遍：填充到相同长度
        if max_length > 0:
            tgt = torch.zeros(batch_size, max_length, dtype=torch.float32)
            for i, tgt_item in enumerate(tgt_list):
                tgt[i, :tgt_item.shape[0]] = tgt_item
        else:
            tgt = torch.zeros(batch_size, 1, dtype=torch.float32)

        print(f"src: {src.shape}, tgt: {tgt.shape}")

        
        return src, tgt


    
