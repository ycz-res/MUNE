from py_compile import main
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Any
from utils import load_mat_data
import h5py



class Sim(Dataset):
    """仿真数据集类"""
    
    def __init__(self, data_path: str, data_type: str = 'sim', start_percent: float = 0.0, 
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
        if data_type == 'sim':
            return self.__load_sim_data()
        elif data_type == 'real':
            return self.__load_real_data()
        else:
            raise ValueError(f"不支持的数据类型: {data_type}")

    def __load_sim_data(self):
        """加载仿真数据并进行预处理"""
        mat_data = load_mat_data(self.data_path, lazy=False)
        # print(f"MAT文件键值: {list(mat_data.keys())}")
        
        # Step1: 归一化CMAP幅值数据 (N,500)
        cmap_normalized = self._normalize_cmap_data(mat_data['data'])
        
        # Step2: 加载运动单位数量标签
        mu_counts = np.array(mat_data['label_num']).squeeze().astype(np.float32)
        
        # Step3: 加载原始运动单位阈值
        mu_thresholds_raw = np.array(mat_data['muThr']).squeeze()

        # Step4: 将阈值映射到 x 轴对应的位置
        mu_thresholds_aligned = self._map_mu_thresholds(mat_data['data'], mu_thresholds_raw)  # (N, 500)


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
        将每个样本的 MU 阈值 (muThr) 映射到对应的 x 轴位置 (500维)。

        参数:
            data: (N, 500, 2)
                每个样本的电刺激序列和幅值。
                data[n, :, 0] 表示刺激电流序列 x（单位 mA）
            muThr: (N, 160)
                每个样本的运动单位阈值分布（mA），0 表示无效填充。

        输出:
            thr_matrix: (N, 500)
                每个样本的 500 维阈值映射结果：
                    - 若 threshold_mode == 'binary' → 0/1 掩码
                    - 若 threshold_mode == 'value' → 实际阈值
        """
        N, P, _ = data.shape
        thr_matrix = np.zeros((N, P), dtype=np.float32)

        for n in range(N):
            # 电刺激坐标（单调递增）
            x = data[n, :, 0]  # (500,)
            thr_vector = np.zeros(P, dtype=np.float32)

            # 提取该样本有效阈值：去0 → 排序 → 去重
            mu_vals = muThr[n][muThr[n] > 0]
            if mu_vals.size == 0:
                thr_matrix[n] = thr_vector
                continue

            mu_vals = np.unique(np.sort(mu_vals))  # 保证递增顺序与生理一致

            # 将每个阈值映射到 x 轴最近位置
            for val in mu_vals:
                idx = np.searchsorted(x, val)  # 找到第一个 ≥ val 的位置
                if idx < P:  # 只在有效范围内标记
                    if self.threshold_mode == "binary":
                        thr_vector[idx] = 1.0
                    else:
                        thr_vector[idx] = val  # 保留实际阈值（mA）

            thr_matrix[n] = thr_vector

        return thr_matrix

    def __load_real_data(self):
        return {}
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取单个样本数据（自动适配大文件懒加载）
        
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

        # 检查是否是大文件（h5py.Dataset）
        if isinstance(self.data_dict["data"], h5py.Dataset):
            # ---- 大文件懒加载模式 ----
            # 从 HDF5 数据中读取单个样本
            data_item = np.array(self.data_dict["data"][actual_idx])  # (500, 2)
            mu_count_val = np.array(self.data_dict["label_num"][actual_idx]).astype(np.float32)
            mu_thr_item = np.array(self.data_dict["muThr"][actual_idx]).astype(np.float32)

            # 提取电流 (x) 与 幅值 (y)
            x = data_item[:, 0]
            y = data_item[:, 1]

            # 归一化幅值到 [0,1]
            y = (y - y.min()) / (y.max() - y.min() + 1e-8)
            cmap_data = torch.from_numpy(y).float()

            # MU数量标签
            mu_count = torch.tensor(mu_count_val, dtype=torch.float32)

            # 阈值数据（500维映射）
            threshold_data = torch.from_numpy(mu_thr_item).float()

        else:
            # ---- 小文件内存模式 ----
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

    # =====================================================
    # ✅ 静态方法：阈值重复检查（完整逻辑封装）
    # =====================================================
    @staticmethod
    def check_threshold_duplicates(data_path: str, tol: float = 1e-5, verbose: bool = True) -> list:
        """
        检查仿真数据中的 MU 阈值是否存在重复或过近值。
        Args:
            data_path (str): .mat 数据文件路径（需包含 'label_thr' 键）
            tol (float): 判断重复的容差，默认 1e-5
            verbose (bool): 是否打印详细信息

        Returns:
            list: 含有重复阈值的样本索引列表
        """
        try:
            mat_data = load_mat_data(data_path)
        except Exception as e:
            print(f"❌ 加载数据失败：{e}")
            return []

        if "label_thr" not in mat_data:
            print("❌ 数据文件中未找到 'label_thr' 键，无法检测。")
            return []

        muThr = np.array(mat_data["label_thr"]).squeeze()

        # 确保数据为二维格式进行处理
        if muThr.ndim == 1:
            muThr = muThr.reshape(-1, 1)

        print(f"✅ 加载数据成功：{muThr.shape[0]} 个样本，开始检测重复阈值...\n")

        N = muThr.shape[0]
        dup_samples = []

        for n in range(N):
            vals = muThr[n][muThr[n] > 0]
            if len(vals) <= 1:
                continue
            vals_sorted = np.sort(vals)
            diffs = np.diff(vals_sorted)
            if np.any(diffs < tol):
                dup_samples.append(n)
                if verbose:
                    dup_vals = vals_sorted[np.where(diffs < tol)[0]]
                    print(f"⚠️ 样本 {n} 存在重复或过近阈值: {dup_vals}")

        if verbose:
            if len(dup_samples) == 0:
                print("✅ 所有样本阈值均唯一，无重复。")
            else:
                print(f"\n⚠️ 共 {len(dup_samples)} 个样本存在重复阈值: {dup_samples}\n")

        print("—— 检查完成 ——")
        if len(dup_samples) == 0:
            print("✅ 数据通过完整性检查，可安全进入训练阶段。\n")
        else:
            print("⚠️ 请检查上方输出，建议手动或脚本修复重复阈值。\n")

        return dup_samples


if __name__ == "__main__":
    """
    - 检查仿真数据中的运动单位(MU)阈值是否存在重复或过近的值
    """
    Sim.check_threshold_duplicates(
        data_path="./data/SimDataset/data.mat",  # 数据路径
        tol=1e-5,                        # 容差
        verbose=True                     # 打印详情
    )