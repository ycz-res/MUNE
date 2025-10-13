from py_compile import main
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Any
from utils import load_mat_data
import h5py
import os



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
        """严格加载预处理后的数据（.npz），若不存在则报错提示先运行预处理。"""

        # 支持两种默认命名：data.npz（推荐）与历史 *_preprocessed_<mode>.npz
        base, _ = os.path.splitext(self.data_path)
        candidates = [
            os.path.join(os.path.dirname(self.data_path), "data.npz"),
            f"{base}_preprocessed_{self.threshold_mode}.npz",
        ]

        selected = None
        for p in candidates:
            if os.path.isfile(p):
                selected = p
                break

        if selected is not None:
            print(f"📦 检测到预处理文件: {selected}，直接加载以加速训练...")
            npz = np.load(selected, allow_pickle=True)
            cmap = np.array(npz["cmap"]).astype(np.float32)
            mus = np.array(npz["mus"]).astype(np.float32)
            thresholds = np.array(npz["thresholds"]).astype(np.float32)
            result = {"data": cmap, "label_num": mus, "muThr": thresholds}
            print(f"✅ 预处理数据加载完成: data={cmap.shape}, thresholds={thresholds.shape}")
            return result

        raise FileNotFoundError(
            f"未找到预处理文件: {preprocessed_npz}。请先运行预处理脚本生成 .npz 文件。"
        )
    
    

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