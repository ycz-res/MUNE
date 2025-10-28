import os
import argparse
import numpy as np
from typing import Dict
from utils import load_mat_data


def normalize_cmap_data(data: np.ndarray) -> np.ndarray:
    """
    对 CMAP 原始数据进行归一化。

    输入 data 形状为 (N, 500, 2)，最后一维分别是 [x, y]。
    注意：输入的data已经是按x值排好序的。
    返回形状为 (N, 500) 的 y 归一化结果。
    """
    N, P, _ = data.shape
    y_normalized = np.zeros((N, P), dtype=np.float32)

    for sample_index in range(N):
        y = data[sample_index, :, 1]  # 直接取y值，因为x已经排好序

        y_min = y.min()
        y_max = y.max()
        y_range = y_max - y_min + 1e-8
        y_normalized[sample_index] = (y - y_min) / y_range

    return y_normalized


def map_mu_thresholds(data: np.ndarray, mu_thresholds: np.ndarray, mode: str) -> np.ndarray:
    """
    将每个样本的 MU 阈值 (mu_thresholds) 映射到对应的 x 轴位置 (500 维)。
    - data: (N, 500, 2) 其中 data[n, :, 0] 为单调递增电流 x（mA）
    - mu_thresholds: (N, 160) 0 表示无效填充
    - mode: "binary" | "value"
    返回: (N, 500)
    
    逻辑：
    1. 对mu_thresholds进行从小到大排序
    2. 对于每个阈值，在500维向量中找到对应的位置
    3. binary模式：在对应位置计数+1
    4. value模式：暂未实现
    """
    N, P, _ = data.shape
    threshold_matrix = np.zeros((N, P), dtype=np.float32)

    for n in range(N):
        x = data[n, :, 0]  # 电流值序列，如 [0, 1, 2, 3, 4, 5, 6, ...]
        row = np.zeros(P, dtype=np.float32)

        # 提取有效的阈值（非零值）
        mu_vals = mu_thresholds[n][mu_thresholds[n] > 0]
        if mu_vals.size == 0:
            threshold_matrix[n] = row
            continue

        # 对阈值进行从小到大排序
        mu_vals = np.sort(mu_vals)

        if mode == "binary":
            # binary模式：在对应位置计数为1
            for val in mu_vals:
                # 使用就近原则找到最接近的位置
                # 计算与每个位置的差值，找到最小差值的位置
                distances = np.abs(x - val)
                idx = np.argmin(distances)
                if idx < P:
                    row[idx] = 1.0
        elif mode == "value":
            # value模式：暂未实现
            pass
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        threshold_matrix[n] = row

    return threshold_matrix


def fix_transpose_and_extract(mat: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    修复从 .mat 读取后的维度问题，并返回标准化键：
    - data: (N, 500, 2)
    - label_num: (N,)
    - muThr: (N, 160) 或相似，数值为阈值（mA），0 为填充
    """
    print("🔍 处理前的维度信息:")
    print(f"   - mat['data'] shape: {mat['data'].shape}")
    print(f"   - mat['label_num'] shape: {mat['label_num'].shape}")
    print(f"   - mat['muThr'] shape: {mat['muThr'].shape}")
    
    data = np.array(mat["data"])  # 可能是 (2,500,N) 或 (N,500,2)
    label_num = np.array(mat["label_num"]).squeeze()
    mu_thr = np.array(mat["muThr"])  # 可能是 (M,N) 或 (N,M)

    if data.shape[0] < data.shape[-1]:
        data = np.transpose(data, (2, 1, 0))  # (2,500,N) -> (N,500,2)

    if mu_thr.shape[0] < mu_thr.shape[-1]:
        mu_thr = mu_thr.T

    if label_num.ndim == 2 and label_num.shape[0] < label_num.shape[1]:
        label_num = label_num.T.squeeze()

    print("✅ 处理后的最终维度:")
    print(f"   - data shape: {data.shape}")
    print(f"   - label_num shape: {label_num.shape}")
    print(f"   - muThr shape: {mu_thr.shape}")
    print()

    return {"data": data, "label_num": label_num.astype(np.float32), "muThr": mu_thr.astype(np.float32)}


def preprocess(mat_path: str, output_path: str, threshold_mode: str = "binary", 
               start_ratio: float = 0.0, end_ratio: float = 1.0) -> str:
    """
    预处理入口：读取 .mat -> 修复维度 -> 归一化/映射 -> 保存 .npz
    
    Args:
        mat_path: .mat 文件路径
        output_path: 输出 .npz 文件路径
        threshold_mode: 阈值映射模式 ('binary' 或 'value')
        start_ratio: 起始位置比例，范围 [0, 1)，默认 0.0 表示从头开始
        end_ratio: 结束位置比例，范围 (0, 1]，默认 1.0 表示到末尾
    
    Returns:
        保存文件的路径
    """
    if threshold_mode not in ("binary", "value"):
        raise ValueError("threshold_mode 必须为 'binary' 或 'value'")
    
    if not (0.0 <= start_ratio < 1.0):
        raise ValueError("start_ratio 必须在 [0, 1) 范围内")
    if not (0.0 < end_ratio <= 1.0):
        raise ValueError("end_ratio 必须在 (0, 1] 范围内")
    if start_ratio >= end_ratio:
        raise ValueError("start_ratio 必须小于 end_ratio")

    mat = load_mat_data(mat_path, lazy=False, start_ratio=start_ratio, end_ratio=end_ratio)

    fixed = fix_transpose_and_extract(mat)
    data = fixed["data"]
    label_num = fixed["label_num"]
    mu_thr = fixed["muThr"]

    y_norm = normalize_cmap_data(data)
    thr_aligned = map_mu_thresholds(data, mu_thr, threshold_mode)

    # 计算mus：从thresholds中统计1的数量
    mus_count = np.sum(thr_aligned, axis=1)  # (N,) 每个样本的阈值位置数量
    
    # 保存数据
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir:  # 如果路径包含目录
        os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(
        output_path,
        cmap=y_norm.astype(np.float32),
        label_num=label_num.astype(np.float32),  # 原始MU数量标签
        mus=mus_count.astype(np.float32),  # 从thresholds统计的实际阈值位置数量
        thresholds=thr_aligned.astype(np.float32),
    )

    return output_path


def main():
    parser = argparse.ArgumentParser(description="预处理仿真数据，生成可快速加载的 .npz 文件")
    parser.add_argument("--mat", default="./data.mat", help="源 .mat 路径（默认: ./data.mat）")
    parser.add_argument("--mode", default="binary", choices=["binary", "value"], help="阈值映射模式")
    parser.add_argument("--out", default="./data_all.npz", help="输出文件路径 (默认: ./data.npz)")
    parser.add_argument("--start", type=float, default=0.0, help="起始位置比例，范围 [0, 1)，默认 0.0 表示从头开始")
    parser.add_argument("--end", type=float, default=1.0, help="结束位置比例，范围 (0, 1]，默认 1.0 表示到末尾")
    args = parser.parse_args()

    saved = preprocess(args.mat, args.out, args.mode, args.start, args.end)
    print(f"✅ 预处理完成，已保存: {saved}")


if __name__ == "__main__":
    main()


