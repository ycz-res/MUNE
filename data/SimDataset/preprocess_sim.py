import os
import sys
import argparse
import numpy as np
from typing import Dict

# 自动添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils import load_mat_data


def normalize_cmap_data(data: np.ndarray) -> np.ndarray:
    """
    对 CMAP 原始数据进行归一化。

    输入 data 形状为 (N, 500, 2)，最后一维分别是 [x, y]。
    返回形状为 (N, 500) 的 y 归一化结果。
    """
    N, P, _ = data.shape
    y_normalized = np.zeros((N, P), dtype=np.float32)

    for sample_index in range(N):
        x = data[sample_index, :, 0]
        y = data[sample_index, :, 1]

        sort_index = np.argsort(x)
        y_sorted = y[sort_index]
        y_min = y_sorted.min()
        y_max = y_sorted.max()
        y_range = y_max - y_min + 1e-8
        y_normalized[sample_index] = (y_sorted - y_min) / y_range

    return y_normalized


def map_mu_thresholds(data: np.ndarray, mu_thresholds: np.ndarray, mode: str) -> np.ndarray:
    """
    将每个样本的 MU 阈值 (mu_thresholds) 映射到对应的 x 轴位置 (500 维)。
    - data: (N, 500, 2) 其中 data[n, :, 0] 为单调递增电流 x（mA）
    - mu_thresholds: (N, 160) 0 表示无效填充
    - mode: "binary" | "value"
    返回: (N, 500)
    """
    N, P, _ = data.shape
    threshold_matrix = np.zeros((N, P), dtype=np.float32)

    for n in range(N):
        x = data[n, :, 0]
        row = np.zeros(P, dtype=np.float32)

        mu_vals = mu_thresholds[n][mu_thresholds[n] > 0]
        if mu_vals.size == 0:
            threshold_matrix[n] = row
            continue

        mu_vals = np.unique(np.sort(mu_vals))

        for val in mu_vals:
            idx = np.searchsorted(x, val)
            if idx < P:
                if mode == "binary":
                    row[idx] = 1.0
                else:
                    row[idx] = float(val)

        threshold_matrix[n] = row

    return threshold_matrix


def fix_transpose_and_extract(mat: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    修复从 .mat 读取后的维度问题，并返回标准化键：
    - data: (N, 500, 2)
    - label_num: (N,)
    - muThr: (N, 160) 或相似，数值为阈值（mA），0 为填充
    """
    data = np.array(mat["data"])  # 可能是 (2,500,N) 或 (N,500,2)
    label_num = np.array(mat["label_num"]).squeeze()
    mu_thr = np.array(mat["muThr"])  # 可能是 (M,N) 或 (N,M)

    if data.shape[0] < data.shape[-1]:
        data = np.transpose(data, (2, 1, 0))  # (2,500,N) -> (N,500,2)

    if mu_thr.shape[0] < mu_thr.shape[-1]:
        mu_thr = mu_thr.T

    if label_num.ndim == 2 and label_num.shape[0] < label_num.shape[1]:
        label_num = label_num.T.squeeze()

    return {"data": data, "label_num": label_num.astype(np.float32), "muThr": mu_thr.astype(np.float32)}


def preprocess(mat_path: str, output_path: str, threshold_mode: str = "binary") -> str:
    """
    预处理入口：读取 .mat -> 修复维度 -> 归一化/映射 -> 保存 .npz
    返回保存文件的路径。
    """
    if threshold_mode not in ("binary", "value"):
        raise ValueError("threshold_mode 必须为 'binary' 或 'value'")

    mat = load_mat_data(mat_path, lazy=False)

    fixed = fix_transpose_and_extract(mat)
    data = fixed["data"]
    label_num = fixed["label_num"]
    mu_thr = fixed["muThr"]

    y_norm = normalize_cmap_data(data)
    thr_aligned = map_mu_thresholds(data, mu_thr, threshold_mode)

    # 计算默认输出路径并保存
    output_path = build_default_output_path(mat_path, threshold_mode)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        cmap=y_norm.astype(np.float32),
        mus=label_num.astype(np.float32),
        thresholds=thr_aligned.astype(np.float32),
    )

    return output_path


def build_default_output_path(mat_path: str, threshold_mode: str) -> str:
    # 默认固定输出为当前工作目录下的 data.npz
    return os.path.join(".", "data.npz")


def main():
    parser = argparse.ArgumentParser(description="预处理仿真数据，生成可快速加载的 .npz 文件")
    parser.add_argument("--mat", default="./data.mat", help="源 .mat 路径（默认: ./data.mat）")
    parser.add_argument("--mode", default="binary", choices=["binary", "value"], help="阈值映射模式")
    parser.add_argument("--out", default=None, help="输出 .npz 路径（可选）")
    args = parser.parse_args()

    output_path = args.out or build_default_output_path(args.mat, args.mode)
    saved = preprocess(args.mat, output_path, args.mode)
    print(f"✅ 预处理完成，已保存: {saved}")


if __name__ == "__main__":
    main()


