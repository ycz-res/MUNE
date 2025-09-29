#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MUNE CMAP扫描仿真器 - Python版本
转换自MATLAB原始代码

功能：模拟CMAP扫描数据生成
参考：Bostock H. Estimating motor unit numbers from a CMAP scan. Muscle and Nerve. 2016;53:889–896.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from scipy.stats import norm
import os
from typing import Dict, List, Tuple, Optional, Union

def truncnormrnd(n: int, mu: float, sigma: float, bounds: List[float]) -> np.ndarray:
    """
    生成截断正态分布随机数
    
    Args:
        n: 生成数量
        mu: 均值
        sigma: 标准差
        bounds: 边界 [min, max]
    
    Returns:
        截断正态分布随机数数组
    """
    result = np.zeros(n)
    lower, upper = bounds[0], bounds[1]
    
    for i in range(n):
        while True:
            x = np.random.normal(mu, sigma)
            if lower <= x <= upper:
                break
        result[i] = x
    
    return result

def cmap_scan_simulation_rein_cjj(
    mu_count: int,
    reinnervation_count: int = 0, 
    stim_count: int = 500,
    direction: Union[int, str] = 1,
    visualization: int = 0,
    custom_model: Optional[Dict] = None
) -> Tuple[np.ndarray, Dict]:
    """
    CMAP扫描仿真函数
    
    Args:
        mu_count: 需要模拟多少个运动单位
        reinnervation_count: 需要模拟多少个再支配的运动单位，0=不模拟
        stim_count: 需要模拟多少次电刺激
        direction: 1或'up'=刺激电流从小到大，-1或'down'=从大到小
        visualization: 是否可视化，0=不绘图，1=在当前图窗绘图，2=绘图+MU分布，-1=新建图窗绘图
        custom_model: 自定义模型参数字典
    
    Returns:
        (oAmplitude, oModel): CMAP扫描数据和模型参数
    """
    
    # 参数设置
    oModel = {}
    
    # 处理direction参数
    if isinstance(direction, str):
        if direction.lower() == 'up':
            direction = 1
        elif direction.lower() == 'down':
            direction = -1
        else:
            direction = 1
            print("direction只接受'up'、'down'、1、-1四种输入")
    elif direction not in [1, -1]:
        direction = 1
        print("direction只接受'up'、'down'、1、-1四种输入")
    
    # 设置默认值
    oModel['AmplitudeParameter'] = [0.025, 0.1]
    oModel['ThresholdParameter'] = [12, 2]
    oModel['ThresholdVariationParameter'] = [0.015, 0.005]
    oModel['NoiseParameter'] = [0.01, 0.01]
    oModel['DynamicNoise'] = 0
    oModel['ScanRange'] = None
    oModel['Mu'] = None
    
    # 用自定义参数替换默认值
    if custom_model is not None:
        for key in ['AmplitudeParameter', 'ThresholdParameter', 'ThresholdVariationParameter', 
                   'NoiseParameter', 'DynamicNoise']:
            if key in custom_model:
                oModel[key] = custom_model[key]
    
    # 仿真
    # 根据分布参数生成运动单位数据
    muAmplitudes = oModel['AmplitudeParameter'][0] + np.random.exponential(
        oModel['AmplitudeParameter'][1], mu_count
    )
    
    muThresholds = np.abs(oModel['ThresholdParameter'][0] + 
                         oModel['ThresholdParameter'][1] * np.random.randn(mu_count))
    
    muThresholdVariations = muThresholds * np.abs(
        oModel['ThresholdVariationParameter'][0] + 
        oModel['ThresholdVariationParameter'][1] * np.random.rand(mu_count)
    )
    
    if reinnervation_count > 0:
        oModel['ReinnervatedMu'] = np.random.exponential(
            oModel['AmplitudeParameter'][1] * 0.7, reinnervation_count
        )
        
        # 高斯分布分配再支配MU
        a = truncnormrnd(reinnervation_count, 0.5, 0.03, [0, 1])
        b = np.floor(a * mu_count).astype(int)
        b[b <= 0] = 1
        b[b > mu_count] = mu_count
        oModel['ReinnervatedBy'] = b
        
        for reinnervated in range(reinnervation_count):
            muAmplitudes[oModel['ReinnervatedBy'][reinnervated]] += oModel['ReinnervatedMu'][reinnervated]
    
    oModel['Mu'] = np.column_stack([muAmplitudes, muThresholds, muThresholdVariations])
    
    # 生成电刺激数据
    if oModel['ScanRange'] is None:
        p0 = truncnormrnd(1, 0.75, 0.1, [0.6, 0.85])[0]
        tmp_p = truncnormrnd(1, 0.65, 0.2, [0.1, 0.9])[0]
        p1 = (1 - p0) * tmp_p
        p2 = (1 - p0) * (1 - tmp_p)
        
        assert 0.99 < p0 + p1 + p2 < 1.01
        
        floor_body = np.min(muThresholds - 1 * muThresholdVariations)
        ceil_body = np.max(muThresholds + 1 * muThresholdVariations)
        range_body = ceil_body - floor_body
        
        range_pre = range_body / p0 * p1
        range_post = range_body / p0 * p2
        
        floor_x = floor_body - range_pre
        ceil_x = ceil_body + range_post
        
        oModel['ScanRange'] = [floor_x, ceil_x]
    
    oModel['stim'] = np.linspace(
        oModel['ScanRange'][0], 
        oModel['ScanRange'][1], 
        stim_count
    )
    
    # 模拟电刺激
    mu_activate_probabilities = norm.cdf(
        oModel['stim'][np.newaxis, :] - muThresholds[:, np.newaxis], 
        0, 
        muThresholdVariations[:, np.newaxis]
    )
    
    oModel['ActivateLog'] = np.random.rand(*mu_activate_probabilities.shape) < mu_activate_probabilities
    
    # 加入噪声
    pureCmap = muAmplitudes @ oModel['ActivateLog']
    
    dynamic_noise_factor = np.maximum(1, oModel['DynamicNoise'] * np.sqrt(np.sum(oModel['ActivateLog'], axis=0)))
    noise_std = oModel['NoiseParameter'][1] * dynamic_noise_factor
    
    cmap = np.abs(np.random.normal(
        pureCmap + oModel['NoiseParameter'][0], 
        noise_std
    ))
    
    # 输出结果
    if direction == -1:
        oAmplitude = np.column_stack([oModel['stim'][::-1], cmap[::-1]])
    else:
        oAmplitude = np.column_stack([oModel['stim'], cmap])
    
    # 可视化模拟结果
    if visualization != 0:
        if visualization < 0:
            plt.figure()
        else:
            plt.clf()
        
        if abs(visualization) == 2:
            plt.subplot(2, 1, 1)
        
        plt.plot(oModel['stim'], pureCmap, '.-', color=[0.8, 0.8, 0.8], label='Pure CMAP')
        plt.plot(oModel['stim'], cmap, 'k.', label='CMAP')
        
        y_min, y_max = plt.ylim()
        for threshold in muThresholds:
            plt.axvline(threshold, color='k', alpha=0.5)
        
        plt.legend(loc='lower right')
        plt.xlabel('Stimulation (mA)')
        plt.ylabel('CMAP Amplitude (mV)')
        
        if abs(visualization) == 2:
            plt.subplot(2, 1, 2)
            for i in range(mu_count):
                pdf_values = muAmplitudes[i] * norm.pdf(
                    oModel['stim'], muThresholds[i], muThresholdVariations[i]
                )
                plt.plot(oModel['stim'], pdf_values, 'k')
            plt.xlabel('Stimulation (mA)')
            plt.ylabel('Amplitude * Probability Density')
    
    return oAmplitude, oModel

def generate_batch_data(
    mus_range: Union[Tuple[int, int], List[int]],
    num_per_mu: int,
    amp_range: Tuple[float, float] = (0.1, 0.2),
    noise_range: Tuple[float, float] = (0.01, 0.02),
    thr_range: Tuple[float, float] = (1, 2),
    thr_var_range: Tuple[float, float] = (0.01, 0.02),
    patient_ratio: float = 0.5
) -> Dict[str, np.ndarray]:
    """
    批量生成数据，对应MATLAB中的mlx逻辑
    
    Args:
        mus_range: MU数量范围，tuple(min,max)或list [5,10,15,...]
        num_per_mu: 每个MU数量生成的样本数
        amp_range: 幅度范围
        noise_range: 噪声范围
        thr_range: 阈值范围
        thr_var_range: 阈值变异性范围
        patient_ratio: 患者比例（需要再支配的比例）
    
    Returns:
        包含所有数据的字典
    """
    
    # 设置随机种子
    np.random.seed(4)
    
    # 生成MU列表
    if isinstance(mus_range, tuple):
        MUs = np.arange(mus_range[0], mus_range[1] + 1)
    else:
        MUs = np.array(mus_range)
    
    n_all = len(MUs) * num_per_mu
    max_mus = np.max(MUs)
    
    # 初始化数据数组
    data = np.zeros((n_all, 500, 2), dtype=np.float32)
    muThr = np.zeros((n_all, max_mus), dtype=np.float32)
    label_num = np.zeros(n_all, dtype=np.float32)
    label_noise = np.zeros(n_all, dtype=np.float32)
    label_amp = np.zeros(n_all, dtype=np.float32)
    label_thr = np.zeros(n_all, dtype=np.float32)
    label_thr_var = np.zeros(n_all, dtype=np.float32)
    label_HP = np.zeros(n_all, dtype=np.float32)  # healthy 0, patient 1
    
    # 生成参数列表
    noise_list = np.random.uniform(noise_range[0], noise_range[1], n_all)
    amp_list = np.random.uniform(amp_range[0], amp_range[1], n_all)
    thr_list = np.random.uniform(thr_range[0], thr_range[1], n_all)
    thr_var_list = np.random.uniform(thr_var_range[0], thr_var_range[1], n_all)
    
    # 生成MU列表（重复每个MU数量）
    mu_list = np.repeat(MUs, num_per_mu)
    
    print(f"开始生成数据: {len(mu_list)} 个样本")
    
    # 逐个生成样本
    for index in range(len(mu_list)):
        mu = mu_list[index]
        
        if index % 100 == 0 or index == len(mu_list) - 1:
            print(f"Index {index+1}/{len(mu_list)}")
        
        # 构建自定义模型参数
        custom_model = {
            'AmplitudeParameter': [0.025, amp_list[index]],
            'ThresholdParameter': [12, thr_list[index]],
            'ThresholdVariationParameter': [thr_var_list[index], 0.005],
            'NoiseParameter': [0.01, noise_list[index]]
        }
        
        # 决定是否为患者（需要再支配）
        is_patient = (index % 2 == 1) and (mu <= 100)
        
        if is_patient:
            # 患者：有再支配
            mu_rein = np.random.randint(
                max(1, int(mu / 5)), 
                max(2, int(mu / 2)) + 1
            )
            cmap, oModel = cmap_scan_simulation_rein_cjj(
                mu, mu_rein, 500, 'down', 0, custom_model
            )
            label_HP[index] = 1
        else:
            # 健康：无再支配
            cmap, oModel = cmap_scan_simulation_rein_cjj(
                mu, 0, 500, 'down', 0, custom_model
            )
            label_HP[index] = 0
        
        # 保存数据
        data[index, :, :] = cmap
        label_num[index] = mu
        label_noise[index] = noise_list[index]
        label_amp[index] = amp_list[index]
        label_thr[index] = thr_list[index]
        label_thr_var[index] = thr_var_list[index]
        
        # 保存MU阈值
        mu_thresholds = oModel['Mu'][:, 1]  # 第二列是阈值
        muThr[index, :len(mu_thresholds)] = mu_thresholds
    
    return {
        'data': data,
        'label_num': label_num,
        'label_noise': label_noise,
        'label_amp': label_amp,
        'label_thr': label_thr,
        'label_thr_var': label_thr_var,
        'label_HP': label_HP,
        'muThr': muThr
    }

def main():
    """
    主函数：复制MATLAB mlx的执行逻辑
    """
    print("MUNE数据生成器 - Python版本")
    
    # 参数设置（对应MATLAB中的参数）
    r_amp = [0.1, 0.2]      # 指数分布，均值
    r_noise = [0.01, 0.02]  # 高斯分布，标准差
    r_thr = [1, 2]          # 高斯分布，标准差
    r_thr_var = [0.01, 0.02]  # 均匀分布，最大值
    
    # 训练和验证数据集参数
    print("请选择数据类型:")
    print("1. 训练集 (train)")
    print("2. 验证集 (val)")
    print("3. 自定义")
    
    choice = input("请输入选择 (1/2/3): ").strip()
    
    if choice == "1":
        # 训练集
        MUs = np.arange(5, 161)  # 150种情况
        Num = 5  # 每个MU数量生成5个样本
        dataset_name = "train_dataset1_HP_better_range_5"
    elif choice == "2":
        # 验证集
        MUs = np.arange(5, 161, 10)  # 5:10:160
        Num = 100
        dataset_name = "val_dataset1_HP_better_range_1000"
    else:
        # 自定义
        mu_min = int(input("最小MU数量 (默认5): ") or "5")
        mu_max = int(input("最大MU数量 (默认160): ") or "160")
        mu_step = int(input("MU数量步长 (默认1): ") or "1")
        num_per_mu = int(input("每个MU数量的样本数 (默认5): ") or "5")
        
        MUs = np.arange(mu_min, mu_max + 1, mu_step)
        Num = num_per_mu
        dataset_name = input("数据集文件名 (不含.mat): ") or "custom_dataset"
    
    print(f"生成配置:")
    print(f"  - MU数量: {MUs}")
    print(f"  - 每个MU样本数: {Num}")
    print(f"  - 总样本数: {len(MUs) * Num}")
    print(f"  - 数据集名称: {dataset_name}")
    
    # 生成数据
    try:
        results = generate_batch_data(
            mus_range=list(MUs),
            num_per_mu=Num,
            amp_range=tuple(r_amp),
            noise_range=tuple(r_noise),
            thr_range=tuple(r_thr),
            thr_var_range=tuple(r_thr_var),
            patient_ratio=0.5
        )
        
        # 转换为单精度
        for key in results:
            if isinstance(results[key], np.ndarray):
                results[key] = results[key].astype(np.float32)
        
        # 创建输出目录
        output_dir = "./data/SimDataset"
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存MAT文件
        output_path = os.path.join(output_dir, f"{dataset_name}.mat")
        savemat(output_path, results)
        
        print(f"\n数据生成完成！")
        print(f"保存路径: {output_path}")
        print(f"数据统计:")
        print(f"  - 样本总数: {len(results['label_num'])}")
        print(f"  - MU数量范围: [{results['label_num'].min():.0f}, {results['label_num'].max():.0f}]")
        print(f"  - 健康样本: {np.sum(results['label_HP'] == 0)}")
        print(f"  - 患者样本: {np.sum(results['label_HP'] == 1)}")
        
        # 可选可视化
        if input("\n是否生成可视化图像? (y/n): ").lower().startswith('y'):
            visualize_samples(results, save_path=os.path.join(output_dir, f"visualization_{dataset_name}.png"))
    
    except Exception as e:
        print(f"生成数据时出错: {e}")
        import traceback
        traceback.print_exc()

def visualize_samples(results: Dict[str, np.ndarray], num_samples: int = 6, save_path: str = None):
    """
    可视化生成的样本
    
    Args:
        results: 生成的数据字典
        num_samples: 显示的样本数量
        save_path: 保存路径（可选）
    """
    data = results['data']
    label_num = results['label_num']
    label_HP = results['label_HP']
    
    # 选择不同类型的样本进行展示
    healthy_samples = np.where(label_HP == 0)[0]
    patient_samples = np.where(label_HP == 1)[0]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # 绘制第一个子图
    for i in range(min(num_samples, len(axes))):
        if i < len(healthy_samples):
            idx = healthy_samples[i]
            axes[i].plot(data[idx, :, 0], data[idx, :, 1], 'b-', linewidth=2)
            axes[i].set_title(f'Healthy: {int(label_num[idx])} MUs')
        elif i - len(healthy_samples) < len(patient_samples):
            idx = patient_samples[i - len(healthy_samples)]
            axes[i].plot(data[idx, :, 0], data[idx, :, 1], 'r-', linewidth=2)
            axes[i].set_title(f'Patient: {int(label_num[idx])} MUs')
        
        axes[i].set_xlabel('Stimulation (mA)')
        axes[i].set_ylabel('CMAP Amplitude (mV)')
        axes[i].grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for i in range(num_samples, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化图像已保存到: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    main()
