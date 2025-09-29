#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MUNE数据生成器使用示例
演示如何使用generate_data.py生成CMAP扫描数据
"""

import numpy as np
from generate_data import cmap_scan_simulation_rein_cjj, generate_batch_data, visualize_samples
from scipy.io import savemat
import os

def example_single_sample():
    """示例1：生成单个CMAP样本"""
    print("=== 示例1：生成单个CMAP样本 ===")
    
    # 生成一个20个MU的健康样本
    cmap_healthy, model_healthy = cmap_scan_simulation_rein_cjj(
        mu_count=20,
        reinnervation_count=0,
        stim_count=500,
        direction='down',
        visualization=1  # 显示图像
    )
    
    print(f"健康样本CMAP数据形状: {cmap_healthy.shape}")
    print(f"MU参数形状: {model_healthy['Mu'].shape}")
    print(f"扫描范围: {model_healthy['ScanRange']}")
    
    input("按回车键继续到下一个示例...")
    
    # 生成一个15个MU的患者样本（有5个再支配MU）
    cmap_patient, model_patient = cmap_scan_simulation_rein_cjj(
        mu_count=15,
        reinnervation_count=5,
        stim_count=500,
        direction='down',
        visualization=-1  # 新窗口显示图像
    )
    
    print(f"患者样本CMAP数据形状: {cmap_patient.shape}")
    print(f"再支配MU分配: {model_patient['ReinnervatedBy']}")

def example_batch_generation():
    """示例2：批量生成数据"""
    print("\n=== 示例2：批量生成小量数据 ===")
    
    # 生成小量数据用于测试
    results = generate_batch_data(
        mus_range=[5, 10, 15],  # MU数量：5, 10, 15
        num_per_mu=3,           # 每个数量3个样本
        amp_range=(0.1, 0.15),
        noise_range=(0.01, 0.015),
        thr_range=(1.5, 2.0),
        thr_var_range=(0.01, 0.015)
    )
    
    print(f"生成的数据统计:")
    print(f"  - 总样本数: {len(results['label_num'])}")
    print(f"  - MU数量: {np.unique(results['label_num'])}")
    print(f"  - 健康样本: {np.sum(results['label_HP'] == 0)}")
    print(f"  - 患者样本: {np.sum(results['label_HP'] == 1)}")
    
    # 保存数据
    os.makedirs("./example_output", exist_ok=True)
    savemat("./example_output/test_dataset.mat", results)
    print("数据已保存到: ./example_output/test_dataset.mat")
    
    # 可视化
    visualize_samples(results, save_path="./example_output/test_visualization.png")

def example_custom_parameters():
    """示例3：自定义参数生成"""
    print("\n=== 示例3：使用自定义参数 ===")
    
    custom_model = {
        'AmplitudeParameter': [0.03, 0.15],        # 更大的偏移和均值
        'ThresholdParameter': [15, 2.5],           # 更高的阈值
        'ThresholdVariationParameter': [0.02, 0.008], # 更大的变异性
        'NoiseParameter': [0.005, 0.015]           # 更小的噪声
    }
    
    cmap, model = cmap_scan_simulation_rein_cjj(
        mu_count=25,
        custom_model=custom_model,
        visualization=1
    )
    
    print("使用自定义参数的CMAP生成完成")
    print(f"自定义幅度参数: {custom_model['AmplitudeParameter']}")
    print(f"自定义阈值参数: {custom_model['ThresholdParameter']}")

def example_data_format():
    """示例4：展示数据格式"""
    print("\n=== 示例4：数据格式展示 ===")
    
    # 生成一个简单的样本
    cmap, model = cmap_scan_simulation_rein_cjj(
        mu_count=5,
        stim_count=100,  # 减少数据点便于查看
        visualization=0   # 不显示图像
    )
    
    print("CMAP数据格式:")
    print(f"  - 数据形状: {cmap.shape} (刺激点数, [x坐标, y幅值])")
    print(f"  - X范围: [{cmap[:,0].min():.2f}, {cmap[:,0].max():.2f}] mA")
    print(f"  - Y范围: [{cmap[:,1].min():.4f}, {cmap[:,1].max():.4f}] mV")
    
    print("\nMU模型参数格式:")
    print(f"  - MU数量: {model['Mu'].shape[0]}")
    print(f"  - 参数矩阵形状: {model['Mu'].shape} [幅度, 阈值, 阈值变异性]")
    
    print(f"\n激活概率矩阵:")
    print(f"  - 形状: {model['ActivateLog'].shape} [MU数量, 刺激点数]")
    print(f"  - 激活情况: {np.sum(model['ActivateLog'], axis=0)[:10]} ... (前10个刺激点)")

if __name__ == "__main__":
    print("MUNE数据生成器使用示例")
    print("=" * 50)
    
    try:
        # 运行示例
        example_single_sample()
        example_data_format()
        example_custom_parameters()
        example_batch_generation()
        
        print("\n" + "=" * 50)
        print("所有示例运行完成！")
        print("\n使用说明:")
        print("1. 单个样本生成：直接调用cmap_scan_simulation_rein_cjj()")
        print("2. 批量生成：调用generate_batch_data()")
        print("3. 可视化：调用visualize_samples()")
        print("4. 保存数据：使用scipy.io.savemat()")
        
    except KeyboardInterrupt:
        print("\n用户中断示例运行")
    except Exception as e:
        print(f"运行示例时出错: {e}")
        import traceback
        traceback.print_exc()
