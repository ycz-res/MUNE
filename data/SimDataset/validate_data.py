#!/usr/bin/env python3
"""
数据校验函数
校验每个样本的muThr中值为1的数量和label_num是否对应
注意：只统计值为1.0的情况，忽略其他大于0的值
"""

import numpy as np
import os
import json
from datetime import datetime

def validate_mu_thresholds(npz_path: str, log_file: str = None) -> dict:
    """
    校验每个样本的muThr中值为1的数量和label_num是否对应
    
    Args:
        npz_path: NPZ文件路径
        log_file: 日志文件路径，如果为None则自动生成1.
    
    Returns:
        校验结果字典
    """
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"./validation_log_{timestamp}.json"
    
    print(f"📊 开始校验数据: {npz_path}")
    
    # 加载数据
    data = np.load(npz_path)
    thresholds = data['thresholds']  # (N, 500) 阈值映射
    mus = data['mus']  # (N,) MU数量
    
    print(f"✅ 数据加载成功:")
    print(f"   - 总样本数: {thresholds.shape[0]:,}")
    print(f"   - 阈值维度: {thresholds.shape[1]}")
    print(f"   - MU数量范围: [{mus.min():.1f}, {mus.max():.1f}]")
    
    # 统计结果
    total_samples = thresholds.shape[0]
    mismatched_samples = []
    matched_samples = 0
    
    print(f"\n🔍 开始逐样本校验...")
    
    # 逐样本校验
    for i in range(total_samples):
        # 计算该样本中1的数量（只统计值为1的情况）
        threshold_count = np.sum(thresholds[i] == 1.0)
        # 获取该样本的MU数量
        mu_count = mus[i]
        
        # 检查是否匹配
        if threshold_count != mu_count:
            mismatched_samples.append({
                'sample_index': i,
                'threshold_count': threshold_count,
                'mu_count': mu_count,
                'threshold': thresholds[i]
            })
        else:
            matched_samples += 1
        
        # 每10000个样本打印一次进度
        if (i + 1) % 10000 == 0:
            print(f"   已校验 {i + 1:,}/{total_samples:,} 样本...")
    
    # 生成校验报告
    validation_result = {
        'total_samples': total_samples,
        'matched_samples': matched_samples,
        'mismatched_samples': len(mismatched_samples),
        'mismatch_rate': len(mismatched_samples) / total_samples * 100,
        'mismatched_details': mismatched_samples
    }
    
    # 打印校验结果
    print(f"\n📋 校验结果:")
    print(f"   - 总样本数: {validation_result['total_samples']:,}")
    print(f"   - 匹配样本数: {validation_result['matched_samples']:,}")
    print(f"   - 不匹配样本数: {validation_result['mismatched_samples']:,}")
    print(f"   - 不匹配率: {validation_result['mismatch_rate']:.2f}%")
    
    # 如果有不匹配的样本，写入日志文件
    if mismatched_samples:
        print(f"\n⚠️  发现 {len(mismatched_samples)} 个不匹配样本，写入JSON日志文件: {log_file}")
        write_validation_json(log_file, validation_result, npz_path)
    else:
        print(f"\n✅ 所有样本都匹配！")
    
    return validation_result

def write_validation_json(json_file: str, validation_result: dict, npz_path: str):
    """
    将校验结果写入JSON文件
    
    Args:
        json_file: JSON文件路径
        validation_result: 校验结果
        npz_path: 原始NPZ文件路径
    """
    json_data = {
        "total_samples": validation_result['total_samples'],
        "matched_samples": validation_result['matched_samples'],
        "mismatched_samples": validation_result['mismatched_samples'],
        "mismatch_rate": validation_result['mismatch_rate'],
        "mismatched_details": []
    }
    
    # 保存前1000个不匹配样本的详细信息
    max_samples = 1000
    for mismatch in validation_result['mismatched_details'][:max_samples]:
        json_data["mismatched_details"].append({
            "sample_index": mismatch['sample_index'],
            "threshold_count": int(mismatch['threshold_count']),
            "mu_count": float(mismatch['mu_count']),
            "difference": int(mismatch['threshold_count'] - mismatch['mu_count'])
        })
    
    # 如果有更多不匹配样本，添加说明
    if len(validation_result['mismatched_details']) > max_samples:
        json_data["note"] = f"Only first {max_samples} mismatched samples are saved. Total mismatched: {len(validation_result['mismatched_details'])}"
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="校验NPZ数据中muThr和label_num的对应关系")
    parser.add_argument("--npz", default="data.npz", help="NPZ文件路径")
    parser.add_argument("--log", default="./validation_log.json", help="日志文件路径（可选）")
    args = parser.parse_args()
    
    if not os.path.exists(args.npz):
        print(f"❌ 文件不存在: {args.npz}")
        return
    
    # 执行校验
    result = validate_mu_thresholds(args.npz, args.log)

if __name__ == "__main__":
    main()
