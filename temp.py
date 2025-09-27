import numpy as np
import matplotlib.pyplot as plt

# 模拟数据
np.random.seed(42)
stim_intensity = np.linspace(0, 20, 200)  # 刺激强度 (0-20mA)
num_MUs = 6                               # 假设有6个运动单位
thresholds = np.sort(np.random.uniform(3, 15, num_MUs))  # MU阈值
amplitudes = np.random.uniform(0.5, 2.0, num_MUs)        # 每个MU贡献幅度

# 构建CMAP曲线
cmap = np.zeros_like(stim_intensity)
for thr, amp in zip(thresholds, amplitudes):
    cmap += amp * (stim_intensity > thr)

# 绘图
plt.figure(figsize=(8, 5))
plt.plot(stim_intensity, cmap, lw=2, color='b')
plt.scatter(thresholds, np.cumsum(amplitudes), color='r', marker='o', label="MU recruit points")

plt.xlabel("Stimulation Intensity (mA)", fontsize=12)
plt.ylabel("CMAP Amplitude (mV)", fontsize=12)
plt.title("CMAP Scan Curve", fontsize=14)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
