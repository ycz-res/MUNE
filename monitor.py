"""
GPU/CPU 动态监控工具
实时显示GPU和CPU的使用情况曲线图
支持图片保存模式（适用于无GUI环境）
"""

import matplotlib
matplotlib.use('Agg')  # 非交互式后端

import matplotlib.pyplot as plt
import psutil
import time
from collections import deque
import numpy as np
import os

# 配置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 尝试导入GPU监控库
GPU_AVAILABLE = False
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    try:
        import pynvml
        pynvml.nvmlInit()
        GPU_AVAILABLE = True
        USE_PYNVML = True
    except:
        USE_PYNVML = False
        print("警告: 未检测到GPU监控库，仅显示CPU监控")


class SystemMonitor:
    """系统资源监控类"""
    
    def __init__(self, max_points=100):
        """
        初始化监控器
        
        Args:
            max_points: 曲线图显示的最大数据点数
        """
        self.max_points = max_points
        
        # CPU数据存储
        self.cpu_times = deque(maxlen=max_points)
        self.cpu_usage = deque(maxlen=max_points)
        self.cpu_memory = deque(maxlen=max_points)
        
        # GPU数据存储
        self.gpu_times = deque(maxlen=max_points)
        self.gpu_usage = deque(maxlen=max_points)
        self.gpu_memory = deque(maxlen=max_points)
        self.gpu_temp = deque(maxlen=max_points)
        
        self.start_time = time.time()
        
        # 检测GPU数量
        self.gpu_count = 0
        if GPU_AVAILABLE:
            try:
                if 'USE_PYNVML' in globals() and USE_PYNVML:
                    self.gpu_count = pynvml.nvmlDeviceGetCount()
                else:
                    gpus = GPUtil.getGPUs()
                    self.gpu_count = len(gpus)
            except:
                pass
    
    def get_cpu_stats(self):
        """获取CPU统计信息"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        return cpu_percent, memory_percent
    
    def get_gpu_stats(self):
        """获取GPU统计信息"""
        if not GPU_AVAILABLE or self.gpu_count == 0:
            return 0, 0, 0
        
        try:
            if 'USE_PYNVML' in globals() and USE_PYNVML:
                # 使用pynvml
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                
                gpu_percent = util.gpu
                gpu_mem_percent = (mem_info.used / mem_info.total) * 100
                gpu_temp = temp
            else:
                # 使用GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    gpu_percent = gpu.load * 100
                    gpu_mem_percent = gpu.memoryUtil * 100
                    gpu_temp = gpu.temperature
                else:
                    return 0, 0, 0
            
            return gpu_percent, gpu_mem_percent, gpu_temp
        except Exception as e:
            return 0, 0, 0
    
    def update_data(self):
        """更新监控数据"""
        current_time = time.time() - self.start_time
        
        # 更新CPU数据
        cpu_percent, memory_percent = self.get_cpu_stats()
        self.cpu_times.append(current_time)
        self.cpu_usage.append(cpu_percent)
        self.cpu_memory.append(memory_percent)
        
        # 更新GPU数据
        if GPU_AVAILABLE and self.gpu_count > 0:
            gpu_percent, gpu_mem_percent, gpu_temp = self.get_gpu_stats()
            self.gpu_times.append(current_time)
            self.gpu_usage.append(gpu_percent)
            self.gpu_memory.append(gpu_mem_percent)
            self.gpu_temp.append(gpu_temp)


def create_monitor_plot(monitor, save_path='monitor_output.png'):
    """创建监控图表"""
    
    plt.clf()  # 清除之前的图
    
    if GPU_AVAILABLE and monitor.gpu_count > 0:
        # 有GPU: 创建2x2子图
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle('System Resource Monitor (GPU + CPU)', fontsize=16, fontweight='bold')
        
        # CPU子图
        ax_cpu = axes[0, 0]
        ax_memory = axes[0, 1]
        
        # GPU子图
        ax_gpu = axes[1, 0]
        ax_gpu_mem = axes[1, 1]
        
    else:
        # 无GPU: 创建1x2子图
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('System Resource Monitor (CPU)', fontsize=16, fontweight='bold')
        
        ax_cpu = axes[0]
        ax_memory = axes[1]
    
    # 绘制CPU使用率
    if len(monitor.cpu_times) > 0:
        cpu_times = list(monitor.cpu_times)
        
        ax_cpu.plot(cpu_times, list(monitor.cpu_usage), 'b-', linewidth=2, label='CPU')
        ax_cpu.set_title(f'CPU Usage (Current: {monitor.cpu_usage[-1]:.1f}%)', 
                        fontsize=12, fontweight='bold')
        ax_cpu.set_xlabel('Time (seconds)')
        ax_cpu.set_ylabel('Usage (%)')
        ax_cpu.set_ylim(0, 100)
        ax_cpu.grid(True, alpha=0.3)
        ax_cpu.legend(loc='upper right')
        
        # 内存使用率
        ax_memory.plot(cpu_times, list(monitor.cpu_memory), 'g-', linewidth=2, label='Memory')
        ax_memory.set_title(f'Memory Usage (Current: {monitor.cpu_memory[-1]:.1f}%)', 
                           fontsize=12, fontweight='bold')
        ax_memory.set_xlabel('Time (seconds)')
        ax_memory.set_ylabel('Usage (%)')
        ax_memory.set_ylim(0, 100)
        ax_memory.grid(True, alpha=0.3)
        ax_memory.legend(loc='upper right')
    
    # 绘制GPU曲线
    if GPU_AVAILABLE and monitor.gpu_count > 0 and len(monitor.gpu_times) > 0:
        gpu_times = list(monitor.gpu_times)
        
        # GPU使用率和温度
        ax_gpu.plot(gpu_times, list(monitor.gpu_usage), 'r-', linewidth=2, label='GPU Usage')
        ax_gpu.set_title(f'GPU Usage & Temperature (Current: {monitor.gpu_usage[-1]:.1f}% / {monitor.gpu_temp[-1]:.1f}°C)', 
                        fontsize=12, fontweight='bold')
        ax_gpu.set_xlabel('Time (seconds)')
        ax_gpu.set_ylabel('Usage (%)', color='r')
        ax_gpu.set_ylim(0, 100)
        ax_gpu.tick_params(axis='y', labelcolor='r')
        ax_gpu.grid(True, alpha=0.3)
        
        # 温度（第二Y轴）
        ax_gpu_temp = ax_gpu.twinx()
        ax_gpu_temp.plot(gpu_times, list(monitor.gpu_temp), 'orange', linewidth=2, 
                        linestyle='--', label='Temperature')
        ax_gpu_temp.set_ylabel('Temperature (°C)', color='orange')
        ax_gpu_temp.set_ylim(0, 100)
        ax_gpu_temp.tick_params(axis='y', labelcolor='orange')
        
        # GPU显存使用率
        ax_gpu_mem.plot(gpu_times, list(monitor.gpu_memory), 'm-', linewidth=2, label='VRAM')
        ax_gpu_mem.set_title(f'GPU Memory Usage (Current: {monitor.gpu_memory[-1]:.1f}%)', 
                            fontsize=12, fontweight='bold')
        ax_gpu_mem.set_xlabel('Time (seconds)')
        ax_gpu_mem.set_ylabel('Usage (%)')
        ax_gpu_mem.set_ylim(0, 100)
        ax_gpu_mem.grid(True, alpha=0.3)
        ax_gpu_mem.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def monitor_system(duration=120, update_interval=1, output_dir='./plot'):
    """
    启动系统监控
    
    Args:
        duration: 监控持续时间（秒），默认120秒，设为0表示持续运行直到Ctrl+C
        update_interval: 更新间隔（秒）
        output_dir: 输出目录
    """
    print("=" * 60)
    print("System Resource Monitor")
    print("=" * 60)
    
    # 显示系统信息
    print(f"\nCPU Cores: {psutil.cpu_count(logical=False)} physical")
    print(f"CPU Threads: {psutil.cpu_count(logical=True)} logical")
    
    mem = psutil.virtual_memory()
    print(f"Total Memory: {mem.total / (1024**3):.2f} GB")
    
    if GPU_AVAILABLE:
        try:
            if 'USE_PYNVML' in globals() and USE_PYNVML:
                gpu_count = pynvml.nvmlDeviceGetCount()
                for i in range(gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle)
                    if isinstance(name, bytes):
                        name = name.decode('utf-8')
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    print(f"GPU {i}: {name}")
                    print(f"  VRAM: {mem_info.total / (1024**3):.2f} GB")
            else:
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    print(f"GPU {i}: {gpu.name}")
                    print(f"  VRAM: {gpu.memoryTotal} MB")
        except Exception as e:
            print(f"Error getting GPU info: {e}")
    else:
        print("No GPU detected")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'system_monitor.png')
    
    print("\n" + "=" * 60)
    print(f"Monitoring started... (Updating every {update_interval}s)")
    print(f"Output: {os.path.abspath(output_path)}")
    if duration > 0:
        print(f"Duration: {duration} seconds ({duration//60}min {duration%60}s)")
    else:
        print("Duration: Continuous (Press Ctrl+C to stop)")
    print("=" * 60)
    print("\nTIP: Open the image file with an image viewer to see live updates!")
    print(f"     eog {output_path}\n")
    
    # 创建监控器
    monitor = SystemMonitor(max_points=100)
    
    start_time = time.time()
    iteration = 0
    
    try:
        while True:
            # 更新数据
            monitor.update_data()
            
            # 生成图表
            create_monitor_plot(monitor, output_path)
            
            iteration += 1
            elapsed = time.time() - start_time
            
            # 打印状态
            if len(monitor.cpu_usage) > 0:
                status = f"[{elapsed:.1f}s] CPU: {monitor.cpu_usage[-1]:.1f}% | "
                status += f"MEM: {monitor.cpu_memory[-1]:.1f}%"
                
                if GPU_AVAILABLE and monitor.gpu_count > 0 and len(monitor.gpu_usage) > 0:
                    status += f" | GPU: {monitor.gpu_usage[-1]:.1f}% | "
                    status += f"VRAM: {monitor.gpu_memory[-1]:.1f}% | "
                    status += f"TEMP: {monitor.gpu_temp[-1]:.1f}°C"
                
                print(f"\r{status}", end='', flush=True)
            
            # 检查是否到达持续时间
            if duration > 0 and elapsed >= duration:
                break
            
            # 等待下一次更新
            time.sleep(update_interval)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")
    finally:
        if GPU_AVAILABLE and 'USE_PYNVML' in globals() and USE_PYNVML:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
        
        print(f"\n\nTotal iterations: {iteration}")
        print(f"Final output saved to: {os.path.abspath(output_path)}")
        print("\nYou can:")
        print(f"  1. Open the image: eog {output_path}")
        print(f"  2. Or use any image viewer to monitor real-time updates")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='GPU/CPU Real-time Monitor')
    parser.add_argument('--duration', type=int, default=120, 
                       help='Monitoring duration in seconds (0 = continuous, default: 120)')
    parser.add_argument('--interval', type=int, default=1,
                       help='Update interval in seconds (default: 1)')
    parser.add_argument('--output-dir', type=str, default='./plot',
                       help='Output directory (default: ./plot)')
    
    args = parser.parse_args()
    
    monitor_system(duration=args.duration, update_interval=args.interval, 
                   output_dir=args.output_dir)
