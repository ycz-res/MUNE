"""
完整的训练-测试-可视化-通知 Pipeline
"""

import os
import sys
import shutil
import subprocess
import json
import glob
import time
import requests
from datetime import datetime
import argparse


class TrainingPipeline:
    """训练流程管理器"""
    
    def __init__(self, 
                 epochs=150,
                 use_weighted_loss=True,
                 pos_weight=50.0,
                 metrics_threshold=0.5,
                 feishu_webhook=None,
                 clean_old_data=True):
        """
        初始化 Pipeline
        
        Args:
            epochs: 训练轮数
            use_weighted_loss: 是否使用加权损失
            pos_weight: 正样本权重
            metrics_threshold: 指标计算阈值
            feishu_webhook: 飞书 webhook URL
            clean_old_data: 是否清理旧数据
        """
        self.epochs = epochs
        self.use_weighted_loss = use_weighted_loss
        self.pos_weight = pos_weight
        self.metrics_threshold = metrics_threshold
        self.feishu_webhook = feishu_webhook
        self.clean_old_data = clean_old_data
        
        self.timestamp = None
        self.train_time = 0
        self.test_time = 0
        
    def log(self, message):
        """打印带时间戳的日志"""
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")
        
    def clean_directories(self):
        """清空旧数据"""
        if not self.clean_old_data:
            self.log("跳过清理旧数据")
            return
            
        self.log("开始清理旧数据...")
        
        # 清空 curve_data
        curve_data_dir = 'plot/curve_data'
        if os.path.exists(curve_data_dir):
            for file in os.listdir(curve_data_dir):
                file_path = os.path.join(curve_data_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    self.log(f"  删除: {file_path}")
        
        # 清空 visual_res
        visual_res_dir = 'plot/visual_res'
        if os.path.exists(visual_res_dir):
            shutil.rmtree(visual_res_dir)
            self.log(f"  删除目录: {visual_res_dir}")
        
        self.log("✅ 清理完成")
        
    def run_train(self):
        """运行训练"""
        self.log("="*60)
        self.log("开始训练...")
        self.log("="*60)
        
        # 构建训练命令
        cmd = [
            'python3', 'train.py',
            '--epochs', str(self.epochs),
            '--metrics_threshold', str(self.metrics_threshold)
        ]
        
        if self.use_weighted_loss:
            cmd.extend(['--use_weighted_loss', 'True'])
            cmd.extend(['--pos_weight', str(self.pos_weight)])
        
        self.log(f"训练命令: {' '.join(cmd)}")
        
        # 运行训练
        start_time = time.time()
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.train_time = time.time() - start_time
            self.log(f"✅ 训练完成，耗时: {self.train_time/60:.2f} 分钟")
            
            # 从输出中提取时间戳
            output = result.stdout
            for line in output.split('\n'):
                if 'best_model_' in line and '.pth' in line:
                    # 提取时间戳
                    import re
                    match = re.search(r'best_model_(\d{8}_\d{6})\.pth', line)
                    if match:
                        self.timestamp = match.group(1)
                        self.log(f"📅 训练时间戳: {self.timestamp}")
                        break
            
            if not self.timestamp:
                # 如果没找到，从文件系统获取最新的
                self.timestamp = self.get_latest_timestamp()
                self.log(f"📅 自动获取时间戳: {self.timestamp}")
                
        except subprocess.CalledProcessError as e:
            self.log(f"❌ 训练失败: {e}")
            self.log(f"错误输出: {e.stderr}")
            raise
            
    def get_latest_timestamp(self):
        """获取最新的时间戳"""
        checkpoint_files = glob.glob('checkpoints/best_model_*.pth')
        if not checkpoint_files:
            raise FileNotFoundError("未找到任何 checkpoint 文件")
        
        latest = max(checkpoint_files, key=os.path.getmtime)
        timestamp = os.path.basename(latest).replace('best_model_', '').replace('.pth', '')
        return timestamp
        
    def run_test(self):
        """运行测试"""
        self.log("="*60)
        self.log("开始测试...")
        self.log("="*60)
        
        cmd = [
            'python3', 'test.py',
            '--timestamp', self.timestamp,
            '--metrics_threshold', str(self.metrics_threshold)
        ]
        
        self.log(f"测试命令: {' '.join(cmd)}")
        
        start_time = time.time()
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.test_time = time.time() - start_time
            self.log(f"✅ 测试完成，耗时: {self.test_time:.2f} 秒")
        except subprocess.CalledProcessError as e:
            self.log(f"❌ 测试失败: {e}")
            self.log(f"错误输出: {e.stderr}")
            raise
            
    def run_visualization(self):
        """运行可视化"""
        self.log("="*60)
        self.log("生成可视化图表...")
        self.log("="*60)
        
        cmd = [
            'python3', 'plot/visualization.py',
            '--timestamp', self.timestamp
        ]
        
        self.log(f"可视化命令: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.log("✅ 可视化完成")
        except subprocess.CalledProcessError as e:
            self.log(f"❌ 可视化失败: {e}")
            self.log(f"错误输出: {e.stderr}")
            raise
            
    def send_to_feishu(self):
        """发送结果到飞书"""
        if not self.feishu_webhook:
            self.log("⚠️  未配置飞书 webhook，跳过通知")
            return
            
        self.log("="*60)
        self.log("发送结果到飞书...")
        self.log("="*60)
        
        # 读取训练和测试结果
        train_json = f'plot/curve_data/train_{self.timestamp}.json'
        test_json = f'plot/curve_data/test_{self.timestamp}.json'
        
        train_data = {}
        test_data = {}
        
        if os.path.exists(train_json):
            with open(train_json, 'r') as f:
                train_data = json.load(f)
        
        if os.path.exists(test_json):
            with open(test_json, 'r') as f:
                test_data = json.load(f)
        
        # 准备文本消息
        message = self.prepare_feishu_message(train_data, test_data)
        
        # 发送文本消息
        self.send_feishu_text(message)
        
        # 发送图片
        image_files = [
            'plot/visual_res/loss_curves.png',
            'plot/visual_res/val_metrics_curves.png',
            'plot/visual_res/test_metrics_summary.png',
            'plot/visual_res/random_test_samples.png'
        ]
        
        for img_path in image_files:
            if os.path.exists(img_path):
                self.send_feishu_image(img_path)
            else:
                self.log(f"⚠️  图片不存在: {img_path}")
        
        self.log("✅ 飞书通知发送完成")
        
    def prepare_feishu_message(self, train_data, test_data):
        """准备飞书消息内容"""
        # 获取最终指标
        if train_data and 'training_history' in train_data:
            last_epoch = train_data['training_history'][-1]
            train_loss = last_epoch.get('train_loss', 0)
            val_loss = last_epoch.get('val_loss', 0)
            val_metrics = last_epoch.get('val_metrics', {})
        else:
            train_loss = val_loss = 0
            val_metrics = {}
        
        if test_data and 'test_metrics' in test_data:
            test_loss = test_data.get('test_loss', 0)
            test_metrics = test_data.get('test_metrics', {})
        else:
            test_loss = 0
            test_metrics = {}
        
        message = f"""🎉 模型训练完成通知

📅 时间戳: {self.timestamp}
⏱️  训练耗时: {self.train_time/60:.2f} 分钟
🔢 训练轮数: {self.epochs}

📊 训练结果:
• 训练损失: {train_loss:.6f}
• 验证损失: {val_loss:.6f}
• Precision: {val_metrics.get('Precision', 0):.4f}
• Recall: {val_metrics.get('Recall', 0):.4f}
• F1: {val_metrics.get('F1', 0):.4f}
• IoU: {val_metrics.get('IoU', 0):.4f}

🧪 测试结果:
• 测试损失: {test_loss:.6f}
• Precision: {test_metrics.get('Precision', 0):.4f}
• Recall: {test_metrics.get('Recall', 0):.4f}
• F1: {test_metrics.get('F1', 0):.4f}
• IoU: {test_metrics.get('IoU', 0):.4f}

📁 文件位置:
• 模型: checkpoints/best_model_{self.timestamp}.pth
• 训练数据: plot/curve_data/train_{self.timestamp}.json
• 测试数据: plot/curve_data/test_{self.timestamp}.json
• 可视化: plot/visual_res/
"""
        return message
        
    def send_feishu_text(self, text):
        """发送文本消息到飞书"""
        payload = {
            "msg_type": "text",
            "content": {
                "text": text
            }
        }
        
        try:
            response = requests.post(self.feishu_webhook, json=payload)
            response.raise_for_status()
            self.log("✅ 文本消息发送成功")
        except Exception as e:
            self.log(f"❌ 发送文本消息失败: {e}")
            
    def send_feishu_image(self, image_path):
        """发送图片到飞书"""
        # 飞书需要先上传图片获取 image_key，这里简化处理
        # 实际使用时需要先调用飞书 API 上传图片
        self.log(f"📷 发送图片: {image_path}")
        
        # 注意：这里需要实现飞书图片上传的完整流程
        # 1. 上传图片到飞书获取 image_key
        # 2. 使用 image_key 发送图片消息
        # 具体实现需要飞书的 app_id 和 app_secret
        
        # 这里仅发送文本提示
        text = f"📊 可视化图表: {os.path.basename(image_path)}"
        payload = {
            "msg_type": "text",
            "content": {
                "text": text
            }
        }
        
        try:
            response = requests.post(self.feishu_webhook, json=payload)
            response.raise_for_status()
        except Exception as e:
            self.log(f"❌ 发送图片提示失败: {e}")
            
    def run(self):
        """运行完整流程"""
        start_time = time.time()
        
        try:
            # 1. 清理旧数据
            if self.clean_old_data:
                self.clean_directories()
            
            # 2. 训练
            self.run_train()
            
            # 3. 测试
            self.run_test()
            
            # 4. 可视化
            self.run_visualization()
            
            # 5. 发送到飞书
            self.send_to_feishu()
            
            total_time = time.time() - start_time
            
            self.log("="*60)
            self.log(f"🎉 Pipeline 完成! 总耗时: {total_time/60:.2f} 分钟")
            self.log("="*60)
            
        except Exception as e:
            self.log(f"❌ Pipeline 失败: {e}")
            raise


def main():
    parser = argparse.ArgumentParser('Training Pipeline')
    parser.add_argument('--epochs', type=int, default=150, help='训练轮数')
    parser.add_argument('--use_weighted_loss', type=bool, default=True, help='使用加权损失')
    parser.add_argument('--pos_weight', type=float, default=50.0, help='正样本权重')
    parser.add_argument('--metrics_threshold', type=float, default=0.5, help='指标计算阈值')
    parser.add_argument('--feishu_webhook', type=str, default=None, help='飞书 webhook URL')
    parser.add_argument('--clean', type=bool, default=True, help='是否清理旧数据')
    
    args = parser.parse_args()
    
    # 创建并运行 pipeline
    pipeline = TrainingPipeline(
        epochs=args.epochs,
        use_weighted_loss=args.use_weighted_loss,
        pos_weight=args.pos_weight,
        metrics_threshold=args.metrics_threshold,
        feishu_webhook=args.feishu_webhook,
        clean_old_data=args.clean
    )
    
    pipeline.run()


if __name__ == '__main__':
    main()

