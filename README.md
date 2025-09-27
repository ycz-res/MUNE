# MUNE - 深度学习训练框架

一个精简但功能完整的深度学习训练框架，支持多种模型、数据集和训练策略。

## 特性

- 🚀 **模块化设计**: 清晰的模块分离，易于扩展
- 📊 **多种模型**: 支持CNN、MLP等常见架构
- 🗂️ **数据集支持**: 内置MNIST、CIFAR-10等数据集
- 📈 **丰富指标**: 准确率、F1分数、AUC等多种评估指标
- ⚙️ **灵活配置**: 支持YAML/JSON配置文件
- 📝 **完整日志**: 详细的训练日志和指标记录
- 🔄 **早停机制**: 防止过拟合的早停策略
- 💾 **检查点**: 自动保存和恢复训练状态

## 项目结构

```
MUNE/
├── model.py          # 模型定义
├── dataset.py        # 数据集处理
├── train.py          # 训练逻辑
├── loss.py           # 损失函数
├── metrics.py        # 评估指标
├── config.py         # 配置管理
├── utils.py          # 工具函数
├── logger.py         # 日志系统
├── main.py           # 主训练脚本
├── requirements.txt  # 依赖包
├── data/             # 数据目录
├── logs/             # 日志目录
└── checkpoints/      # 模型检查点
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练仿真数据集

```bash
# 使用默认配置训练仿真数据集
python main.py --model simple_mlp --epochs 10

# 使用自定义参数
python main.py --model simple_mlp --epochs 50 --batch_size 128 --lr 0.01
```

### 3. 使用配置文件

```python
# 创建配置文件
from config import create_sim_config
config = create_sim_config()
config.to_yaml('config.yaml')

# 使用配置文件训练
python main.py --config config.yaml
```

### 4. 优雅的配置语法

```python
from config import Config

# 创建配置
config = Config()

# 使用点号语法访问和设置配置
config['model.name'] = 'simple_mlp'
config['model.config.input_size'] = 1000
config['dataset.batch_size'] = 32
config['optimizer.lr'] = 0.001

# 获取配置
print(f"模型名称: {config['model.name']}")
print(f"学习率: {config['optimizer.lr']}")

# 使用get方法（带默认值）
device = config.get('training.device', 'cpu')

# 更新配置
config.update({
    'training.num_epochs': 100,
    'dataset.batch_size': 64
})

# 保存配置
config.to_yaml('my_config.yaml')
```

### 5. 训练HP数据集

```bash
# 训练HP数据集（需要将MATLAB文件放在./data目录下）
python train_hp.py --data_root ./data --model simple_mlp --epochs 100

# 在真实数据上测试
python train_hp.py --data_root ./data --eval_only --resume ./checkpoints/best_model.pth --test_real_data
```

## 使用示例

### 基本训练

```python
from config import create_sim_config
from train import Trainer
from dataset import create_dataset, create_dataloader

# 创建配置
config = create_sim_config()

# 创建数据集
train_dataset = create_dataset('sim', config['dataset'], train=True)
train_loader = create_dataloader(train_dataset, config['dataset'], train=True)

# 创建训练器
trainer = Trainer(config)

# 开始训练
results = trainer.train(train_loader, val_loader, num_epochs=50)
```

### 自定义模型

```python
from model import BaseModel, create_model

# 定义自定义模型
class MyModel(BaseModel):
    def _build_model(self):
        # 实现模型架构
        pass
    
    def forward(self, x):
        # 实现前向传播
        pass

# 注册模型
models = {'my_model': MyModel}

# 使用模型
model = create_model('my_model', config)
```

### 自定义损失函数

```python
from loss import BaseLoss, create_loss

# 定义自定义损失
class MyLoss(BaseLoss):
    def forward(self, predictions, targets):
        # 实现损失计算
        pass

# 注册损失函数
losses = {'my_loss': MyLoss}

# 使用损失函数
criterion = create_loss('my_loss', config)
```

## 配置说明

### 模型配置

```yaml
model:
  name: simple_mlp  # 模型名称
  config:
    input_size: 1000
    hidden_sizes: [512, 256, 128]
    num_classes: 2
    dropout_rate: 0.5
```

### 数据集配置

```yaml
dataset:
  name: sim
  data_path: ./data
  batch_size: 32
  num_workers: 4
```

### 训练配置

```yaml
training:
  num_epochs: 100
  device: auto
  save_dir: ./checkpoints
  early_stopping_patience: 10
```

## 支持的模型

- `simple_mlp`: 简单多层感知机
- `simple_cnn`: 简单卷积神经网络

## 支持的数据集

- `sim`: 仿真数据集（MAT文件格式）

## 支持的损失函数

- `cross_entropy`: 交叉熵损失
- `focal`: Focal Loss
- `label_smoothing`: 标签平滑损失
- `mse`: 均方误差损失
- `combined`: 组合损失

## 支持的指标

- `accuracy`: 准确率
- `precision_recall_f1`: 精确率、召回率、F1分数
- `confusion_matrix`: 混淆矩阵
- `auroc`: AUC-ROC
- `loss`: 损失值

## 命令行参数

```bash
python main.py --help
```

主要参数：
- `--config`: 配置文件路径
- `--dataset`: 数据集名称
- `--model`: 模型名称
- `--epochs`: 训练轮数
- `--batch_size`: 批次大小
- `--lr`: 学习率
- `--device`: 设备选择
- `--resume`: 恢复训练的检查点

## 扩展指南

### 添加新模型

1. 继承 `BaseModel` 类
2. 实现 `_build_model` 和 `forward` 方法
3. 在 `create_model` 函数中注册

### 添加新数据集

1. 继承 `BaseDataset` 类
2. 实现 `_load_data` 方法
3. 在 `create_dataset` 函数中注册

### 添加新损失函数

1. 继承 `BaseLoss` 类
2. 实现 `forward` 方法
3. 在 `create_loss` 函数中注册

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！
