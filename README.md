# MUNE - CMAP图预测MU和阈值深度学习框架

基于深度学习的CMAP（Compound Map）图像分析框架，用于预测MU（Motor Unit）数量和对应的不定长MU阈值。

## 任务描述

本项目的主要任务是：
- **输入**: CMAP图像数据
- **输出1**: 预测MU数量
- **输出2**: 预测不定长MU对应的阈值

这是一个多任务学习问题，需要同时预测连续值（MU数量）和分类/回归值（阈值）。

## 项目结构

```
MUNE/
├── model.py              # 深度学习模型定义
│   ├── BaseModel         # 基础模型类
│   ├── SimpleMLP         # 多层感知机
│   └── SimpleCNN         # 卷积神经网络
├── dataset.py            # 数据集处理模块
│   ├── BaseDataset       # 基础数据集类
│   ├── SimDataset        # 仿真数据集
│   └── HPDataset         # HP数据集（CMAP图像）
├── train.py              # 训练逻辑
│   ├── Trainer           # 训练器类
│   ├── 早停机制          # Early Stopping
│   └── 检查点管理        # Checkpoint Management
├── train_hp.py           # HP数据集专用训练脚本
├── hp_dataset_manager.py # HP数据集管理器
├── loss.py               # 损失函数定义
│   ├── CrossEntropyLoss  # 交叉熵损失
│   ├── FocalLoss         # Focal Loss
│   └── CombinedLoss      # 组合损失（MU数量+阈值）
├── metrics.py            # 评估指标
│   ├── Accuracy          # 准确率
│   ├── Precision/Recall  # 精确率/召回率
│   ├── F1Score           # F1分数
│   └── AUROC             # AUC-ROC
├── config.py             # 配置管理系统
│   ├── Config            # 配置类
│   └── 点号语法访问      # config['model.name']
├── utils.py              # 工具函数
├── logger.py             # 日志系统
├── main.py               # 主训练脚本
├── test_dataset.py       # 数据集测试脚本
├── config.yaml           # 配置文件示例
├── requirements.txt      # Python依赖包
├── data/                 # 数据目录（已加入.gitignore）
│   ├── train/            # 训练数据
│   ├── val/              # 验证数据
│   └── test/             # 测试数据
├── logs/                 # 训练日志（已加入.gitignore）
│   ├── tensorboard/      # TensorBoard日志
│   └── training.log      # 训练日志文件
├── checkpoints/          # 模型检查点（已加入.gitignore）
│   ├── best_model.pth    # 最佳模型
│   └── latest_model.pth  # 最新模型
└── .gitignore            # Git忽略文件
```

## 核心功能

### 🎯 多任务学习
- **MU数量预测**: 回归任务，预测CMAP图中的运动单元数量
- **阈值预测**: 分类/回归任务，预测不定长MU对应的阈值

### 🧠 模型架构
- **SimpleMLP**: 适用于特征向量的多层感知机
- **SimpleCNN**: 适用于CMAP图像数据的卷积神经网络
- **多输出头**: 支持同时预测MU数量和阈值

### 📊 数据处理
- **CMAP图像预处理**: 图像标准化、增强等
- **MATLAB文件支持**: 读取.mat格式的CMAP数据
- **数据增强**: 旋转、翻转、噪声添加等

### 📈 评估指标
- **回归指标**: MAE、MSE、R²（用于MU数量预测）
- **分类指标**: 准确率、精确率、召回率、F1分数（用于阈值预测）
- **综合指标**: 多任务学习的加权损失

## 快速开始

### 1. 环境安装

```bash
# 克隆项目
git clone <repository-url>
cd MUNE

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

```bash
# 将CMAP数据文件放在data目录下
mkdir -p data/train data/val data/test

# 数据格式：.mat文件，包含CMAP图像和标签
# 标签格式：[MU数量, 阈值]
```

### 3. 训练模型

```bash
# 使用HP数据集训练
python train_hp.py --data_root ./data --model simple_cnn --epochs 100

# 使用配置文件训练
python train_hp.py --config config.yaml

# 恢复训练
python train_hp.py --resume ./checkpoints/latest_model.pth
```

### 4. 模型评估

```bash
# 仅评估模式
python train_hp.py --data_root ./data --eval_only --resume ./checkpoints/best_model.pth

# 在真实数据上测试
python train_hp.py --data_root ./data --eval_only --resume ./checkpoints/best_model.pth --test_real_data
```

## 配置说明

### 模型配置

```yaml
model:
  name: simple_cnn  # 模型名称
  config:
    input_channels: 1      # 输入通道数
    num_classes_mu: 1     # MU数量输出维度
    num_classes_threshold: 10  # 阈值分类数量
    dropout_rate: 0.5
```

### 数据集配置

```yaml
dataset:
  name: hp               # 数据集名称
  data_path: ./data      # 数据路径
  batch_size: 32         # 批次大小
  num_workers: 4         # 数据加载线程数
  image_size: [224, 224] # 图像尺寸
```

### 训练配置

```yaml
training:
  num_epochs: 100
  device: auto
  save_dir: ./checkpoints
  early_stopping_patience: 10
  learning_rate: 0.001
  weight_decay: 1e-4
```

## 支持的模型

- `simple_mlp`: 多层感知机（适用于特征向量）
- `simple_cnn`: 卷积神经网络（适用于CMAP图像）

## 支持的数据集

- `sim`: 仿真数据集（用于测试）
- `hp`: HP数据集（CMAP图像数据）

## 支持的损失函数

- `cross_entropy`: 交叉熵损失
- `focal`: Focal Loss
- `mse`: 均方误差损失
- `combined`: 组合损失（MU数量MSE + 阈值交叉熵）

## 命令行参数

```bash
python train_hp.py --help
```

主要参数：
- `--data_root`: 数据根目录
- `--model`: 模型名称
- `--epochs`: 训练轮数
- `--batch_size`: 批次大小
- `--lr`: 学习率
- `--device`: 设备选择
- `--resume`: 恢复训练的检查点
- `--eval_only`: 仅评估模式
- `--test_real_data`: 在真实数据上测试

## 扩展指南

### 添加新模型

1. 继承 `BaseModel` 类
2. 实现 `_build_model` 和 `forward` 方法
3. 支持多输出（MU数量 + 阈值）
4. 在 `create_model` 函数中注册

### 添加新数据集

1. 继承 `BaseDataset` 类
2. 实现 `_load_data` 方法
3. 处理CMAP图像和标签
4. 在 `create_dataset` 函数中注册

### 自定义损失函数

1. 继承 `BaseLoss` 类
2. 实现 `forward` 方法
3. 支持多任务损失计算
4. 在 `create_loss` 函数中注册

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！
