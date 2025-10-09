import torch
import torch.nn as nn


class Linear(nn.Module):
    """
    简单线性网络：用于 CMAP → 阈值位置预测

    输入:
        x: (batch_size, 500)
            每个样本的 CMAP 曲线幅值序列

    输出:
        logits: (batch_size, 500)
            未经过 sigmoid 的阈值预测分数（logits）
            - 训练阶段: 直接给 BCEWithLogitsLoss 或 FocalLoss
            - 推理阶段: 由外部脚本控制 sigmoid / 可视化
    """
    def __init__(self, d_model: int = 128):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(500, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )

        self.threshold_head = nn.Linear(d_model, 500)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        logits = self.threshold_head(feat)  # (batch_size, 500)
        return logits

class CNN(nn.Module):
    """
    CNN模型：基于局部卷积特征提取
    输入:
        x: (batch_size, 500)
    输出:
        logits: (batch_size, 500)
    """
    def __init__(self, d_model: int = 64):
        super().__init__()
        
        self.conv_net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),  # [B, 1, 500] → [B, 32, 500]
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=5, padding=2), # [B, 32, 500] → [B, 64, 500]
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, d_model, kernel_size=3, padding=1), # [B, 64, 500] → [B, d_model, 500]
            nn.ReLU()
        )

        self.output_head = nn.Conv1d(d_model, 1, kernel_size=1)  # [B, d_model, 500] → [B, 1, 500]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)             # [B, 500] → [B, 1, 500]
        feat = self.conv_net(x)        # [B, d_model, 500]
        logits = self.output_head(feat).squeeze(1)  # [B, 500]
        return logits

class LSTM(nn.Module):
    """
    LSTM模型：用于捕获CMAP序列的动态变化
    输入:
        x: (batch_size, 500)
    输出:
        logits: (batch_size, 500)
    """
    def __init__(self, d_model: int = 128, num_layers: int = 2, bidirectional: bool = True):
        super().__init__()
        
        self.hidden_dim = d_model
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.fc = nn.Linear(d_model * self.num_directions, 1)  # 每个时间步输出1个logit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)  # [B, 500] → [B, 500, 1]
        lstm_out, _ = self.lstm(x)  # [B, 500, d_model * num_directions]
        logits = self.fc(lstm_out).squeeze(-1)  # [B, 500]
        return logits
