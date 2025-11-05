import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, d_model: int = 128, dropout: float = 0.1):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(500, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
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
    def __init__(self, d_model: int = 64, dropout: float = 0.1):
        super().__init__()
        
        self.conv_net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),  # [B, 1, 500] → [B, 32, 500]
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(32, 64, kernel_size=5, padding=2), # [B, 32, 500] → [B, 64, 500]
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

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
    def __init__(self, d_model: int = 64, num_layers: int = 2, bidirectional: bool = True, dropout: float = 0.1):
        super().__init__()
        
        self.hidden_dim = d_model
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,  # LSTM的dropout只在多层时生效
        )
        
        # 添加额外的dropout层
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(d_model * self.num_directions, 1)  # 每个时间步输出1个logit
        
        # 改进初始化：给最后一层更大的权重和bias
        nn.init.xavier_uniform_(self.fc.weight, gain=2.0)
        nn.init.constant_(self.fc.bias, 0.1)  # 给一个小的正bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)  # [B, 500] → [B, 500, 1]
        lstm_out, _ = self.lstm(x)  # [B, 500, d_model * num_directions]
        lstm_out = self.dropout(lstm_out)  # 应用dropout
        logits = self.fc(lstm_out).squeeze(-1)  # [B, 500]
        return logits


class ResidualBlock1D(nn.Module):
    """1D残差块：Conv1d → BN → ReLU → Conv1d → BN → 残差相加 → ReLU"""
    def __init__(self, channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class MUNECNN(nn.Module):
    """
    MUNECNN模型：基于残差CNN的CMAP阈值位置预测
    输入:
        x: (batch_size, 500)
    输出:
        logits: (batch_size, 500)
    """
    def __init__(
        self,
        input_channels: int = 3,
        base_channels: int = 64,
        seq_len: int = 500,
        d_model: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        # 如果提供了 d_model，使用它作为 base_channels
        if d_model is not None:
            base_channels = d_model
        self.seq_len = seq_len
        
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        self.residual_blocks = nn.Sequential(
            ResidualBlock1D(base_channels, kernel_size=3, padding=1),
            ResidualBlock1D(base_channels, kernel_size=3, padding=1),
            ResidualBlock1D(base_channels, kernel_size=3, padding=1),
        )
        
        self.output_head = nn.Conv1d(base_channels, 1, kernel_size=1)
    
    def _compute_differences(self, x: torch.Tensor) -> torch.Tensor:
        """计算0阶、1阶、2阶差分并concat"""
        order0 = x
        order1 = F.pad(torch.diff(x, dim=1), (0, 1), mode='constant', value=0)
        order2 = F.pad(torch.diff(order1, dim=1), (0, 1), mode='constant', value=0)
        return torch.stack([order0, order1, order2], dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = self._compute_differences(x)
        elif x.dim() != 3:
            raise ValueError(f"Expected input shape (batch_size, seq_len) or (batch_size, 3, seq_len), got {x.shape}")
        
        x = self.stem(x)
        x = self.residual_blocks(x)
        logits = self.output_head(x).squeeze(1)
        return logits


class Transformer(nn.Module):
    """
    Transformer模型：基于自注意力机制的CMAP序列建模
    输入:
        x: (batch_size, 500)
    输出:
        logits: (batch_size, 500)
    """
    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        seq_len: int = 500,
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        
        # 输入嵌入层：将标量值映射到 d_model 维
        self.input_embedding = nn.Linear(1, d_model)
        
        # 位置编码（可学习的）
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout)
        
        # 输出头：每个时间步输出1个logit
        self.output_head = nn.Linear(d_model, 1)
        
        # 改进初始化：给最后一层更大的权重和bias
        nn.init.xavier_uniform_(self.output_head.weight, gain=2.0)
        nn.init.constant_(self.output_head.bias, 0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, 500)
        x = x.unsqueeze(-1)  # (batch_size, 500, 1)
        
        # 输入嵌入
        x = self.input_embedding(x)  # (batch_size, 500, d_model)
        
        # 添加位置编码
        x = x + self.pos_embedding
        
        # Transformer编码
        x = self.dropout(x)
        x = self.transformer_encoder(x)  # (batch_size, 500, d_model)
        
        # 输出头
        logits = self.output_head(x).squeeze(-1)  # (batch_size, 500)
        return logits


