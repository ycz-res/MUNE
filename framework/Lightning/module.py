"""
PyTorch Lightning Module
"""

import os
import sys

import torch
import pytorch_lightning as pl

import model
import loss
from metrics import b_v_metrics


class MUNEModule(pl.LightningModule):
    """
    MUNA/CMAP 500维 0/1 序列预测的 Lightning 封装

    Lightning 内置的 hook 方法有：
      - forward
      - training_step
      - validation_step
      - on_validation_epoch_end
      - configure_optimizers
    其它（_compute_loss 等）都是你自定义的工具函数。
    """

    def __init__(
        self,
        model_type: str = "CNN",
        d_model: int = 128,
        dropout: float = 0.0,
        loss_type: str = "ce",           # 可是 "ce" / "mixed" 等
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        grad_clip: float = 1.0,
        lr_scheduler: str = "plateau",   # "plateau" / "cosine" / "none"
        warmup_epochs: int = 5,
        threshold_mode: str = "binary",
        metrics_threshold: float = 0.5,
        **kwargs,
    ):
        super().__init__()

        # 1) 动态获取模型
        model_class = getattr(model, model_type, None)
        if model_class is None:
            raise ValueError(f"未知的模型类型: {model_type}")
        self.model = model_class(d_model=d_model, dropout=dropout)

        # 2) 设置损失函数
        self.loss_fn = getattr(loss, loss_type, None)
        if self.loss_fn is None:
            raise ValueError(f"未知的损失类型 {loss_type}")
        self.loss_type = loss_type  
        
        # 3) 一些超参数
        self.lr = lr
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.lr_scheduler_type = lr_scheduler
        self.warmup_epochs = warmup_epochs
        self.threshold_mode = threshold_mode
        self.metrics_threshold = metrics_threshold

        # 用于 val 阶段聚合输出
        self._val_outputs = []
        self.current_epoch_global = 0
        self.latest_train_loss = None
        self.latest_val_loss = None
        self.latest_val_metrics = {}
        self.training_history = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor):
        loss_result = self.loss_fn(pred, target)
        if isinstance(loss_result, dict) and "total" in loss_result:
            return loss_result["total"], loss_result.get("losses", {})
        return loss_result, {}

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        x = src["cmap"].float()          # (B, 500)
        y = tgt["thresholds"].float()    # (B, 500)

        pred = self(x)                   # logits
        loss, extra_losses = self._compute_loss(pred, y)
        self.latest_train_loss = loss.detach().item()

        # 记录训练损失，供回调（如 EarlyStopping / ReduceLROnPlateau）使用
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        result = {
            "loss": loss,
            "train_loss": loss.detach(),
        }
        for name, val in extra_losses.items():
            result[f"train_loss_{name}"] = val.detach()

        return result

    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        x = src["cmap"].float()
        y = tgt["thresholds"].float()

        pred = self(x)
        loss, extra_losses = self._compute_loss(pred, y)
        self.latest_val_loss = loss.detach().item()

        # 记录验证损失，供 EarlyStopping 等回调监控
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        self._val_outputs.append(
            {
                "pred": pred.detach().cpu(),
                "target": y.detach().cpu(),
            }
        )

        result = {
            "pred": pred.detach(),
            "target": y.detach(),
            "val_loss": loss.detach(),
        }
        for name, val in extra_losses.items():
            result[f"val_loss_{name}"] = val.detach()

        return result

    # ========= Lightning 内置：每个验证 epoch 结束后调用 =========
    def on_validation_epoch_end(self):
        """
        在这里聚合整个验证集的 pred/target，计算一次全局 metrics。
        """
        if not self._val_outputs:
            return

        if self.trainer and getattr(self.trainer, "sanity_checking", False):
            self._val_outputs.clear()
            return

        all_pred = torch.cat([o["pred"] for o in self._val_outputs], dim=0)
        all_target = torch.cat([o["target"] for o in self._val_outputs], dim=0)

        metrics = b_v_metrics(
            all_pred,
            all_target,
            mode=self.threshold_mode,
            threshold=self.metrics_threshold,
        )
        # 将 metrics 转为 python float，便于序列化
        self.latest_val_metrics = {
            k: (v.item() if torch.is_tensor(v) else float(v))
            for k, v in metrics.items()
        }
        # 记录当前 epoch（给可视化命名用）
        self.current_epoch_global = self.current_epoch

        history_entry = {
            "epoch": self.current_epoch + 1,
            "train_loss": self.latest_train_loss,
            "val_loss": self.latest_val_loss,
            "metrics": self.latest_val_metrics,
        }
        self.training_history.append(history_entry)

        # 清理缓存，防止显存/内存增长
        self._val_outputs.clear()

    # ========= Lightning 内置：配置优化器 & 学习率调度 =========
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.95),
            weight_decay=self.weight_decay,
        )

        # lr_scheduler 选择：cosine / plateau / none
        if self.lr_scheduler_type == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

            warmup = LinearLR(optimizer, start_factor=0.1, total_iters=self.warmup_epochs)
            cosine = CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs - self.warmup_epochs,
                eta_min=self.lr * 0.01,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[self.warmup_epochs],
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }

        elif self.lr_scheduler_type == "plateau":
            from torch.optim.lr_scheduler import ReduceLROnPlateau

            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=5,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                },
            }

        # 不用 scheduler 的情况
        return optimizer

    # ========= 梯度裁剪：推荐用 Trainer 自带，而不是手动 =========
    # 你可以删掉原来的 on_train_batch_end 手动裁剪；
    # 更推荐在 Trainer 里这样写：
    #   trainer = pl.Trainer(..., gradient_clip_val=self.grad_clip, gradient_clip_algorithm="norm")
    #
    # 如果你一定要在 Module 里自定义裁剪，可以用 configure_gradient_clipping：
    #
    # def configure_gradient_clipping(self, optimizer, optimizer_idx, gradient_clip_val, gradient_clip_algorithm):
    #     if self.grad_clip and self.grad_clip > 0:
    #         torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
