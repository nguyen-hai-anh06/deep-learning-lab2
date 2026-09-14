"""TensorBoard logging isolated by experiment run."""

import os
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    def __init__(self, experiment_dir: str) -> None:
        self.log_dir = os.path.join(experiment_dir, "tensorboard")
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        print(f"[TensorBoard] {self.log_dir}")

    def log_scalars(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        lr: Optional[float] = None,
    ) -> None:
        self.writer.add_scalars(
            "Loss", {"train": train_loss, "validation": val_loss}, epoch
        )
        self.writer.add_scalars(
            "Accuracy", {"train": train_acc, "validation": val_acc}, epoch
        )
        if lr is not None:
            self.writer.add_scalar("LearningRate", lr, epoch)
        self.writer.flush()

    def log_model_graph(self, model: nn.Module, input_to_model: torch.Tensor) -> None:
        try:
            self.writer.add_graph(model, input_to_model)
            self.writer.flush()
        except Exception as exc:
            print(f"[TensorBoard] Không thể ghi graph: {exc}")

    def close(self) -> None:
        self.writer.close()
