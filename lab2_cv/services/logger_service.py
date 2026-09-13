"""
Module logger_service: Quản lý việc ghi nhật ký huấn luyện (Experiment Tracking)
sử dụng TensorBoard (SummaryWriter) để trực quan hóa Loss, Accuracy và Model Graph.
"""

import os
from typing import Optional
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from lab2_cv.config import LOG_DIR


class TensorBoardLogger:
    """
    Quản lý phiên ghi chép dữ liệu TensorBoard cho từng lần thí nghiệm huấn luyện.
    Hỗ trợ ghi nhận:
    - Scalar metrics: Train/Loss, Train/Accuracy, Val/Loss, Val/Accuracy, Learning Rate.
    - Model Graph: Đồ thị luồng tensor qua các tầng của mạng.
    """

    def __init__(
        self,
        model_name: str,
        log_dir: str = LOG_DIR,
        comment: Optional[str] = None,
    ) -> None:
        """
        Khởi tạo TensorBoard SummaryWriter với cấu trúc thư mục riêng cho từng thí nghiệm.
        
        Args:
            model_name (str): Tên mô hình (ví dụ: 'resnet18', 'mobilenet_v2').
            log_dir (str): Thư mục gốc chứa logs của TensorBoard.
            comment (Optional[str]): Ghi chú bổ sung (ví dụ: 'freeze_lr0.001').
        """
        self.model_name = model_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{model_name}_{timestamp}"
        if comment:
            exp_name += f"_{comment}"
            
        self.experiment_dir = os.path.join(log_dir, exp_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=self.experiment_dir)
        print(f"[TensorBoard] Đã khởi tạo logger tại: {self.experiment_dir}")

    def log_scalars(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        lr: Optional[float] = None,
    ) -> None:
        """
        Ghi lại các giá trị Loss và Accuracy của cả hai tập Train và Validation sau mỗi epoch.
        
        Cách tổ chức tag trên giao diện TensorBoard:
        - Loss/Train vs Loss/Validation (hiển thị chung biểu đồ so sánh Overfitting/Underfitting)
        - Accuracy/Train vs Accuracy/Validation
        
        Args:
            epoch (int): Số thứ tự epoch hiện tại (bắt đầu từ 1).
            train_loss (float): Giá trị hàm mất mát trên tập huấn luyện.
            train_acc (float): Độ chính xác (%) trên tập huấn luyện.
            val_loss (float): Giá trị hàm mất mát trên tập validation/test.
            val_acc (float): Độ chính xác (%) trên tập validation/test.
            lr (Optional[float]): Tốc độ học hiện tại của Optimizer.
        """
        # Ghi biểu đồ Loss kết hợp (so sánh trực quan train vs val)
        self.writer.add_scalars(
            "Loss/Epoch",
            {"Train": train_loss, "Validation": val_loss},
            global_step=epoch,
        )
        
        # Ghi biểu đồ Accuracy kết hợp
        self.writer.add_scalars(
            "Accuracy/Epoch",
            {"Train": train_acc, "Validation": val_acc},
            global_step=epoch,
        )

        # Ghi riêng từng scalar để dễ lọc đồ thị
        self.writer.add_scalar(f"Loss/Train_{self.model_name}", train_loss, epoch)
        self.writer.add_scalar(f"Loss/Val_{self.model_name}", val_loss, epoch)
        self.writer.add_scalar(f"Accuracy/Train_{self.model_name}", train_acc, epoch)
        self.writer.add_scalar(f"Accuracy/Val_{self.model_name}", val_acc, epoch)

        if lr is not None:
            self.writer.add_scalar("Learning_Rate", lr, epoch)

        self.writer.flush()

    def log_model_graph(self, model: nn.Module, input_to_model: torch.Tensor) -> None:
        """
        Vẽ đồ thị luồng xử lý dữ liệu của kiến trúc mô hình lên TensorBoard.
        
        Args:
            model (nn.Module): Mô hình mạng nơ-ron.
            input_to_model (torch.Tensor): Tensor đầu vào mẫu (dummy tensor).
        """
        try:
            self.writer.add_graph(model, input_to_model)
            self.writer.flush()
        except Exception as e:
            print(f"[TensorBoard] Cảnh báo: Không thể vẽ model graph: {e}")

    def close(self) -> None:
        """Đóng kết nối SummaryWriter và lưu toàn bộ buffer xuống đĩa."""
        self.writer.close()
        print(f"[TensorBoard] Đã lưu và đóng logger tại: {self.experiment_dir}")
