"""
Module trainer_service: Quản lý vòng lặp huấn luyện (Training Loop), đánh giá (Evaluation),
tính toán hàm mất mát (CrossEntropyLoss), tối ưu hóa trọng số (Optimizer),
và lưu trữ checkpoint mô hình có độ chính xác cao nhất.
"""

import os
import time
from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from lab2_cv.config import DEVICE, CHECKPOINT_DIR, LEARNING_RATE
from lab2_cv.services.logger_service import TensorBoardLogger


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device = DEVICE,
    epoch_idx: int = 1,
    total_epochs: int = 10,
) -> Tuple[float, float]:
    """
    Thực thi 1 epoch huấn luyện trên tập dữ liệu train:
    1. Chuyển mô hình sang chế độ train (model.train()).
    2. Duyệt qua từng batch dữ liệu:
       - Đẩy dữ liệu (inputs, labels) lên thiết bị tính toán (GPU/CPU).
       - Xóa gradient cũ (optimizer.zero_grad()).
       - Forward pass: Dự đoán logits đầu ra.
       - Tính loss: criterion(outputs, labels).
       - Backward pass: Tính gradient (loss.backward()).
       - Optimizer step: Cập nhật trọng số (optimizer.step()).
    3. Tính trung bình Loss và Accuracy (%) của toàn bộ epoch.
    
    Args:
        model (nn.Module): Mô hình nơ-ron cần train.
        dataloader (DataLoader): DataLoader chứa dữ liệu huấn luyện.
        criterion (nn.Module): Hàm mất mát (CrossEntropyLoss).
        optimizer (torch.optim.Optimizer): Thuật toán tối ưu (Adam hoặc SGD).
        device (torch.device): Thiết bị tính toán (CUDA hoặc CPU).
        epoch_idx (int): Số thứ tự epoch hiện tại.
        total_epochs (int): Tổng số epoch cần chạy.
        
    Returns:
        Tuple[float, float]: (train_loss_trung_binh, train_accuracy_phan_tram)
    """
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # Thanh tiến trình hiển thị thông tin từng batch trực quan
    pbar = tqdm(
        dataloader,
        desc=f"Epoch [{epoch_idx:02d}/{total_epochs:02d}] (Train)",
        leave=False,
    )

    for inputs, targets in pbar:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # 1. Reset gradients từ bước lặp trước
        optimizer.zero_grad()

        # 2. Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 3. Backward pass & Optimizer step
        loss.backward()
        optimizer.step()

        # 4. Thống kê loss và số lượng mẫu dự đoán chính xác
        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == targets).item()
        total_samples += batch_size

        # Cập nhật thông số hiển thị thời gian thực trên thanh tiến trình
        current_loss = running_loss / total_samples
        current_acc = (correct_predictions / total_samples) * 100.0
        pbar.set_postfix({"Loss": f"{current_loss:.4f}", "Acc": f"{current_acc:.2f}%"})

    epoch_loss = running_loss / total_samples
    epoch_acc = (correct_predictions / total_samples) * 100.0
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device = DEVICE,
) -> Tuple[float, float]:
    """
    Đánh giá độ chính xác và hàm mất mát của mô hình trên tập kiểm thử (Test / Validation):
    1. Chuyển mô hình sang chế độ eval (model.eval()).
    2. Tắt tính gradient với torch.no_grad() để tiết kiệm bộ nhớ GPU và tăng tốc.
    3. Tính trung bình Loss và Accuracy (%) trên toàn bộ tập dữ liệu.
    
    Args:
        model (nn.Module): Mô hình nơ-ron cần đánh giá.
        dataloader (DataLoader): DataLoader chứa dữ liệu kiểm thử.
        criterion (nn.Module): Hàm mất mát (CrossEntropyLoss).
        device (torch.device): Thiết bị tính toán.
        
    Returns:
        Tuple[float, float]: (eval_loss, eval_accuracy_phan_tram)
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == targets).item()
            total_samples += batch_size

    eval_loss = running_loss / total_samples if total_samples > 0 else 0.0
    eval_acc = (correct_predictions / total_samples) * 100.0 if total_samples > 0 else 0.0
    return eval_loss, eval_acc


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_name: str,
    num_epochs: int = 10,
    learning_rate: float = LEARNING_RATE,
    optimizer_type: str = "adam",
    device: torch.device = DEVICE,
    checkpoint_dir: str = CHECKPOINT_DIR,
    logger: Optional[TensorBoardLogger] = None,
) -> Dict[str, Any]:
    """
    Hàm tổng hợp quản lý toàn bộ vòng lặp huấn luyện qua nhiều epochs:
    - Tự động cấu hình optimizer (chỉ đưa các parameters có requires_grad=True).
    - Thực hiện huấn luyện và kiểm thử sau mỗi epoch.
    - Lưu file trọng số tốt nhất (Best Checkpoint: best_<model_name>.pth).
    - Tích hợp ghi log liên tục lên TensorBoard.
    
    Args:
        model (nn.Module): Mô hình cần huấn luyện.
        train_loader (DataLoader): Dữ liệu tập train.
        val_loader (DataLoader): Dữ liệu tập validation/test.
        model_name (str): Tên định danh mô hình (vd: 'resnet18').
        num_epochs (int): Tổng số epoch huấn luyện.
        learning_rate (float): Tốc độ học.
        optimizer_type (str): Loại optimizer ('adam' hoặc 'sgd').
        device (torch.device): Thiết bị tính toán.
        checkpoint_dir (str): Thư mục lưu checkpoint.
        logger (Optional[TensorBoardLogger]): Đối tượng ghi log TensorBoard.
        
    Returns:
        Dict[str, Any]: Kết quả tổng kết bao gồm best accuracy, thời gian train và history.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # LỌC CÁC THAM SỐ CẦN UPDATE: Chỉ đưa những tham số có requires_grad=True vào optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"[{model_name.upper()}] Số lượng tham số cần tối ưu: {sum(p.numel() for p in trainable_params):,}")

    # Khởi tạo Optimizer theo lựa chọn của người dùng
    if optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD(
            trainable_params,
            lr=learning_rate,
            momentum=0.9,
            weight_decay=1e-4,
        )
    else:
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=learning_rate,
            weight_decay=1e-4,
        )

    # Learning rate scheduler: Giảm LR sau mỗi 5 epochs để hội tụ ổn định
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_val_acc = 0.0
    best_checkpoint_path = os.path.join(checkpoint_dir, f"best_{model_name}.pth")
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_times": [],
    }

    start_total_time = time.time()
    print(f"\n{'='*70}")
    print(f"BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH: {model_name.upper()} TRÊN {device.type.upper()}")
    print(f"Hyperparameters: Epochs={num_epochs}, LR={learning_rate}, Optimizer={optimizer_type.upper()}")
    print(f"{'='*70}")

    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()

        # 1. Huấn luyện 1 epoch
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch_idx=epoch,
            total_epochs=num_epochs,
        )

        # 2. Đánh giá trên tập validation
        val_loss, val_acc = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        # Cập nhật scheduler
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        epoch_duration = time.time() - epoch_start_time

        # Lưu lại lịch sử
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_times"].append(epoch_duration)

        # Ghi nhận vào TensorBoard
        if logger is not None:
            logger.log_scalars(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                lr=current_lr,
            )

        # In kết quả epoch ra màn hình
        is_best = val_acc > best_val_acc
        best_marker = " [★ BEST CHECKPOINT]" if is_best else ""
        print(
            f"Epoch [{epoch:02d}/{num_epochs:02d}] ({epoch_duration:.1f}s) | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%{best_marker}"
        )

        # 3. Lưu checkpoint khi đạt Accuracy cao nhất
        if is_best:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_name": model_name,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                },
                best_checkpoint_path,
            )

    total_training_time = time.time() - start_total_time
    print(f"{'='*70}")
    print(f"HOÀN THÀNH HUẤN LUYỆN: {model_name.upper()}")
    print(f"Tổng thời gian: {total_training_time:.2f}s (~{total_training_time/60:.2f} phút)")
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"Checkpoint lưu tại: {best_checkpoint_path}")
    print(f"{'='*70}\n")

    return {
        "model_name": model_name,
        "best_val_acc": best_val_acc,
        "total_time_seconds": total_training_time,
        "checkpoint_path": best_checkpoint_path,
        "history": history,
    }
