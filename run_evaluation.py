"""
Script run_evaluation.py: Đánh giá độc lập và so sánh hiệu năng của 4 mô hình:
['resnet18', 'vgg16', 'densenet121', 'mobilenet_v2'] trên tập kiểm thử CIFAR-10 (10,000 ảnh).

Tính toán các chỉ số so sánh chuyên sâu:
1. Test Accuracy (%)
2. Test Loss
3. Total Parameters (Triệu tham số - M)
4. Trainable Parameters (nghìn tham số - K)
5. Model Size (Dung lượng MB trên đĩa)
6. Inference Latency (Độ trễ suy luận trung bình: ms/batch)

Xuất bảng so sánh tổng hợp định dạng đẹp bằng thư viện tabulate ra màn hình console
và lưu dữ liệu ra file JSON tại results/evaluation_results.json.
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, Any, List
import torch
import torch.nn as nn
from tabulate import tabulate

# Đảm bảo in tiếng Việt chuẩn trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lab2_cv.config import (
    SUPPORTED_MODELS,
    NUM_CLASSES,
    DEVICE,
    CHECKPOINT_DIR,
    RESULTS_DIR,
    DATA_DIR,
    BATCH_SIZE,
)
from lab2_cv.services.data_service import get_cifar10_dataloaders
from lab2_cv.services.model_service import build_model, count_parameters
from lab2_cv.services.trainer_service import evaluate


def measure_inference_latency(
    model: nn.Module,
    device: torch.device,
    batch_size: int = 64,
    num_warmup: int = 5,
    num_runs: int = 20,
) -> float:
    """
    Đo thời gian suy luận (Inference Latency) trung bình của mô hình trên 1 batch dữ liệu:
    - Chạy warmup một số lần để kích hoạt cache và CUDA context.
    - Đồng bộ CUDA (torch.cuda.synchronize) nếu chạy trên GPU để đo đạc chuẩn xác.
    
    Args:
        model (nn.Module): Mô hình nơ-ron cần đo.
        device (torch.device): Thiết bị tính toán.
        batch_size (int): Kích thước batch mẫu.
        num_warmup (int): Số lần chạy khởi động.
        num_runs (int): Số lần đo để lấy trung bình.
        
    Returns:
        float: Thời gian suy luận trung bình tính bằng mili-giây (ms/batch).
    """
    model.eval()
    dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)

    with torch.no_grad():
        # Warmup runs
        for _ in range(num_warmup):
            _ = model(dummy_input)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.time()
        for _ in range(num_runs):
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()

        elapsed_ms = ((time.time() - start_time) / num_runs) * 1000.0

    return round(elapsed_ms, 2)


def evaluate_single_model(
    model_name: str,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    checkpoint_dir: str = CHECKPOINT_DIR,
) -> Dict[str, Any]:
    """Đánh giá một mô hình cụ thể và trả về đầy đủ các chỉ số hiệu năng."""
    print(f"[*] Đang nạp mô hình: {model_name.upper()}...")
    
    # 1. Xây dựng kiến trúc mô hình
    model = build_model(model_name=model_name, num_classes=NUM_CLASSES, freeze_backbone=True)
    
    # 2. Kiểm tra xem có file checkpoint đã huấn luyện không
    ckpt_path = os.path.join(checkpoint_dir, f"best_{model_name}.pth")
    loaded_from_checkpoint = False
    
    if os.path.exists(ckpt_path):
        print(f"    -> Đang nạp trọng số tối ưu từ checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        loaded_from_checkpoint = True
    else:
        print(f"    -> [LƯU Ý] Không tìm thấy checkpoint ({ckpt_path}). Đang dùng trọng số pre-trained mặc định.")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # 3. Đánh giá Accuracy và Loss trên tập kiểm thử
    test_loss, test_acc = evaluate(model=model, dataloader=test_loader, criterion=criterion, device=device)

    # 4. Thống kê số lượng tham số
    param_counts = count_parameters(model)
    total_params = param_counts["total_params"]
    trainable_params = param_counts["trainable_params"]

    # 5. Đo Inference Latency
    latency_ms = measure_inference_latency(model=model, device=device, batch_size=test_loader.batch_size)

    # 6. Kích thước file ước tính (MB): 1 param float32 = 4 bytes
    estimated_size_mb = round((total_params * 4) / (1024 * 1024), 2)

    return {
        "Model Name": model_name,
        "Checkpoint Loaded": loaded_from_checkpoint,
        "Test Accuracy (%)": round(test_acc, 2),
        "Test Loss": round(test_loss, 4),
        "Total Params (M)": round(total_params / 1e6, 2),
        "Trainable Params (K)": round(trainable_params / 1e3, 2),
        "Model Size (MB)": estimated_size_mb,
        "Latency (ms/batch)": latency_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy đánh giá và so sánh 4 mô hình pre-trained trên CIFAR-10")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size cho tập test")
    parser.add_argument("--device", type=str, default=str(DEVICE), help="Thiết bị tính toán: 'cuda' hoặc 'cpu'")
    parser.add_argument("--subset", type=int, default=None, help="Giới hạn số mẫu test để đánh giá nhanh")
    args = parser.parse_args()

    device = torch.device(args.device)
    print("\n" + "=" * 80)
    print("      BẮT ĐẦU QUY TRÌNH ĐÁNH GIÁ & SO SÁNH 4 MÔ HÌNH PRE-TRAINED")
    print("=" * 80)
    print(f" Thiết bị đánh giá : {device}")
    print(f" Thư mục Checkpoint: {CHECKPOINT_DIR}")
    print("=" * 80 + "\n")

    # Nạp dữ liệu kiểm thử
    _, test_loader = get_cifar10_dataloaders(
        data_dir=DATA_DIR,
        batch_size=args.batch_size,
        subset_size=args.subset,
    )

    results: List[Dict[str, Any]] = []

    # Đánh giá tuần tự từng mô hình
    for model_name in SUPPORTED_MODELS:
        res = evaluate_single_model(
            model_name=model_name,
            test_loader=test_loader,
            device=device,
            checkpoint_dir=CHECKPOINT_DIR,
        )
        results.append(res)
        print(f"    [OK] Test Accuracy: {res['Test Accuracy (%)']}% | Loss: {res['Test Loss']} | Latency: {res['Latency (ms/batch)']}ms\n")

    # In bảng so sánh trực quan ra Terminal bằng tabulate
    table_headers = [
        "Model Name",
        "Test Acc (%)",
        "Test Loss",
        "Total Params",
        "Trainable Params",
        "Model Size",
        "Latency (ms/batch)",
        "Tình trạng Checkpoint",
    ]
    table_rows = []
    for r in results:
        table_rows.append([
            r["Model Name"].upper(),
            f"{r['Test Accuracy (%)']}%",
            f"{r['Test Loss']:.4f}",
            f"{r['Total Params (M)']}M",
            f"{r['Trainable Params (K)']}K",
            f"{r['Model Size (MB)']} MB",
            f"{r['Latency (ms/batch)']} ms",
            "Đã train (Best)" if r["Checkpoint Loaded"] else "Pre-trained mặc định",
        ])

    print("\n" + "#" * 85)
    print("             BẢNG TỔNG HỢP SO SÁNH HIỆU NĂNG 4 MÔ HÌNH (CIFAR-10)")
    print("#" * 85)
    print(tabulate(table_rows, headers=table_headers, tablefmt="fancy_grid"))
    print("#" * 85 + "\n")

    # Lưu kết quả ra file JSON
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_json_path = os.path.join(RESULTS_DIR, "evaluation_results.json")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"[THÀNH CÔNG] Dữ liệu đánh giá chi tiết đã được lưu tại: {out_json_path}\n")


if __name__ == "__main__":
    main()
