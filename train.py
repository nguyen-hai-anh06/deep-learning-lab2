"""
Script train.py: Chạy quá trình Fine-tuning / Transfer Learning trên tập dữ liệu CIFAR-10.
Hỗ trợ huấn luyện từng mô hình riêng biệt hoặc chạy tự động lần lượt cả 4 mô hình:
['resnet18', 'vgg16', 'densenet121', 'mobilenet_v2'].

Cách sử dụng dòng lệnh (CLI):
1. Huấn luyện một mô hình cụ thể:
   python train.py --model resnet18 --epochs 10 --batch_size 64

2. Huấn luyện toàn bộ 4 mô hình:
   python train.py --model all --epochs 10

3. Chạy thử nghiệm nhanh (Dry run) với tập mẫu nhỏ (ví dụ 500 ảnh):
   python train.py --model mobilenet_v2 --subset 500 --epochs 2
"""

import os
import sys
import json
import argparse
import torch

# Đảm bảo in tiếng Việt chuẩn trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Đảm bảo import được module lab2_cv khi chạy trực tiếp từ thư mục gốc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lab2_cv.config import (
    SUPPORTED_MODELS,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    NUM_CLASSES,
    DEVICE,
    CHECKPOINT_DIR,
    LOG_DIR,
    RESULTS_DIR,
    DATA_DIR,
)
from lab2_cv.services.data_service import get_cifar10_dataloaders
from lab2_cv.services.model_service import build_model, inspect_model
from lab2_cv.services.trainer_service import train_model
from lab2_cv.services.logger_service import TensorBoardLogger


def parse_arguments() -> argparse.Namespace:
    """Định nghĩa và phân tích các tham số dòng lệnh."""
    parser = argparse.ArgumentParser(
        description="Huấn luyện mô hình Pre-trained trên CIFAR-10 (Practice 2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help=f"Tên mô hình cần train: {SUPPORTED_MODELS} hoặc 'all' để chạy tuần tự cả 4 models",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help="Số lượng epochs huấn luyện cho mỗi mô hình",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Kích thước mini-batch",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help="Tốc độ học (Learning Rate)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adam", "sgd"],
        default="adam",
        help="Thuật toán tối ưu hóa (Optimizer)",
    )
    parser.add_argument(
        "--freeze",
        action="store_true",
        default=True,
        help="Đóng băng Feature Extractor, chỉ huấn luyện lớp Classifier cuối",
    )
    parser.add_argument(
        "--unfreeze",
        action="store_false",
        dest="freeze",
        help="Mở khóa toàn bộ mạng để Fine-tune tất cả các tầng",
    )
    parser.add_argument(
        "--fine_tune_last",
        action="store_true",
        default=False,
        help="Mở khóa một vài tầng cuối của backbone để Fine-tuning (Step 4 của Lab)",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Số lượng mẫu train giới hạn (hữu ích khi muốn chạy thử nhanh)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=str(DEVICE),
        help="Thiết bị tính toán: 'cuda', 'cpu', hoặc 'cuda:0'",
    )

    return parser.parse_args()


def run_training_pipeline(args: argparse.Namespace) -> None:
    """Quản lý toàn bộ tiến trình huấn luyện các mô hình theo tham số được chỉ định."""
    device = torch.device(args.device)
    strategy_str = "Fine-tune Last Layers (Step 4)" if args.fine_tune_last else ("Freeze Backbone (Feature Extraction)" if args.freeze else "Unfreeze All (Full Fine-tuning)")
    print(f"\n=======================================================")
    print(f"      PRACTICE 2 - CIFAR-10 TRANSFER LEARNING          ")
    print(f"=======================================================")
    print(f" Thiết bị (Device)      : {device}")
    print(f" Chiến lược huấn luyện  : {strategy_str}")
    print(f" Epochs                  : {args.epochs}")
    print(f" Batch size              : {args.batch_size}")
    print(f" Learning rate           : {args.lr}")
    print(f" Optimizer               : {args.optimizer.upper()}")
    print(f" Dữ liệu mẫu (Subset)    : {args.subset if args.subset else 'Toàn bộ CIFAR-10 (50k train, 10k test)'}")
    print(f"=======================================================\n")

    # 1. Tải dữ liệu CIFAR-10
    print("[1/3] Đang chuẩn bị dữ liệu CIFAR-10 với Transforms ImageNet...")
    train_loader, test_loader = get_cifar10_dataloaders(
        data_dir=DATA_DIR,
        batch_size=args.batch_size,
        subset_size=args.subset,
    )
    print(f" -> Train Batches: {len(train_loader)} (khoảng {len(train_loader.dataset):,} ảnh)")
    print(f" -> Test Batches : {len(test_loader)} (khoảng {len(test_loader.dataset):,} ảnh)")

    # 2. Xác định danh sách các mô hình cần huấn luyện
    if args.model.lower() == "all":
        models_to_train = SUPPORTED_MODELS
    else:
        if args.model.lower() not in SUPPORTED_MODELS:
            print(f"[LỖI] Mô hình '{args.model}' không hợp lệ. Chọn từ: {SUPPORTED_MODELS}")
            sys.exit(1)
        models_to_train = [args.model.lower()]

    summary_results = []

    # 3. Lần lượt khởi tạo và huấn luyện từng mô hình
    for idx, model_name in enumerate(models_to_train, 1):
        print(f"\n>>> TIẾN HÀNH [{idx}/{len(models_to_train)}]: MÔ HÌNH {model_name.upper()} <<<")

        # Khởi tạo mô hình
        model = build_model(
            model_name=model_name,
            num_classes=NUM_CLASSES,
            freeze_backbone=args.freeze,
            fine_tune_last_layers=args.fine_tune_last,
        )

        # In thông tin kiến trúc nhanh
        print(f"\n--- THÔNG SỐ KIẾN TRÚC MÔ HÌNH {model_name.upper()} ---")
        inspect_model(model, input_size=(args.batch_size, 3, 224, 224), device="cpu", verbose=1)

        # Khởi tạo logger TensorBoard cho mô hình này
        if args.fine_tune_last:
            comment_str = f"finetunelast_opt_{args.optimizer}"
        else:
            comment_str = f"freeze_{args.freeze}_opt_{args.optimizer}"
        logger = TensorBoardLogger(
            model_name=model_name,
            log_dir=LOG_DIR,
            comment=comment_str,
        )

        # Ghi log kiến trúc mạng (Graph) vào TensorBoard
        dummy_input = torch.randn(2, 3, 224, 224).to(device)
        logger.log_model_graph(model.to(device), dummy_input)

        # Huấn luyện mô hình
        res = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            model_name=model_name,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            optimizer_type=args.optimizer,
            device=device,
            checkpoint_dir=CHECKPOINT_DIR,
            logger=logger,
        )

        logger.close()
        summary_results.append({
            "model_name": res["model_name"],
            "best_val_acc": round(res["best_val_acc"], 2),
            "total_time_seconds": round(res["total_time_seconds"], 2),
            "checkpoint_path": res["checkpoint_path"],
        })

    # 4. Lưu tổng kết huấn luyện ra file JSON
    output_summary_file = os.path.join(RESULTS_DIR, "training_summary.json")
    with open(output_summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, ensure_ascii=False, indent=4)

    print(f"\n{'#'*70}")
    print(f"   ĐÃ HOÀN TẤT HUẤN LUYỆN TOÀN BỘ CÁC MÔ HÌNH YÊU CẦU!")
    print(f"   Bảng tóm tắt đã được lưu tại: {output_summary_file}")
    print(f"   Để khởi động TensorBoard, chạy lệnh: tensorboard --logdir runs")
    print(f"   Để chạy đánh giá và xuất bảng so sánh, chạy: python run_evaluation.py")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    cli_args = parse_arguments()
    run_training_pipeline(cli_args)
