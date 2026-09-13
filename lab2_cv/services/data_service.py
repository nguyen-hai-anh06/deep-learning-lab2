"""
Module data_service: Quản lý toàn bộ quy trình tiền xử lý, tăng cường dữ liệu (Data Augmentation)
và nạp dữ liệu (Data Loading) cho tập dữ liệu CIFAR-10 chuẩn hóa theo ImageNet.
"""

import os
from typing import Tuple, Optional
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from lab2_cv.config import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    BATCH_SIZE,
    NUM_WORKERS,
    DATA_DIR,
)


def get_cifar10_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Tạo pipeline biến đổi hình ảnh (Transforms) cho tập Train và Test.
    
    Lý do thiết kế:
    1. CIFAR-10 có kích thước gốc 32x32 pixel.
    2. Các mô hình Pre-trained trên ImageNet (ResNet, VGG, DenseNet, MobileNet) được thiết kế
       với receptive field và các lớp stride/pooling tối ưu cho ảnh kích thước 224x224.
    3. Do đó, cần Resize về (224, 224) và chuẩn hóa theo đúng mean & std của ImageNet
       để tận dụng tối đa trọng số đã học (Transfer Learning).
    
    Returns:
        Tuple[transforms.Compose, transforms.Compose]: (train_transforms, test_transforms)
    """
    train_transforms = transforms.Compose([
        # Bước 1: Resize từ 32x32 lên 224x224 để phù hợp kiến trúc mạng pre-trained
        transforms.Resize(IMAGE_SIZE),
        
        # Bước 2: Tăng cường dữ liệu (Data Augmentation) chống overfitting
        transforms.RandomHorizontalFlip(p=0.5), # Lật ngang ngẫu nhiên
        transforms.RandomRotation(degrees=10),   # Xoay nhẹ góc tối đa 10 độ
        
        # Bước 3: Chuyển đổi định dạng PIL Image sang Tensor [C, H, W] trong dải [0.0, 1.0]
        transforms.ToTensor(),
        
        # Bước 4: Chuẩn hóa theo phân phối thống kê của tập ImageNet-1k
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    test_transforms = transforms.Compose([
        # Bước 1: Resize kích thước tương tự tập train
        transforms.Resize(IMAGE_SIZE),
        
        # Bước 2: Không áp dụng data augmentation ngẫu nhiên cho tập kiểm thử (chỉ giữ nguyên dữ liệu gốc)
        transforms.ToTensor(),
        
        # Bước 3: Chuẩn hóa cùng tham số ImageNet
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transforms, test_transforms


def get_cifar10_dataloaders(
    data_dir: str = DATA_DIR,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    subset_size: Optional[int] = None,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Tải tập dữ liệu CIFAR-10 từ torchvision và đóng gói vào PyTorch DataLoader.
    
    Args:
        data_dir (str): Thư mục lưu dữ liệu CIFAR-10.
        batch_size (int): Kích thước mỗi mini-batch. Mặc định lấy từ config.
        num_workers (int): Số lượng workers phụ trợ nạp dữ liệu.
        subset_size (Optional[int]): Lấy mẫu một phần dữ liệu nhỏ để test nhanh pipeline.
        pin_memory (bool): Đẩy tensor vào pinned memory trên host để truyền nhanh sang GPU.
        
    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, test_loader)
    """
    train_transform, test_transform = get_cifar10_transforms()

    # Tải tập huấn luyện CIFAR-10 (50,000 ảnh)
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )

    # Tải tập kiểm thử CIFAR-10 (10,000 ảnh)
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    # Nếu người dùng chỉ định subset_size (thường dùng trong Unit Test hoặc Dry Run)
    if subset_size is not None and subset_size > 0:
        train_indices = list(range(min(subset_size, len(train_dataset))))
        test_indices = list(range(min(max(1, subset_size // 5), len(test_dataset))))
        train_dataset = Subset(train_dataset, train_indices)
        test_dataset = Subset(test_dataset, test_indices)

    # Tạo DataLoader cho tập Train (xáo trộn dữ liệu shuffle=True)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False,
    )

    # Tạo DataLoader cho tập Test (không xáo trộn shuffle=False)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False,
    )

    return train_loader, test_loader
