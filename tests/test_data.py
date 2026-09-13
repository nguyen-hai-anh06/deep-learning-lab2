"""
Unit tests cho module data_service:
Kiểm tra cấu hình transforms và shapes của DataLoader trên CIFAR-10.
"""

import pytest
import torch
from PIL import Image

from lab2_cv.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from lab2_cv.services.data_service import get_cifar10_transforms, get_cifar10_dataloaders


def test_transforms_shape_and_type():
    """Kiểm tra Pipeline transforms chuyển đổi ảnh PIL kích thước bất kỳ về đúng (3, 224, 224)."""
    train_transform, test_transform = get_cifar10_transforms()

    # Tạo ảnh PIL giả lập kích thước gốc của CIFAR-10 (32x32)
    dummy_cifar_img = Image.new("RGB", (32, 32), color=(128, 64, 200))

    # Áp dụng transforms
    train_tensor = train_transform(dummy_cifar_img)
    test_tensor = test_transform(dummy_cifar_img)

    # 1. Kiểm tra kiểu dữ liệu
    assert isinstance(train_tensor, torch.Tensor), "Kết quả train transform phải là torch.Tensor"
    assert isinstance(test_tensor, torch.Tensor), "Kết quả test transform phải là torch.Tensor"

    # 2. Kiểm tra shape phải là [C, H, W] = [3, 224, 224]
    expected_shape = (3, IMAGE_SIZE[0], IMAGE_SIZE[1])
    assert train_tensor.shape == expected_shape, f"Train tensor shape sai: {train_tensor.shape} != {expected_shape}"
    assert test_tensor.shape == expected_shape, f"Test tensor shape sai: {test_tensor.shape} != {expected_shape}"

    # 3. Kiểm tra kiểu số thực float32
    assert train_tensor.dtype == torch.float32, "Tensor train phải có dtype float32"
    assert test_tensor.dtype == torch.float32, "Tensor test phải có dtype float32"


def test_normalization_values():
    """Kiểm tra dữ liệu sau chuẩn hóa ImageNet không bị giá trị NaN hoặc Inf."""
    _, test_transform = get_cifar10_transforms()
    dummy_img = Image.new("RGB", (32, 32), color=(255, 255, 255))
    tensor = test_transform(dummy_img)

    assert not torch.isnan(tensor).any(), "Tensor sau chuẩn hóa không được chứa giá trị NaN"
    assert not torch.isinf(tensor).any(), "Tensor sau chuẩn hóa không được chứa giá trị Inf"


def test_dataloader_batch_shapes(monkeypatch):
    """Kiểm tra DataLoader trả về đúng kích thước batch [Batch_Size, 3, 224, 224] và nhãn [Batch_Size]."""
    test_batch_size = 4

    # Tạo lớp Dataset giả lập ảnh CIFAR-10 (PIL Image 32x32) để test hoàn toàn offline
    class MockCIFAR10(torch.utils.data.Dataset):
        def __init__(self, root, train=True, download=False, transform=None):
            self.transform = transform
            self.data = [Image.new("RGB", (32, 32), color=(i * 20, 50, 100)) for i in range(16)]
            self.targets = [i % 10 for i in range(16)]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            img = self.data[idx]
            target = self.targets[idx]
            if self.transform is not None:
                img = self.transform(img)
            return img, target

    # Áp dụng monkeypatch thay thế datasets.CIFAR10 bằng MockCIFAR10
    monkeypatch.setattr("torchvision.datasets.CIFAR10", MockCIFAR10)

    train_loader, test_loader = get_cifar10_dataloaders(
        batch_size=test_batch_size,
        num_workers=0,
        subset_size=8,
    )

    # Lấy 1 batch từ train_loader
    images, labels = next(iter(train_loader))

    assert images.shape == (test_batch_size, 3, 224, 224), f"Kích thước batch ảnh sai: {images.shape}"
    assert labels.shape == (test_batch_size,), f"Kích thước nhãn sai: {labels.shape}"
    assert labels.dtype == torch.int64, "Nhãn phân loại phải là kiểu torch.int64 (LongTensor)"

