"""
Package lab2_cv.services: Chứa các module chuyên biệt thực hiện các tác vụ cụ thể:
- data_service: Data loading, data augmentation, transforms.
- model_service: Xây dựng 4 pre-trained models, sửa classifier, inspect_model với torchinfo.
- trainer_service: Vòng lặp training, evaluation, loss, optimizer, checkpoint.
- logger_service: Quản lý ghi nhận logs với TensorBoard.
"""

from .data_service import get_cifar10_transforms, get_cifar10_dataloaders
from .model_service import build_model, inspect_model, freeze_features, unfreeze_last_layers
from .trainer_service import train_one_epoch, evaluate, train_model
from .logger_service import TensorBoardLogger

__all__ = [
    "get_cifar10_transforms",
    "get_cifar10_dataloaders",
    "build_model",
    "inspect_model",
    "freeze_features",
    "unfreeze_last_layers",
    "train_one_epoch",
    "evaluate",
    "train_model",
    "TensorBoardLogger",
]
