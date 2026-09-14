"""Deterministic CIFAR-10 train/validation/test input pipeline."""

from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from lab2_cv.config import (
    BATCH_SIZE,
    DATA_DIR,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_WORKERS,
    RANDOM_SEED,
    VALIDATION_RATIO,
)


def get_cifar10_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Return stochastic train transforms and deterministic evaluation transforms."""
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_transform, eval_transform


def create_split_indices(
    dataset_size: int = 50_000,
    validation_ratio: float = VALIDATION_RATIO,
    seed: int = RANDOM_SEED,
) -> Tuple[list[int], list[int]]:
    """Create a reproducible, non-overlapping train/validation split."""
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio phải nằm trong khoảng (0, 1)")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_size, generator=generator).tolist()
    validation_size = int(dataset_size * validation_ratio)
    return indices[validation_size:], indices[:validation_size]


def get_split_metadata(
    train_indices: list[int], val_indices: list[int], test_size: int, seed: int
) -> Dict[str, object]:
    """Metadata used to verify that every teammate used the same split."""
    checksum = sum((position + 1) * index for position, index in enumerate(val_indices))
    return {
        "dataset": "CIFAR10",
        "seed": seed,
        "train_size": len(train_indices),
        "validation_size": len(val_indices),
        "test_size": test_size,
        "validation_indices_checksum": str(checksum),
    }


def get_cifar10_dataloaders(
    data_dir: str = DATA_DIR,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    validation_ratio: float = VALIDATION_RATIO,
    seed: int = RANDOM_SEED,
    subset_size: Optional[int] = None,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, object]]:
    """Return independent train, validation and test loaders plus split metadata."""
    train_transform, eval_transform = get_cifar10_transforms()

    # Separate objects ensure validation never receives random augmentation.
    train_source = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform)
    val_source = datasets.CIFAR10(data_dir, train=True, download=True, transform=eval_transform)
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=eval_transform)

    train_indices, val_indices = create_split_indices(
        len(train_source), validation_ratio=validation_ratio, seed=seed
    )
    if subset_size is not None:
        if subset_size <= 0:
            raise ValueError("subset_size phải lớn hơn 0")
        train_indices = train_indices[: min(subset_size, len(train_indices))]
        val_target = max(1, int(subset_size * validation_ratio))
        val_indices = val_indices[: min(val_target, len(val_indices))]
        test_target = max(1, subset_size // 5)
        test_dataset = Subset(test_dataset, range(min(test_target, len(test_dataset))))

    train_dataset = Subset(train_source, train_indices)
    val_dataset = Subset(val_source, val_indices)
    use_pin_memory = pin_memory and torch.cuda.is_available()
    loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": use_pin_memory,
        "persistent_workers": num_workers > 0,
    }
    train_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=train_generator, drop_last=False, **loader_args
    )
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, **loader_args)
    metadata = get_split_metadata(train_indices, val_indices, len(test_dataset), seed)
    return train_loader, val_loader, test_loader, metadata


def get_cifar10_test_loader(
    data_dir: str = DATA_DIR,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> DataLoader:
    """Load the untouched official test set for the final evaluation only."""
    _, eval_transform = get_cifar10_transforms()
    dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=eval_transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
