"""Training, validation, atomic checkpoints and Colab resume support."""

import csv
import json
import os
import random
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from lab2_cv.config import DEVICE, LEARNING_RATE, MOMENTUM, WEIGHT_DECAY
from lab2_cv.services.logger_service import TensorBoardLogger
from lab2_cv.services.model_service import set_frozen_batchnorm_eval


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device = DEVICE,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    epoch_idx: int = 1,
    total_epochs: int = 10,
) -> Tuple[float, float]:
    model.train()
    set_frozen_batchnorm_eval(model)
    running_loss = 0.0
    correct = 0
    total = 0
    use_amp = scaler is not None and scaler.is_enabled()
    progress = tqdm(dataloader, desc=f"Epoch {epoch_idx}/{total_epochs} train", leave=False)
    for inputs, targets in progress:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        count = targets.size(0)
        running_loss += loss.item() * count
        correct += (outputs.argmax(dim=1) == targets).sum().item()
        total += count
        progress.set_postfix(loss=f"{running_loss / total:.4f}", acc=f"{100 * correct / total:.2f}%")
    return running_loss / total, 100.0 * correct / total


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device = DEVICE,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.inference_mode():
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            count = targets.size(0)
            running_loss += loss.item() * count
            correct += (outputs.argmax(dim=1) == targets).sum().item()
            total += count
    return running_loss / total, 100.0 * correct / total


def _atomic_torch_save(payload: Dict[str, Any], path: str) -> None:
    temporary_path = f"{path}.tmp"
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def _write_history(history: list[Dict[str, Any]], run_dir: str) -> None:
    json_path = os.path.join(run_dir, "history.json")
    temporary_json = f"{json_path}.tmp"
    with open(temporary_json, "w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)
    os.replace(temporary_json, json_path)
    csv_path = os.path.join(run_dir, "history.csv")
    temporary_csv = f"{csv_path}.tmp"
    with open(temporary_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    os.replace(temporary_csv, csv_path)


def _checkpoint_payload(
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    best_val_acc: float,
    history: list[Dict[str, Any]],
    run_config: Dict[str, Any],
    train_loader: DataLoader,
) -> Dict[str, Any]:
    generator_state = None
    if getattr(train_loader, "generator", None) is not None:
        generator_state = train_loader.generator.get_state()
    return {
        "format_version": 2,
        "epoch": epoch,
        "model_name": run_config["model"],
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val_acc": best_val_acc,
        "history": history,
        "run_config": run_config,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "loader_generator_state": generator_state,
    }


def _restore_training_state(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    train_loader: DataLoader,
    device: torch.device,
    run_config: Dict[str, Any],
) -> Tuple[int, float, list[Dict[str, Any]]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_config = checkpoint.get("run_config", {})
    protected_keys = (
        "run_id", "member_id", "model", "strategy", "batch_size", "learning_rate",
        "optimizer", "weight_decay", "seed", "validation_ratio", "subset_size",
    )
    changed = [key for key in protected_keys if checkpoint_config.get(key) != run_config.get(key)]
    if changed:
        raise ValueError(f"Checkpoint không tương thích với cấu hình hiện tại: {changed}")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    if checkpoint.get("torch_rng_state") is not None:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if device.type == "cuda" and checkpoint.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
    if checkpoint.get("python_rng_state") is not None:
        random.setstate(checkpoint["python_rng_state"])
    if checkpoint.get("numpy_rng_state") is not None:
        np.random.set_state(checkpoint["numpy_rng_state"])
    if checkpoint.get("loader_generator_state") is not None and train_loader.generator is not None:
        train_loader.generator.set_state(checkpoint["loader_generator_state"].cpu())
    return (
        int(checkpoint["epoch"]) + 1,
        float(checkpoint.get("best_val_acc", float("-inf"))),
        checkpoint.get("history", []),
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    run_config: Dict[str, Any],
    run_dir: str,
    logger: Optional[TensorBoardLogger] = None,
    resume_path: Optional[str] = None,
    device: torch.device = DEVICE,
) -> Dict[str, Any]:
    """Train using validation only; save last.pt and best.pt inside one immutable run."""
    os.makedirs(run_dir, exist_ok=True)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    last_path = os.path.join(checkpoint_dir, "last.pt")
    best_path = os.path.join(checkpoint_dir, "best.pt")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if run_config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(
            trainable,
            lr=run_config["learning_rate"],
            momentum=run_config.get("momentum", MOMENTUM),
            weight_decay=run_config["weight_decay"],
        )
    else:
        optimizer = torch.optim.AdamW(
            trainable,
            lr=run_config["learning_rate"],
            weight_decay=run_config["weight_decay"],
        )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=run_config.get("scheduler_step_size", 5),
        gamma=run_config.get("scheduler_gamma", 0.1),
    )
    amp_enabled = bool(run_config.get("amp", True) and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    start_epoch = 1
    best_val_acc = float("-inf")
    history: list[Dict[str, Any]] = []
    if resume_path:
        start_epoch, best_val_acc, history = _restore_training_state(
            resume_path, model, optimizer, scheduler, scaler, train_loader, device, run_config
        )
        print(f"[Resume] Tiếp tục từ epoch {start_epoch}: {resume_path}")

    started = time.time()
    for epoch in range(start_epoch, int(run_config["epochs"]) + 1):
        epoch_started = time.time()
        learning_rate = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, epoch, run_config["epochs"]
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "validation_loss": val_loss,
            "validation_accuracy": val_acc,
            "learning_rate": learning_rate,
            "epoch_seconds": time.time() - epoch_started,
        }
        history.append(record)
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
        payload = _checkpoint_payload(
            epoch, model, optimizer, scheduler, scaler, best_val_acc, history, run_config, train_loader
        )
        _atomic_torch_save(payload, last_path)
        if is_best:
            _atomic_torch_save(
                {
                    "format_version": payload["format_version"],
                    "epoch": epoch,
                    "model_name": run_config["model"],
                    "model_state_dict": model.state_dict(),
                    "best_val_acc": best_val_acc,
                    "run_config": run_config,
                },
                best_path,
            )
        _write_history(history, run_dir)
        if logger:
            logger.log_scalars(epoch, train_loss, train_acc, val_loss, val_acc, learning_rate)
        print(
            f"Epoch {epoch:02d}/{run_config['epochs']:02d} | "
            f"train {train_loss:.4f}/{train_acc:.2f}% | "
            f"val {val_loss:.4f}/{val_acc:.2f}% | {record['epoch_seconds']:.1f}s"
        )

    summary = {
        "status": "completed",
        "run_id": run_config["run_id"],
        "member_id": run_config["member_id"],
        "model": run_config["model"],
        "strategy": run_config["strategy"],
        "best_validation_accuracy": best_val_acc,
        "best_epoch": max(history, key=lambda row: row["validation_accuracy"])["epoch"],
        "total_training_seconds_this_session": time.time() - started,
        "epochs_completed": len(history),
        "best_checkpoint": best_path,
        "last_checkpoint": last_path,
    }
    summary_path = os.path.join(run_dir, "summary.json")
    with open(f"{summary_path}.tmp", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    os.replace(f"{summary_path}.tmp", summary_path)
    return summary
