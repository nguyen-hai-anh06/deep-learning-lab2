"""Select runs by validation accuracy, then evaluate each chosen checkpoint once on test."""

import argparse
import csv
import glob
import json
import os
import sys
import time
from typing import Any, Dict

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tabulate import tabulate

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lab2_cv.config import CIFAR10_CLASSES, DATA_DIR, NUM_CLASSES, RESULTS_DIR, SUPPORTED_MODELS  # noqa: E402
from lab2_cv.services.data_service import get_cifar10_test_loader  # noqa: E402
from lab2_cv.services.model_service import build_model, count_parameters  # noqa: E402


def discover_best_runs(experiments_root: str) -> Dict[str, Dict[str, Any]]:
    """Choose exactly one completed run per model using validation accuracy only."""
    selected: Dict[str, Dict[str, Any]] = {}
    split_signatures: set[tuple] = set()
    source_fingerprints: set[str] = set()
    for path in glob.glob(os.path.join(experiments_root, "**", "summary.json"), recursive=True):
        with open(path, encoding="utf-8") as file:
            summary = json.load(file)
        if summary.get("status") != "completed" or summary.get("model") not in SUPPORTED_MODELS:
            continue
        run_dir = os.path.dirname(path)
        config_path = os.path.join(run_dir, "config.json")
        split_path = os.path.join(run_dir, "split.json")
        history_path = os.path.join(run_dir, "history.json")
        environment_path = os.path.join(run_dir, "environment.json")
        checkpoint_path = os.path.join(run_dir, "checkpoints", "best.pt")
        if not all(os.path.exists(item) for item in (config_path, split_path, history_path, environment_path, checkpoint_path)):
            print(f"[Skip incomplete artifacts] {run_dir}")
            continue
        with open(config_path, encoding="utf-8") as file:
            config = json.load(file)
        with open(split_path, encoding="utf-8") as file:
            split = json.load(file)
        with open(environment_path, encoding="utf-8") as file:
            environment = json.load(file)
        if environment.get("source_fingerprint"):
            source_fingerprints.add(environment["source_fingerprint"])
        split_signatures.add((
            str(split["validation_indices_checksum"]),
            split["train_size"], split["validation_size"], split["test_size"], split["seed"],
        ))
        candidate = {
            "summary": summary,
            "config": config,
            "split": split,
            "run_dir": run_dir,
            "checkpoint_path": checkpoint_path,
        }
        previous = selected.get(summary["model"])
        if previous is None or summary["best_validation_accuracy"] > previous["summary"]["best_validation_accuracy"]:
            selected[summary["model"]] = candidate
    if len(split_signatures) > 1:
        raise RuntimeError("Các run dùng data split khác nhau; không thể so sánh công bằng")
    if len(source_fingerprints) > 1:
        raise RuntimeError("Các run không dùng cùng phiên bản source code")
    if not selected:
        raise FileNotFoundError(f"Không tìm thấy run hoàn chỉnh trong {experiments_root}")
    return selected


def evaluate_checkpoint(
    candidate: Dict[str, Any], test_loader, device: torch.device
) -> Dict[str, Any]:
    config = candidate["config"]
    strategy = config["strategy"]
    model = build_model(
        config["model"],
        num_classes=NUM_CLASSES,
        pretrained=False,
        freeze_backbone=True,
        fine_tune_last_layers=strategy == "finetune_last",
    )
    checkpoint = torch.load(candidate["checkpoint_path"], map_location=device, weights_only=False)
    checkpoint_config = checkpoint.get("run_config", {})
    identity_keys = ("run_id", "member_id", "model", "strategy", "seed", "validation_ratio")
    changed = [key for key in identity_keys if checkpoint_config.get(key) != config.get(key)]
    if changed:
        raise RuntimeError(f"Checkpoint không khớp config ở các trường: {changed}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)
    total_loss = 0.0
    total = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for inputs, targets in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            total_loss += criterion(outputs, targets).item()
            predictions = outputs.argmax(dim=1)
            total += targets.numel()
            indices = (targets * NUM_CLASSES + predictions).cpu()
            confusion += torch.bincount(indices, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    true_positive = confusion.diag().float()
    precision = true_positive / confusion.sum(dim=0).clamp(min=1)
    recall = true_positive / confusion.sum(dim=1).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
    counts = count_parameters(model)
    latency_batch_size = min(16, test_loader.batch_size or 16)
    latency_input = torch.randn(latency_batch_size, 3, 224, 224, device=device)
    with torch.inference_mode():
        for _ in range(5):
            model(latency_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency_started = time.perf_counter()
        for _ in range(20):
            model(latency_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
    latency_ms = 1000.0 * (time.perf_counter() - latency_started) / 20
    history_path = os.path.join(candidate["run_dir"], "history.json")
    with open(history_path, encoding="utf-8") as file:
        history = json.load(file)
    return {
        "model": config["model"],
        "run_id": config["run_id"],
        "member_id": config["member_id"],
        "strategy": strategy,
        "validation_accuracy": candidate["summary"]["best_validation_accuracy"],
        "test_accuracy": 100.0 * true_positive.sum().item() / total,
        "test_loss": total_loss / total,
        "macro_precision": precision.mean().item(),
        "macro_recall": recall.mean().item(),
        "macro_f1": f1.mean().item(),
        "total_parameters": counts["total_params"],
        "trainable_parameters": counts["trainable_params"],
        "training_seconds": sum(row["epoch_seconds"] for row in history),
        "test_seconds": elapsed,
        "latency_ms_per_batch": latency_ms,
        "latency_batch_size": latency_batch_size,
        "estimated_model_size_mb": counts["total_params"] * 4 / (1024 ** 2),
        "confusion_matrix": confusion.tolist(),
        "checkpoint": candidate["checkpoint_path"],
    }


def save_confusion_matrix(result: Dict[str, Any], output_dir: str) -> None:
    matrix = result["confusion_matrix"]
    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_xticks(range(NUM_CLASSES), CIFAR10_CLASSES, rotation=45, ha="right")
    axis.set_yticks(range(NUM_CLASSES), CIFAR10_CLASSES)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(f"{result['model']} — test confusion matrix")
    for row in range(NUM_CLASSES):
        for column in range(NUM_CLASSES):
            axis.text(column, row, matrix[row][column], ha="center", va="center", fontsize=7)
    figure.tight_layout()
    figure.savefig(os.path.join(output_dir, f"confusion_matrix_{result['model']}.png"), dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-root", required=True)
    parser.add_argument("--output-dir", default=os.path.join(RESULTS_DIR, "final_evaluation"))
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    selected = discover_best_runs(args.experiments_root)
    missing = [model for model in SUPPORTED_MODELS if model not in selected]
    if missing:
        raise RuntimeError(f"Chưa đủ run để đánh giá test. Còn thiếu: {missing}")
    test_loader = get_cifar10_test_loader(args.data_dir, args.batch_size, args.num_workers)
    results = [evaluate_checkpoint(selected[name], test_loader, device) for name in SUPPORTED_MODELS if name in selected]
    os.makedirs(args.output_dir, exist_ok=True)
    for result in results:
        save_confusion_matrix(result, args.output_dir)
    with open(os.path.join(args.output_dir, "test_results.json"), "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    csv_fields = [key for key in results[0] if key not in ("confusion_matrix", "checkpoint")]
    with open(os.path.join(args.output_dir, "test_results.csv"), "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    rows = [[r["model"], r["strategy"], f"{r['validation_accuracy']:.2f}", f"{r['test_accuracy']:.2f}", f"{r['macro_f1']:.4f}", f"{r['training_seconds']:.1f}"] for r in results]
    print(tabulate(rows, headers=["Model", "Strategy", "Val Acc %", "Test Acc %", "Macro F1", "Train s"], tablefmt="github"))


if __name__ == "__main__":
    main()
