"""Colab-friendly experiment runner with isolated logs and resumable checkpoints."""

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lab2_cv.config import (  # noqa: E402
    BATCH_SIZE,
    DATA_DIR,
    LEARNING_RATE,
    NUM_CLASSES,
    NUM_EPOCHS,
    RANDOM_SEED,
    RESULTS_DIR,
    SUPPORTED_MODELS,
    VALIDATION_RATIO,
    WEIGHT_DECAY,
)
from lab2_cv.services.data_service import get_cifar10_dataloaders  # noqa: E402
from lab2_cv.services.logger_service import TensorBoardLogger  # noqa: E402
from lab2_cv.services.model_service import build_model, count_parameters, inspect_model  # noqa: E402
from lab2_cv.services.trainer_service import train_model  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one reproducible CIFAR-10 experiment")
    parser.add_argument("--model", choices=[*SUPPORTED_MODELS, "all"], default="resnet18")
    parser.add_argument("--strategy", choices=["freeze", "finetune_last"], default="freeze")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--validation-ratio", type=float, default=VALIDATION_RATIO)
    parser.add_argument("--member-id", default="manager")
    parser.add_argument("--run-id", default=None, help="Stable ID required when using --resume auto")
    parser.add_argument("--output-root", default=os.path.join(RESULTS_DIR, "experiments"))
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--resume", default=None, help="Checkpoint path, or 'auto' for run_dir/checkpoints/last.pt")
    parser.add_argument("--inspect", action="store_true", help="Print model and torchinfo summary")
    parser.add_argument("--log-graph", action="store_true", help="Add model graph to TensorBoard")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def atomic_json(payload: Dict[str, Any], path: str) -> None:
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def source_fingerprint() -> str:
    project_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(project_dir, "train.py"),
        os.path.join(project_dir, "requirements.txt"),
    ]
    for root, _, filenames in os.walk(os.path.join(project_dir, "lab2_cv")):
        candidates.extend(os.path.join(root, name) for name in filenames if name.endswith(".py"))
    digest = hashlib.sha256()
    for path in sorted(candidates):
        relative = os.path.relpath(path, project_dir).replace(os.sep, "/")
        digest.update(relative.encode("utf-8"))
        with open(path, "rb") as file:
            digest.update(file.read())
    return digest.hexdigest()


def environment_info(device: torch.device) -> Dict[str, Any]:
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)), text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        git_commit = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "timm": __import__("timm").__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "git_commit": git_commit,
        "source_fingerprint": source_fingerprint(),
    }


def run_one(args: argparse.Namespace, model_name: str) -> Dict[str, Any]:
    set_seed(args.seed)
    device = torch.device(args.device)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id or f"{args.member_id}_{model_name}_{args.strategy}_{timestamp}"
    if args.model == "all" and args.run_id:
        run_id = f"{args.run_id}_{model_name}"
    run_dir = os.path.abspath(os.path.join(args.output_root, args.member_id, run_id))
    os.makedirs(run_dir, exist_ok=True)

    run_config = {
        "run_id": run_id,
        "member_id": args.member_id,
        "model": model_name,
        "strategy": args.strategy,
        "pretrained": True,
        "num_classes": NUM_CLASSES,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "optimizer": args.optimizer,
        "weight_decay": args.weight_decay,
        "momentum": 0.9,
        "scheduler": "StepLR",
        "scheduler_step_size": 5,
        "scheduler_gamma": 0.1,
        "seed": args.seed,
        "validation_ratio": args.validation_ratio,
        "amp": not args.no_amp,
        "subset_size": args.subset,
    }
    config_path = os.path.join(run_dir, "config.json")
    if args.resume and os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as file:
            previous_config = json.load(file)
        protected_keys = (
            "run_id", "member_id", "model", "strategy", "batch_size", "learning_rate",
            "optimizer", "weight_decay", "seed", "validation_ratio", "subset_size",
        )
        changed = [key for key in protected_keys if previous_config.get(key) != run_config.get(key)]
        if changed:
            raise ValueError(f"Không thể resume vì cấu hình đã đổi: {changed}")
    atomic_json(run_config, config_path)
    atomic_json(environment_info(device), os.path.join(run_dir, "environment.json"))

    train_loader, val_loader, _, split_metadata = get_cifar10_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        subset_size=args.subset,
    )
    atomic_json(split_metadata, os.path.join(run_dir, "split.json"))

    resume_path = args.resume
    if resume_path == "auto":
        resume_path = os.path.join(run_dir, "checkpoints", "last.pt")
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Không có checkpoint để resume: {resume_path}")
    elif resume_path:
        resume_path = os.path.abspath(resume_path)
        if not os.path.exists(resume_path):
            raise FileNotFoundError(resume_path)

    fine_tune_last = args.strategy == "finetune_last"
    model = build_model(
        model_name,
        num_classes=NUM_CLASSES,
        freeze_backbone=True,
        pretrained=not bool(resume_path),
        fine_tune_last_layers=fine_tune_last,
    )
    model.eval()
    parameter_counts = count_parameters(model)
    model_statistics = inspect_model(
        model, input_size=(1, 3, 224, 224), device="cpu", verbose=0
    )
    with open(os.path.join(run_dir, "architecture.txt"), "w", encoding="utf-8") as file:
        file.write(str(model))
        file.write("\n\n")
        file.write(json.dumps(parameter_counts, indent=2))
        file.write("\n\n")
        file.write(str(model_statistics))
    print(f"Run: {run_id}\nOutput: {run_dir}\nParameters: {parameter_counts}")
    print(model)
    if args.inspect:
        print(model_statistics)

    logger = TensorBoardLogger(run_dir)
    if args.log_graph:
        logger.log_model_graph(model.to(device), torch.randn(1, 3, 224, 224, device=device))
    try:
        return train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            run_config=run_config,
            run_dir=run_dir,
            logger=logger,
            resume_path=resume_path,
            device=device,
        )
    finally:
        logger.close()


def main() -> None:
    args = parse_arguments()
    models_to_train = SUPPORTED_MODELS if args.model == "all" else [args.model]
    summaries = [run_one(args, model_name) for model_name in models_to_train]
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
