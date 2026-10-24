from __future__ import annotations

import csv
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dsr.data import CsvImageDataset, create_transforms, read_split_csv
from dsr.metrics import classification_metrics


def set_seed(seed: int) -> torch.Generator:
    """Set seeds for reproducibility across random, numpy, and torch.

    Returns a PyTorch Generator seeded for DataLoaders.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    """Worker init function to guarantee deterministic DataLoader multiprocessing."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_scheduler(optimizer: torch.optim.Optimizer, protocol: dict, epochs: int) -> torch.optim.lr_scheduler.LRScheduler:
    """Build standardized Warmup + Cosine Annealing learning rate scheduler."""
    warmup_epochs = int(protocol.get("lr_schedule", {}).get("warmup_epochs", 5))
    eta_min = float(protocol.get("lr_schedule", {}).get("min_lr", 1e-6))

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs - warmup_epochs),
        eta_min=eta_min,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module | None,
    device: torch.device,
    num_classes: int,
) -> dict[str, float | list[float]]:
    """Evaluate model on a DataLoader, returning classification metrics and loss."""
    model.eval()
    total_loss = 0.0
    total_examples = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            if criterion is not None:
                loss = criterion(logits, labels)
                batch_size = labels.size(0)
                total_loss += float(loss.detach().cpu()) * batch_size
                total_examples += batch_size

            preds = logits.argmax(dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    metrics = classification_metrics(y_true, y_pred, num_classes=num_classes)
    metrics["loss"] = total_loss / total_examples if total_examples > 0 else 0.0
    return metrics


def write_kd_history(path: Path, rows: list[dict[str, float | int]], classes: list[str]) -> None:
    """Write training history to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    per_class_fields = [f"val_f1_{name}" for name in classes] + [f"val_support_{name}" for name in classes]
    fieldnames = [
        "epoch",
        "train_loss",
        "train_ce_loss",
        "train_kd_loss",
        "val_loss",
        "val_accuracy",
        "val_macro_f1",
        "val_balanced_accuracy",
        *per_class_fields,
    ]
    # Filter fieldnames based on available keys in first row
    if rows:
        available = set(rows[0].keys())
        fieldnames = [f for f in fieldnames if f in available] + [f for f in rows[0].keys() if f not in fieldnames]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def get_git_commit() -> str:
    """Safely get current git commit hash."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def write_manifest(path: Path, manifest_data: dict) -> None:
    """Write run manifest JSON for experiment provenance and traceability."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(manifest_data)
    data.setdefault("git_commit", get_git_commit())
    data.setdefault("created_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def make_holdout_loader(config: dict, batch_size: int = 32) -> DataLoader:
    """Create DataLoader for the Dev/Corruption-Holdout dataset (379 images).

    This dataset is strictly isolated from CV-folds and training.
    """
    data_root = Path(config["data"]["trashnet_root"])
    split_dir = Path(config["data"]["split_dir"])
    classes = list(config["classes"])
    image_size = int(config["data"]["image_size"])
    num_workers = int(config["data"].get("num_workers", 0))

    holdout_file = split_dir / "trashnet_dev_corruption_holdout.csv"
    if not holdout_file.exists():
        raise FileNotFoundError(f"Holdout file not found: {holdout_file}")

    rows = read_split_csv(holdout_file)
    dataset = CsvImageDataset(
        data_root=data_root,
        rows=rows,
        classes=classes,
        transform=create_transforms(image_size=image_size, train=False),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
