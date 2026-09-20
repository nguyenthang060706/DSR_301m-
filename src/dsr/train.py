from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from dsr.data import compute_class_weights, make_week1_loaders
from dsr.metrics import classification_metrics
from dsr.models import create_model


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_examples += batch_size

    return total_loss / total_examples if total_examples else 0.0


def evaluate(model, loader, criterion, device, num_classes: int) -> dict[str, float]:
    import torch

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
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)

            batch_size = labels.size(0)
            total_loss += float(loss.detach().cpu()) * batch_size
            total_examples += batch_size
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    metrics = classification_metrics(y_true, y_pred, num_classes=num_classes)
    metrics["loss"] = total_loss / total_examples if total_examples else 0.0
    return metrics


def write_history(path: Path, rows: list[dict[str, float | int]], classes: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    per_class_fields = [f"val_f1_{name}" for name in classes] + [f"val_support_{name}" for name in classes]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "val_accuracy",
                "val_macro_f1",
                "val_balanced_accuracy",
                *per_class_fields,
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Week-1 baseline training entrypoint.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--model", 
        required=True, 
        choices=["resnet50", "resnet18", "resnet18_eca", "mobilenet_v3_large", "mobilenet_v3_small", "efficientnet_b0", "efficientformer_l1"]
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/week1"), help="Output directory for history CSV")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    num_classes = len(config["classes"])

    if args.dry_run:
        print(f"run_name={args.run_name}")
        print(f"model={args.model}")
        print(f"num_classes={num_classes}")
        print(f"trashnet_root={config['data']['trashnet_root']}")
        print(f"split_dir={config['data']['split_dir']}")
        print(f"protocol_status={config['training_protocol_status']}")
        return

    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for training. Install requirements.txt first.") from exc

    protocol = config["internal_ablation_protocol"]
    batch_size = args.batch_size or int(protocol["batch_size"])
    epochs = args.epochs or int(protocol["max_epochs"])
    lr = args.lr or float(protocol["lr_schedule"]["base_lr"])

    classes = list(config["classes"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = make_week1_loaders(config, batch_size=batch_size)
    model = create_model(
        args.model,
        num_classes=num_classes,
        pretrained=not args.no_pretrained,
    ).to(device)

    class_weights = compute_class_weights(config)
    print("Class weights (inverse-frequency, computed from trashnet_cv_folds.csv):")
    for name, weight in zip(classes, class_weights):
        print(f"  {name:>10s}: {weight:.4f}")
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=float(protocol["momentum"]),
        weight_decay=float(protocol["weight_decay"]),
    )
    warmup_epochs = int(protocol["lr_schedule"].get("warmup_epochs", 5))
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs - warmup_epochs,
        eta_min=float(protocol["lr_schedule"]["min_lr"]),
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
    )

    history: list[dict[str, float | int]] = []
    best_macro_f1 = -1.0
    checkpoint_dir = Path("checkpoints") / args.run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training {args.model} on {device} for {epochs} epochs.")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device, num_classes=num_classes)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        }
        for name, f1 in zip(classes, val_metrics["per_class_f1"]):
            row[f"val_f1_{name}"] = f1
        for name, support in zip(classes, val_metrics["per_class_support"]):
            row[f"val_support_{name}"] = support
        history.append(row)
        out_dir = args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        write_history(out_dir / f"{args.run_name}_history.csv", history, classes=classes)

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            torch.save(
                {
                    "model": args.model,
                    "run_name": args.run_name,
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "metrics": val_metrics,
                    "config": config,
                    "class_weights": dict(zip(classes, class_weights)),
                },
                checkpoint_dir / "best.pt",
            )

        minority_class = classes[class_weights.index(max(class_weights))]
        print(
            "epoch={epoch} train_loss={train_loss:.4f} "
            "val_loss={val_loss:.4f} val_acc={val_accuracy:.4f} "
            "val_macro_f1={val_macro_f1:.4f} "
            "val_f1[{minority_class}]={minority_f1:.4f} (n={minority_n})".format(
                minority_class=minority_class,
                minority_f1=row[f"val_f1_{minority_class}"],
                minority_n=row[f"val_support_{minority_class}"],
                **row,
            )
        )

    torch.save(model.state_dict(), checkpoint_dir / "last_state_dict.pt")


if __name__ == "__main__":
    main()


