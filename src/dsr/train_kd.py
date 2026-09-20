from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from dsr.data import compute_class_weights, make_week1_loaders
from dsr.metrics import classification_metrics
from dsr.models import create_model


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def train_kd_one_epoch(student, teacher, loader, criterion_ce, optimizer, device, alpha: float, temperature: float) -> tuple[float, float, float]:
    student.train()
    teacher.eval()
    
    total_loss = 0.0
    total_ce_loss = 0.0
    total_kd_loss = 0.0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        
        # Student forward
        logits_S = student(images)
        
        # Teacher forward (no grad)
        with torch.no_grad():
            logits_T = teacher(images)
            
        # 1. Cross Entropy Loss (Hard Labels)
        loss_ce = criterion_ce(logits_S, labels)
        
        # 2. Knowledge Distillation Loss (Soft Labels)
        # Warning from Master Plan: input = log_prob(student), target = prob(teacher)
        student_log_prob = F.log_softmax(logits_S / temperature, dim=1)
        teacher_prob = F.softmax(logits_T / temperature, dim=1).detach()
        
        # reduction='batchmean' is required for KL divergence to be mathematically correct over a batch
        loss_kd = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (temperature ** 2)
        
        # 3. Total Loss (Cố định cứng theo chuẩn Additive của Master Plan §5)
        # MASTER: L_total = L_CE + alpha*L_logit + ... -> Không scale L_CE để tránh rủi ro gradient khi so sánh chéo dòng 4a-8.
        loss = loss_ce + alpha * loss_kd
        
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_ce_loss += float(loss_ce.detach().cpu()) * batch_size
        total_kd_loss += float(loss_kd.detach().cpu()) * batch_size
        total_examples += batch_size

    return (
        total_loss / total_examples if total_examples else 0.0,
        total_ce_loss / total_examples if total_examples else 0.0,
        total_kd_loss / total_examples if total_examples else 0.0
    )


def evaluate(model, loader, criterion, device, num_classes: int) -> dict[str, float]:
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


def write_kd_history(path: Path, rows: list[dict[str, float | int]], classes: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    per_class_fields = [f"val_f1_{name}" for name in classes] + [f"val_support_{name}" for name in classes]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_ce_loss",
                "train_kd_loss",
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
    parser = argparse.ArgumentParser(description="Vanilla KD training entrypoint.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--student-model", default="resnet18", choices=["resnet18"])
    parser.add_argument("--teacher-model", default="resnet50", choices=["resnet50"])
    parser.add_argument("--teacher-ckpt", required=True, type=Path, help="Path to best teacher checkpoint")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/week2"), help="Output directory for history CSV")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint and history")
    
    # KD Hyperparameters
    parser.add_argument("--alpha", type=float, default=0.7, help="KD loss weight")
    parser.add_argument("--temperature", type=float, default=3.0, help="Softmax temperature")
    
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    args = parser.parse_args()

    config = load_config(args.config)
    num_classes = len(config["classes"])
    protocol = config["internal_ablation_protocol"]
    
    batch_size = args.batch_size or int(protocol["batch_size"])
    epochs = args.epochs or int(protocol["max_epochs"])
    lr = args.lr or float(protocol["lr_schedule"]["base_lr"])
    classes = list(config["classes"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, val_loader = make_week1_loaders(config, batch_size=batch_size)
    
    # Init Teacher
    print(f"Loading Teacher {args.teacher_model} from {args.teacher_ckpt}...")
    teacher = create_model(args.teacher_model, num_classes=num_classes, pretrained=False)
    t_ckpt = torch.load(args.teacher_ckpt, map_location=device)
    teacher.load_state_dict(t_ckpt["state_dict"])
    teacher.to(device)
    teacher.eval() # Freeze teacher
    for param in teacher.parameters():
        param.requires_grad = False
        
    # Init Student
    print(f"Loading Student {args.student_model} (pretrained ImageNet weights)...")
    student = create_model(args.student_model, num_classes=num_classes, pretrained=True)
    student.to(device)

    class_weights = compute_class_weights(config)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion_ce = nn.CrossEntropyLoss(weight=weight_tensor)
    
    optimizer = torch.optim.SGD(
        student.parameters(),
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
    start_epoch = 1
    checkpoint_dir = Path("checkpoints") / args.run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    history_file = args.out_dir / f"{args.run_name}_history.csv"

    if args.resume and history_file.exists():
        print(f"Resuming from existing history: {history_file}")
        with history_file.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # convert string to float/int where appropriate
                parsed_row = {}
                for k, v in row.items():
                    if k == "epoch":
                        parsed_row[k] = int(v)
                    else:
                        parsed_row[k] = float(v)
                history.append(parsed_row)
        if history:
            start_epoch = history[-1]["epoch"] + 1
            best_macro_f1 = max(r["val_macro_f1"] for r in history)
            
        last_ckpt = checkpoint_dir / "last_state_dict.pt"
        if last_ckpt.exists():
            print(f"Loading last state dict from {last_ckpt}")
            ckpt = torch.load(last_ckpt, map_location=device); student.load_state_dict(ckpt["model"]); optimizer.load_state_dict(ckpt["optimizer"]); scheduler.load_state_dict(ckpt["scheduler"])

    print(f"Training Vanilla KD on {device}. Alpha={args.alpha}, T={args.temperature}")
    for epoch in range(start_epoch, epochs + 1):
        t_loss, ce_loss, kd_loss = train_kd_one_epoch(
            student, teacher, train_loader, criterion_ce, optimizer, device, args.alpha, args.temperature
        )
        val_metrics = evaluate(student, val_loader, criterion_ce, device, num_classes=num_classes)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": t_loss,
            "train_ce_loss": ce_loss,
            "train_kd_loss": kd_loss,
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
        write_kd_history(args.out_dir / f"{args.run_name}_history.csv", history, classes=classes)

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            torch.save(
                {
                    "model": args.student_model,
                    "run_name": args.run_name,
                    "epoch": epoch,
                    "state_dict": student.state_dict(),
                    "metrics": val_metrics,
                    "config": config,
                    "class_weights": dict(zip(classes, class_weights)),
                },
                checkpoint_dir / "best.pt",
            )

        minority_class = classes[class_weights.index(max(class_weights))]
        print(
            "Ep {epoch:3d} | L_CE {train_ce_loss:.4f} L_KD {train_kd_loss:.4f} "
            "| Val Acc {val_accuracy:.4f} F1 {val_macro_f1:.4f} F1[{minority_class}] {minority_f1:.4f}".format(
                minority_class=minority_class,
                minority_f1=row[f"val_f1_{minority_class}"],
                **row,
            )
        )

    torch.save({
        "model": student.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, checkpoint_dir / "last_state_dict.pt")


if __name__ == "__main__":
    main()

