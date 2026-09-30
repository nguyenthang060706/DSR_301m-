"""Đánh giá hiệu năng trên tập Dev/Corruption-Holdout (379 ảnh).

Chấm điểm cho:
1. Teacher sạch (ResNet50)
2. Student sạch (ResNet18)
3. Pilot Vanilla KD sạch (ResNet18)
Ghi kết quả ra reports/week3/week3_holdout_eval.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Thêm src vào sys.path
sys.path.append(str(Path("src").resolve()))

from dsr.data import compute_class_weights
from dsr.kd_common import evaluate, make_holdout_loader
from dsr.models import create_model


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on Dev/Corruption-Holdout set.")
    parser.add_argument("--config", type=Path, default=Path("configs/week1_protocol.json"))
    parser.add_argument("--out-csv", type=Path, default=Path("reports/week3/week3_holdout_eval.csv"))
    args = parser.parse_args()

    config = load_config(args.config)
    num_classes = len(config["classes"])
    classes = list(config["classes"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    holdout_loader = make_holdout_loader(config, batch_size=32)

    class_weights = compute_class_weights(config)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    models_to_eval = [
        {
            "run_name": "teacher_resnet50_clean",
            "model_type": "resnet50",
            "role": "Teacher Clean",
            "ckpt_path": Path("checkpoints/teacher_resnet50_clean/best.pt"),
        },
        {
            "run_name": "student_resnet18_clean",
            "model_type": "resnet18",
            "role": "Student Clean (Dòng 1)",
            "ckpt_path": Path("checkpoints/student_resnet18_clean/best.pt"),
        },
        {
            "run_name": "week2_vanilla_kd_clean",
            "model_type": "resnet18",
            "role": "Vanilla KD (PILOT)",
            "ckpt_path": Path("checkpoints/week2_vanilla_kd_clean/best.pt"),
        },
        {
            "run_name": "w3_vkd_tau3_s42",
            "model_type": "resnet18",
            "role": "R1: Vanilla KD (tau=3, seed=42)",
            "ckpt_path": Path("checkpoints/w3_vkd_tau3_s42/best.pt"),
        },
        {
            "run_name": "w3_vkd_tau2_s42",
            "model_type": "resnet18",
            "role": "R2: Vanilla KD (tau=2, seed=42)",
            "ckpt_path": Path("checkpoints/w3_vkd_tau2_s42/best.pt"),
        },
        {
            "run_name": "w3_vkd_tau6_s42",
            "model_type": "resnet18",
            "role": "R3: Vanilla KD (tau=6, seed=42)",
            "ckpt_path": Path("checkpoints/w3_vkd_tau6_s42/best.pt"),
        },
    ]

    results = []
    print(f"Evaluating {len(models_to_eval)} models on Holdout set ({len(holdout_loader.dataset)} images) on {device}...\n")

    for item in models_to_eval:
        ckpt_path = item["ckpt_path"]
        if not ckpt_path.exists():
            print(f"WARNING: Checkpoint not found: {ckpt_path}, skipping.")
            continue

        print(f"Loading {item['role']} from {ckpt_path}...")
        model = create_model(item["model_type"], num_classes=num_classes, pretrained=False)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)

        metrics = evaluate(model, holdout_loader, criterion, device, num_classes=num_classes)

        trash_idx = classes.index("trash") if "trash" in classes else -1
        trash_f1 = metrics["per_class_f1"][trash_idx] if trash_idx >= 0 else 0.0

        row = {
            "run_name": item["run_name"],
            "role": item["role"],
            "model_type": item["model_type"],
            "holdout_loss": metrics["loss"],
            "holdout_accuracy": metrics["accuracy"],
            "holdout_macro_f1": metrics["macro_f1"],
            "holdout_balanced_accuracy": metrics["balanced_accuracy"],
            "holdout_trash_f1": trash_f1,
        }
        for cls_name, f1 in zip(classes, metrics["per_class_f1"]):
            row[f"holdout_f1_{cls_name}"] = f1
        results.append(row)

        print(
            f"  -> Acc: {metrics['accuracy']*100:.2f}% | "
            f"Macro-F1: {metrics['macro_f1']*100:.2f}% | "
            f"Balanced-Acc: {metrics['balanced_accuracy']*100:.2f}% | "
            f"Trash F1: {trash_f1*100:.2f}%\n"
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    if results:
        fieldnames = list(results[0].keys())
        with args.out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Results successfully saved to {args.out_csv}")


if __name__ == "__main__":
    main()
