"""Đo KL Divergence đa nhiệt độ (T=1,2,3,6) cho cặp Teacher/Student sạch.
Ghi kết quả ra reports/week2/teacher_student_kl.csv
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(str(Path("src").resolve()))
from dsr.data import make_week1_loaders
from dsr.models import create_model

def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def measure_kl_at_temperature(teacher, student, val_loader, device, temperature: float):
    """Return per-sample KL divergences at given temperature."""
    all_kl = []
    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device)
            logits_T = teacher(images)
            logits_S = student(images)

            p_T = F.softmax(logits_T / temperature, dim=1)
            log_p_S = F.log_softmax(logits_S / temperature, dim=1)

            # Per-sample KL: sum over classes, no batch reduction
            kl_per_sample = F.kl_div(log_p_S, p_T, reduction='none').sum(dim=1) * (temperature ** 2)
            all_kl.extend(kl_per_sample.cpu().tolist())
    return np.array(all_kl)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(Path("configs/week1_protocol.json"))
    num_classes = len(config["classes"])
    _, val_loader = make_week1_loaders(config, batch_size=32)

    # Load Teacher (clean)
    teacher_ckpt = torch.load("checkpoints/teacher_resnet50_clean/best.pt", map_location=device)
    teacher = create_model("resnet50", num_classes, pretrained=False)
    teacher.load_state_dict(teacher_ckpt["state_dict"])
    teacher.to(device).eval()

    # Load Student (clean)
    student_ckpt = torch.load("checkpoints/student_resnet18_clean/best.pt", map_location=device)
    student = create_model("resnet18", num_classes, pretrained=False)
    student.load_state_dict(student_ckpt["state_dict"])
    student.to(device).eval()

    temperatures = [1, 2, 3, 6]
    rows = []

    print(f"Evaluating KL(p_T || p_S) on {device} at temperatures {temperatures}...")
    for T in temperatures:
        kl_arr = measure_kl_at_temperature(teacher, student, val_loader, device, T)
        row = {
            'Temperature': T,
            'Mean': kl_arr.mean(),
            'Median': np.median(kl_arr),
            'Std': kl_arr.std(),
            'Min': kl_arr.min(),
            'Max': kl_arr.max(),
            'p90': np.percentile(kl_arr, 90),
            'p95': np.percentile(kl_arr, 95),
            'Non-zero (%)': (kl_arr > 1e-10).mean() * 100,
        }
        rows.append(row)
        print(f"  T={T}: Mean={row['Mean']:.4f}, Median={row['Median']:.4f}, Std={row['Std']:.4f}")

    out = Path("reports/week2/teacher_student_kl.csv")
    with out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nDa ghi: {out}")

if __name__ == "__main__":
    main()
