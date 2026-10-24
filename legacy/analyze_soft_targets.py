import json
import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import pandas as pd

import sys
sys.path.append(str(Path("src").resolve()))
from dsr.data import make_week1_loaders
from dsr.models import create_model

def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(Path("configs/week1_protocol.json"))
    num_classes = len(config["classes"])
    
    _, val_loader = make_week1_loaders(config, batch_size=32)
    
    # Load Teacher (ResNet50)
    print("Loading Teacher (ResNet50)...")
    teacher_ckpt = torch.load("checkpoints/week1_resnet50_baseline/best.pt", map_location=device) if Path("checkpoints/week1_resnet50_baseline/best.pt").exists() else torch.load("checkpoints/teacher_resnet50_week1/best.pt", map_location=device)
    teacher = create_model("resnet50", num_classes, pretrained=False)
    teacher.load_state_dict(teacher_ckpt["state_dict"])
    teacher.to(device).eval()
    
    # Load Student (ResNet18)
    print("Loading Student (ResNet18)...")
    student_ckpt = torch.load("checkpoints/week1_resnet18_baseline/best.pt", map_location=device) if Path("checkpoints/week1_resnet18_baseline/best.pt").exists() else torch.load("checkpoints/student_resnet18_week1/best.pt", map_location=device)
    student = create_model("resnet18", num_classes, pretrained=False)
    student.load_state_dict(student_ckpt["state_dict"])
    student.to(device).eval()

    temperatures = [1, 2, 3, 6]
    kl_records = {T: [] for T in temperatures}
    
    teacher_correct = 0
    student_correct = 0
    agreement = 0
    teacher_right_student_wrong = 0
    total_samples = 0
    
    teacher_entropies = []

    print(f"Running inference on validation set ({device})...")
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            logits_T = teacher(images)
            logits_S = student(images)
            
            # 1. Accuracy & Agreement Stats
            preds_T = logits_T.argmax(dim=1)
            preds_S = logits_S.argmax(dim=1)
            
            teacher_correct += (preds_T == labels).sum().item()
            student_correct += (preds_S == labels).sum().item()
            agreement += (preds_T == preds_S).sum().item()
            
            t_right_s_wrong = ((preds_T == labels) & (preds_S != labels)).sum().item()
            teacher_right_student_wrong += t_right_s_wrong
            
            total_samples += labels.size(0)
            
            # 2. Teacher Entropy at T=1
            p_T_1 = F.softmax(logits_T, dim=1)
            log_p_T_1 = F.log_softmax(logits_T, dim=1)
            entropy = -(p_T_1 * log_p_T_1).sum(dim=1)
            teacher_entropies.extend(entropy.cpu().tolist())
            
            # 3. KL Divergence at different Temperatures
            for T in temperatures:
                # Target: Teacher probability
                p_T = F.softmax(logits_T / T, dim=1)
                # Input: Student log-probability
                log_p_S = F.log_softmax(logits_S / T, dim=1)
                
                # KL(P || Q) = sum(P * log(P/Q)). F.kl_div with reduction='none' returns per-class components
                # Sum over classes (dim=1) to get per-sample KL
                kl_per_sample = F.kl_div(log_p_S, p_T, reduction='none').sum(dim=1)
                kl_records[T].extend(kl_per_sample.cpu().tolist())

    # Compile Report
    print("\n" + "="*50)
    print("DAY 4: SOFT-TARGET & KL DIVERGENCE ANALYSIS")
    print("="*50)
    
    print(f"\n[1] Agreement Metrics (N={total_samples})")
    print(f"Teacher Accuracy: {teacher_correct/total_samples*100:.2f}%")
    print(f"Student Accuracy: {student_correct/total_samples*100:.2f}%")
    print(f"Top-1 Agreement (T == S): {agreement/total_samples*100:.2f}%")
    print(f"Teacher Correct but Student Wrong: {teacher_right_student_wrong/total_samples*100:.2f}%")
    
    t_entropy_arr = np.array(teacher_entropies)
    print(f"\n[2] Teacher Entropy (T=1)")
    print(f"Mean: {np.mean(t_entropy_arr):.4f}, Std: {np.std(t_entropy_arr):.4f}")
    
    print(f"\n[3] KL Divergence KL(p_T || p_S) across Temperatures")
    results = []
    for T in temperatures:
        arr = np.array(kl_records[T])
        metrics = {
            "Temperature": T,
            "Mean": np.mean(arr),
            "Median": np.median(arr),
            "Std": np.std(arr),
            "Min": np.min(arr),
            "Max": np.max(arr),
            "p90": np.percentile(arr, 90),
            "p95": np.percentile(arr, 95),
            "Non-zero (%)": (arr > 1e-6).mean() * 100
        }
        results.append(metrics)
        print(f"--- T={T} ---")
        print(f"  Mean: {metrics['Mean']:.4f} | Median: {metrics['Median']:.4f} | Std: {metrics['Std']:.4f}")
        print(f"  p90:  {metrics['p90']:.4f} | p95:    {metrics['p95']:.4f}")
        print(f"  Min:  {metrics['Min']:.4f} | Max:    {metrics['Max']:.4f}")
    
    # Save to CSV
    df = pd.DataFrame(results)
    out_dir = Path("reports/week2")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "teacher_student_kl.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved detailed KL stats to {out_path}")

if __name__ == "__main__":
    main()
