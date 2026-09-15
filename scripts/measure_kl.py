import json
import torch
import torch.nn.functional as F
from pathlib import Path

# Thêm đường dẫn src để import
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
    
    # Load Validation Data
    _, val_loader = make_week1_loaders(config, batch_size=32)
    
    # Load Teacher
    teacher_ckpt = torch.load("checkpoints/teacher_resnet50_week1/best.pt", map_location=device)
    teacher = create_model("resnet50", num_classes, pretrained=False)
    teacher.load_state_dict(teacher_ckpt["state_dict"])
    teacher.to(device)
    teacher.eval()
    
    # Load Student
    student_ckpt = torch.load("checkpoints/student_resnet18_week1/best.pt", map_location=device)
    student = create_model("resnet18", num_classes, pretrained=False)
    student.load_state_dict(student_ckpt["state_dict"])
    student.to(device)
    student.eval()

    total_kl = 0.0
    total_examples = 0
    
    print(f"Evaluating KL(p_T || p_S) on {device}...")
    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device)
            
            logits_T = teacher(images)
            logits_S = student(images)
            
            # p_T: Teacher probabilities
            p_T = F.softmax(logits_T, dim=1)
            # p_S: Student log-probabilities (required by F.kl_div)
            log_p_S = F.log_softmax(logits_S, dim=1)
            
            # PyTorch kl_div(input, target) -> input is log_prob (Student), target is prob (Teacher)
            # reduction='batchmean' divides by batch size
            batch_kl = F.kl_div(log_p_S, p_T, reduction='batchmean')
            
            batch_size = images.size(0)
            total_kl += batch_kl.item() * batch_size
            total_examples += batch_size

    avg_kl = total_kl / total_examples
    print(f"Average KL Divergence KL(p_T || p_S) over validation set: {avg_kl:.4f}")

if __name__ == "__main__":
    main()
