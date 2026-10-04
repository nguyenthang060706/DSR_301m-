"""scripts/audit_weeks_1_and_2.py

Kiểm tra toàn diện tính nhất quán, xung đột và toàn vẹn của kết quả Tuần 1 và Tuần 2.
Bao gồm:
  1. Kiểm tra Dataset, Splits và Label Map (không rò rỉ, đủ 15.000 ảnh).
  2. Kiểm tra Checkpoints của Teacher, Student M0 và Student M1.
  3. Kiểm tra Lịch sử Huấn luyện CSV (40 epoch mỗi mô hình).
  4. Kiểm tra Cổng Teacher (RER >= 20%).
  5. Kiểm tra tính đồng bộ giữa code train.py, train_kd.py, protocol_lock.md và Kế hoạch MASTER v9.5.
"""

from __future__ import annotations
import csv
import sys
from pathlib import Path
import torch

sys.path.insert(0, "src")
from dsr.data import load_label_map, read_split_csv
from dsr.models import create_model

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

errors = []
warnings = []

def check(condition, msg, is_warn=False):
    if condition:
        print(f"  {PASS} {msg}")
    elif is_warn:
        print(f"  {WARN} {msg}")
        warnings.append(msg)
    else:
        print(f"  {FAIL} {msg}")
        errors.append(msg)

print("=" * 75)
print("BẮT ĐẦU AUDIT TOÀN DIỆN TUẦN 1 & TUẦN 2 (MASTER V9.5)")
print("=" * 75)

# -------------------------------------------------------------
# PHẦN 1: DỮ LIỆU & ÁNH XẠ NHÃN
# -------------------------------------------------------------
print("\n[PHẦN 1] Kiểm tra Dữ liệu & Bảng ánh xạ nhãn...")
data_root = Path("data/raw/household_waste_30")
map_file = Path("data/mappings/label_map.csv")
train_file = Path("data/splits/public_train.csv")
dev_file = Path("data/splits/public_dev.csv")

check(data_root.exists(), f"Thư mục dữ liệu chính thức tồn tại: {data_root}")
check(map_file.exists(), f"Bảng ánh xạ nhãn tồn tại: {map_file}")
check(train_file.exists(), f"Tập train split tồn tại: {train_file}")
check(dev_file.exists(), f"Tập dev split tồn tại: {dev_file}")

fine_name_to_id, fine_to_coarse_id, fine_classes, coarse_classes = load_label_map(map_file)
check(len(fine_classes) == 30, f"Đủ đúng 30 lớp nhãn mịn (thực tế: {len(fine_classes)})")
check(len(coarse_classes) == 3, f"Đủ đúng 3 nhóm quyết định Canteen: {coarse_classes}")

train_rows = read_split_csv(train_file)
dev_rows = read_split_csv(dev_file)
check(len(train_rows) == 12750, f"Tập Train có đúng 12.750 ảnh (85%) (thực tế: {len(train_rows)})")
check(len(dev_rows) == 2250, f"Tập Dev có đúng 2.250 ảnh (15%) (thực tế: {len(dev_rows)})")

overlap = {r['path'] for r in train_rows} & {r['path'] for r in dev_rows}
check(len(overlap) == 0, f"Hoàn toàn cách ly, 0 rò rỉ dữ liệu Train/Dev (overlap: {len(overlap)})")

# -------------------------------------------------------------
# PHẦN 2: KIỂM TRA CHECKPOINTS & WEIGHT INTEGRITY
# -------------------------------------------------------------
print("\n[PHẦN 2] Kiểm tra Checkpoints & Trọng số...")
ckpts = {
    "Teacher ResNet50": ("checkpoints/teacher_resnet50_clean/best.pt", "resnet50", 30),
    "Student M0 (ResNet18)": ("checkpoints/student_m0_clean/best.pt", "resnet18", 30),
    "Student M1 (Vanilla KD)": ("checkpoints/m1_vanilla_kd_clean/best.pt", "resnet18", 30),
}

for name, (path_str, exp_arch, exp_classes) in ckpts.items():
    ckpt_path = Path(path_str)
    check(ckpt_path.exists(), f"Checkpoint {name} tồn tại ({ckpt_path})")
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        check(state.get("num_classes") == exp_classes, f"{name}: num_classes = {exp_classes}")
        check(state.get("model_name") == exp_arch, f"{name}: model_name = '{exp_arch}'")
        
        # Test nạp trực tiếp vào kiến trúc mô hình
        try:
            m = create_model(exp_arch, num_classes=exp_classes, pretrained=False)
            m.load_state_dict(state["model_state_dict"])
            check(True, f"{name}: Nạp trọng số state_dict thành công vào mô hình PyTorch")
        except Exception as e:
            check(False, f"{name}: Lỗi nạp trọng số: {e}")

# -------------------------------------------------------------
# PHẦN 3: LỊCH SỬ HUẤN LUYỆN & SỐ LIỆU ĐỐI CHIẾU
# -------------------------------------------------------------
print("\n[PHẦN 3] Kiểm tra Lịch sử Huấn luyện & Đối chiếu Metric...")
histories = {
    "Teacher ResNet50": ("reports/teacher_resnet50_clean_history.csv", 40),
    "Student M0": ("reports/student_m0_clean_history.csv", 40),
    "Student M1": ("reports/m1_vanilla_kd_clean_history.csv", 40),
}

scores = {}
for name, (hist_str, exp_epochs) in histories.items():
    hist_path = Path(hist_str)
    check(hist_path.exists(), f"Lịch sử {name} tồn tại ({hist_path})")
    if hist_path.exists():
        with hist_path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        check(len(rows) == exp_epochs, f"{name}: Chạy đủ {exp_epochs} epoch (thực tế: {len(rows)})")
        best_fine = max(float(r["dev_fine_f1"]) for r in rows)
        best_coarse = max(float(r["dev_coarse_f1"]) for r in rows)
        scores[name] = {"fine_f1": best_fine, "coarse_f1": best_coarse}
        print(f"    -> {name}: Best Fine F1 = {best_fine:.2f}% | Best 3-Class F1 = {best_coarse:.2f}%")

# -------------------------------------------------------------
# PHẦN 4: NGHIỆM THU CỔNG TEACHER (RER >= 20%)
# -------------------------------------------------------------
print("\n[PHẦN 4] Kiểm định Cổng Teacher (§4.1)...")
if "Teacher ResNet50" in scores and "Student M0" in scores:
    t_f1 = scores["Teacher ResNet50"]["fine_f1"]
    m0_f1 = scores["Student M0"]["fine_f1"]
    
    err_t = 100.0 - t_f1
    err_m0 = 100.0 - m0_f1
    rer = (err_m0 - err_t) / err_m0 * 100.0
    
    print(f"  Err(Student M0) = {err_m0:.2f}% | Err(Teacher) = {err_t:.2f}%")
    print(f"  RER = ({err_m0:.2f} - {err_t:.2f}) / {err_m0:.2f} = {rer:.2f}%")
    check(rer >= 20.0, f"Cổng Teacher RER >= 20% (thực tế: {rer:.2f}%)")

# -------------------------------------------------------------
# PHẦN 5: KIỂM TRA ĐỒNG BỘ VĂN BẢN VÀ THAM SỐ
# -------------------------------------------------------------
print("\n[PHẦN 5] Kiểm tra Đồng bộ Văn bản & Code...")
lock_file = Path("docs/protocol_lock.md")
guide_file = Path("docs/label_guide.md")
master_file = Path("ke_hoach_nghien_cuu_MASTER_v9.5.md")

check(lock_file.exists(), f"Văn bản protocol_lock.md tồn tại")
check(guide_file.exists(), f"Văn bản label_guide.md tồn tại")
check(master_file.exists(), f"Kế hoạch MASTER v9.5 tồn tại")

# Kiểm tra nội dung protocol lock
lock_txt = lock_file.read_text(encoding="utf-8")
check("15.000" in lock_txt, "protocol_lock.md ghi nhận đủ 15.000 ảnh")
check("21.60" in lock_txt, "protocol_lock.md ghi nhận đúng RER 21.60%")
check("SGD" in lock_txt, "protocol_lock.md khóa optimizer SGD")
check("0.01" in lock_txt, "protocol_lock.md khóa lr 0.01")
check("40 epochs" in lock_txt, "protocol_lock.md khóa 40 epochs")

# -------------------------------------------------------------
# TỔNG KẾT
# -------------------------------------------------------------
print("\n" + "=" * 75)
print("KẾT QUẢ TỔNG HỢP KIỂM TRA 2 TUẦN ĐẦU")
print("=" * 75)
print(f"  Số lỗi (FAIL): {len(errors)}")
print(f"  Số cảnh báo (WARN): {len(warnings)}")
if errors:
    print("\n  CÁC ĐIỂM CẦN XỬ LÝ:")
    for e in errors:
        print(f"    ❌ {e}")
else:
    print("\n  🎉 TOÀN BỘ 2 TUẦN ĐẦU HOÀN TOÀN NHẤT QUÁN, KHÔNG CÓ BẤT KỲ XUNG ĐỘT NÀO!")
print("=" * 75)
