"""scripts/audit_week1.py

Kiểm tra toàn diện logic, tính nhất quán và xung đột
trong toàn bộ pipeline Tuần 1 (V9.5).
"""
from __future__ import annotations
import csv
import os
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, "src")
from dsr.data import load_label_map, read_split_csv

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

errors = []
warnings = []

def check(condition, msg, is_warning=False):
    if condition:
        print(f"  {PASS} {msg}")
    elif is_warning:
        print(f"  {WARN} {msg}")
        warnings.append(msg)
    else:
        print(f"  {FAIL} {msg}")
        errors.append(msg)

# ========================================================
# 1. LABEL MAP INTEGRITY
# ========================================================
print("\n" + "=" * 70)
print("1. KIỂM TRA LABEL MAP (data/mappings/label_map.csv)")
print("=" * 70)

map_file = Path("data/mappings/label_map.csv")
fine_name_to_id, fine_to_coarse_id, fine_classes, coarse_classes = load_label_map(map_file)

check(len(fine_classes) == 30, f"Có đúng 30 fine classes (thực tế: {len(fine_classes)})")
check(len(coarse_classes) == 3, f"Có đúng 3 coarse classes (thực tế: {len(coarse_classes)})")

# Kiểm tra fine_id liên tục 0..29
expected_ids = set(range(30))
actual_ids = set(fine_name_to_id.values())
check(actual_ids == expected_ids, f"fine_id liên tục 0..29 (thực tế: min={min(actual_ids)}, max={max(actual_ids)}, count={len(actual_ids)})")

# Kiểm tra coarse_id chỉ có 0, 1, 2
coarse_ids_used = set(fine_to_coarse_id.values())
check(coarse_ids_used == {0, 1, 2}, f"coarse_id chỉ có {{0, 1, 2}} (thực tế: {coarse_ids_used})")

# Kiểm tra coarse_classes thứ tự
check(coarse_classes[0] == "organic", f"coarse_id=0 là 'organic' (thực tế: '{coarse_classes[0]}')")
check(coarse_classes[1] == "recyclable", f"coarse_id=1 là 'recyclable' (thực tế: '{coarse_classes[1]}')")
check(coarse_classes[2] == "other_landfill", f"coarse_id=2 là 'other_landfill' (thực tế: '{coarse_classes[2]}')")

# Đếm phân bổ fine → coarse
coarse_dist = Counter(fine_to_coarse_id.values())
print(f"\n  Phân bổ fine classes → coarse:")
for cid, cname in enumerate(coarse_classes):
    count = coarse_dist[cid]
    print(f"    {cname} (id={cid}): {count} fine classes")

check(coarse_dist[0] == 4, f"organic có 4 fine classes (thực tế: {coarse_dist[0]})")
check(coarse_dist[1] == 16, f"recyclable có 16 fine classes (thực tế: {coarse_dist[1]})")
check(coarse_dist[2] == 10, f"other_landfill có 10 fine classes (thực tế: {coarse_dist[2]})")

# ========================================================
# 2. DATA SPLIT INTEGRITY
# ========================================================
print("\n" + "=" * 70)
print("2. KIỂM TRA DATA SPLITS (data/splits/)")
print("=" * 70)

data_root = Path("data/raw/household_waste_30")
train_rows = read_split_csv(Path("data/splits/public_train.csv"))
dev_rows = read_split_csv(Path("data/splits/public_dev.csv"))

check(len(train_rows) == 12750, f"Train có 12,750 ảnh (thực tế: {len(train_rows)})")
check(len(dev_rows) == 2250, f"Dev có 2,250 ảnh (thực tế: {len(dev_rows)})")
check(len(train_rows) + len(dev_rows) == 15000, f"Tổng = 15,000 (thực tế: {len(train_rows) + len(dev_rows)})")

# Kiểm tra data leakage
train_paths = {r["path"] for r in train_rows}
dev_paths = {r["path"] for r in dev_rows}
overlap = train_paths & dev_paths
check(len(overlap) == 0, f"Không có data leakage giữa Train và Dev (overlap: {len(overlap)})")

# Kiểm tra mỗi file tồn tại trên ổ đĩa (chọn 100 mẫu ngẫu nhiên)
import random
random.seed(42)
sample_rows = random.sample(train_rows + dev_rows, min(200, len(train_rows) + len(dev_rows)))
missing_files = []
for r in sample_rows:
    fpath = data_root / r["path"]
    if not fpath.exists():
        missing_files.append(str(fpath))
check(len(missing_files) == 0, f"200 ảnh mẫu đều tồn tại trên ổ đĩa (missing: {len(missing_files)})")
if missing_files:
    for mf in missing_files[:5]:
        print(f"    Missing: {mf}")

# Kiểm tra SPLIT CSV header có cột 'class' (dùng cho WasteDataset via get_item → row["class"])
csv_header = list(train_rows[0].keys())
check("class" in csv_header, f"CSV header có cột 'class' (header: {csv_header})")
check("path" in csv_header, f"CSV header có cột 'path' (header: {csv_header})")

# Kiểm tra mọi class trong CSV đều có trong label_map
classes_in_csv = {r["class"] for r in train_rows + dev_rows}
classes_in_map = set(fine_name_to_id.keys())
missing_in_map = classes_in_csv - classes_in_map
extra_in_map = classes_in_map - classes_in_csv
check(len(missing_in_map) == 0, f"Mọi class trong CSV đều có trong label_map (missing: {missing_in_map})")
check(len(extra_in_map) == 0, f"Mọi class trong label_map đều có trong CSV (extra: {extra_in_map})")

# ========================================================
# 3. WASTEDATASET ↔ LABEL MAP CONSISTENCY
# ========================================================
print("\n" + "=" * 70)
print("3. KIỂM TRA WASTEDATASET ↔ LABEL MAP CONSISTENCY")
print("=" * 70)

# WasteDataset dùng row.get("fine_label") or row.get("class") (line 117 data.py)
# CSV dùng cột "class" (không có cột "fine_label")
# → WasteDataset sẽ fallback sang row["class"] → khớp với fine_name_to_id
check(
    "fine_label" not in csv_header and "class" in csv_header,
    f"WasteDataset fallback: CSV không có 'fine_label', có 'class' → đúng logic (line 117 data.py)"
)

# Kiểm tra phân bổ nhãn theo Class trong Train
train_class_dist = Counter(r["class"] for r in train_rows)
dev_class_dist = Counter(r["class"] for r in dev_rows)
print(f"\n  Phân bổ Train (top 5): {train_class_dist.most_common(5)}")
print(f"  Phân bổ Dev (top 5): {dev_class_dist.most_common(5)}")

# Kiểm tra phân tầng: mỗi class có 85% train, 15% dev
print(f"\n  Kiểm tra tỷ lệ phân tầng từng class (kỳ vọng ≈85/15):")
bad_ratio_classes = []
for cname in sorted(fine_name_to_id.keys()):
    n_train = train_class_dist.get(cname, 0)
    n_dev = dev_class_dist.get(cname, 0)
    total = n_train + n_dev
    ratio = n_train / total if total > 0 else 0
    if abs(ratio - 0.85) > 0.03:  # cho phép sai lệch ±3%
        bad_ratio_classes.append((cname, ratio, n_train, n_dev))
if bad_ratio_classes:
    for bc in bad_ratio_classes:
        print(f"    {WARN} {bc[0]}: train_ratio={bc[1]:.2%} ({bc[2]}/{bc[2]+bc[3]})")
check(len(bad_ratio_classes) == 0, f"Mọi class đều có tỷ lệ train ≈85% (vi phạm: {len(bad_ratio_classes)})", is_warning=True)

# ========================================================
# 4. PILOT RECIPE KẾT QUẢ CONSISTENCY
# ========================================================
print("\n" + "=" * 70)
print("4. KIỂM TRA PILOT RECIPE KẾT QUẢ")
print("=" * 70)

pilot_csv = Path("reports/pilot_recipe_summary.csv")
if pilot_csv.exists():
    with pilot_csv.open("r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    check(len(reader) == 2, f"Pilot có đúng 2 dòng kết quả (SGD + AdamW) (thực tế: {len(reader)})")
    sgd_row = [r for r in reader if r["optimizer"] == "sgd"]
    adamw_row = [r for r in reader if r["optimizer"] == "adamw"]
    check(len(sgd_row) == 1 and len(adamw_row) == 1, "Có đúng 1 dòng SGD và 1 dòng AdamW")
    if sgd_row and adamw_row:
        sgd_f1 = float(sgd_row[0]["best_fine_f1"])
        adamw_f1 = float(adamw_row[0]["best_fine_f1"])
        check(sgd_f1 > adamw_f1, f"SGD ({sgd_f1:.2f}) > AdamW ({adamw_f1:.2f}) → quyết định chọn SGD nhất quán")
else:
    check(False, "File pilot_recipe_summary.csv không tồn tại")

# ========================================================
# 5. TEACHER vs STUDENT M0 KẾT QUẢ & RER GATE
# ========================================================
print("\n" + "=" * 70)
print("5. KIỂM TRA TEACHER vs STUDENT M0 & CỔNG RER")
print("=" * 70)

teacher_csv = Path("reports/teacher_resnet50_clean_history.csv")
student_csv = Path("reports/student_m0_clean_history.csv")

if teacher_csv.exists() and student_csv.exists():
    with teacher_csv.open("r", encoding="utf-8") as f:
        teacher_rows = list(csv.DictReader(f))
    with student_csv.open("r", encoding="utf-8") as f:
        student_rows = list(csv.DictReader(f))
    
    check(len(teacher_rows) == 40, f"Teacher chạy đủ 40 epoch (thực tế: {len(teacher_rows)})")
    check(len(student_rows) == 40, f"Student M0 chạy đủ 40 epoch (thực tế: {len(student_rows)})")
    
    teacher_best_f1 = max(float(r["dev_fine_f1"]) for r in teacher_rows)
    student_best_f1 = max(float(r["dev_fine_f1"]) for r in student_rows)
    
    check(teacher_best_f1 > student_best_f1, 
          f"Teacher ({teacher_best_f1:.2f}%) > Student ({student_best_f1:.2f}%) → hợp lý")
    
    # Tính RER
    err_student = 100.0 - student_best_f1
    err_teacher = 100.0 - teacher_best_f1
    rer = (err_student - err_teacher) / err_student * 100.0 if err_student > 0 else 0
    
    print(f"\n  Err(Student M0) = {err_student:.2f}%")
    print(f"  Err(Teacher)    = {err_teacher:.2f}%")
    print(f"  RER = ({err_student:.2f} - {err_teacher:.2f}) / {err_student:.2f} = {rer:.2f}%")
    
    check(rer >= 20.0, f"Cổng RER ≥ 20% PASS (thực tế: {rer:.2f}%)")
    
    # Kiểm tra Teacher KHÔNG dùng label smoothing (train loss phải tiến về ~0)
    teacher_last_train_loss = float(teacher_rows[-1]["train_loss"])
    check(teacher_last_train_loss < 0.15, 
          f"Teacher train loss cuối = {teacher_last_train_loss:.4f} → nhất quán với label_smoothing=0.0")
    
    # Kiểm tra Student CÓ dùng label smoothing (train loss không thể < ~0.7 vì entropy cơ sở cao hơn)
    student_last_train_loss = float(student_rows[-1]["train_loss"])
    check(student_last_train_loss > 0.5,
          f"Student M0 train loss cuối = {student_last_train_loss:.4f} → nhất quán với label_smoothing=0.1")
else:
    if not teacher_csv.exists():
        check(False, "File teacher_resnet50_clean_history.csv không tồn tại")
    if not student_csv.exists():
        check(False, "File student_m0_clean_history.csv không tồn tại")

# ========================================================
# 6. CHECKPOINT FILE INTEGRITY
# ========================================================
print("\n" + "=" * 70)
print("6. KIỂM TRA CHECKPOINT FILES")
print("=" * 70)

teacher_ckpt = Path("checkpoints/teacher_resnet50_clean/best.pt")
student_ckpt = Path("checkpoints/student_m0_clean/best.pt")

check(teacher_ckpt.exists(), f"Teacher best.pt tồn tại ({teacher_ckpt})")
check(student_ckpt.exists(), f"Student M0 best.pt tồn tại ({student_ckpt})")

if teacher_ckpt.exists():
    import torch
    t_ckpt = torch.load(teacher_ckpt, map_location="cpu", weights_only=False)
    check(t_ckpt.get("model_name") == "resnet50", 
          f"Teacher checkpoint ghi model_name='resnet50' (thực tế: '{t_ckpt.get('model_name')}')")
    check(t_ckpt.get("num_classes") == 30,
          f"Teacher checkpoint ghi num_classes=30 (thực tế: {t_ckpt.get('num_classes')})")
    
if student_ckpt.exists():
    s_ckpt = torch.load(student_ckpt, map_location="cpu", weights_only=False)
    check(s_ckpt.get("model_name") == "resnet18",
          f"Student M0 checkpoint ghi model_name='resnet18' (thực tế: '{s_ckpt.get('model_name')}')")
    check(s_ckpt.get("num_classes") == 30,
          f"Student M0 checkpoint ghi num_classes=30 (thực tế: {s_ckpt.get('num_classes')})")

# ========================================================
# 7. CROSS-CHECK: TRANSFORMS ORDER
# ========================================================
print("\n" + "=" * 70)
print("7. KIỂM TRA THỨ TỰ TRANSFORMS")
print("=" * 70)

from dsr.data import create_transforms
train_t = create_transforms(224, train=True, use_trivial_augment=True)
val_t = create_transforms(224, train=False)

train_names = [type(t).__name__ for t in train_t.transforms]
val_names = [type(t).__name__ for t in val_t.transforms]
print(f"  Train transforms: {train_names}")
print(f"  Val transforms: {val_names}")

# TrivialAugment PHẢI trước ToTensor (vì nó hoạt động trên PIL Image)
ta_idx = train_names.index("TrivialAugmentWide") if "TrivialAugmentWide" in train_names else -1
tt_idx = train_names.index("ToTensor") if "ToTensor" in train_names else -1
check(ta_idx >= 0 and tt_idx >= 0 and ta_idx < tt_idx,
      f"TrivialAugmentWide (idx={ta_idx}) trước ToTensor (idx={tt_idx}) → đúng thứ tự")

# Normalize phải là cuối cùng
check(train_names[-1] == "Normalize", f"Train: Normalize là transform cuối cùng")
check(val_names[-1] == "Normalize", f"Val: Normalize là transform cuối cùng")

# Val không có augmentation
check("TrivialAugmentWide" not in val_names, "Val KHÔNG có TrivialAugmentWide → đúng")
check("RandomResizedCrop" not in val_names, "Val KHÔNG có RandomResizedCrop → đúng")

# ========================================================
# 8. CROSS-CHECK: KD LOSS DIRECTION
# ========================================================
print("\n" + "=" * 70)
print("8. KIỂM TRA KD LOSS DIRECTION (kd_loss)")
print("=" * 70)

import torch
torch.manual_seed(42)
student_logits = torch.randn(4, 30)
teacher_logits = torch.randn(4, 30) + 2.0  # teacher tự tin hơn
from dsr.losses import kd_loss

loss_val = kd_loss(student_logits, teacher_logits, tau=3.0)
check(loss_val.item() > 0, f"KD loss > 0 khi student ≠ teacher (loss = {loss_val.item():.4f})")

# KL(T||S) nên giảm khi student → teacher
loss_same = kd_loss(teacher_logits, teacher_logits, tau=3.0)
check(loss_same.item() < loss_val.item(), 
      f"KD loss giảm khi student = teacher (same={loss_same.item():.6f} < diff={loss_val.item():.4f})")

# Kiểm tra gradient flows qua student, KHÔNG qua teacher
student_logits_grad = torch.randn(4, 30, requires_grad=True)
teacher_logits_grad = torch.randn(4, 30, requires_grad=True)
loss_g = kd_loss(student_logits_grad, teacher_logits_grad, tau=3.0)
loss_g.backward()
check(student_logits_grad.grad is not None, "Gradient flows qua student logits → đúng")
check(teacher_logits_grad.grad is None, "Gradient KHÔNG flows qua teacher logits (.detach()) → đúng")

# ========================================================
# TỔNG KẾT
# ========================================================
print("\n" + "=" * 70)
print("TỔNG KẾT KIỂM TRA TUẦN 1")
print("=" * 70)
print(f"  Tổng PASS: phía trên")
print(f"  Tổng FAIL: {len(errors)}")
print(f"  Tổng WARN: {len(warnings)}")
if errors:
    print(f"\n  LỖI CẦN SỬA:")
    for e in errors:
        print(f"    ❌ {e}")
if warnings:
    print(f"\n  CẢNH BÁO:")
    for w in warnings:
        print(f"    ⚠️  {w}")
if not errors:
    print(f"\n  ✅ TOÀN BỘ KIỂM TRA TUẦN 1 ĐỀU PASS. Không phát hiện lỗi logic hoặc xung đột.")
