"""scripts/make_public_split.py

Tạo phân chia Train / Dev-Holdout (15%) phân tầng cố định cho bộ dữ liệu 30 lớp.
Khóa seed: 20261001 (theo Kế hoạch MASTER v9.5 §3.5).
Phân tầng theo cả [Class] và [Domain: default / real_world].
"""

from __future__ import annotations

import csv
from pathlib import Path
from sklearn.model_selection import train_test_split


def make_splits(
    data_root: Path,
    out_dir: Path,
    holdout_ratio: float = 0.15,
    seed: int = 20261001,
):
    print(f"Quét dữ liệu từ: {data_root}")
    rows = []

    # Cấu trúc: data_root / class_name / [default | real_world] / image_file
    for class_dir in sorted(data_root.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        for domain_dir in sorted(class_dir.iterdir()):
            if not domain_dir.is_dir():
                continue
            domain_name = domain_dir.name  # 'default' hoặc 'real_world'
            for img_file in sorted(domain_dir.glob("*.jpg")) + sorted(domain_dir.glob("*.png")):
                rel_path = img_file.relative_to(data_root).as_posix()
                rows.append({
                    "path": rel_path,
                    "class": class_name,
                    "domain": domain_name,
                    # Nhãn phân tầng kết hợp Class + Domain để phân bổ đều
                    "strat_label": f"{class_name}_{domain_name}",
                })

    print(f"Tổng số ảnh quét được: {len(rows):,}")
    assert len(rows) == 15000, f"Kỳ vọng 15.000 ảnh, thực tế có {len(rows)}"

    strat_labels = [r["strat_label"] for r in rows]

    train_rows, dev_rows = train_test_split(
        rows,
        test_size=holdout_ratio,
        random_state=seed,
        stratify=strat_labels,
    )

    for r in train_rows:
        r["split"] = "train"
    for r in dev_rows:
        r["split"] = "dev"

    print(f"Tập Train: {len(train_rows):,} ảnh ({len(train_rows)/len(rows):.1%})")
    print(f"Tập Dev/Holdout: {len(dev_rows):,} ảnh ({len(dev_rows)/len(rows):.1%})")

    out_dir.mkdir(parents=True, exist_ok=True)
    split_file = out_dir / "public_split.csv"
    train_file = out_dir / "public_train.csv"
    dev_file = out_dir / "public_dev.csv"

    fieldnames = ["path", "class", "domain", "split"]

    # 1. Ghi toàn bộ split file
    all_rows = train_rows + dev_rows
    with split_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_rows:
            writer.writerow({k: r[k] for k in fieldnames})

    # 2. Ghi train file
    with train_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in train_rows:
            writer.writerow({k: r[k] for k in fieldnames})

    # 3. Ghi dev file
    with dev_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in dev_rows:
            writer.writerow({k: r[k] for k in fieldnames})

    print(f"\n[PASS] Đã tạo thành công các file phân chia:")
    print(f"  - {split_file}")
    print(f"  - {train_file}")
    print(f"  - {dev_file}")


if __name__ == "__main__":
    data_root = Path("data/raw/household_waste_30")
    out_dir = Path("data/splits")
    make_splits(data_root=data_root, out_dir=out_dir)
