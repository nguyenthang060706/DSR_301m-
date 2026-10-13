"""scripts/split_canteen_by_cluster.py

Script phân chia dữ liệu khay canteen thật vào Train / Val / Test theo cụm ngày/ca (MASTER v10.0 §3.3).
Nguyên tắc:
- Phân chia theo cụm (ngày/ca), tuyệt đối không chia ngẫu nhiên từng ảnh.
- Dữ liệu thuộc cùng một ngày/ca chỉ được nằm trọn trong 1 tập duy nhất.
- Tập Test Canteen giữ nguyên các ngày riêng biệt hoàn toàn (đóng băng đánh giá).
"""

import sys
import io
import csv
import shutil
import argparse
from pathlib import Path
import random

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description="Chia dữ liệu khay canteen theo cụm ngày/ca")
    parser.add_argument("--metadata", type=str, default="data/canteen_metadata.csv", help="Đường dẫn file metadata")
    parser.add_argument("--images_dir", type=str, default="data/raw/canteen_trays", help="Thư mục ảnh gốc")
    parser.add_argument("--labels_dir", type=str, default="data/raw/canteen_labels", help="Thư mục nhãn txt đã gán")
    parser.add_argument("--out_dir", type=str, default="data/processed/canteen_yolo", help="Thư mục đích chuẩn YOLO")
    parser.add_argument("--seed", type=int, default=20261012, help="Seed ngẫu nhiên")
    args = parser.parse_args()

    meta_path = Path(args.metadata)
    if not meta_path.exists():
        print(f"[CẢNH BÁO] Chưa tìm thấy file metadata: {meta_path}")
        print("Khi gán nhãn xong, hãy cập nhật metadata và chạy script này.")
        return

    random.seed(args.seed)
    with open(meta_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    if not reader:
        print("[LỖI] File metadata trống!")
        return

    # Gom nhóm theo cluster_id (ngày/ca)
    clusters = {}
    for row in reader:
        cid = row.get("cluster_id", "default_cluster")
        clusters.setdefault(cid, []).append(row)

    cluster_keys = list(clusters.keys())
    random.shuffle(cluster_keys)
    print(f"[*] Tìm thấy {len(cluster_keys)} cụm ngày/ca: {cluster_keys}")

    # Tỉ lệ phân chia cụm: ~60% Train, ~15% Val, ~25% Test
    n = len(cluster_keys)
    if n >= 4:
        n_test = max(1, int(round(n * 0.25)))
        n_val = max(1, int(round(n * 0.15)))
        test_clusters = set(cluster_keys[:n_test])
        val_clusters = set(cluster_keys[n_test:n_test + n_val])
        train_clusters = set(cluster_keys[n_test + n_val:])
    else:
        # Nếu ít cụm (đợt đầu)
        test_clusters = set([cluster_keys[0]])
        val_clusters = set([cluster_keys[1]]) if n > 1 else set()
        train_clusters = set(cluster_keys[2:]) if n > 2 else set([cluster_keys[0]])

    print(f"[*] Phân cụm: Train ({len(train_clusters)}) | Val ({len(val_clusters)}) | Test ({len(test_clusters)})")

    out_base = Path(args.out_dir)
    src_img = Path(args.images_dir)
    src_lbl = Path(args.labels_dir)

    copied = {"train": 0, "val": 0, "test": 0}

    for cid, rows in clusters.items():
        if cid in test_clusters:
            split = "test"
        elif cid in val_clusters:
            split = "val"
        else:
            split = "train"

        dest_img_dir = out_base / "images" / split
        dest_lbl_dir = out_base / "labels" / split
        dest_img_dir.mkdir(parents=True, exist_ok=True)
        dest_lbl_dir.mkdir(parents=True, exist_ok=True)

        for r in rows:
            img_file = r["image_id"]
            lbl_file = Path(img_file).stem + ".txt"

            img_p = src_img / img_file
            lbl_p = src_lbl / lbl_file

            if img_p.exists():
                shutil.copy2(img_p, dest_img_dir / img_file)
            if lbl_p.exists():
                shutil.copy2(lbl_p, dest_lbl_dir / lbl_file)
            copied[split] += 1

    print(f"[THÀNH CÔNG] Phân chia hoàn tất:")
    print(f"  - Train: {copied['train']} ảnh")
    print(f"  - Val:   {copied['val']} ảnh")
    print(f"  - Test:  {copied['test']} ảnh")

if __name__ == "__main__":
    main()
