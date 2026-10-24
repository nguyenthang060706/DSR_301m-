import argparse
import io
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path


def process_taco(
    raw_taco_dir="data/raw/TACO (Trash Annotations in Context)/data",
    output_dir="data/processed/taco_yolo_agnostic",
    split_ratio=0.8,
    val_ratio=0.10,
    seed=20261012,
):
    """Processes TACO dataset into YOLO class-agnostic format with train/val/heldout split.

    Fixes validation leakage: val is carved from the last 10% of the training pool,
    preserving exact heldout membership (OOD test set) with seed=20261012.
    """
    random.seed(seed)
    raw_dir = Path(raw_taco_dir)
    out_dir = Path(output_dir)
    anno_file = raw_dir / "annotations.json"

    if not anno_file.exists():
        print(f"[WARN] File annotations TACO không tồn tại: {anno_file}")
        return

    print("Loading annotations.json...")
    with open(anno_file, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    images = {img["id"]: img for img in coco_data["images"]}
    annotations_by_img = defaultdict(list)
    for ann in coco_data["annotations"]:
        annotations_by_img[ann["image_id"]].append(ann)

    print(f"Total images: {len(images)}")
    print(f"Total bounding boxes: {len(coco_data['annotations'])}")

    # Group image IDs with at least 1 annotation
    valid_img_ids = [img_id for img_id, anns in annotations_by_img.items() if len(anns) > 0]
    print(f"Images with at least 1 box: {len(valid_img_ids)}")

    random.shuffle(valid_img_ids)
    num_train_pool = int(len(valid_img_ids) * split_ratio)
    train_pool = valid_img_ids[:num_train_pool]
    heldout_ids = set(valid_img_ids[num_train_pool:])

    # Val is carved from the last val_ratio (10%) of train_pool
    num_val = int(len(train_pool) * val_ratio)
    train_ids = set(train_pool[:-num_val])
    val_ids = set(train_pool[-num_val:])

    # Verify sets are mutually disjoint
    assert train_ids.isdisjoint(val_ids), "Train and Val must be disjoint!"
    assert train_ids.isdisjoint(heldout_ids), "Train and Heldout must be disjoint!"
    assert val_ids.isdisjoint(heldout_ids), "Val and Heldout must be disjoint!"

    print(f"Split sạch (v10): Train = {len(train_ids)} ({len(train_ids)/len(valid_img_ids):.1%}), "
          f"Val = {len(val_ids)} ({len(val_ids)/len(valid_img_ids):.1%}), "
          f"Heldout = {len(heldout_ids)} ({len(heldout_ids)/len(valid_img_ids):.1%})")

    # Create target YOLO folders
    for split in ["train", "val", "heldout"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {
        "train_boxes": 0, "val_boxes": 0, "heldout_boxes": 0,
        "train_imgs": 0, "val_imgs": 0, "heldout_imgs": 0
    }

    for img_id in valid_img_ids:
        img_info = images[img_id]
        file_name = img_info["file_name"]  # e.g. "batch_1/000000.jpg"
        img_w = img_info["width"]
        img_h = img_info["height"]

        if img_id in train_ids:
            split = "train"
        elif img_id in val_ids:
            split = "val"
        else:
            split = "heldout"

        src_img_path = raw_dir / file_name
        if not src_img_path.exists():
            continue

        # Target file names: batch_1_000000.jpg
        flat_name = file_name.replace("/", "_").replace("\\", "_")
        stem_name = Path(flat_name).stem
        dst_img_path = out_dir / "images" / split / f"{stem_name}{src_img_path.suffix}"
        dst_label_path = out_dir / "labels" / split / f"{stem_name}.txt"

        # Copy image file
        shutil.copy2(src_img_path, dst_img_path)
        stats[f"{split}_imgs"] += 1

        # Write YOLO labels (class_id=0 for all boxes)
        yolo_lines = []
        for ann in annotations_by_img[img_id]:
            x_min, y_min, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue

            # Normalize to [0, 1]
            x_center = (x_min + w / 2.0) / img_w
            y_center = (y_min + h / 2.0) / img_h
            norm_w = w / img_w
            norm_h = h / img_h

            x_center = min(max(x_center, 0.0), 1.0)
            y_center = min(max(y_center, 0.0), 1.0)
            norm_w = min(max(norm_w, 0.0), 1.0)
            norm_h = min(max(norm_h, 0.0), 1.0)

            # Class-agnostic: class_id is always 0 (waste)
            yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
            stats[f"{split}_boxes"] += 1

        with open(dst_label_path, "w", encoding="utf-8") as lf:
            lf.write("\n".join(yolo_lines) + "\n")

    # 1. Create standard training dataset yaml: val points to images/val (NO LEAKAGE)
    yaml_train_content = f"""# TACO Class-Agnostic Dataset Configuration (Training with Clean Validation)
path: {out_dir.resolve().as_posix()}
train: images/train
val: images/val

# Classes
names:
  0: waste
"""
    yaml_path = out_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as yf:
        yf.write(yaml_train_content)

    # 2. Create heldout evaluation dataset yaml: val points to images/heldout
    yaml_heldout_content = f"""# TACO Class-Agnostic Heldout OOD Evaluation Configuration
path: {out_dir.resolve().as_posix()}
train: images/train
val: images/heldout

# Classes
names:
  0: waste
"""
    heldout_yaml_path = out_dir / "taco_heldout.yaml"
    with open(heldout_yaml_path, "w", encoding="utf-8") as yf:
        yf.write(yaml_heldout_content)

    print(f"\n[PASS] Đã tạo thành công dữ liệu TACO sạch tại: {out_dir}")
    print(f"  - Config train/val sạch: {yaml_path}")
    print(f"  - Config heldout OOD:    {heldout_yaml_path}")
    print(f"  - Thống kê ảnh:  Train={stats['train_imgs']}, Val={stats['val_imgs']}, Heldout={stats['heldout_imgs']}")
    print(f"  - Thống kê box:  Train={stats['train_boxes']}, Val={stats['val_boxes']}, Heldout={stats['heldout_boxes']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process TACO dataset into YOLO format")
    parser.add_argument("--raw-dir", default="data/raw/TACO (Trash Annotations in Context)/data", help="Raw TACO data folder")
    parser.add_argument("--out-dir", default="data/processed/taco_yolo_agnostic", help="Output YOLO folder")
    parser.add_argument("--split-ratio", type=float, default=0.8, help="Train pool ratio")
    parser.add_argument("--val-ratio", type=float, default=0.10, help="Val ratio from train pool")
    parser.add_argument("--seed", type=int, default=20261012, help="Random seed")
    args = parser.parse_args()

    process_taco(
        raw_taco_dir=args.raw_dir,
        output_dir=args.out_dir,
        split_ratio=args.split_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
