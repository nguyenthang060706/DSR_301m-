import os
import sys
import io
import json
import random
import shutil
from pathlib import Path
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def process_taco(
    raw_taco_dir="data/raw/TACO (Trash Annotations in Context)/data",
    output_dir="data/processed/taco_yolo_agnostic",
    split_ratio=0.8,
    seed=20261012
):
    random.seed(seed)
    raw_dir = Path(raw_taco_dir)
    out_dir = Path(output_dir)
    anno_file = raw_dir / "annotations.json"

    if not anno_file.exists():
        print(f"Error: {anno_file} does not exist!")
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
    num_train = int(len(valid_img_ids) * split_ratio)
    train_ids = set(valid_img_ids[:num_train])
    heldout_ids = set(valid_img_ids[num_train:])

    print(f"Split: Train = {len(train_ids)} images, Heldout = {len(heldout_ids)} images")

    # Create target YOLO folders
    for split in ["train", "heldout"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {"train_boxes": 0, "heldout_boxes": 0, "train_imgs": 0, "heldout_imgs": 0}

    for img_id in valid_img_ids:
        img_info = images[img_id]
        file_name = img_info["file_name"] # e.g. "batch_1/000000.jpg"
        img_w = img_info["width"]
        img_h = img_info["height"]

        split = "train" if img_id in train_ids else "heldout"

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
            # COCO bbox: [x_min, y_min, width, height]
            x_min, y_min, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue

            # Normalize to [0, 1]
            x_center = (x_min + w / 2.0) / img_w
            y_center = (y_min + h / 2.0) / img_h
            norm_w = w / img_w
            norm_h = h / img_h

            # Clip within [0, 1]
            x_center = min(max(x_center, 0.0), 1.0)
            y_center = min(max(y_center, 0.0), 1.0)
            norm_w = min(max(norm_w, 0.0), 1.0)
            norm_h = min(max(norm_h, 0.0), 1.0)

            # Class-agnostic: class_id is always 0 (waste)
            yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
            stats[f"{split}_boxes"] += 1

        with open(dst_label_path, "w", encoding="utf-8") as lf:
            lf.write("\n".join(yolo_lines) + "\n")

    # Create dataset.yaml for YOLO
    yaml_content = f"""# TACO Class-Agnostic Dataset Configuration
path: {out_dir.resolve().as_posix()}
train: images/train
val: images/heldout

# Classes
names:
  0: waste
"""
    yaml_path = out_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as yf:
        yf.write(yaml_content)

    print("\n--- HOÀN TẤT XỬ LÝ TACO CHO DETECTOR D1 ---")
    print(f"Tập Train:   {stats['train_imgs']} ảnh, {stats['train_boxes']} bounding boxes")
    print(f"Tập Heldout: {stats['heldout_imgs']} ảnh, {stats['heldout_boxes']} bounding boxes")
    print(f"File cấu hình YOLO: {yaml_path}")

if __name__ == "__main__":
    process_taco()
