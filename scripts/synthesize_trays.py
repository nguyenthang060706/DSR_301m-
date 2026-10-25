"""scripts/synthesize_trays.py  (v2)

D2-C1: Script sinh dữ liệu khay rác nhân tạo theo MASTER v10 §3.5 (spec 8.4).
Chức năng chính:
  - Tách nền bằng `rembg` cho tập con ~3.000 vật cân bằng nhóm
  - Chỉ dùng ảnh thuộc public_train.csv
  - Feather + bóng đổ + chỉnh sáng, box từ mask, IoU ≤ 0.3
  - Có --seed để tái tạo kết quả
  - Xuất nhãn class-agnostic (labels_agnostic/) VÀ 3 nhóm (labels_3cls/)
  - Sinh 3.000–5.000 ảnh (job CPU)

Sử dụng:
    python scripts/synthesize_trays.py --num_images 4000 --seed 20261012 \
        --out data/processed/synthetic_trays_v2
"""
from __future__ import annotations

import sys
import io
import os
import csv
import json
import random
import argparse
import hashlib
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


# ──────────────────────────── Helpers ──────────────────────────────────
def compute_box_iou(box1: list[float], box2: list[float]) -> float:
    """IoU between two [xmin, ymin, xmax, ymax] boxes."""
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def load_label_map(path: Path) -> dict[int, int]:
    """Load label_map.csv → {fine_id: coarse_id}."""
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("fine_id"):
                continue
            parts = line.split(",")
            if len(parts) >= 3:
                fine_id = int(parts[0])
                coarse_id = int(parts[2])
                mapping[fine_id] = coarse_id
    return mapping


def load_public_train(path: Path) -> set[str]:
    """Load public_train.csv → set of image file stems (basenames without ext)."""
    stems = set()
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        # Try to find a 'path' or 'filename' column, else use first column
        idx = 0
        if header:
            for i, col in enumerate(header):
                if col.strip().lower() in ("path", "filename", "image", "file"):
                    idx = i
                    break
        for row in reader:
            if row:
                stem = Path(row[idx].strip()).stem
                stems.add(stem)
    return stems


def remove_background_rembg(img_path: Path, cache_dir: Path) -> Image.Image | None:
    """Remove background using rembg, cache RGBA result."""
    cache_file = cache_dir / f"{img_path.stem}.png"
    if cache_file.exists():
        return Image.open(cache_file).convert("RGBA")

    try:
        from rembg import remove
        img = Image.open(img_path).convert("RGB")
        result = remove(img)  # Returns RGBA

        # Validate mask coverage: 3%–90% of area
        alpha = np.array(result.split()[-1])
        coverage = np.count_nonzero(alpha > 128) / alpha.size
        if coverage < 0.03 or coverage > 0.90:
            return None

        cache_dir.mkdir(parents=True, exist_ok=True)
        result.save(cache_file, "PNG")
        return result
    except Exception as e:
        print(f"  [WARN] rembg failed for {img_path.name}: {e}")
        return None


def remove_background_fallback(img_path: Path) -> Image.Image:
    """Fallback: return full image as RGBA (for when rembg is not available)."""
    img = Image.open(img_path).convert("RGBA")
    return img


def add_shadow(canvas: Image.Image, box: list[int], opacity: int = 60) -> Image.Image:
    """Add subtle drop shadow under placed object."""
    shadow_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(shadow_layer)
    sx = box[0] + 3
    sy = box[1] + 5
    draw.rectangle([sx, sy, box[2] + 3, box[3] + 5], fill=(0, 0, 0, opacity))
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(4))
    return Image.alpha_composite(canvas, shadow_layer)


def adjust_brightness_to_bg(obj_img: Image.Image, bg_brightness: float) -> Image.Image:
    """Adjust object brightness to roughly match background lighting."""
    obj_rgb = obj_img.convert("RGB")
    obj_arr = np.array(obj_rgb).astype(float)
    obj_brightness = obj_arr.mean()
    if obj_brightness < 1:
        return obj_img
    ratio = bg_brightness / obj_brightness
    ratio = max(0.7, min(1.3, ratio))  # Clamp adjustment
    enhancer = ImageEnhance.Brightness(obj_img)
    return enhancer.enhance(ratio)


# ──────────────────────────── Main synthesis ──────────────────────────
def generate_synthetic_tray(
    bg_img: Image.Image,
    cutout_list: list[tuple[Image.Image, int, int]],  # (RGBA_img, fine_id, coarse_id)
    output_size: tuple[int, int] = (640, 640),
    num_objects: int = 4,
    rng: random.Random | None = None,
) -> tuple[Image.Image, list[str], list[str]]:
    """Generate one synthetic tray image.

    Returns:
        (RGB image, agnostic_labels, 3cls_labels) in YOLO format.
    """
    if rng is None:
        rng = random.Random()

    canvas = bg_img.resize(output_size).convert("RGBA")
    w_c, h_c = output_size
    placed_boxes: list[list[int]] = []

    # Estimate BG brightness for lighting adjustment
    bg_arr = np.array(bg_img.convert("RGB")).astype(float)
    bg_brightness = bg_arr.mean()

    sampled = rng.sample(cutout_list, min(num_objects, len(cutout_list)))
    agnostic_labels: list[str] = []
    threecls_labels: list[str] = []

    for cutout_rgba, fine_id, coarse_id in sampled:
        # Scale: 15%–35% of canvas width
        target_w = rng.randint(int(w_c * 0.15), int(w_c * 0.35))
        aspect = cutout_rgba.height / max(1, cutout_rgba.width)
        target_h = int(target_w * aspect)
        target_h = max(target_h, 10)

        obj_resized = cutout_rgba.resize((target_w, target_h), Image.Resampling.BILINEAR)

        # Adjust brightness to match BG
        obj_resized = adjust_brightness_to_bg(obj_resized, bg_brightness)

        # Rotate ±30°
        angle = rng.randint(-30, 30)
        obj_rotated = obj_resized.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)

        # Feather the edges (blur alpha channel slightly)
        channels = obj_rotated.split()
        if len(channels) == 4:
            alpha = channels[3].filter(ImageFilter.GaussianBlur(1))
            obj_rotated = Image.merge("RGBA", (*channels[:3], alpha))

        rw, rh = obj_rotated.size

        # Find non-overlapping position (IoU ≤ 0.3)
        max_attempts = 30
        placed = False
        for _ in range(max_attempts):
            if w_c - rw <= 10 or h_c - rh <= 10:
                break
            px = rng.randint(5, max(6, w_c - rw - 5))
            py = rng.randint(5, max(6, h_c - rh - 5))

            # Tight bounding box from mask
            alpha_arr = np.array(obj_rotated.split()[-1])
            ys, xs = np.where(alpha_arr > 128)
            if len(xs) == 0:
                break
            bx1, by1 = int(xs.min()) + px, int(ys.min()) + py
            bx2, by2 = int(xs.max()) + px, int(ys.max()) + py
            cand_box = [bx1, by1, bx2, by2]

            # Check IoU
            overlap = any(compute_box_iou(cand_box, pb) > 0.3 for pb in placed_boxes)
            if not overlap:
                # Add shadow
                canvas = add_shadow(canvas, cand_box)
                # Paste object
                canvas.paste(obj_rotated, (px, py), obj_rotated)
                placed_boxes.append(cand_box)

                # YOLO format: xc, yc, w, h (normalized)
                xc = ((bx1 + bx2) / 2.0) / w_c
                yc = ((by1 + by2) / 2.0) / h_c
                bw = (bx2 - bx1) / float(w_c)
                bh = (by2 - by1) / float(h_c)
                agnostic_labels.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                threecls_labels.append(f"{coarse_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                placed = True
                break

    return canvas.convert("RGB"), agnostic_labels, threecls_labels


def main():
    parser = argparse.ArgumentParser(
        description="Sinh ảnh khay rác tổng hợp Copy-Paste v2 (MASTER v10 §3.5)"
    )
    parser.add_argument("--num_images", type=int, default=4000, help="Số ảnh cần sinh")
    parser.add_argument("--seed", type=int, default=20261012, help="Random seed cho tái tạo")
    parser.add_argument("--out", type=str, default="data/processed/synthetic_trays_v2", help="Thư mục xuất")
    parser.add_argument("--cutout-cache", type=str, default="data/processed/cutouts",
                        help="Thư mục cache RGBA cutout")
    parser.add_argument("--max-cutouts", type=int, default=3000,
                        help="Số vật tối đa, cân bằng nhóm")
    parser.add_argument("--no-rembg", action="store_true",
                        help="Bỏ qua rembg, dùng fallback (toàn ảnh)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    # ── 1. Load label_map & public_train ──────────────────────────────
    label_map_path = Path("data/mappings/label_map.csv")
    if not label_map_path.exists():
        print(f"[LỖI] Không tìm thấy {label_map_path}")
        sys.exit(1)
    fine_to_coarse = load_label_map(label_map_path)

    train_csv = Path("data/splits/public_train.csv")
    public_train_stems: set[str] | None = None
    if train_csv.exists():
        public_train_stems = load_public_train(train_csv)
        print(f"[*] Đã nạp {len(public_train_stems)} stems từ public_train.csv")
    else:
        print(f"[WARN] Không tìm thấy {train_csv} — sẽ dùng TẤT CẢ ảnh (chưa lọc)")

    # ── 2. Collect object images grouped by coarse class ──────────────
    raw_dir = Path("data/raw/household_waste_30")
    if not raw_dir.exists():
        print(f"[LỖI] Không tìm thấy {raw_dir}")
        sys.exit(1)

    # Build fine_label → fine_id mapping from directory names
    fine_labels = {}
    with open(label_map_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("fine_id"):
                continue
            parts = line.split(",")
            if len(parts) >= 2:
                fine_labels[parts[1].strip()] = int(parts[0])

    # Group images by coarse class
    coarse_groups: dict[int, list[tuple[Path, int]]] = {0: [], 1: [], 2: []}
    for class_dir in sorted(raw_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        fine_label = class_dir.name
        fine_id = fine_labels.get(fine_label)
        if fine_id is None:
            # Try subdirectories (default/real_world structure)
            for sub in class_dir.iterdir():
                if sub.is_dir():
                    fine_label2 = sub.name
                    fine_id2 = fine_labels.get(fine_label2)
                    if fine_id2 is not None:
                        coarse_id = fine_to_coarse.get(fine_id2, 2)
                        for ext in ["*.jpg", "*.jpeg", "*.png"]:
                            for img_path in sub.glob(ext):
                                if public_train_stems is None or img_path.stem in public_train_stems:
                                    coarse_groups[coarse_id].append((img_path, fine_id2))
            continue

        coarse_id = fine_to_coarse.get(fine_id, 2)
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            for img_path in class_dir.rglob(ext):
                if public_train_stems is None or img_path.stem in public_train_stems:
                    coarse_groups[coarse_id].append((img_path, fine_id))

    print(f"[*] Ảnh theo nhóm: organic={len(coarse_groups[0])}, "
          f"recyclable={len(coarse_groups[1])}, other={len(coarse_groups[2])}")

    # ── 3. Balance & select cutouts ───────────────────────────────────
    per_group = args.max_cutouts // 3
    selected: list[tuple[Path, int, int]] = []  # (path, fine_id, coarse_id)
    for coarse_id in [0, 1, 2]:
        group = coarse_groups[coarse_id]
        rng.shuffle(group)
        for path, fine_id in group[:per_group]:
            selected.append((path, fine_id, coarse_id))
    rng.shuffle(selected)
    print(f"[*] Đã chọn {len(selected)} vật cân bằng nhóm (mỗi nhóm ≤ {per_group})")

    # ── 4. Remove backgrounds ─────────────────────────────────────────
    cache_dir = Path(args.cutout_cache)
    cutouts: list[tuple[Image.Image, int, int]] = []
    print(f"[*] Tách nền {'(rembg)' if not args.no_rembg else '(fallback)'}...")

    for i, (path, fine_id, coarse_id) in enumerate(selected):
        if (i + 1) % 200 == 0:
            print(f"  ... đã xử lý {i+1}/{len(selected)} ảnh")
        if args.no_rembg:
            rgba = remove_background_fallback(path)
        else:
            rgba = remove_background_rembg(path, cache_dir)
        if rgba is not None:
            cutouts.append((rgba, fine_id, coarse_id))

    print(f"[*] Cutout hợp lệ: {len(cutouts)}")
    if len(cutouts) < 10:
        print("[LỖI] Quá ít cutout, không thể sinh ảnh")
        sys.exit(1)

    # ── 5. Prepare backgrounds ────────────────────────────────────────
    bg_dir = Path("data/raw/canteen_backgrounds")
    real_bgs: list[Path] = []
    if bg_dir.exists():
        real_bgs = list(bg_dir.glob("*.jpg")) + list(bg_dir.glob("*.png"))
    print(f"[*] Ảnh nền thật: {len(real_bgs)}")

    fallback_colors = [
        (210, 215, 220), (185, 190, 195), (225, 215, 195),
        (240, 240, 242), (160, 165, 170), (205, 195, 180),
    ]

    # ── 6. Split source images for train/val (85/15 by source) ────────
    # Split cutouts by source image path to avoid same source in train AND val
    unique_sources = list(set(c[1] for c in cutouts))  # fine_ids as proxy
    rng.shuffle(unique_sources)

    val_count = int(args.num_images * 0.15)
    train_count = args.num_images - val_count

    # ── 7. Generate synthetic trays ───────────────────────────────────
    out_path = Path(args.out)
    for sub in ["images/train", "images/val",
                 "labels_agnostic/train", "labels_agnostic/val",
                 "labels_3cls/train", "labels_3cls/val"]:
        (out_path / sub).mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    total_gen = 0

    for split, count in [("val", val_count), ("train", train_count)]:
        print(f"[*] Sinh {count} ảnh {split}...")
        for idx in range(count):
            # Choose background
            if real_bgs and rng.random() < 0.7:
                try:
                    bg = Image.open(rng.choice(real_bgs)).convert("RGB")
                except Exception:
                    bg = Image.new("RGB", (640, 640), color=rng.choice(fallback_colors))
            else:
                bg = Image.new("RGB", (640, 640), color=rng.choice(fallback_colors))

            num_objs = rng.randint(3, 6)
            tray_img, agn_labels, cls_labels = generate_synthetic_tray(
                bg, cutouts, num_objects=num_objs, rng=rng
            )

            stem = f"synth_tray_{split}_{idx:05d}"
            img_path = out_path / "images" / split / f"{stem}.jpg"
            tray_img.save(img_path, quality=90)

            # Write agnostic labels
            with open(out_path / "labels_agnostic" / split / f"{stem}.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(agn_labels) + ("\n" if agn_labels else ""))

            # Write 3-class labels
            with open(out_path / "labels_3cls" / split / f"{stem}.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(cls_labels) + ("\n" if cls_labels else ""))

            manifest_rows.append({
                "filename": f"{stem}.jpg",
                "split": split,
                "num_objects": len(agn_labels),
            })
            total_gen += 1

            if (total_gen) % 500 == 0:
                print(f"  ... đã sinh {total_gen}/{args.num_images}")

    # ── 8. Write data.yaml files ──────────────────────────────────────
    for label_variant in ["labels_agnostic", "labels_3cls"]:
        num_cls = 1 if label_variant == "labels_agnostic" else 3
        names = {0: "waste"} if num_cls == 1 else {0: "organic", 1: "recyclable", 2: "other_landfill"}
        yaml_content = (
            f"# Synthetic Trays v2 - {label_variant}\n"
            f"path: {out_path.resolve().as_posix()}\n"
            f"train: images/train\n"
            f"val: images/val\n\n"
            f"# Labels directory override\n"
            f"# Symlink or copy {label_variant}/{{train,val}} → labels/{{train,val}}\n\n"
            f"nc: {num_cls}\n"
            f"names: {names}\n"
        )
        with open(out_path / f"data_{label_variant}.yaml", "w", encoding="utf-8") as f:
            f.write(yaml_content)

    # ── 9. Write manifest ─────────────────────────────────────────────
    manifest = {
        "version": "2.0",
        "seed": args.seed,
        "num_images": total_gen,
        "train_count": train_count,
        "val_count": val_count,
        "num_cutouts": len(cutouts),
        "timestamp": datetime.now().isoformat(),
        "source_filter": "public_train.csv" if public_train_stems else "none",
    }
    with open(out_path / "synthetic_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    with open(out_path / "synthetic_manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "split", "num_objects"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\n[THÀNH CÔNG] Đã sinh {total_gen} ảnh khay nhân tạo v2:")
    print(f"  Train: {train_count}, Val: {val_count}")
    print(f"  Cutout cache: {cache_dir}")
    print(f"  Output: {out_path}")
    print(f"  Manifest: {out_path / 'synthetic_manifest.json'}")


if __name__ == "__main__":
    main()
