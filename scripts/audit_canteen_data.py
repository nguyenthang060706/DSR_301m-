"""scripts/audit_canteen_data.py

Kiểm kê và audit toàn diện dữ liệu canteen theo Kế hoạch MASTER v10.0 (§8.1 & D1-C1).
Spec:
--metadata data/canteen_metadata.csv --images data/raw/canteen_trays --labels data/raw/canteen_labels --out reports/canteen_audit.csv [--no-hash]
Xuất định dạng: metric,value,note
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

CLUSTER_REGEX = re.compile(r"^\d{8}_[a-z]+$")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def compute_md5(file_path: Path) -> str:
    hasher = hashlib.md5()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def audit_canteen(
    metadata_path: Path | None,
    images_dir: Path,
    labels_dir: Path | None,
    out_csv: Path,
    no_hash: bool = False,
) -> list[dict[str, str]]:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    # 1. Kiểm tra tồn tại thư mục ảnh
    if not images_dir.exists() or not any(images_dir.iterdir()):
        print("[WARN] chưa có dữ liệu ảnh canteen tại:", images_dir)
        rows.extend([
            {"metric": "total_images", "value": "0", "note": "Thư mục ảnh trống hoặc không tồn tại"},
            {"metric": "total_labels", "value": "0", "note": "Chưa có nhãn"},
            {"metric": "status", "value": "EMPTY", "note": "[WARN] chưa có dữ liệu"},
        ])
        _write_csv(out_csv, rows)
        return rows

    # 2. Quét file ảnh
    image_files = sorted([
        f for f in images_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ])
    total_images = len(image_files)
    image_stems = {f.stem: f for f in image_files}
    image_names = {f.name: f for f in image_files}

    if total_images == 0:
        print("[WARN] chưa có dữ liệu ảnh hợp lệ")
        rows.extend([
            {"metric": "total_images", "value": "0", "note": "Không tìm thấy file ảnh hợp lệ"},
            {"metric": "status", "value": "EMPTY", "note": "[WARN] chưa có dữ liệu"},
        ])
        _write_csv(out_csv, rows)
        return rows

    # 3. Quét metadata nếu có
    metadata_by_id: dict[str, dict[str, str]] = {}
    invalid_clusters: list[str] = []
    missing_annotators: list[str] = []
    ambiguous_count = 0
    clusters_from_meta: set[str] = set()

    if metadata_path and metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                img_id = r.get("image_id", "").strip()
                if not img_id:
                    continue
                metadata_by_id[img_id] = r
                cid = r.get("cluster_id", "").strip()
                if cid:
                    clusters_from_meta.add(cid)
                    if not CLUSTER_REGEX.match(cid):
                        invalid_clusters.append(f"{img_id}:{cid}")
                ann1 = r.get("annotator_1", "").strip()
                ann2 = r.get("annotator_2", "").strip()
                if not ann1 or not ann2:
                    missing_annotators.append(img_id)
                amb = r.get("ambiguous", "0").strip()
                if amb in {"1", "true", "True"}:
                    ambiguous_count += 1

    # 4. Quét labels
    label_stems: dict[str, Path] = {}
    if labels_dir and labels_dir.exists():
        for lf in labels_dir.glob("*.txt"):
            label_stems[lf.stem] = lf

    images_with_label = 0
    empty_label_files = 0
    missing_label_images = 0
    orphan_labels = [stem for stem in label_stems if stem not in image_stems]

    total_boxes = 0
    boxes_by_group: Counter[int] = Counter()

    for stem, img_path in image_stems.items():
        if stem in label_stems:
            lbl_file = label_stems[stem]
            lines = [l.strip() for l in lbl_file.read_text(encoding="utf-8").splitlines() if l.strip()]
            if not lines:
                empty_label_files += 1
            else:
                images_with_label += 1
                for line in lines:
                    parts = line.split()
                    if parts:
                        try:
                            cls_id = int(float(parts[0]))
                            boxes_by_group[cls_id] += 1
                            total_boxes += 1
                        except ValueError:
                            pass
        else:
            missing_label_images += 1

    # 5. Phân tích cụm và ngày
    clusters = clusters_from_meta.copy()
    if not clusters:
        # Dự đoán từ image stem / name nếu không có metadata
        for stem in image_stems:
            parts = stem.split("_")
            if len(parts) >= 2 and len(parts[0]) == 8 and parts[0].isdigit():
                clusters.add(f"{parts[0]}_{parts[1]}")

    dates = {c[:8] for c in clusters if len(c) >= 8 and c[:8].isdigit()}
    trays_per_cluster: defaultdict[str, int] = defaultdict(int)
    for stem in image_stems:
        # Tìm cluster
        cid = None
        if stem in metadata_by_id:
            cid = metadata_by_id[stem].get("cluster_id")
        elif f"{stem}.jpg" in metadata_by_id:
            cid = metadata_by_id[f"{stem}.jpg"].get("cluster_id")
        elif f"{stem}.png" in metadata_by_id:
            cid = metadata_by_id[f"{stem}.png"].get("cluster_id")

        if not cid:
            parts = stem.split("_")
            if len(parts) >= 2 and len(parts[0]) == 8 and parts[0].isdigit():
                cid = f"{parts[0]}_{parts[1]}"

        if cid:
            trays_per_cluster[cid] += 1

    # 6. Hash & trùng lặp
    dup_md5_count = 0
    phash_near_count = 0

    if not no_hash:
        md5_map: defaultdict[str, list[str]] = defaultdict(list)
        for img_path in image_files:
            h = compute_md5(img_path)
            md5_map[h].append(img_path.name)
        dup_md5_count = sum(len(names) - 1 for names in md5_map.values() if len(names) > 1)

        # pHash nếu có imagehash
        try:
            import imagehash
            from PIL import Image
            phashes = []
            for img_path in image_files:
                try:
                    with Image.open(img_path) as im:
                        ph = imagehash.phash(im)
                        phashes.append((img_path.name, ph))
                except Exception:
                    pass
            for i in range(len(phashes)):
                for j in range(i + 1, len(phashes)):
                    if phashes[i][1] - phashes[j][1] <= 5:
                        phash_near_count += 1
        except ImportError:
            phash_near_count = -1  # Chưa cài imagehash

    # 7. Định dạng kết quả audit theo spec
    rows.append({"metric": "total_images", "value": str(total_images), "note": "Tổng số ảnh khay quét được"})
    rows.append({"metric": "images_with_labels", "value": str(images_with_label), "note": "Số ảnh có nhãn box"})
    rows.append({"metric": "empty_labels", "value": str(empty_label_files), "note": "Số file nhãn rỗng (0 box)"})
    rows.append({"metric": "missing_labels", "value": str(missing_label_images), "note": "Số ảnh chưa có file nhãn"})
    rows.append({"metric": "orphan_labels", "value": str(len(orphan_labels)), "note": "File nhãn không có ảnh tương ứng"})
    rows.append({"metric": "total_boxes", "value": str(total_boxes), "note": "Tổng số bounding box"})

    # Nhóm nhãn
    if total_boxes > 0:
        if set(boxes_by_group.keys()) == {0} and len(boxes_by_group) == 1:
            box_note = "không biết nhóm (mọi box là class 0, class-agnostic)"
            rows.append({"metric": "boxes_by_group", "value": f"class_0:{boxes_by_group[0]}", "note": box_note})
        else:
            group_strs = [f"group_{k}:{v}" for k, v in sorted(boxes_by_group.items())]
            rows.append({"metric": "boxes_by_group", "value": "; ".join(group_strs), "note": "Phân bổ box theo nhóm (0: hữu cơ, 1: tái chế, 2: còn lại)"})
    else:
        rows.append({"metric": "boxes_by_group", "value": "none", "note": "Chưa có box"})

    rows.append({"metric": "num_clusters", "value": str(len(clusters)), "note": "Số cụm nhận diện được"})
    rows.append({"metric": "num_dates", "value": str(len(dates)), "note": "Số ngày chụp khác nhau"})

    if trays_per_cluster:
        min_trays = min(trays_per_cluster.values())
        max_trays = max(trays_per_cluster.values())
        mean_trays = sum(trays_per_cluster.values()) / len(trays_per_cluster)
        rows.append({"metric": "trays_per_cluster", "value": f"min={min_trays},max={max_trays},mean={mean_trays:.1f}", "note": "Số khay trên mỗi cụm"})
    else:
        rows.append({"metric": "trays_per_cluster", "value": "0", "note": "Chưa rõ số khay/cụm"})

    rows.append({"metric": "invalid_cluster_ids", "value": str(len(invalid_clusters)), "note": f"Sai định dạng regex ^\\d{{8}}_[a-z]+$: {invalid_clusters[:3]}"})
    rows.append({"metric": "md5_duplicates", "value": str(dup_md5_count) if not no_hash else "skipped", "note": "Số ảnh trùng hoàn toàn MD5"})
    phash_val = "skipped" if no_hash else (str(phash_near_count) if phash_near_count >= 0 else "imagehash_not_installed")
    rows.append({"metric": "phash_near_duplicates", "value": phash_val, "note": "Số cặp ảnh gần trùng pHash (Hamming <= 5)"})
    rows.append({"metric": "ambiguous_count", "value": str(ambiguous_count), "note": "Số ảnh hoặc mẫu đánh dấu ambiguous"})
    rows.append({"metric": "missing_annotators", "value": str(len(missing_annotators)), "note": "Số dòng metadata thiếu annotator_1 hoặc annotator_2"})

    _write_csv(out_csv, rows)
    return rows


def _write_csv(out_csv: Path, rows: list[dict[str, str]]):
    fieldnames = ["metric", "value", "note"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[PASS] Đã ghi kết quả audit vào: {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="Audit dữ liệu khay canteen theo Kế hoạch MASTER v10.0 §8.1")
    parser.add_argument("--metadata", type=Path, default=Path("data/canteen_metadata.csv"), help="File CSV metadata canteen")
    parser.add_argument("--images", type=Path, default=Path("data/raw/canteen_trays"), help="Thư mục chứa ảnh khay canteen")
    parser.add_argument("--labels", type=Path, default=Path("data/raw/canteen_labels"), help="Thư mục chứa file nhãn YOLO")
    parser.add_argument("--out", type=Path, default=Path("reports/canteen_audit.csv"), help="File CSV báo cáo đầu ra")
    parser.add_argument("--no-hash", action="store_true", help="Bỏ qua tính hash MD5/pHash (tiết kiệm thời gian hoặc sau khi Test đóng băng)")
    args = parser.parse_args()

    audit_canteen(
        metadata_path=args.metadata,
        images_dir=args.images,
        labels_dir=args.labels,
        out_csv=args.out,
        no_hash=args.no_hash,
    )


if __name__ == "__main__":
    main()
