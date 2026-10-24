"""Count TACO instances per category and apply TrashNet-6 mapping.

Usage:
    python scripts/count_taco_instances.py \
        --annotations path/to/taco/annotations.json \
        --mapping data/mappings/taco_to_trashnet6.csv \
        --out reports/week1/taco_mapping_instance_counts.csv

Requires:  pip install pycocotools  (or just json + pandas)
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count TACO instances per category and apply TrashNet-6 mapping."
    )
    parser.add_argument(
        "--annotations",
        required=True,
        type=Path,
        help="Path to TACO annotations.json (COCO format).",
    )
    parser.add_argument(
        "--mapping",
        required=True,
        type=Path,
        help="Path to taco_to_trashnet6.csv (60-category level).",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output CSV with per-category instance counts and mapping.",
    )
    args = parser.parse_args()

    # ---- Load TACO annotations (COCO format) ----
    with args.annotations.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    id_to_cat: dict[int, dict] = {}
    for cat in coco["categories"]:
        id_to_cat[cat["id"]] = {
            "name": cat["name"],
            "supercategory": cat.get("supercategory", ""),
        }

    ann_counts: Counter[int] = Counter()
    for ann in coco["annotations"]:
        ann_counts[ann["category_id"]] += 1

    # ---- Load mapping CSV ----
    mapping: dict[str, str] = {}  # exact name → class
    mapping_lower: dict[str, tuple[str, str]] = {}  # lowercase name → (original_csv_name, class)
    with args.mapping.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_name = row["taco_category"]
            cls = row["trashnet6_class"]
            mapping[csv_name] = cls
            mapping_lower[csv_name.lower()] = (csv_name, cls)

    # ---- Build per-category report ----
    case_warnings: list[str] = []
    rows: list[dict] = []
    for cat_id in sorted(id_to_cat.keys()):
        cat_name = id_to_cat[cat_id]["name"]
        supercategory = id_to_cat[cat_id]["supercategory"]
        count = ann_counts.get(cat_id, 0)

        # Exact match first, then case-insensitive fallback
        if cat_name in mapping:
            mapped_class = mapping[cat_name]
        elif cat_name.lower() in mapping_lower:
            csv_name, mapped_class = mapping_lower[cat_name.lower()]
            case_warnings.append(
                f"  CASE MISMATCH: TACO has '{cat_name}', CSV has '{csv_name}' (matched via lowercase)"
            )
        else:
            mapped_class = "UNMAPPED"
        rows.append(
            {
                "taco_category_id": cat_id,
                "taco_category": cat_name,
                "taco_supercategory": supercategory,
                "instance_count": count,
                "trashnet6_class": mapped_class,
            }
        )

    # ---- Write per-category CSV ----
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "taco_category_id",
                "taco_category",
                "taco_supercategory",
                "instance_count",
                "trashnet6_class",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    # ---- Print summary ----
    total_instances = sum(r["instance_count"] for r in rows)
    keep_instances = sum(
        r["instance_count"] for r in rows if r["trashnet6_class"] not in ("ignore", "UNMAPPED")
    )
    ignore_instances = sum(
        r["instance_count"] for r in rows if r["trashnet6_class"] == "ignore"
    )
    unmapped_instances = sum(
        r["instance_count"] for r in rows if r["trashnet6_class"] == "UNMAPPED"
    )

    print(f"\n{'='*60}")
    print(f"TACO → TrashNet-6 Mapping Instance Summary")
    print(f"{'='*60}")
    print(f"Total TACO instances:    {total_instances:>6}")
    print(f"  Kept (mapped):         {keep_instances:>6}  ({100*keep_instances/total_instances:.1f}%)")
    print(f"  Ignored:               {ignore_instances:>6}  ({100*ignore_instances/total_instances:.1f}%)")
    if unmapped_instances:
        print(f"  UNMAPPED (check CSV!): {unmapped_instances:>6}  ({100*unmapped_instances/total_instances:.1f}%)")
    print()

    # Per TrashNet-6 class breakdown
    class_counts: dict[str, int] = {}
    for r in rows:
        cls = r["trashnet6_class"]
        class_counts[cls] = class_counts.get(cls, 0) + r["instance_count"]

    print(f"{'TrashNet-6 class':<20} {'Instances':>10} {'% of total':>12}")
    print(f"{'-'*20} {'-'*10} {'-'*12}")
    for cls in ["cardboard", "glass", "metal", "paper", "plastic", "trash", "ignore", "UNMAPPED"]:
        if cls in class_counts:
            c = class_counts[cls]
            print(f"{cls:<20} {c:>10} {100*c/total_instances:>11.1f}%")
    print()

    # Flag warnings
    for cls in ["glass", "cardboard", "metal"]:
        if cls in class_counts and class_counts[cls] < 100:
            print(f"⚠️  WARNING: {cls} has only {class_counts[cls]} instances — low statistical power for eval.")

    # List unmapped categories
    unmapped = [r for r in rows if r["trashnet6_class"] == "UNMAPPED" and r["instance_count"] > 0]
    if unmapped:
        print(f"\n⚠️  UNMAPPED categories with instances:")
        for r in unmapped:
            print(f"   {r['taco_category']} (supercategory: {r['taco_supercategory']}, count: {r['instance_count']})")

    # Report case mismatches
    if case_warnings:
        print(f"\n⚠️  CASE MISMATCHES ({len(case_warnings)}):")
        for w in case_warnings:
            print(w)
        print("  → Fix these in taco_to_trashnet6.csv to ensure exact matching.")
    else:
        print("✅ All category names match exactly (case-sensitive).")

    print(f"\nDetailed output saved to: {args.out}")


if __name__ == "__main__":
    main()
