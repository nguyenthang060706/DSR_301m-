from __future__ import annotations

import argparse
import csv
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TRASHNET_CLASSES = ("cardboard", "glass", "metal", "paper", "plastic", "trash")


def iter_images(class_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def inspect_dataset(data_root: Path) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    counts: dict[str, int] = {}

    for class_name in TRASHNET_CLASSES:
        class_dir = data_root / class_name
        count = len(iter_images(class_dir)) if class_dir.exists() else 0
        counts[class_name] = count

    total = sum(counts.values())
    for class_name, count in counts.items():
        rows.append(
            {
                "class": class_name,
                "count": count,
                "fraction": round(count / total, 6) if total else 0.0,
            }
        )

    rows.append({"class": "__total__", "count": total, "fraction": 1.0 if total else 0.0})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect TrashNet class distribution.")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    rows = inspect_dataset(args.data_root)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["class", "count", "fraction"])
            writer.writeheader()
            writer.writerows(rows)

    for row in rows:
        print(f"{row['class']}: {row['count']} ({row['fraction']})")


if __name__ == "__main__":
    main()
