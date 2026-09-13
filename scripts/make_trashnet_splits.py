from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TRASHNET_CLASSES = ("cardboard", "glass", "metal", "paper", "plastic", "trash")


def collect_images(data_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for class_name in TRASHNET_CLASSES:
        class_dir = data_root / class_name
        if not class_dir.exists():
            continue
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append(
                    {
                        "path": image_path.relative_to(data_root).as_posix(),
                        "class": class_name,
                    }
                )
    return rows


def stratified_holdout(
    rows: list[dict[str, str]], holdout_ratio: float, seed: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rng = random.Random(seed)
    holdout: list[dict[str, str]] = []
    cv_pool: list[dict[str, str]] = []

    for class_name in TRASHNET_CLASSES:
        class_rows = [row for row in rows if row["class"] == class_name]
        rng.shuffle(class_rows)
        holdout_n = max(1, round(len(class_rows) * holdout_ratio)) if class_rows else 0
        holdout.extend(class_rows[:holdout_n])
        cv_pool.extend(class_rows[holdout_n:])

    return sorted(holdout, key=lambda row: row["path"]), sorted(cv_pool, key=lambda row: row["path"])


def assign_stratified_folds(rows: list[dict[str, str]], folds: int, seed: int) -> list[dict[str, str | int]]:
    rng = random.Random(seed)
    output: list[dict[str, str | int]] = []

    for class_name in TRASHNET_CLASSES:
        class_rows = [row for row in rows if row["class"] == class_name]
        rng.shuffle(class_rows)
        for index, row in enumerate(class_rows):
            output.append({**row, "fold": index % folds})

    return sorted(output, key=lambda row: (int(row["fold"]), str(row["class"]), str(row["path"])))


def write_csv(path: Path, rows: list[dict[str, str | int]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create fixed TrashNet holdout and CV splits.")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--holdout-ratio", default=0.15, type=float)
    parser.add_argument("--folds", default=5, type=int)
    parser.add_argument("--seed", default=20260913, type=int)
    args = parser.parse_args()

    if not 0 < args.holdout_ratio < 1:
        raise ValueError("--holdout-ratio must be between 0 and 1")
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")

    rows = collect_images(args.data_root)
    if not rows:
        raise SystemExit(f"No images found under {args.data_root}")

    holdout, cv_pool = stratified_holdout(rows, args.holdout_ratio, args.seed)
    fold_rows = assign_stratified_folds(cv_pool, args.folds, args.seed)

    write_csv(args.out_dir / "trashnet_dev_corruption_holdout.csv", holdout, ["path", "class"])
    write_csv(args.out_dir / "trashnet_cv_folds.csv", fold_rows, ["path", "class", "fold"])

    print(f"Total images: {len(rows)}")
    print(f"Dev/Corruption-holdout: {len(holdout)}")
    print(f"CV pool: {len(cv_pool)}")
    print(f"Folds: {args.folds}")


if __name__ == "__main__":
    main()
