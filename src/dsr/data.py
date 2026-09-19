from __future__ import annotations

import csv
from pathlib import Path

try:
    from torch.utils.data import Dataset
except ImportError:
    Dataset = object


def read_split_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_class_weights(config: dict) -> list[float]:
    """Inverse-frequency class weights computed from the actual train split
    (trashnet_cv_folds.csv), aligned 1:1 with config["classes"] order.

    weight_c = N / (K * count_c)  ?" same convention as sklearn's
    class_weight="balanced". Computed from the real split file (not a
    hard-coded/estimated distribution) so it stays correct if the split,
    dedup, or holdout ratio ever changes.
    """
    classes = list(config["classes"])
    split_dir = Path(config["data"]["split_dir"])
    all_cv_rows = read_split_csv(split_dir / "trashnet_cv_folds.csv")
    val_fold = str(config.get("data", {}).get("val_fold", 0))
    train_rows = [r for r in all_cv_rows if str(r.get("fold", "")) != val_fold]

    counts = {name: 0 for name in classes}
    for row in train_rows:
        counts[row["class"]] += 1

    total = sum(counts.values())
    num_classes = len(classes)
    if total == 0:
        return [1.0 for _ in classes]

    weights: list[float] = []
    for name in classes:
        count = counts[name]
        if count == 0:
            raise ValueError(
                f"Class '{name}' has 0 examples in the train split ({split_dir / 'trashnet_cv_folds.csv'}); "
                "cannot compute a finite inverse-frequency weight for it."
            )
        weights.append(total / (num_classes * count))
    return weights


def create_transforms(image_size: int, train: bool):
    try:
        from torchvision import transforms
    except ImportError as exc:
        raise RuntimeError("torchvision is required for data transforms.") from exc

    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


class CsvImageDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        rows: list[dict[str, str]],
        classes: list[str],
        transform,
    ) -> None:
        if Dataset is object:
            raise RuntimeError("torch is required for image datasets.")

        self.data_root = data_root
        self.rows = rows
        self.classes = classes
        self.class_to_index = {name: index for index, name in enumerate(classes)}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required for image datasets.") from exc

        row = self.rows[index]
        image_path = self.data_root / row["path"]
        image = Image.open(image_path).convert("RGB")
        label = self.class_to_index[row["class"]]
        return self.transform(image), label


def make_week1_loaders(config: dict, batch_size: int):
    try:
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError("torch is required for DataLoader.") from exc

    data_root = Path(config["data"]["trashnet_root"])
    split_dir = Path(config["data"]["split_dir"])
    classes = list(config["classes"])
    image_size = int(config["data"]["image_size"])
    num_workers = int(config["data"].get("num_workers", 0))

    all_cv_rows = read_split_csv(split_dir / "trashnet_cv_folds.csv")
    
    # Lấy val_fold từ config (mặc định là 0 cho các thí nghiệm exploratory Tuần 1-7)
    val_fold = str(config.get("data", {}).get("val_fold", 0))
    
    # MASTER PLAN §6.4 FIX: Tuyệt đối KHÔNG dùng dev_corruption_holdout làm val set hàng ngày.
    train_rows = [r for r in all_cv_rows if str(r.get("fold", "")) != val_fold]
    val_rows = [r for r in all_cv_rows if str(r.get("fold", "")) == val_fold]

    train_dataset = CsvImageDataset(
        data_root=data_root,
        rows=train_rows,
        classes=classes,
        transform=create_transforms(image_size=image_size, train=True),
    )
    val_dataset = CsvImageDataset(
        data_root=data_root,
        rows=val_rows,
        classes=classes,
        transform=create_transforms(image_size=image_size, train=False),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
