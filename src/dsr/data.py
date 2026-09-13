from __future__ import annotations

import csv
from pathlib import Path


def read_split_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


class CsvImageDataset:
    def __init__(
        self,
        data_root: Path,
        rows: list[dict[str, str]],
        classes: list[str],
        transform,
    ) -> None:
        try:
            from PIL import Image
            from torch.utils.data import Dataset
        except ImportError as exc:
            raise RuntimeError("Pillow and torch are required for image datasets.") from exc

        class _Dataset(Dataset):
            def __init__(self, outer: CsvImageDataset) -> None:
                self.outer = outer

            def __len__(self) -> int:
                return len(self.outer.rows)

            def __getitem__(self, index: int):
                row = self.outer.rows[index]
                image_path = self.outer.data_root / row["path"]
                image = Image.open(image_path).convert("RGB")
                label = self.outer.class_to_index[row["class"]]
                return self.outer.transform(image), label

        self.data_root = data_root
        self.rows = rows
        self.classes = classes
        self.class_to_index = {name: index for index, name in enumerate(classes)}
        self.transform = transform
        self.dataset = _Dataset(self)


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

    train_rows = read_split_csv(split_dir / "trashnet_cv_folds.csv")
    val_rows = read_split_csv(split_dir / "trashnet_dev_corruption_holdout.csv")

    train_dataset = CsvImageDataset(
        data_root=data_root,
        rows=train_rows,
        classes=classes,
        transform=create_transforms(image_size=image_size, train=True),
    ).dataset
    val_dataset = CsvImageDataset(
        data_root=data_root,
        rows=val_rows,
        classes=classes,
        transform=create_transforms(image_size=image_size, train=False),
    ).dataset

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
