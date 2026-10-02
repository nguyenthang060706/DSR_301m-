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


def load_label_map(path: Path) -> tuple[dict[str, int], dict[int, int], list[str], list[str]]:
    """Loads label_map.csv defining fine-grained to coarse 3-class mapping.

    Returns:
        fine_name_to_id: dict[str, int]
        fine_to_coarse_id: dict[int, int]
        fine_classes: list[str]
        coarse_classes: list[str] (length 3: ['organic', 'recyclable', 'other_landfill'])
    """
    fine_name_to_id: dict[str, int] = {}
    fine_to_coarse_id: dict[int, int] = {}
    fine_classes_dict: dict[int, str] = {}
    coarse_classes_dict: dict[int, str] = {}

    with path.open("r", newline="", encoding="utf-8") as f:
        # Filter comment lines before parsing CSV
        lines = [line for line in f if not line.strip().startswith("#")]
        for row in csv.DictReader(lines):
            if not row or not row.get("fine_id"):
                continue
            f_id = int(row["fine_id"])
            f_name = row["fine_label"].strip()
            c_id = int(row["coarse_id"])
            c_name = row["coarse_label"].strip()

            fine_name_to_id[f_name] = f_id
            fine_to_coarse_id[f_id] = c_id
            fine_classes_dict[f_id] = f_name
            coarse_classes_dict[c_id] = c_name

    fine_classes = [fine_classes_dict[i] for i in sorted(fine_classes_dict.keys())]
    coarse_classes = [coarse_classes_dict[i] for i in sorted(coarse_classes_dict.keys())]

    return fine_name_to_id, fine_to_coarse_id, fine_classes, coarse_classes


def create_transforms(image_size: int = 224, train: bool = True, use_trivial_augment: bool = True):
    """Creates modern transforms conforming to Kế hoạch MASTER v9.5 §5.0.

    - Train: RandomResizedCrop (preserves aspect ratio) + RandomHorizontalFlip + TrivialAugmentWide
    - Val/Test: Resize(256) + CenterCrop(224) (no distortion)
    """
    try:
        from torchvision import transforms
    except ImportError as exc:
        raise RuntimeError("torchvision is required for data transforms.") from exc

    if train:
        t_list = [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
        if use_trivial_augment:
            t_list.append(transforms.TrivialAugmentWide())
        t_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        return transforms.Compose(t_list)

    # Standard non-distorted validation/test transform
    resize_dim = int(image_size * 256.0 / 224.0)
    return transforms.Compose([
        transforms.Resize(resize_dim),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


class WasteDataset(Dataset):
    """Dataset supporting fine-grained labels and automatic coarse 3-class mapping."""

    def __init__(
        self,
        data_root: Path,
        rows: list[dict[str, str]],
        fine_name_to_id: dict[str, int],
        fine_to_coarse_id: dict[int, int],
        transform,
    ) -> None:
        self.data_root = data_root
        self.rows = rows
        self.fine_name_to_id = fine_name_to_id
        self.fine_to_coarse_id = fine_to_coarse_id
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

        fine_name = row.get("fine_label") or row.get("class")
        fine_id = self.fine_name_to_id[fine_name]
        coarse_id = self.fine_to_coarse_id[fine_id]

        # Returns image, fine_id, coarse_id
        return self.transform(image), fine_id, coarse_id
