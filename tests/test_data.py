from pathlib import Path
from dsr.data import load_label_map, create_transforms, read_split_csv, WasteDataset


def test_public_dataset_loading():
    map_file = Path("data/mappings/label_map.csv")
    train_file = Path("data/splits/public_train.csv")
    dev_file = Path("data/splits/public_dev.csv")
    data_root = Path("data/raw/household_waste_30")

    assert map_file.exists()
    assert train_file.exists()
    assert dev_file.exists()
    assert data_root.exists()

    fine_name_to_id, fine_to_coarse_id, fine_classes, coarse_classes = load_label_map(map_file)
    assert len(fine_classes) == 30, f"Expected 30 fine classes, got {len(fine_classes)}"
    assert len(coarse_classes) == 3, f"Expected 3 coarse classes, got {len(coarse_classes)}"

    train_rows = read_split_csv(train_file)
    dev_rows = read_split_csv(dev_file)

    assert len(train_rows) == 12750
    assert len(dev_rows) == 2250

    # Ensure no data leakage between train and dev
    train_paths = {r["path"] for r in train_rows}
    dev_paths = {r["path"] for r in dev_rows}
    overlap = train_paths.intersection(dev_paths)
    assert len(overlap) == 0, f"Data leakage detected! {len(overlap)} images overlap between Train and Dev"

    # Test loading a few images through WasteDataset
    transform = create_transforms(image_size=224, train=False)
    dataset = WasteDataset(
        data_root=data_root,
        rows=train_rows[:10],
        fine_name_to_id=fine_name_to_id,
        fine_to_coarse_id=fine_to_coarse_id,
        transform=transform,
    )

    assert len(dataset) == 10
    img, fine_id, coarse_id = dataset[0]
    assert img.shape == (3, 224, 224)
    assert 0 <= fine_id < 30
    assert 0 <= coarse_id < 3
    print(f"Sample 0: fine_id={fine_id} ({fine_classes[fine_id]}), coarse_id={coarse_id} ({coarse_classes[coarse_id]})")


if __name__ == "__main__":
    print("Testing public dataset loading...")
    test_public_dataset_loading()
    print("PASS! Public dataset, splits, and label mapping are fully verified.")
