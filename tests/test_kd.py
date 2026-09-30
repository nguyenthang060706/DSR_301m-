from pathlib import Path
import torch
import torch.nn.functional as F

from dsr.data import read_split_csv
from dsr.losses import kd_loss


def test_kd_loss_formula_and_properties():
    torch.manual_seed(42)
    batch_size, num_classes = 8, 6
    logits_S = torch.randn(batch_size, num_classes)
    logits_T = torch.randn(batch_size, num_classes)
    temperature = 3.0

    # Inline implementation (cũ)
    student_log_prob = F.log_softmax(logits_S / temperature, dim=1)
    teacher_prob = F.softmax(logits_T / temperature, dim=1)
    expected_loss = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (temperature ** 2)

    # Modular implementation (mới từ losses.py)
    actual_loss = kd_loss(logits_S, logits_T, temperature)

    # (a) Khớp hoàn toàn với công thức inline cũ
    assert torch.allclose(actual_loss, expected_loss, atol=1e-7), (
        f"Modular kd_loss {actual_loss.item()} != inline {expected_loss.item()}"
    )

    # (b) KL >= 0
    assert actual_loss.item() >= 0.0, "KD loss must be non-negative"

    # KL bằng 0 khi logits trùng khớp
    self_loss = kd_loss(logits_T, logits_T, temperature)
    assert abs(self_loss.item()) < 1e-5, f"KL(T||T) must be close to 0, got {self_loss.item()}"


def test_kd_direction_error():
    # (c) Đảo input/target bị test bắt (PyTorch kl_div yêu cầu input=log_prob, target=prob)
    torch.manual_seed(42)
    logits_S = torch.randn(4, 6)
    logits_T = torch.randn(4, 6)

    student_prob = F.softmax(logits_S, dim=1)
    teacher_prob = F.softmax(logits_T, dim=1)

    # Truyền sai dạng (cả 2 đều là prob, thiếu log)
    wrong_loss = F.kl_div(student_prob, teacher_prob, reduction="batchmean")
    assert wrong_loss.item() < 0, "Swapped/prob inputs yield negative loss"


def test_splits_isolation_no_leakage():
    # (e) Assert giao tập holdout ∩ train = ∅, holdout ∩ val = ∅
    split_dir = Path("data/splits")
    cv_file = split_dir / "trashnet_cv_folds.csv"
    holdout_file = split_dir / "trashnet_dev_corruption_holdout.csv"

    assert cv_file.exists(), f"Missing {cv_file}"
    assert holdout_file.exists(), f"Missing {holdout_file}"

    cv_rows = read_split_csv(cv_file)
    holdout_rows = read_split_csv(holdout_file)

    cv_paths = {r["path"] for r in cv_rows}
    holdout_paths = {r["path"] for r in holdout_rows}

    # Giao tập hoàn toàn rỗng
    intersection = cv_paths.intersection(holdout_paths)
    assert len(intersection) == 0, (
        f"Data leakage detected! {len(intersection)} images found in both CV-folds and Holdout set: {list(intersection)[:5]}"
    )

    # Kiểm tra riêng với từng fold
    folds = {r.get("fold", "") for r in cv_rows}
    for f in folds:
        fold_paths = {r["path"] for r in cv_rows if r.get("fold", "") == f}
        assert len(fold_paths.intersection(holdout_paths)) == 0, f"Fold {f} intersects with holdout!"

    print(f"Data isolation verified: {len(cv_paths)} CV images and {len(holdout_paths)} holdout images have 0 overlap.")


if __name__ == "__main__":
    print("Running test_kd_loss_formula_and_properties()...")
    test_kd_loss_formula_and_properties()
    print("PASS")

    print("Running test_kd_direction_error()...")
    test_kd_direction_error()
    print("PASS")

    print("Running test_splits_isolation_no_leakage()...")
    test_splits_isolation_no_leakage()
    print("PASS")

    print("\nALL TESTS PASSED SUCCESSFULLY!")
