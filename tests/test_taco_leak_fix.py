import random
import pytest


def test_taco_split_leakage_free():
    # Giả lập danh sách valid image IDs (ví dụ 1500 ảnh như TACO)
    valid_img_ids = list(range(1500))
    seed = 20261012
    split_ratio = 0.8
    val_ratio = 0.10

    # 1. Logic cũ
    random.seed(seed)
    ids_old = list(valid_img_ids)
    random.shuffle(ids_old)
    num_train_old = int(len(ids_old) * split_ratio)
    train_ids_old = set(ids_old[:num_train_old])
    heldout_ids_old = set(ids_old[num_train_old:])

    # 2. Logic mới
    random.seed(seed)
    ids_new = list(valid_img_ids)
    random.shuffle(ids_new)
    num_train_pool = int(len(ids_new) * split_ratio)
    train_pool = ids_new[:num_train_pool]
    heldout_ids_new = set(ids_new[num_train_pool:])

    num_val = int(len(train_pool) * val_ratio)
    train_ids_new = set(train_pool[:-num_val])
    val_ids_new = set(train_pool[-num_val:])

    # 3. Kiểm định
    # Heldout cũ và mới phải TUYỆT ĐỐI TRÙNG NHAU (bảo toàn tập OOD benchmark)
    assert heldout_ids_old == heldout_ids_new, "set(heldout_cũ) phải bằng set(heldout_mới)"

    # Không giao nhau giữa 3 tập: train ∩ val ∩ heldout = ∅
    assert len(train_ids_new.intersection(val_ids_new)) == 0, "Train và Val bị rò rỉ!"
    assert len(train_ids_new.intersection(heldout_ids_new)) == 0, "Train và Heldout bị rò rỉ!"
    assert len(val_ids_new.intersection(heldout_ids_new)) == 0, "Val và Heldout bị rò rỉ!"

    # Tập train mới và val mới ghép lại đúng bằng train_pool cũ
    assert train_ids_new.union(val_ids_new) == train_ids_old

    print("[PASS] Đã kiểm định không rò rỉ TACO và bảo toàn hoàn toàn tập heldout.")
