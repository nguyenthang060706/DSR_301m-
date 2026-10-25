"""scripts/split_canteen_by_cluster.py

Phân chia dữ liệu khay ăn canteen vào Pool (Train) / Val / Test theo cụm ngày/ca (MASTER v10.0 §3.3 & Spec 8.6).

Nguyên tắc:
1. Phân chia theo ngày (date-level cluster), tuyệt đối không chia ngẫu nhiên từng ảnh để chống rò rỉ.
2. Đọc bắt buộc kế hoạch ngày từ configs/canteen_date_plan.json nếu được cung cấp.
3. Ràng buộc cứng: assert không có bất kỳ ngày nào hay cụm nào cùng xuất hiện ở 2 tập.
4. Cảnh báo/dừng: Nếu số ngày chụp < 5, raise RuntimeError (chỉ cho phép chia theo ca khi bật --allow-shift-split).
5. Hỗ trợ --dry-run: Chỉ quét và kiểm tra quota/rò rỉ, không copy ảnh và không ghi manifest Test.
6. Khi chạy thật:
   - Thư mục đích: images/{pool, val, test}
   - Xuất song song 2 bộ nhãn:
     + labels_agnostic/ (class 0 cho detector D1)
     + labels_3cls/ (class 0, 1, 2 cho detector D2/tham chiếu)
   - Ghi mã SHA-256 từng file Test vào data/splits/canteen_test_manifest.json
   - Xuất báo cáo reports/canteen_split_report.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def normalize_date(val: str) -> str:
    """Chuẩn hóa ngày sang định dạng YYYY-MM-DD và YYYYMMDD để khớp an toàn."""
    val = val.strip().replace("/", "-")
    parts = val.split("_")[0].split()
    date_str = parts[0]
    # Nếu là YYYYMMDD (8 chữ số liền)
    if len(date_str) == 8 and date_str.isdigit():
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    return date_str


def compute_sha256(file_path: Path) -> str:
    """Tính mã SHA-256 của file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def count_labels_in_file(lbl_path: Path) -> dict[int, int]:
    """Đếm số instance theo từng class trong 1 file nhãn YOLO."""
    counts: dict[int, int] = {}
    if not lbl_path.exists():
        return counts
    try:
        with open(lbl_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        cls_id = int(parts[0])
                        counts[cls_id] = counts.get(cls_id, 0) + 1
                    except ValueError:
                        continue
    except Exception:
        pass
    return counts


def parse_args():
    parser = argparse.ArgumentParser(description="Chia dữ liệu khay canteen theo cụm ngày/ca (Spec 8.6)")
    parser.add_argument("--metadata", type=Path, default=Path("data/canteen_metadata.csv"), help="File metadata khay canteen")
    parser.add_argument("--images-dir", type=Path, default=Path("data/raw/canteen_trays"), help="Thư mục ảnh gốc")
    parser.add_argument("--labels-dir", type=Path, default=Path("data/raw/canteen_labels"), help="Thư mục nhãn gốc")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/canteen_yolo"), help="Thư mục đích")
    parser.add_argument("--date-plan", type=Path, default=Path("configs/canteen_date_plan.json"), help="Kế hoạch ngày Test/Val/Pool")
    parser.add_argument("--manifest-out", type=Path, default=Path("data/splits/canteen_test_manifest.json"), help="File manifest Test")
    parser.add_argument("--report-out", type=Path, default=Path("reports/canteen_split_report.csv"), help="Báo cáo phân chia")
    parser.add_argument("--seed", type=int, default=20261012, help="Seed ngẫu nhiên")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ kiểm tra và in thống kê, không copy và không ghi manifest")
    parser.add_argument("--allow-shift-split", action="store_true", help="Cho phép chia theo ca nếu số ngày < 5 (yêu cầu ghi vào Limitations)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 68)
    print("PHÂN CHIA DỮ LIỆU CANTEEN THEO CỤM NGÀY/CA (MASTER v10.0 §3.3)")
    print(f"Chế độ: {'[DRY-RUN - KHÔNG GHI FILE]' if args.dry_run else '[THỰC THI CHÍNH THỨC]'}")
    print(f"Metadata:  {args.metadata}")
    print(f"Date Plan: {args.date_plan}")
    print("=" * 68)

    if not args.metadata.exists():
        raise FileNotFoundError(f"Không tìm thấy file metadata: {args.metadata}")

    with open(args.metadata, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError("File metadata trống!")

    # 1. Trích xuất danh sách ngày và cụm
    for r in rows:
        # Lấy ngày từ cột 'date' hoặc từ 8 ký tự đầu của 'cluster_id'
        d = r.get("date", "").strip()
        if not d and "cluster_id" in r:
            cid = r["cluster_id"].strip()
            d = cid.split("_")[0]
        r["norm_date"] = normalize_date(d)
        r["norm_cluster"] = r.get("cluster_id", f"{r['norm_date']}_trua").strip()

    all_dates = sorted(list(set(r["norm_date"] for r in rows)))
    all_clusters = sorted(list(set(r["norm_cluster"] for r in rows)))
    num_dates = len(all_dates)
    num_clusters = len(all_clusters)

    print(f"[*] Quét được: {len(rows)} ảnh | {num_clusters} cụm ca | {num_dates} ngày chụp")

    # Kiểm tra an toàn số ngày (< 5 ngày)
    if num_dates < 5:
        if not args.allow_shift_split:
            raise RuntimeError(
                f"[LỖI AN TOÀN SPEC 8.6] Số ngày chụp quá ít ({num_dates} < 5 ngày). "
                f"Phân chia theo cụm đòi hỏi >= 5 ngày để tránh tương quan thời gian. "
                f"Chỉ cho phép chia theo ca chụp khi bạn đồng ý bằng cờ --allow-shift-split "
                f"và ghi nhận rõ ràng vào phần Limitations của bài báo."
            )
        else:
            print("[CẢNH BÁO] Số ngày < 5 nhưng cờ --allow-shift-split được bật. Chia theo ca (Limitations).")

    # 2. Xác định tập split cho từng ngày (date-level partition)
    test_dates: set[str] = set()
    val_dates: set[str] = set()
    pool_dates: set[str] = set()

    if args.date_plan and args.date_plan.exists():
        print(f"[*] Nạp kế hoạch phân chia ngày từ: {args.date_plan}")
        with open(args.date_plan, "r", encoding="utf-8") as f:
            plan_data = json.load(f)

        test_dates = set(normalize_date(d) for d in plan_data.get("test", []))
        val_dates = set(normalize_date(d) for d in plan_data.get("val", []))
        pool_dates = set(normalize_date(d) for d in plan_data.get("pool", []))
    else:
        print(f"[*] Không có date-plan, tự động phân chia ngày theo seed {args.seed}...")
        rng = random.Random(args.seed)
        shuffled_dates = list(all_dates)
        rng.shuffle(shuffled_dates)

        # Tính số khay theo ngày để phân bổ đủ quota
        trays_by_date = {d: 0 for d in all_dates}
        for r in rows:
            trays_by_date[r["norm_date"]] += 1

        acc_test = 0
        acc_val = 0
        for d in shuffled_dates:
            c = trays_by_date[d]
            if acc_test < 150:
                test_dates.add(d)
                acc_test += c
            elif acc_val < 50:
                val_dates.add(d)
                acc_val += c
            else:
                pool_dates.add(d)

    # 3. Assert bảo vệ tính toàn vẹn: Không ngày nào thuộc 2 tập
    assert test_dates.isdisjoint(val_dates), f"RÒ RỈ: Test và Val giao nhau tại ngày {test_dates & val_dates}"
    assert test_dates.isdisjoint(pool_dates), f"RÒ RỈ: Test và Pool giao nhau tại ngày {test_dates & pool_dates}"
    assert val_dates.isdisjoint(pool_dates), f"RÒ RỈ: Val và Pool giao nhau tại ngày {val_dates & pool_dates}"

    # Gán split cho từng dòng metadata
    split_rows: dict[str, list[dict]] = {"test": [], "val": [], "pool": []}
    for r in rows:
        d = r["norm_date"]
        if d in test_dates:
            split_rows["test"].append(r)
        elif d in val_dates:
            split_rows["val"].append(r)
        else:
            split_rows["pool"].append(r)

    # Assert cụm không bị chia vào 2 tập
    test_cids = set(r["norm_cluster"] for r in split_rows["test"])
    val_cids = set(r["norm_cluster"] for r in split_rows["val"])
    pool_cids = set(r["norm_cluster"] for r in split_rows["pool"])

    assert test_cids.isdisjoint(val_cids), f"RÒ RỈ CỤM: Test và Val giao cụm {test_cids & val_cids}"
    assert test_cids.isdisjoint(pool_cids), f"RÒ RỈ CỤM: Test và Pool giao cụm {test_cids & pool_cids}"
    assert val_cids.isdisjoint(pool_cids), f"RÒ RỈ CỤM: Val và Pool giao cụm {val_cids & pool_cids}"

    # 4. Thống kê số khay và đếm instance/nhóm
    stats = {}
    report_rows = []

    for sp in ["test", "val", "pool"]:
        cur_rows = split_rows[sp]
        cids = set(r["norm_cluster"] for r in cur_rows)
        dates = set(r["norm_date"] for r in cur_rows)

        group_counts: dict[int, int] = {0: 0, 1: 0, 2: 0}
        total_boxes = 0

        for r in cur_rows:
            img_file = r["image_id"]
            lbl_file = Path(img_file).stem + ".txt"
            lbl_path = args.labels_dir / lbl_file
            if lbl_path.exists():
                f_counts = count_labels_in_file(lbl_path)
                for c_id, cnt in f_counts.items():
                    group_counts[c_id] = group_counts.get(c_id, 0) + cnt
                    total_boxes += cnt

        stats[sp] = {
            "trays": len(cur_rows),
            "clusters": len(cids),
            "dates": len(dates),
            "group_0": group_counts.get(0, 0),
            "group_1": group_counts.get(1, 0),
            "group_2": group_counts.get(2, 0),
            "total_boxes": total_boxes,
        }

        report_rows.append({
            "split": sp,
            "trays": len(cur_rows),
            "clusters": len(cids),
            "dates": len(dates),
            "group_0_organic": group_counts.get(0, 0),
            "group_1_recyclable": group_counts.get(1, 0),
            "group_2_other": group_counts.get(2, 0),
            "total_boxes": total_boxes,
            "dates_list": ";".join(sorted(dates)),
        })

    # In kết quả thống kê
    print("\n" + "=" * 68)
    print("KẾT QUẢ PHÂN CHIA VÀ KIỂM TRA QUOTA:")
    print("-" * 68)
    for sp in ["test", "val", "pool"]:
        st = stats[sp]
        quota_mark = ""
        if sp == "test":
            quota_mark = " [QUOTA KHAY: ĐẠT]" if st["trays"] >= 150 else " [QUOTA KHAY: CHƯA ĐỦ]"
        elif sp == "val":
            quota_mark = " [QUOTA KHAY: ĐẠT]" if st["trays"] >= 50 else " [QUOTA KHAY: CHƯA ĐỦ]"

        print(
            f"• Tập {sp.upper():4s}: {st['trays']:3d} khay ({st['dates']:2d} ngày, {st['clusters']:2d} cụm){quota_mark}\n"
            f"   Nhãn hiện tại: {st['total_boxes']} box (Hữu cơ: {st['group_0']}, Tái chế: {st['group_1']}, Còn lại: {st['group_2']})"
        )
    print("=" * 68)

    # 5. Xử lý xuất báo cáo và file (khi không bật dry-run)
    if args.dry_run:
        print("\n[DRY-RUN HOÀN TẤT] Assert kiểm tra rò rỉ thành công. Không có file nào bị ghi đè.")
        return

    # THỰC HIỆN COPY VÀ XUẤT NHÃN KHI CHẠY THẬT
    print("\n[*] Đang tiến hành copy ảnh và tạo nhãn song song (agnostic + 3cls)...")
    out_dir = args.out_dir
    img_src_dir = args.images_dir
    lbl_src_dir = args.labels_dir

    manifest_entries = []

    for sp in ["pool", "val", "test"]:
        dest_img_dir = out_dir / "images" / sp
        dest_lbl_agnostic_dir = out_dir / "labels_agnostic" / sp
        dest_lbl_3cls_dir = out_dir / "labels_3cls" / sp

        dest_img_dir.mkdir(parents=True, exist_ok=True)
        dest_lbl_agnostic_dir.mkdir(parents=True, exist_ok=True)
        dest_lbl_3cls_dir.mkdir(parents=True, exist_ok=True)

        for r in split_rows[sp]:
            img_file = r["image_id"]
            lbl_file = Path(img_file).stem + ".txt"

            src_img_p = img_src_dir / img_file
            src_lbl_p = lbl_src_dir / lbl_file

            # 1. Copy ảnh
            if src_img_p.exists():
                shutil.copy2(src_img_p, dest_img_dir / img_file)

                # Nếu là tập Test: tính SHA-256 để ghi manifest đóng băng
                if sp == "test":
                    sha = compute_sha256(src_img_p)
                    manifest_entries.append({
                        "path": f"images/test/{img_file}",
                        "image_id": img_file,
                        "cluster_id": r["norm_cluster"],
                        "date": r["norm_date"],
                        "sha256": sha,
                    })

            # 2. Xử lý nhãn song song
            if src_lbl_p.exists():
                with open(src_lbl_p, "r", encoding="utf-8") as f_in:
                    lines = [ln.strip() for ln in f_in if ln.strip()]

                # labels_3cls (giữ nguyên class 0, 1, 2)
                with open(dest_lbl_3cls_dir / lbl_file, "w", encoding="utf-8") as f_out:
                    f_out.write("\n".join(lines) + ("\n" if lines else ""))

                # labels_agnostic (mọi box đều gán class 0)
                agnostic_lines = []
                for ln in lines:
                    parts = ln.split()
                    if len(parts) >= 5:
                        parts[0] = "0"
                        agnostic_lines.append(" ".join(parts))

                with open(dest_lbl_agnostic_dir / lbl_file, "w", encoding="utf-8") as f_out:
                    f_out.write("\n".join(agnostic_lines) + ("\n" if agnostic_lines else ""))

    # Ghi file manifest Test
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_data = {
        "dataset": "canteen_test_frozen",
        "created_at": datetime.now().isoformat(),
        "total_test_trays": len(manifest_entries),
        "files": manifest_entries,
    }
    with open(args.manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"[THÀNH CÔNG] Đã ghi Test Manifest ({len(manifest_entries)} ảnh) tại: {args.manifest_out}")

    # Ghi file báo cáo CSV
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split", "trays", "clusters", "dates",
        "group_0_organic", "group_1_recyclable", "group_2_other",
        "total_boxes", "dates_list"
    ]
    with open(args.report_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)
    print(f"[THÀNH CÔNG] Đã xuất báo cáo phân chia tại: {args.report_out}")


if __name__ == "__main__":
    main()
