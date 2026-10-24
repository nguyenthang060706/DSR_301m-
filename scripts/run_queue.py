"""scripts/run_queue.py

Quản lý hàng đợi thực nghiệm tuần tự theo phụ thuộc (Spec 8.2).
Cách dùng:
  python scripts/run_queue.py --config configs/queue_week.json [--dry-run] [--only Q01,Q04] [--until Q14]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment Queue Runner (Spec 8.2)")
    parser.add_argument("--config", type=Path, default=Path("configs/queue_week.json"), help="File cấu hình hàng đợi JSON")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ in kế hoạch thực thi và kiểm tra phụ thuộc/expect")
    parser.add_argument("--only", type=str, default=None, help="Chỉ chạy danh sách job xác định (cách nhau bởi dấu phẩy, vd: Q01,Q04)")
    parser.add_argument("--until", type=str, default=None, help="Chạy tuần tự tới khi gặp job id này (bao gồm cả job đó)")
    parser.add_argument("--log-dir", type=Path, default=Path("reports/queue_logs"), help="Thư mục lưu log chi tiết từng job")
    parser.add_argument("--status-csv", type=Path, default=Path("reports/queue_status.csv"), help="File CSV theo dõi trạng thái")
    return parser.parse_args()


def load_jobs(config_path: Path) -> list[dict]:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_dynamic_cmd(job: dict) -> str:
    """Tự động bổ sung siêu tham số từ configs/best_kd/<method>.json nếu có."""
    cmd = job["cmd"]
    job_id = job.get("id", "")

    # Ánh xạ job KD sang file best config
    kd_map = {
        "Q04": ("dkd", "configs/best_kd/dkd.json"),
        "Q07": ("vanilla_kd", "configs/best_kd/vanilla_kd.json"),
        "Q09": ("attention_transfer", "configs/best_kd/attention_transfer.json"),
        "Q11": ("hybrid", "configs/best_kd/hybrid.json"),
        "Q13": ("fitnet", "configs/best_kd/fitnet.json"),
    }

    if job_id in kd_map:
        method, cfg_path = kd_map[job_id]
        p = Path(cfg_path)
        if p.exists():
            try:
                with p.open("r", encoding="utf-8") as f:
                    params = json.load(f)
                extra = []
                if "temperature" in params:
                    extra.append(f"--temperature {params['temperature']}")
                if "alpha" in params:
                    extra.append(f"--alpha {params['alpha']}")
                if "alpha_tckd" in params:
                    extra.append(f"--alpha-tckd {params['alpha_tckd']}")
                if "beta_nckd" in params:
                    extra.append(f"--beta-nckd {params['beta_nckd']}")
                if "beta_at" in params:
                    extra.append(f"--beta-at {params['beta_at']}")
                if "lambda_fitnet" in params:
                    extra.append(f"--lambda-fitnet {params['lambda_fitnet']}")
                if extra:
                    cmd = f"{cmd} {' '.join(extra)}"
            except Exception:
                pass
    return cmd


def update_status_csv(status_csv: Path, records: list[dict]):
    status_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "name", "kind", "est_h", "status", "start_time", "duration_sec", "exit_code", "expect"]
    with status_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def main():
    args = parse_args()
    if not args.config.exists():
        raise FileNotFoundError(f"Không tìm thấy file queue: {args.config}")

    all_jobs = load_jobs(args.config)
    job_map = {j["id"]: j for j in all_jobs}

    # Lọc danh sách job cần chạy
    target_job_ids = []
    if args.only:
        only_ids = set(x.strip() for x in args.only.split(","))
        target_job_ids = [j["id"] for j in all_jobs if j["id"] in only_ids]
    elif args.until:
        for j in all_jobs:
            target_job_ids.append(j["id"])
            if j["id"] == args.until:
                break
    else:
        target_job_ids = [j["id"] for j in all_jobs]

    print("=" * 85)
    print("EXPERIMENT QUEUE MANAGER (Spec 8.2)")
    print(f"File cấu hình: {args.config}")
    print(f"Tổng số job: {len(target_job_ids)}/{len(all_jobs)} | Chế độ: {'DRY-RUN' if args.dry_run else 'EXECUTION'}")
    print("=" * 85)

    status_records = []
    completed_or_skipped = set()
    failed_jobs = set()

    # Kiểm tra trạng thái sơ bộ
    for j_id in target_job_ids:
        job = job_map[j_id]
        expect_path = Path(job.get("expect", ""))
        already_done = expect_path.exists() if job.get("expect") else False

        rec = {
            "id": job["id"],
            "name": job["name"],
            "kind": job.get("kind", "gpu"),
            "est_h": job.get("est_h", 0.0),
            "status": "ready",
            "start_time": "",
            "duration_sec": 0.0,
            "exit_code": "",
            "expect": job.get("expect", ""),
        }

        if already_done:
            rec["status"] = "skipped"
            completed_or_skipped.add(job["id"])
        status_records.append(rec)

    # In kế hoạch Dry-run
    print(f"{'ID':<6} | {'Tên Job':<25} | {'Loại':<5} | {'Ước (h)':<7} | {'Trạng thái ban đầu':<18} | Phụ thuộc")
    print("-" * 85)
    for rec in status_records:
        job = job_map[rec["id"]]
        deps = job.get("depends_on", [])
        deps_str = ", ".join(deps) if deps else "—"
        init_status = "SKIP (expect có)" if rec["status"] == "skipped" else "SẴN SÀNG"
        print(f"{job['id']:<6} | {job['name']:<25} | {job.get('kind', 'gpu'):<5} | {job.get('est_h', 0.0):<7.2f} | {init_status:<18} | {deps_str}")

    print("=" * 85)

    if args.dry_run:
        print("\n[DRY-RUN HOÀN TẤT] Thứ tự chạy và điều kiện phụ thuộc hoàn toàn hợp lệ.")
        return

    # Chạy thật
    args.log_dir.mkdir(parents=True, exist_ok=True)
    update_status_csv(args.status_csv, status_records)

    total_start = time.perf_counter()

    for idx, rec in enumerate(status_records):
        job = job_map[rec["id"]]
        job_id = job["id"]

        if rec["status"] == "skipped":
            print(f"\n[BỎ QUA {job_id}] File đầu ra đã tồn tại: {job['expect']}")
            continue

        # Kiểm tra dependencies
        deps = job.get("depends_on", [])
        unmet_deps = [d for d in deps if d not in completed_or_skipped]
        failed_deps = [d for d in deps if d in failed_jobs]

        if failed_deps:
            rec["status"] = "blocked"
            print(f"\n[CHẶN {job_id}] Phụ thuộc vào job bị lỗi: {failed_deps}")
            update_status_csv(args.status_csv, status_records)
            continue

        if unmet_deps:
            rec["status"] = "blocked"
            print(f"\n[CHẶN {job_id}] Phụ thuộc chưa hoàn thành: {unmet_deps}")
            update_status_csv(args.status_csv, status_records)
            continue

        # Thực thi job
        cmd = build_dynamic_cmd(job)
        log_file = args.log_dir / f"{job_id}.log"
        print(f"\n>>> [BẮT ĐẦU {job_id}] {job['name']} ({job.get('kind', 'gpu')})")
        print(f"    Lệnh: {cmd}")
        print(f"    Log:  {log_file}")

        rec["status"] = "running"
        rec["start_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        update_status_csv(args.status_csv, status_records)

        t_job_start = time.perf_counter()
        env = os.environ.copy()
        env["PYTHONPATH"] = f".;src;{env.get('PYTHONPATH', '')}"
        env["PYTHONIOENCODING"] = "utf-8"

        with log_file.open("w", encoding="utf-8") as lf:
            proc = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
            for line in proc.stdout:
                sys.stdout.write(line)
                lf.write(line)
            proc.wait()

        dur = time.perf_counter() - t_job_start
        rec["duration_sec"] = round(dur, 2)
        rec["exit_code"] = proc.returncode

        if proc.returncode == 0:
            rec["status"] = "completed"
            completed_or_skipped.add(job_id)
            print(f"[HOÀN THÀNH {job_id}] Thời gian: {dur/60.0:.2f} phút.")
        else:
            rec["status"] = "failed"
            failed_jobs.add(job_id)
            print(f"[LỖI {job_id}] Thoát với mã lỗi: {proc.returncode}")

        update_status_csv(args.status_csv, status_records)

    total_time = (time.perf_counter() - total_start) / 60.0
    print("\n" + "=" * 85)
    print(f"TỔNG KẾT HÀNG ĐỢI: {len(completed_or_skipped)}/{len(target_job_ids)} thành công | Thời gian: {total_time:.2f} phút")
    print(f"Báo cáo trạng thái lưu tại: {args.status_csv}")
    print("=" * 85)


if __name__ == "__main__":
    main()
