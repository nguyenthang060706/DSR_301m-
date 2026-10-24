# Kế hoạch 7 ngày hoàn tất thực nghiệm + draft thô (T7 24/10 → T6 30/10/2026)

- **Phiên bản:** 1.0 · rút gọn từ `Kế hoạch từng ngày đến hoàn thành paper (24/10/2026 – 03/01/2027).docx`
- **Căn cứ:** `ke_hoach_nghien_cuu_MASTER_v10.0.md`, `docs/protocol_lock.md`, `docs/label_guide.md` (v2.0)
- **Đọc bởi:** người (nhóm nghiên cứu) và AI agent (Antigravity). Mỗi task có ID để agent tham chiếu.
- **Nguyên tắc nén:** không làm vội từng việc mà (1) chạy **hàng đợi GPU liên tục**, (2) viết code **song song** với gán nhãn, (3) **cắt** hạng mục mở rộng. Nút thắt thật là *gán nhãn người* và *code còn thiếu*, không phải GPU.

---

## 0. Cách dùng file này với agent (đọc trước)

Đặt file tại `docs/PLAN_7_NGAY.md`. Mỗi sáng dán prompt sau, đổi `<N>` thành số ngày (1–7):

```text
Bạn là agent kỹ thuật cho repo DSR_301m-. Đọc docs/PLAN_7_NGAY.md, ke_hoach_nghien_cuu_MASTER_v10.0.md,
docs/protocol_lock.md, docs/label_guide.md. Hôm nay là Ngày <N> của kế hoạch.
1. Chạy "Bước 0" của ngày: kiểm tra trạng thái repo/reports. KHÔNG giả định file tồn tại (data/, reports/, checkpoints/ bị gitignore).
2. Liệt kê các task [C] của ngày + file sẽ tạo/sửa. Task nào chạm mục 4 (Quyết định chờ xác nhận) thì DỪNG và hỏi tôi.
3. Làm lần lượt. Mỗi task: viết code + test, chạy lệnh ở cột "Kiểm tra", dán kết quả thật (không tóm tắt "đã pass").
4. Cuối ngày ghi reports/progress_log.md (xong gì, lệch gì, số liệu lấy từ file nào).
CẤM: đọc/đếm nội dung/huấn luyện/tune/calibrate bằng ảnh Test canteen; hard-code số liệu kết quả; đổi recipe đã khóa;
sửa docs/protocol_lock.md khi tôi chưa xác nhận; chia dữ liệu canteen theo từng ảnh.
```

**Ký hiệu task:** `[H]` việc của người (chụp, gán nhãn, họp, đo thiết bị, xác nhận quyết định) · `[C]` việc code của agent · `[G]` job trong hàng đợi GPU · `[W]` viết paper/slide.

**Quy ước chung cho agent:** Windows + PowerShell, RTX 4050 6GB (một job GPU nặng tại một thời điểm). Mở file luôn `encoding="utf-8"`; script in tiếng Việt cần `sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')`. Chạy module bằng `$env:PYTHONPATH="src"; python -m dsr.<module>`. Mỗi run lưu seed + git commit (xem `write_manifest` ở `kd_common.py`, copy hàm sang module mới vì `kd_common.py` là legacy).

---

## 1. Phạm vi và định nghĩa "xong"

| Có trong 7 ngày | Không có trong 7 ngày |
|---|---|
| Kappa box, chia cụm, **đóng băng Test canteen** | Paper hoàn chỉnh (còn 8 tuần để viết/đệm đến 03/01/2027) |
| Main Tier A: M0, M1, M1b, M2, M3, M4 × 3 seed (có tune N=5) | Mở rộng M5, M7/Canteen-C, KD-det, E3-size |
| Detector E3a–d, E3-curve rút gọn, D2 | Thí nghiệm độ mịn nhãn (~10h GPU) — hoãn, xem mục 9 |
| Tier B {M0, M1, KD tốt nhất} × 3 seed | Few-shot Tier B |
| Lớp `background`, pha 2, E4 oracle vs detected, few-shot Tier A | Slide Report 3 (dựng từ `collect_results.py` sau tuần này) |
| INT8 + đo latency (thiết bị thật nếu có, CPU ONNX nếu không) | Hoàn thiện Intro/Related Work/Discussion |
| Bootstrap, McNemar+Holm, bảng/hình, **draft thô Methods + Results** | |

**Cuối Ngày 7 bạn có:** toàn bộ số liệu thực nghiệm + log kiểm rò rỉ + draft thô. **Không có:** paper hoàn chỉnh.

---

## 2. Giả định và ẩn số (trả lời sau audit Ngày 1)

| Mục | Trạng thái | Cách biết |
|---|---|---|
| Số khay canteen, instance/nhóm, số cụm, số ngày chụp | **CHƯA BIẾT** | `reports/canteen_audit.csv` (D1-C1) |
| Nhãn canteen đã là 3 nhóm hay chỉ class-agnostic | **CHƯA BIẾT** | audit |
| Có điện thoại Android/Raspberry Pi để đo thiết bị thật | **CHƯA BIẾT** | bạn trả lời |
| `m1_vanilla_kd_clean` là run đã tune hay chỉ pilot | **CHƯA BIẾT** | đối chiếu `reports/*history.csv` + manifest |
| Đã có `reports/tune_dkd_summary.csv` (tune M1b xong chưa) | **CHƯA BIẾT** | `Test-Path` ở Bước 0 Ngày 1 |
| Số giờ/epoch thực | Ước lượng | lấy cột `train_time_sec` trong `reports/*_history.csv` |
| Hai người gán nhãn độc lập | Giả định có | bạn xác nhận |

---

## 3. Luật bất khả xâm phạm (trích từ protocol, KHÔNG thương lượng)

1. **Test canteen đóng băng:** không dùng ở bất kỳ bước nào ngoài lần chạy `eval_test_once.py` (Ngày 6): không train, tune, chọn ngưỡng, hard-negative, calibration INT8. Trước Ngày 4 chỉ được *đếm file*.
2. **Chia dữ liệu canteen theo cụm** `cluster_id = YYYYMMDD_Ca`, không bao giờ theo từng ảnh/box. Test gồm các **ngày** chưa xuất hiện ở Train/Pool.
3. **Classifier:** chọn model/siêu tham số/checkpoint theo **Macro-F1 nhãn mịn 30 lớp trên Dev** (`public_dev.csv`). Chỉ số 3 nhóm chỉ để báo cáo cuối.
4. **Detector:** checkpoint theo **AP50 trên Val canteen**; ngưỡng conf/NMS chọn trên Val.
5. **Recipe classifier khóa:** SGD momentum 0.9, lr 0.01, wd 1e-4, CosineAnnealingLR(eta_min=1e-5), 40 epoch, batch 32, RandomResizedCrop(224, 0.8–1.0)+HFlip+TrivialAugmentWide, AMP FP16, channels_last, **label smoothing Teacher=0.0 / Student=0.1** (lưu ý: `train.py` mặc định `--label-smoothing 0.0`, phải truyền `0.1` cho Student), Teacher nhận cùng ảnh augment (online), seed gốc 20261001.
6. **Tune công bằng:** N=5 cấu hình/phương pháp, 1 seed, 20 epoch, median pruning ở epoch 10, cấu hình thắng chạy lại ≥3 seed 40 epoch. Không early stopping ở main runs. Cầu chì: sau epoch 20 Dev Fine F1 < 30% hoặc loss NaN → dừng. **Không cắt N=5.**
7. Teacher ResNet50 (RER 21.60%) và Student/D1 đã khóa: không đổi.
8. Dataset trả `(image, fine_id, coarse_id)`; checkpoint dict gồm `model_state_dict`, `model_name`, `num_classes`, `f1_fine`, `f1_coarse`, `epoch` (không dùng key `state_dict`). Teacher luôn frozen/eval/no_grad; target KL phải `.detach()`.
9. Loss mới phải có unit test (không âm, =0 khi self-distill, tương đương công thức gốc nếu có).
10. Không hard-code số liệu vào tài liệu/slide: lấy từ `reports/*.csv`, ghi nguồn. Kết luận âm (KD không thắng M0; pipeline không hơn D2) được báo cáo nguyên trạng. Không dùng "giải quyết triệt để domain shift"; không gọi ResNet18 là "siêu nhẹ".
11. Trước khi đổi bất kỳ tham số nào trong protocol-lock: dừng, nêu tác động, chờ xác nhận.

---

## 4. Quyết định chờ xác nhận (agent dừng và hỏi nếu task chạm các mục này)

Điền `[x]` khi đã xác nhận. **Mặc định** chỉ được dùng để *chuẩn bị code/test*, chưa dùng để chạy kết quả chính khi chưa tick.

| ✔ | ID | Vấn đề | Mặc định đề xuất | Ảnh hưởng |
|---|---|---|---|---|
| [ ] | D1 | v10 §4.2 yêu cầu `background` cho mọi classifier, nhưng Teacher/M0/Tier A đã train 30 lớp | Giữ 30 lớp cho so sánh KD (RQ1). **Pha 2:** fine-tune ≤10 epoch thêm output `background` cho {M0, M1, KD tốt nhất}; ghi là deviation trong protocol_lock v2.0 | Ngày 5 |
| [ ] | D2 | v10 §5.0 ghi warmup 5 epoch + EMA, code/protocol_lock không dùng | Sửa plan theo code (mọi kết quả đã chạy theo code) | Văn bản |
| [ ] | D3 | AT trong code = 0.5·‖·‖², plan = ‖·‖₂; `kd_loss(tau)` ≠ `dkd_loss(temperature)` | Giữ code; sửa công thức trong plan; thống nhất tên `temperature` | Văn bản + gọi hàm |
| [ ] | D4 | Go/No-go cuối Tuần 5 quá sát; định nghĩa "recall@0.5 ≥ 0.80" chưa rõ ngưỡng conf (`metrics.box.r` của ultralytics là recall tại conf tối ưu F1, **không** phải recall trần) | Gate Go/No-go dời sang **sáng Ngày 5**; recall tính bằng `eval_detector.py` tại conf cố định 0.25 (ngoài ra báo cáo 0.05, 0.001) | Ngày 5 |
| [ ] | D5 | `label_map.csv` v2.1 (31 dòng, có `background`) làm hỏng loader/test/audit 30 lớp | Khôi phục `label_map.csv` = 30 lớp đã khóa; chuyển bản có background sang `label_map_v3_bg.csv` | Ngày 1 |
| [ ] | D6 | FitNet gốc có 2 giai đoạn | Làm FitNet **một giai đoạn** (CE + λ·MSE qua regressor 1×1 conv tại `layer3`), ghi rõ là biến thể | Ngày 1 |
| [ ] | D7 | Pre-label bằng D1 làm GT box lệch về phía detector (annotator có xu hướng chấp nhận gợi ý) → **thổi phồng recall/AP** | Val/Test: gán **không pre-label** (hoặc pre-label bằng detector khác với detector được đánh giá); Train/Pool được pre-label. Ghi vào Limitations | Ngày 2–3 |
| [ ] | D8 | Pha 2 và few-shot dùng loss gì | Pha 2: CE 31 lớp, lr 1e-3 cosine, 10 epoch, có crop-jitter. Few-shot: **group-NLL** = −log Σ_{c∈nhóm(y)} softmax(z)_c (giữ nguyên head mịn) | Ngày 4–5 |
| [ ] | D9 | Bộ seed | `{20261001, 20261002, 20261003}`; seed 20261001 = các run đã có | Toàn bộ |
| [ ] | D10 | Tier B khi KD tốt nhất là M2/M3/M4 (hook chỉ hỗ trợ ResNet) | Tier B dùng KD logit tốt nhất giữa M1/M1b; ghi limitation | Ngày 5–6 |
| [ ] | D11 | Lưới tune cho M1, M3, M4, M2 | Mục 8.3; **đóng băng trước khi chạy**, không sửa sau khi thấy kết quả | Ngày 1–2 |

---

## 5. Tổng quan lịch và phụ thuộc

| Ngày | Chủ đề | Cổng cuối ngày |
|---|---|---|
| **1** · T7 24/10 | Kiểm kê, sửa nền (label_map, TACO leak, FitNet, tune_kd, queue) | `pytest` xanh; có `canteen_audit.csv` |
| **2** · CN 25/10 | Ảnh ghép đúng §3.5, kappa pilot 40 khay, `eval_detector.py` | 40 khay gán độc lập xong |
| **3** · T2 26/10 | Kappa → gán nhãn diện rộng, chia cụm, test guard | Box F1 ≥ 0.80 và κ ≥ 0.70 |
| **4** · T3 27/10 | **Đóng băng Test**, E3a–d trên dữ liệu thật, e2e/bootstrap | Manifest SHA256 + tag `test-frozen` |
| **5** · T4 28/10 | **Go/No-go (sáng)**, background, pha 2, D2, E3-curve | `docs/gonogo_decision.md` |
| **6** · T5 29/10 | Khóa ngưỡng → INT8 → **mở Test đúng 1 lần** → đo thiết bị | `reports/test_open_log.json` |
| **7** · T6 30/10 | Bootstrap, bảng/hình, kiểm rò rỉ, draft thô | Checklist mục 11 |

```text
audit ─► split cụm ─► FREEZE TEST (Ng.4) ─┐
label_map/TACO fix ─► D1(a) ─► prelabel   │
tune_kd+FitNet ─► tune M1,M3,M4,M2 ─► main s1 ─► s2 ─► s3 ─► KD tốt nhất ─► Tier B
E3a–d (cần Val canteen) ─► GO/NO-GO (Ng.5) ─► hard-negative ─► background ─► pha 2 ─┐
khóa ngưỡng + INT8 (Ng.6 sáng) ──────────────────────────────────────────────────► eval_test_once (1 lần)
```

**Rule Go/No-go (sáng Ngày 5):** (1) Test ≥150 khay & ≥200 instance/nhóm, kappa đạt; (2) ≥1 seed Main M0–M4/M1b; (3) D1 recall@0.5 class-agnostic ≥ 0.80 trên Val (định nghĩa theo D4). **ĐẠT** → chạy tiếp. **KHÔNG ĐẠT** → vẫn chạy pipeline và báo cáo trung thực recall thấp; tăng khay thật cho detector trước mọi thí nghiệm mới; không thêm hạng mục mở rộng.

---

## 6. Chi tiết từng ngày

### Ngày 1 · T7 24/10 — Kiểm kê và sửa nền

**Mục tiêu:** biết đang có gì, sửa các lỗi nền chặn mọi thứ phía sau, đưa hàng đợi GPU chạy ngay trong ngày.

**Bước 0 (agent chạy trước, dán kết quả):**

```powershell
git status; git log --oneline -5
Test-Path reports/tune_dkd_summary.csv
Get-ChildItem reports/*_history.csv | Select-Object Name
Get-ChildItem checkpoints -Directory | Select-Object Name
Get-ChildItem data/raw -Directory | Select-Object Name
Get-Content data/mappings/label_map.csv | Measure-Object -Line
```

| ID | Việc | Kiểm tra |
|---|---|---|
| **D1-H1** `[H]` | Chạy `audit_canteen_data.py` (D1-C1), dán `reports/canteen_audit.csv` + 3 dòng tóm tắt. Chọn **nhánh dữ liệu**: **A** tổng ≥450 khay (đủ), **B** 150–449 (ưu tiên Test ≥150 + Val ≥50, giảm Pool), **C** <150 hoặc chưa chụp (chỉ làm classifier + detector công khai, chưa đóng băng Test, chụp tiếp, dời lịch) | Bạn ghi nhánh vào `progress_log.md` |
| **D1-H2** `[H]` | Chọn 40 khay pilot kappa: đủ 3 nhóm, ≥100 box, đa dạng ca/sáng/mức chồng lấp. **Không** đưa chúng vào Test (quy ước đã được tinh chỉnh dựa trên chúng) | Danh sách 40 `image_id` lưu ở `data/kappa_pilot_list.txt` |
| **D1-H3** `[H]` | Tick các quyết định D1–D11 ở mục 4 (ít nhất D5, D6, D9, D11 trước khi agent bắt đầu C2/C4) | Mục 4 đã tick |
| **D1-H4** `[H]` | Sau audit, lập **kế hoạch ngày** `configs/canteen_date_plan.json`: ngày nào làm **Test**, **Val**, **Pool** (chia theo ngày, xem 8.6). Việc này phải xong **trước khi gán nhãn diện rộng** để biết ngày nào được/không được pre-label (D7) | File có 3 danh sách ngày, không trùng nhau |
| **D1-C1** `[C]` | `scripts/audit_canteen_data.py` (spec 8.1). Phải chạy được trên thư mục rỗng (in `[WARN] chưa có dữ liệu`, không crash) | `pytest tests/test_audit_canteen.py -q` (dữ liệu giả trong `tmp_path`) |
| **D1-C2** `[C]` | **D5:** `git log -p data/mappings/label_map.csv` để xác nhận bản 30 lớp. Khôi phục `label_map.csv` = 30 lớp (coarse 0–2); tạo `label_map_v3_bg.csv` (31 lớp, `background` id 30, coarse 3). Thêm `load_label_map(path, include_background=False)`; không đổi chữ ký cũ | `pytest tests/test_data.py tests/test_config.py -q`; `python scripts/audit_week1.py` không có FAIL ở mục 1 |
| **D1-C3** `[C]` | **Sửa rò rỉ val TACO.** Hiện `process_taco.py` ghi `val: images/heldout`, và `train_d1_taco.py` chọn `best.pt` theo val = heldout → heldout (tập OOD) bị dùng để chọn checkpoint. Việc: đổi tên `data/processed/taco_yolo_agnostic` → `..._old`; sửa `process_taco.py` tạo `train / val / heldout`, trong đó `val` = 10% **cuối** của danh sách train (giữ nguyên membership `heldout`: cùng seed 20261012, cùng thứ tự shuffle); `data.yaml` → `val: images/val`; thêm `taco_heldout.yaml`; sửa `train_d1_taco.py` nhận `--name` (mặc định `d1a_taco`) và `--data`. Kết quả cũ (mAP50 ≈ 51.1%, recall ≈ 45.6% trong deck) **phải thay bằng số mới** | Script so sánh `set(heldout_cũ) == set(heldout_mới)` in `[PASS]`; `train ∩ val ∩ heldout = ∅` |
| **D1-C4** `[C]` | **FitNet + tune_kd.** (a) `losses.py`: `FitNetRegressor`, `fitnet_loss`; `train_kd.py`: `--kd-method fitnet`, `--lambda-fitnet`, optimizer gồm tham số regressor, regressor lưu riêng (`regressor_state_dict`), **không** lẫn vào `model_state_dict`. (b) `scripts/tune_kd.py --method {vanilla_kd,attention_transfer,hybrid,fitnet}` tổng quát hóa `tune_dkd.py` (spec 8.3), ghi `reports/tune_<method>_summary.csv` + `configs/best_kd/<method>.json` | `pytest tests/test_fitnet.py tests/test_kd.py -q`; thêm case `fitnet` vào `scripts/smoke_test_kd.py` và chạy pass |
| **D1-C5** `[C]` | `scripts/run_queue.py` + `configs/queue_week.json` (spec 8.2), nạp sẵn Q01–Q26 ở mục 7 | `python scripts/run_queue.py --dry-run` in đúng thứ tự, đúng điều kiện skip |
| **D1-C6** `[C]` | Dọn nền: `git mv` legacy sang `legacy/` (`kd_common.py`, `measure_kl.py`, `eval_holdout.py`, `analyze_soft_targets.py`, `make_trashnet_splits.py`, `inspect_dataset.py`, `count_taco_instances.py`, `src/dsr/test_metrics.py`); `calc_kappa.py` thêm `import time`; `requirements.txt` thêm `ultralytics`, `rembg`, `pytest`, `imagehash` (pin theo `pip show` hiện có; ghi chú AGPL-3.0 của ultralytics). Cập nhật `docs/canteen_quick_checklist.md` và `data/canteen_metadata_template.csv` sang v10 (nhiều vật/khay, box + nhóm 0/1/2, `cluster_id = YYYYMMDD_Ca`, **không** hậu tố vật phẩm như `20261006_trua_hopxop`) | `pytest tests -q` xanh |

**GPU (bắt đầu ngay, không chờ code):** Q01 → Q02 → Q03 → Q04; Q05 (D1(a) retrain) chạy khi D1-C3 xong.

**Cổng cuối ngày:** `pytest tests -q` xanh · `canteen_audit.csv` có · mục 4 đã tick tối thiểu D5/D6/D9/D11 · queue đã chạy ≥ 4 giờ.

---

### Ngày 2 · CN 25/10 — Ảnh ghép, kappa pilot, eval detector

| ID | Việc | Kiểm tra |
|---|---|---|
| **D2-H1** `[H]` | **Pilot kappa 40 khay:** 2 người gán độc lập (CVAT/Roboflow), **box + nhóm 0/1/2**, xuất YOLO vào `data/raw/kappa_pilot/annotator_1/` và `annotator_2/`. **Không pre-label** cho pilot này: nếu cả hai khởi đầu từ cùng box gợi ý thì Box F1 chỉ đo mức độ đồng ý khi chỉnh sửa, không đo vẽ box độc lập. Lưu ý `auto_prelabel_canteen.py` ghi `class_id = 0` (waste, **không phải** `organic`) | Mỗi thư mục có đủ file `.txt` của 40 khay |
| **D2-C1** `[C]` | Viết lại `scripts/synthesize_trays.py` theo MASTER v10 §3.5 (spec 8.4): tách nền bằng `rembg` cho tập con ~3.000 vật cân bằng nhóm, **chỉ ảnh thuộc `public_train.csv`**, feather + bóng đổ + chỉnh màu, box từ mask, IoU ≤ 0.3, **có `--seed`**, xuất nhãn class-agnostic *và* 3 nhóm. Sinh 3.000–5.000 ảnh (job CPU, chạy song song GPU) | Test: không có đường dẫn nào thuộc `public_dev.csv` trong manifest; chạy 2 lần cùng seed cho cùng ảnh |
| **D2-C2** `[C]` | Test **DKD ≡ Hinton**: tách `dkd_components(...)` trả TCKD/NCKD *theo mẫu* (không đổi giá trị `dkd_loss`); test `mean(TCKD + (1−p_t^T)·NCKD) == kd_loss` (cùng T², atol 1e-5). Hiện `dkd_loss` chỉ nhận α, β vô hướng nên không test trực tiếp được | `pytest tests/test_kd.py -q` |
| **D2-C3** `[C]` | `scripts/eval_detector.py` (spec 8.5): AP50, AP50–95, precision, **recall@IoU0.5 class-agnostic tại conf cố định** (mặc định 0.25; thêm 0.05 và 0.001), trên `--split val|heldout` | Chạy trên TACO val/heldout mới (D1-C3), in bảng; recall khớp với tự tính tay trên 5 ảnh |
| **D2-C4** `[C]` | `scripts/validate_labels.py`: mọi `.txt` có `class_id ∈ {0,1,2}` (hoặc `{0}` với bản agnostic), tọa độ ∈ [0,1], w,h > 0, box < 16×16 px bị cảnh báo; **báo FAIL nếu 100% box cùng class 0** (dấu hiệu quên gán nhóm sau pre-label) | `pytest tests/test_validate_labels.py -q` |
| **D2-H2** `[H]` | Phần còn lại của khay chưa gán: bắt đầu gán Train/Pool (chỉ các ngày thuộc **Pool** theo `canteen_date_plan.json` mới được pre-label bằng `auto_prelabel_canteen.py`, cần `models/checkpoints/d1b_combined_best.pt`, nếu không có thì dùng D1(a) mới; ngày Test/Val gán không pre-label) | `validate_labels.py` không FAIL |
| **D2-H3** `[H]` | Chụp 30–50 ảnh **nền trống** (khay/bàn không vật) vào `data/raw/canteen_backgrounds/`; gắn `cluster_id` theo ngày/ca như ảnh khay, không chụp vào ngày dự kiến làm Test. Dùng làm nền cho ảnh ghép và làm mẫu `background` | Số ảnh ≥ 30 |

**GPU:** Q05 (nếu chưa), Q06 tune M1 → Q07 main M1 s1 → Q08 tune M3 → Q09 main M3 s1.

**Cổng cuối ngày:** 40 khay có nhãn của cả 2 người · `eval_detector.py` chạy được · ảnh ghép sinh xong (hoặc đang sinh) · tune M1 xong.

---

### Ngày 3 · T2 26/10 — Kappa và chia cụm

| ID | Việc | Kiểm tra |
|---|---|---|
| **D3-H1** `[H]` | Chạy kappa: `python scripts/calc_agreement_multiobject.py --dir1 data/raw/kappa_pilot/annotator_1 --dir2 data/raw/kappa_pilot/annotator_2`. **Cổng: Box F1 ≥ 0.80 và Cohen's κ ≥ 0.70.** Không đạt → họp đối soát từng box bất đồng, bổ sung `label_guide.md` §3–4, gán lại **40 khay mới** (mất ~1 ngày, kích hoạt mục 9) | Dán nguyên output vào `reports/kappa_box_report.md` |
| **D3-H2** `[H]` | Gán nhãn diện rộng theo quota (MASTER v10 §3.3): **Test** ≥150 khay & ≥200 instance/nhóm · **Val** ≥50 khay & ≥50 instance/nhóm · **Pool/Train thật** ≥250 khay. Ngày Test/Val theo `canteen_date_plan.json` **không pre-label** (D7). Mỗi người một nửa; ca khó gắn `ambiguous=True` | `audit_canteen_data.py` báo số khay/instance còn thiếu |
| **D3-C1** `[C]` | Sửa `scripts/split_canteen_by_cluster.py` (spec 8.6): chia ở mức **ngày**, thêm **Pool**, đếm instance/nhóm theo tập, `assert` không cụm nào ở 2 tập, ghi 2 bộ nhãn (`labels_agnostic/` cho D1, `labels_3cls/` cho D2 và classifier), ghi `manifest.json` (SHA256 từng file) | `pytest tests/test_split_cluster.py -q` (có ca ngày chia hai ca, ca ít cụm) |
| **D3-C2** `[C]` | `src/dsr/test_guard.py` + tích hợp (spec 8.7) để **máy chặn** việc đọc Test ngoài `eval_test_once.py` | Test: ép train đọc đường dẫn Test → `RuntimeError` |
| **D3-C3** `[C]` | `scripts/extract_crops.py` + `JitterBoxCrop` (spec 8.8): crop box thật hoặc box detector (padding 10–20%, N_max=8) | Chạy thử 5 khay Train; ảnh crop mở ra đúng vật |
| **D3-C4** `[C]` | `scripts/eval_canteen_classifier.py`: Macro-F1 3 nhóm (từ output 30 hoặc 31 lớp, bỏ `background`), Critical Error (Hữu cơ ↔ Tái chế khô), ECE, confusion matrix; chỉ nhận `--split val` cho đến Ngày 6 | Chạy trên Val bằng M0 s1 |

**GPU:** Q10 tune M4 → Q11 main M4 s1 → Q12 tune M2 → Q13 main M2 s1.

**Cổng cuối ngày:** kappa **đạt** (nếu không, kích hoạt mục 9) · `split_canteen_by_cluster.py` đã sửa và chạy thử trên dữ liệu thật mà **chưa** ghi Test cuối.

---

### Ngày 4 · T3 27/10 — Đóng băng Test, detector trên dữ liệu thật

| ID | Việc | Kiểm tra |
|---|---|---|
| **D4-H1** `[H]` | Hoàn tất gán nhãn, đạt quota. Chạy `split_canteen_by_cluster.py` bản cuối → **đóng băng Test** (`git tag test-frozen`, commit `canteen_test_manifest.json`). Từ lúc này không ai xem Test | `reports/canteen_audit.csv` thể hiện đủ quota; manifest có SHA256 |
| **D4-C1** `[C]` | `scripts/eval_e2e.py` (spec 8.9): ghép box tham lam IoU ≥ 0.5, P/R/F1 theo nhóm, Macro-F1 e2e, mAP 3 lớp, Critical Error mức box, phân rã lỗi, MAE đếm vật; chế độ `--boxes oracle|detected`; xuất bảng **theo từng khay kèm `cluster_id`** để bootstrap | `pytest tests/test_eval_e2e.py -q` (ca thủ công: 1 khay 3 vật, có bỏ sót + nhầm nhóm + FP) |
| **D4-C2** `[C]` | `scripts/stats_bootstrap.py` (spec 8.10): cluster bootstrap B=2000 ghép cặp, McNemar + Holm, effect size, in số cụm | Test trên dữ liệu giả: CI chứa giá trị thật; hai mô hình y hệt → hiệu = 0 |
| **D4-C3** `[C]` | `scripts/few_shot_finetune.py` (spec 8.11) | Chạy k=5 head-only trên Pool, không đụng Test/Val |
| **D4-C4** `[C]` | Sinh `data/processed/e3/*.yaml` cho E3a–d (train/val liệt kê tường minh; `val` = **Val canteen**); mọi yaml đi qua `test_guard.assert_no_test_in_yaml` | `pytest tests/test_test_guard.py -q` |

**GPU:** Q14 (main seed 2 cho M1, M1b, M3, M4, M2) → khi Val canteen đóng băng: Q15 (E3a, E3b) → Q16 (E3d, E3c × 3 seed).

**Cổng cuối ngày:** Test đóng băng, tag có · Q14 chạy ≥ 50% · E3a–b đã bắt đầu.

---

### Ngày 5 · T4 28/10 — Go/No-go, background, pha 2

| ID | Việc | Kiểm tra |
|---|---|---|
| **D5-H1** `[H]` | **Sáng: họp Go/No-go** theo rule ở mục 5. Điền `docs/gonogo_decision.md` (nhánh ĐẠT / KHÔNG ĐẠT, số liệu lấy từ CSV, ký tên người quyết). Recall D1 tính bằng `eval_detector.py` trên **Val canteen** (D4) | File có đủ 3 điều kiện, mỗi điều kiện kèm đường dẫn CSV |
| **D5-H2** `[H]` | Đảm bảo đã có 30–50 ảnh **nền trống** (khay/bàn không vật) thuộc các cụm Train/Pool (không chụp thêm vào ngày Test). Chuẩn bị thiết bị đo (cài ONNX Runtime/app, chạy thử một model nhỏ) | Thư mục `data/raw/canteen_backgrounds/` |
| **D5-C1** `[C]` | `scripts/build_background_crops.py` (spec 8.15): (i) crop ngẫu nhiên từ ảnh nền trống; (ii) **hard-negative:** chạy D1 tốt nhất trên **khay Train** (không Test), lấy box dự đoán có IoU < 0.1 với mọi box thật. Số mẫu `background` ≈ trung bình số mẫu một lớp mịn của Train. Chia theo cụm | Manifest liệt kê `cluster_id`; `test_guard` không báo vi phạm |
| **D5-C2** `[C]` | `scripts/finetune_background.py` (pha 2, spec 8.15, **cần D1 và D8 đã tick**): mở rộng `fc` từ 30 → 31 đầu ra (chép 30 hàng cũ), CE 31 lớp, có `JitterBoxCrop`, lr 1e-3 cosine, 10 epoch, chọn checkpoint theo **Dev Macro-F1 mịn 30 lớp**; ghi thêm recall `background` trên Val bg crops. Báo Δ Macro-F1 Dev trước/sau, không tự đặt ngưỡng chấp nhận mới | `reports/phase2_<run>_history.csv`, checkpoint `num_classes=31` |
| **D5-C3** `[C]` | `scripts/tune_thresholds.py` (spec 8.15): lưới conf × NMS-IoU × công thức điểm (`det×cls` hoặc trung bình), tối đa Macro-F1 e2e trên **Val**, dùng classifier **M0 pha 2** để không thiên vị phương pháp nào; một bộ ngưỡng cho mọi pipeline | `reports/threshold_search.csv`, chỉ đọc Val |
| **D5-C4** `[C]` | `scripts/benchmark_pipeline_latency.py` (spec 8.12), chạy thử CPU mô phỏng | `reports/pipeline_latency_cpu.csv` |
| **D5-C5** `[C]` | Soạn **bản nháp** `docs/protocol_lock.md` v2.0: recipe detector §5.0b, ngưỡng, D1–D3, deviation pha 2, FitNet một giai đoạn. **Chưa ghi đè** cho đến khi tôi xác nhận (xuất ra `docs/protocol_lock_v2_DRAFT.md`) | Diff so với bản hiện tại |

**GPU:** Q17 (main seed 3) · Q18 (D2 × 3 seed) · Q19 (E3-curve) · Q20 (Tier B M0, M1 × 3 seed) · Q21 (pha 2, khi D5-C1/C2 xong). Khi Q17 xong → xác định **KD tốt nhất** = mean Macro-F1 mịn Dev (3 seed) cao nhất trong {M1b, M2, M3, M4}.

**Cổng cuối ngày:** `gonogo_decision.md` có · mean±std 3 seed cho M0–M4/M1b · background crops + pha 2 đã chạy ít nhất M0 s1.

---

### Ngày 6 · T5 29/10 — Khóa → INT8 → mở Test một lần → đo thiết bị

**Thứ tự bắt buộc (không đảo):** khóa ngưỡng → export + INT8 (calibration chỉ Dev + Val) → xác nhận điều kiện → chạy `eval_test_once.py` → đo thiết bị. Lý do: các model INT8 phải nằm trong **cùng một lần** mở Test.

| ID | Việc | Kiểm tra |
|---|---|---|
| **D6-C1** `[C]` | Chạy `tune_thresholds.py` (nếu chưa) → khóa ngưỡng conf/NMS + công thức điểm, checkpoint detector (AP50 Val), KD tốt nhất. Sinh `configs/test_eval_manifest.json`: liệt kê **mọi** checkpoint/ONNX + SHA256 + ngưỡng sẽ đưa vào Test. Sau khi tôi xác nhận: ghi đè `docs/protocol_lock.md` → v2.0 | Manifest có hash; `git tag pre-test-open` |
| **D6-C2** `[C]` | Export ONNX FP32 + INT8 static (QDQ) cho D1 và classifier Tier B (đã pha 2): calibration đúng 200 ảnh phân tầng (classifier: Dev công khai + crop Val; detector: ảnh khay Val). Đo sụt trên **Val**: AP50 D1 ≤ 3.0 điểm, Macro-F1 Tier B ≤ 2.0 điểm. Nếu TFLite không export được trên Windows, dùng ONNX Runtime INT8 và **ghi rõ runtime** | `reports/int8_val_drop.csv`; calibration reader `assert_no_test` |
| **D6-C3** `[C]` | `scripts/eval_test_once.py` (spec 8.13) | Dry-run `--dry-run` liệt kê đúng danh sách, không đọc ảnh Test |
| **D6-H1** `[H]` | **Điều kiện mở Test (tick hết mới chạy):** ☐ ngưỡng đã khóa ☐ KD tốt nhất đã chọn ☐ manifest + tag `pre-test-open` ☐ protocol_lock v2.0 đã ghi ☐ Q17–Q25 đã xong (không còn run nào đổi checkpoint hoặc ONNX) ☐ bạn nói "OK mở Test" | Bạn chạy lệnh ở mục 12.3 |
| **D6-C4** `[C]` | Sau khi Test chạy: **không chạy lại**, không chỉnh ngưỡng. Nếu phát hiện bug sau khi xem, ghi vào `reports/test_open_log.json` (có lý do) và báo cáo trong paper như một lần chạy lại có điều kiện, không âm thầm sửa | `test_open_log.json` có 1 mục |
| **D6-H2** `[H]` | **Đo thiết bị thật** (Android/Raspberry Pi): ghi SoC, runtime + phiên bản, nhiệt độ ban đầu, tắt tiết kiệm pin, độ sáng cố định; ≥50 warm-up, 200 lần đo; N ∈ {1,3,6}; đo cả D2; lặp 3 lần; ghi throttling, RAM đỉnh, kích thước file. **Không có thiết bị:** dùng CPU ONNX (1 thread) và ghi rõ trong bảng là *mô phỏng* | `reports/pipeline_latency_device.csv` |

**GPU:** Q22 (Tier B KD tốt nhất × 3 + pha 2 Tier B) · Q23 (few-shot Tier A trên Val) · Q24 (OOD RealWaste + TACO-heldout) · Q25 (INT8) · Q26 (`eval_test_once`, đúng 1 lần).

**Cổng cuối ngày:** `reports/test_open_log.json` đúng 1 mục · mọi CSV Test có cột `cluster_id` hoặc `image_id` để bootstrap.

---

### Ngày 7 · T6 30/10 — Thống kê, bảng/hình, kiểm rò rỉ, draft thô

| ID | Việc | Kiểm tra |
|---|---|---|
| **D7-C1** `[C]` | Chạy `stats_bootstrap.py`: cluster bootstrap B=2000 ghép cặp (Macro-F1, F1 từng nhóm, AP50, hiệu hai mô hình), McNemar + Holm trên cùng tập crop, effect size; **ghi số cụm Test** | `reports/stats_*.csv` |
| **D7-C2** `[C]` | `scripts/collect_results.py` (spec 8.14): gom `reports/*.csv` → bảng Markdown/CSV (mean±std theo seed, Dev vs Val vs Test, nguồn file dưới mỗi bảng) + hình: confusion matrix, đường cong few-shot, E3-curve, Pareto Macro-F1/mAP vs latency vs size vs RAM, phân rã lỗi | `reports/results/` |
| **D7-C3** `[C]` | `scripts/audit_leakage.py` → `reports/leakage_audit.md`: chứng minh bằng manifest rằng Test không xuất hiện trong mọi file train/val/yaml/calibration/hard-negative; không cụm nào ở 2 tập; `public_train ∩ public_dev = ∅`; ảnh ghép chỉ từ `public_train`; TACO heldout không dùng chọn checkpoint | Mọi dòng `[PASS]` |
| **D7-C4** `[C]` | Viết lại `audit_weeks_1_and_2.py` thành `audit_v10.py` theo label_map 30 lớp + v3_bg; cập nhật `README.md` sang v10 | `python scripts/audit_v10.py` không FAIL |
| **D7-W1** `[W]` | Methods nháp: dữ liệu và chia tập, recipe, KD losses (sửa D3), detector, quy ước nhãn + kappa | Mỗi con số có nguồn `reports/` |
| **D7-W2** `[W]` | Results nháp RQ1–RQ3 từ bảng do `collect_results.py` sinh; báo cáo kết quả âm nguyên trạng | Không gõ tay số liệu |
| **D7-W3** `[W]` | Danh sách Limitations (mục 10) | 1 đoạn/giới hạn |

**Cổng cuối tuần:** mục 11.

---

## 7. Hàng đợi GPU (nạp vào `configs/queue_week.json`)

Giờ là **ước lượng** (nguồn: `protocol_lock.md` ~55 s/epoch cho M0; MASTER v10 §9.1 ~110–130 s/epoch cho KD ≈ 1.3–1.5h/run 40 epoch, tune ≈ 2.7h/phương pháp, detector ≈ 0.63h/run 50 epoch). Thay bằng số thật từ cột `train_time_sec` trong `reports/*_history.csv` ngay Ngày 1. Job bỏ qua nếu `expect` đã tồn tại (an toàn khi chạy lại).

| ID | Job | Phụ thuộc | Ước (h) | Ngày | `expect` (skip nếu có) |
|---|---|---|---|---|---|
| Q01 | `tune_dkd.py` (M1b) | — | 2.7 | 1 | `reports/tune_dkd_summary.csv` |
| Q02 | M0 ResNet18 seed 20261002 | — | 0.65 | 1 | `reports/m0_resnet18_s20261002_history.csv` |
| Q03 | M0 ResNet18 seed 20261003 | — | 0.65 | 1 | `..._s20261003_history.csv` |
| Q04 | Main M1b seed 20261001 (cfg thắng Q01) | Q01 | 1.4 | 1 | `reports/m1b_dkd_s20261001_history.csv` |
| Q05 | D1(a) retrain trên TACO đã sửa val | D1-C3 | 0.6 | 1 đêm | `runs/detect/d1a_taco/weights/best.pt` |
| Q06 | `tune_kd.py --method vanilla_kd` (M1) | D1-C4 | 2.7 | 2 | `reports/tune_vanilla_kd_summary.csv` |
| Q07 | Main M1 s1 | Q06 | 1.4 | 2 | `reports/m1_hinton_s20261001_history.csv` |
| Q08 | tune M3 (`attention_transfer`) | D1-C4 | 2.7 | 2 | `reports/tune_attention_transfer_summary.csv` |
| Q09 | Main M3 s1 | Q08 | 1.4 | 2 | `reports/m3_at_s20261001_history.csv` |
| Q10 | tune M4 (`hybrid`) | D1-C4 | 2.7 | 3 | `reports/tune_hybrid_summary.csv` |
| Q11 | Main M4 s1 | Q10 | 1.4 | 3 | `reports/m4_hybrid_s20261001_history.csv` |
| Q12 | tune M2 (`fitnet`) | D1-C4 | 2.7 | 3 | `reports/tune_fitnet_summary.csv` |
| Q13 | Main M2 s1 | Q12 | 1.4 | 3 | `reports/m2_fitnet_s20261001_history.csv` |
| Q14 | Main seed 2: M1, M1b, M2, M3, M4 (5 job) | Q04,Q07,Q09,Q11,Q13 | 7.0 | 3–4 | 5 file history `_s20261002` |
| Q15 | E3a (TACO), E3b (TACO + ảnh ghép), `val` = Val canteen | Val đóng băng, D2-C1 | 1.3 | 4 | `runs/detect/e3a/…`, `e3b/…` |
| Q16 | E3d (chỉ khay thật), E3c (b + khay thật) × 3 seed | Q15, Pool | 2.5 | 4 | `runs/detect/e3c_s*/…` |
| Q17 | Main seed 3: M1, M1b, M2, M3, M4 | Q14 | 7.0 | 5 | 5 file history `_s20261003` |
| Q18 | D2 (YOLO 3 lớp, không KD) × 3 seed | Pool | 2.0 | 5 | `runs/detect/d2_s*/…` |
| Q19 | E3-curve n ∈ {50, 250} khay × 3 seed (n=0 = Q15/Q16 không fine-tune) | Q16 | 2.0 | 5 | `runs/detect/e3curve_*/…` |
| Q20 | Tier B MobileNetV3-Small: M0 × 3, M1 × 3 | Q06 | 5.4 | 5 | `reports/tierb_*_history.csv` |
| Q21 | Pha 2 `background` Tier A: {M0, M1, KD tốt nhất} × 3 seed | D5-C1, D5-C2, Q17 | 2.0 | 5–6 | `reports/phase2_*_history.csv` |
| Q22 | Tier B KD tốt nhất × 3 + pha 2 Tier B | Q17, Q20 | 3.5 | 6 | `reports/tierb_*_history.csv` |
| Q23 | Few-shot Tier A trên Val: k ∈ {5,10,20,50} × 3 seed lấy mẫu × {head, full} × {M0, M1, M1b, KD tốt nhất}, ckpt gốc 30 lớp seed 20261001 (≈ 96 run nhỏ) | Q17 | 1.5 | 6 | `reports/fewshot_val.csv` |
| Q24 | OOD: RealWaste (classifier), TACO-heldout (D1 E3a–c) | Q15–Q16, Q21 | 0.3 | 6 | `reports/ood_*.csv` |
| Q25 | Export ONNX + INT8 (D1 và Tier B) | D6-C1 | 0.5 | 6 | `deploy/onnx/*_int8.onnx` |
| Q26 | `eval_test_once.py` | tất cả + xác nhận `[H]` | 1.0 | 6 | `reports/test_open_log.json` |
| | **Tổng** | | **≈ 58** | | (≈ 56 nếu Q01 đã có) |

**Đọc kết quả:** GPU **không** phải nút thắt (≈ 58h so với ~140h khả dụng); phần dư dùng để **rerun lỗi**, không thêm thí nghiệm. Con số ~90–100h trong bản trước tính trước khi cắt độ mịn nhãn, few-shot Tier B và E3-curve đầy đủ.

---

## 8. Đặc tả các script mới

### 8.1 `scripts/audit_canteen_data.py`
`--metadata data/canteen_metadata.csv --images data/raw/canteen_trays --labels data/raw/canteen_labels --out reports/canteen_audit.csv [--no-hash]`. Xuất dạng `metric,value,note`: số ảnh, số có nhãn, nhãn rỗng, ảnh thiếu nhãn/nhãn mồ côi, tổng box, **box theo nhóm** (hoặc "không biết nhóm" nếu mọi box là class 0), số cụm, số ngày, số khay/cụm, `cluster_id` sai định dạng (regex `^\d{8}_[a-z]+$`), MD5 trùng, pHash gần trùng (Hamming ≤ 5) trong và giữa cụm, số `ambiguous`, thiếu `annotator_*`. Sau khi Test đóng băng chỉ chạy với `--no-hash` và chỉ đếm file.

### 8.2 `scripts/run_queue.py`
`--config configs/queue_week.json [--dry-run] [--only Q07,Q09] [--until Q14]`. Mỗi job: `{"id","cmd","depends_on":[],"expect":"path","est_h":0.65,"kind":"gpu|cpu"}`. Chạy tuần tự theo phụ thuộc, **một job GPU tại một thời điểm** (job `cpu` có thể chạy song song tối đa 1); bỏ qua nếu `expect` tồn tại; log vào `reports/queue_logs/<id>.log`; job lỗi → đánh dấu `failed`, các job phụ thuộc → `blocked`, job độc lập vẫn chạy; ghi `reports/queue_status.csv` (id, trạng thái, giờ thực, mã thoát). Job `main_kd` đọc `configs/best_kd/<method>.json` để dựng lệnh. Có thể dừng và chạy lại an toàn.

### 8.3 `scripts/tune_kd.py` và lưới tune (D11)
Tổng quát hóa `tune_dkd.py`: 5 cấu hình · 1 seed 20261001 · 20 epoch · CosineAnnealingLR(T_max=20) · mốc epoch 10 cắt cấu hình dưới median · chọn thắng theo **best Dev Fine Macro-F1**. Lưới ghi trong `configs/tune_grid/<method>.json`, **commit trước khi chạy**:

| Phương pháp | Lưới 5 cấu hình |
|---|---|
| `vanilla_kd` (M1) | (τ, α) ∈ {(3,1.0), (2,1.0), (4,1.0), (6,1.0), (3,0.5)} |
| `attention_transfer` (M3) | β_AT ∈ {0.5, 1, 2, 4, 8} |
| `hybrid` (M4) | (τ, α, β_AT) ∈ {(3,1,2), (3,1,1), (3,0.5,2), (3,1,4), (4,1,2)} |
| `fitnet` (M2) | λ ∈ {0.1, 0.5, 1, 2, 5} |
| `dkd` (M1b) | giữ nguyên `CONFIGS` trong `tune_dkd.py` |

Gọi đúng tên tham số: `kd_loss(..., tau=)`, `dkd_loss(..., temperature=)` (D3). AT/hybrid/FitNet chỉ chạy với họ ResNet (hook).

### 8.4 `scripts/synthesize_trays.py` (viết lại)
`--num_images 4000 --seed 20261012 --out data/processed/synthetic_trays_v2`. (1) Chọn ~3.000 vật cân bằng nhóm **chỉ từ `public_train.csv`**, tách nền bằng `rembg`, cache RGBA ở `data/processed/cutouts/`, loại cutout có diện tích mask ngoài 3–90% (ngưỡng đề xuất). (2) Dán 3–6 vật khác loại lên ảnh nền trống thật (nếu chưa có thì dùng màu nền); xoay ±30°, scale 15–35% chiều rộng, feather viền, bóng đổ, chỉnh sáng theo nền. (3) Box = bbox chặt của mask, IoU ≤ 0.3. (4) Chia train/val theo **ảnh nguồn** (85/15). (5) Xuất `labels_agnostic/` và `labels_3cls/` (nhóm lấy từ `label_map.csv` của lớp nguồn) + `synthetic_manifest.csv`. Ảnh ghép **chỉ** là dữ liệu train detector, không bao giờ vào Val/Test.

### 8.5 `scripts/eval_detector.py`
`--weights --data-yaml --split val|heldout|test --conf 0.25 --imgsz 416`. Tự ghép box tham lam theo confidence, IoU ≥ 0.5, mỗi box thật ghép một lần. Báo AP50, AP50–95 (ultralytics), precision, **recall tại conf cố định** (0.25, 0.05, 0.001). `--split test` bị `test_guard` chặn ngoài `eval_test_once.py`.

### 8.6 `scripts/split_canteen_by_cluster.py` (sửa)
Gom khay theo **ngày** (`cluster_id = YYYYMMDD_Ca` → ngày = 8 ký tự đầu). Nếu có `--date-plan configs/canteen_date_plan.json` thì theo đó; nếu không, xáo ngày bằng seed 20261012 và gán lần lượt vào **test** đến khi đạt ≥150 khay & ≥200 instance/nhóm, rồi **val** (≥50 khay & ≥50 instance/nhóm), phần còn lại **pool** (= train detector thật = pool few-shot). Ghi `images/ labels_agnostic/ labels_3cls/` theo tập, `canteen_split_report.csv` (khay, instance/nhóm, số cụm mỗi tập), `data/splits/canteen_test_manifest.json` (SHA256). `assert` không ngày/cụm nào ở hai tập; **nếu số ngày < 5 thì dừng và báo** (chia theo ca chỉ khi bạn chấp nhận và ghi vào Limitations, vì ca cùng ngày có tương quan).

### 8.7 `src/dsr/test_guard.py`
Đọc `data/splits/canteen_test_manifest.json`. `assert_not_test(paths)` và `assert_no_test_in_yaml(yaml)` raise `RuntimeError` nếu có đường dẫn thuộc Test, trừ khi `DSR_ALLOW_TEST=1` (chỉ do `eval_test_once.py` đặt trong tiến trình). Gắn vào: `WasteDataset`/crop dataset canteen, mọi nơi sinh yaml YOLO, calibration reader, `build_background_crops.py`, `few_shot_finetune.py`, `tune_*`.

### 8.8 `scripts/extract_crops.py` và `JitterBoxCrop`
`--split val|pool --boxes gt|det --pad 0.15 --nmax 8 [--det-weights --conf --nms]`. Xuất crop + `crops_manifest.csv` (`crop_id, image_id, cluster_id, box, source, gt_group, iou_with_gt`). Với `det`: sắp theo confidence, cắt N_max=8. `JitterBoxCrop(shift=0.10, scale=0.15, pad=(0.05, 0.20))` chỉ dùng ở **pha 2 và few-shot** (các run Main đã khóa không có jitter; ghi rõ deviation ở D1). Với ảnh Kaggle một vật, "box" = toàn ảnh.

### 8.9 `scripts/eval_e2e.py`
`--boxes oracle|detected --det-weights ... --cls-ckpt ... --split val|test`. Ghép box dự đoán với box thật (IoU ≥ 0.5, tham lam theo confidence); dự đoán đúng khi ghép được **và** nhóm đúng. Xuất: P/R/F1 theo nhóm, Macro-F1 e2e, mAP 3 lớp, Critical Error mức box, MAE đếm vật; **phân rã lỗi** mỗi box thật vào đúng 1 loại: bỏ sót (max IoU < 0.1), định vị kém (0.1–0.5), nhầm nhóm, đúng; dương tính giả: nền / trùng lặp / nhóm sai. Xuất `per_tray.csv` (`image_id, cluster_id, TP/FP/FN theo nhóm`). **Oracle-box vs detected-box** (E4) dùng cùng classifier → hiệu = phần mất do detector.

### 8.10 `scripts/stats_bootstrap.py`
Cluster bootstrap theo `cluster_id`, B=2000, seed cố định, lấy mẫu lại cụm có hoàn lại, **ghép cặp** (cùng tập cụm cho hai mô hình), percentile 95% CI cho Macro-F1, F1 từng nhóm, AP50, hiệu hai mô hình. McNemar ghép cặp trên cùng tập crop + hiệu chỉnh Holm; báo effect size. Luôn in số cụm; ít cụm → CI rộng và nói thẳng.

### 8.11 `scripts/few_shot_finetune.py`
`--ckpt --model --k 0|5|10|20|50 --seed --mode head|full`. Lấy k crop/nhóm từ **pool** theo cụm (xáo cụm, nhặt crop đến đủ k/nhóm; thiếu thì báo lỗi). **head:** đóng băng backbone, lr 1e-3, 10 epoch. **full:** lr 1e-4 cosine, wd 1e-4, 15 epoch. Loss **group-NLL** (D8). Dùng checkpoint **epoch cuối** (không chọn trên Test). `k=0` không huấn luyện (zero-shot). Checkpoint gốc: bản 30 lớp seed 20261001 của mỗi mô hình (few-shot đánh giá trên crop thật, không cần `background`). Lưu checkpoint để `eval_test_once.py` chấm một lần.

### 8.12 `scripts/benchmark_pipeline_latency.py`
`T_pipeline = T_detector + N·T_classifier`, N ∈ {1, 3, 6}, gồm cả tiền xử lý, NMS, crop; ≥50 warm-up, 200 lần đo; Median/P95/Std; thêm biến thể suy luận crop theo lô. Ghi nhiệt độ ban đầu, runtime + phiên bản, thread. D2 đo cùng điều kiện. Cột `platform` ghi rõ `cpu_sim` hay tên SoC.

### 8.13 `scripts/eval_test_once.py`
Đọc `configs/test_eval_manifest.json`; kiểm SHA256 mọi artifact; yêu cầu git sạch và tag `pre-test-open`; **từ chối chạy nếu `reports/test_open_log.json` đã tồn tại** (`--force` phải kèm `--reason` và vẫn được ghi log). Chạy một lượt: classifier trên crop thật (Tier A và Tier B, zero-shot và few-shot), E4 oracle vs detected, D2, E3a–c, FP32 và INT8. Ghi `reports/test/*.csv` kèm `cluster_id` và `image_id`, rồi ghi log (thời gian, commit, hash manifest). `--dry-run` chỉ in danh sách.

### 8.14 `scripts/collect_results.py`
Quét `reports/**/*.csv`, nhóm theo (phương pháp, kiến trúc, seed), tính mean ± std (nêu rõ n seed), tách cột Dev / Val / Test, xuất `reports/results/*.csv|md` với dòng "Nguồn: <file>". Không có dữ liệu → ô ghi `chưa có`, không điền số.

### 8.15 `build_background_crops.py`, `finetune_background.py`, `tune_thresholds.py`
Đã mô tả ở D5-C1/C2/C3. Quy tắc chung: không đọc Test; `background` chia train/val theo cụm; pha 2 lưu `has_background: True` trong checkpoint; không đè lên checkpoint 30 lớp đã khóa (lưu `checkpoints/<run>_bg/`).

---

## 9. Khi trễ: cắt gì, theo thứ tự

| Tình huống | Hành động |
|---|---|
| **Kappa trượt** (+1 ngày) | Gán lại 40 khay mới; dời Go/No-go sang sáng Ngày 6, mở Test ở Ngày 7; cắt E3-curve còn n ∈ {0, 250} |
| **Nhánh dữ liệu B** (150–449 khay) | Giữ Test ≥150 và Val ≥50; giảm Pool; E3-curve n ∈ {0, 50} nếu Pool < 250 |
| **Nhánh dữ liệu C** (<150 khay) | Không đóng băng Test; làm Main Tier A + Tier B + detector công khai; kéo dài thu thập; báo cáo trung thực phần chưa làm |
| **D1 recall < 0.80** | Không thêm hạng mục mở rộng; thêm khay thật cho detector; vẫn chạy E4 và báo cáo recall thấp như một kết quả |
| **GPU/queue lỗi** | Dùng phần GPU dư để rerun; tuyệt đối không giảm N=5 hay bỏ seed của Main |
| **Thiếu giờ** | Cắt theo thứ tự: E3-curve → seed thứ 3 của Tier B → few-shot k=20 → D2 (cắt **cuối cùng**) |
| **Không đo được thiết bị thật** | Báo cáo CPU ONNX và ghi rõ là mô phỏng |

**Không bao giờ cắt:** N=5 tune; đóng băng Test và chia theo cụm; bootstrap ghép cặp theo cụm; log kiểm rò rỉ.
**Đã hoãn:** thí nghiệm độ mịn nhãn (RQ1 phần "30 lớp rồi gộp" vs "3 nhóm trực tiếp"). Nếu hoãn, RQ1 phần này **không có bằng chứng**; `train.py --direct-3class` đã có sẵn để làm sau (Teacher 3 lớp, M0, M1 nhánh 3 lớp ≈ 7 run ≈ 10h).

---

## 10. Rủi ro khoa học phải ghi vào Limitations

1. **Thiên lệch pre-label:** box GT gần với box detector → recall/AP có thể bị thổi phồng (D7). Val/Test gán không pre-label.
2. **Box F1 kappa pilot** chỉ có ý nghĩa nếu hai người vẽ độc lập; nếu dùng pre-label chung, phải ghi rõ.
3. **Ảnh ghép lệch miền** (ánh sáng, bóng, kết cấu): chỉ để pretrain; mọi tuyên bố dựa trên khay thật. Kết quả trên ảnh ghép không dùng để nói về hiệu năng thực.
4. **Test ít cụm:** CI bootstrap rộng; luôn nêu số cụm và số ngày.
5. **TACO ít thức ăn thừa:** detector có thể yếu ở nhóm Hữu cơ; theo dõi recall riêng nhóm này ở E4.
6. **Kết quả TACO cũ không dùng được** (checkpoint chọn trên heldout); chỉ dùng số sau D1-C3.
7. **Pha 2 chỉ CE** và Teacher không có `background`: tác động riêng của KD lên pipeline bị pha loãng; ghi nhận (D1, D8). Main Tier A và pha 2 khác nhau về crop-jitter.
8. **FitNet một giai đoạn** khác bài gốc (D6). AT: code = 0.5·‖·‖² (D3).
9. **Tier B** chỉ có KD logit nếu KD tốt nhất là họ feature/attention (D10).
10. **Số seed = 3** → std ước lượng thô; không kết luận chênh lệch nhỏ hơn std.
11. **`m1_vanilla_kd_clean`** có thể là pilot, không phải cấu hình đã tune: không dùng làm seed 1 của M1 nếu cấu hình khác Q06.
12. **Đo thiết bị:** nếu là CPU ONNX thì không phải thiết bị biên thật.
13. **Domain gap** chỉ được "đánh giá mức độ thu hẹp", không "giải quyết triệt để".

---

## 11. Checklist cuối tuần (T6 30/10)

- [ ] `pytest tests -q` xanh; `python scripts/audit_v10.py` không FAIL
- [ ] Kappa box: Box F1 ≥ 0.80, κ ≥ 0.70, có file `reports/kappa_box_report.md`
- [ ] Test đóng băng: manifest SHA256 + tag `test-frozen`; `reports/test_open_log.json` đúng 1 mục
- [ ] `reports/leakage_audit.md` toàn `[PASS]`
- [ ] Main Tier A: M0, M1, M1b, M2, M3, M4 × 3 seed có `history.csv`; tune N=5 cho 5 phương pháp có `tune_*_summary.csv`
- [ ] Tier B {M0, M1, KD tốt nhất} × 3 seed; INT8: sụt AP50 D1 và Macro-F1 Tier B được báo cáo (so với 3.0 và 2.0 điểm)
- [ ] E3a–d, E3-curve, D2, E4 (oracle vs detected), phân rã lỗi
- [ ] few-shot Tier A (k = 0…50), OOD RealWaste và TACO-heldout
- [ ] Bootstrap theo cụm + McNemar/Holm, có số cụm
- [ ] `docs/protocol_lock.md` v2.0 (đã được xác nhận), `docs/gonogo_decision.md`
- [ ] Latency: thiết bị thật hoặc ghi rõ `cpu_sim`
- [ ] Draft thô Methods + Results; Limitations từ mục 10
- [ ] Mọi bảng/hình sinh từ `collect_results.py`, mỗi bảng có dòng nguồn

---

## 12. Phụ lục

### 12.1 Lệnh mẫu (PowerShell, từ thư mục gốc repo)

```powershell
$env:PYTHONPATH = "src"

# M0 (Student, label smoothing 0.1 — KHÔNG quên, mặc định của train.py là 0.0)
python -m dsr.train --model resnet18 --run-name m0_resnet18_s20261002 --label-smoothing 0.1 --seed 20261002

# KD (tham số lấy từ configs/best_kd/<method>.json)
python -m dsr.train_kd --kd-method vanilla_kd         --run-name m1_hinton_s20261002 --temperature 3 --alpha 1 --seed 20261002
python -m dsr.train_kd --kd-method dkd                --run-name m1b_dkd_s20261002   --temperature 3 --alpha-tckd 1 --beta-nckd 2 --seed 20261002
python -m dsr.train_kd --kd-method attention_transfer --run-name m3_at_s20261002     --beta-at 2 --seed 20261002
python -m dsr.train_kd --kd-method hybrid             --run-name m4_hybrid_s20261002 --temperature 3 --alpha 1 --beta-at 2 --seed 20261002

# Tier B
python -m dsr.train --model mobilenet_v3_small --run-name tierb_m0_s20261001 --label-smoothing 0.1 --seed 20261001
python -m dsr.train_kd --kd-method vanilla_kd --student-model mobilenet_v3_small --run-name tierb_m1_s20261001 --seed 20261001

# Hàng đợi
python scripts/run_queue.py --config configs/queue_week.json --dry-run
python scripts/run_queue.py --config configs/queue_week.json
```

(Số trong lệnh KD chỉ là ví dụ định dạng; **giá trị thật lấy từ tune**.)

### 12.2 File mới/sửa theo ngày

| Ngày | File |
|---|---|
| 1 | `scripts/audit_canteen_data.py`, `data/mappings/label_map.csv` (30 lớp) + `label_map_v3_bg.csv`, `scripts/process_taco.py`, `losses.py` (FitNet), `train_kd.py`, `scripts/tune_kd.py`, `configs/tune_grid/*.json`, `scripts/run_queue.py`, `configs/queue_week.json`, `legacy/`, `requirements.txt`, `docs/canteen_quick_checklist.md`, `data/canteen_metadata_template.csv` |
| 2 | `scripts/synthesize_trays.py`, `scripts/eval_detector.py`, `scripts/validate_labels.py`, test DKD ≡ Hinton |
| 3 | `scripts/split_canteen_by_cluster.py`, `src/dsr/test_guard.py`, `scripts/extract_crops.py`, `scripts/eval_canteen_classifier.py` |
| 4 | `scripts/eval_e2e.py`, `scripts/stats_bootstrap.py`, `scripts/few_shot_finetune.py`, `data/processed/e3/*.yaml` |
| 5 | `scripts/build_background_crops.py`, `scripts/finetune_background.py`, `scripts/tune_thresholds.py`, `scripts/benchmark_pipeline_latency.py`, `docs/gonogo_decision.md`, `docs/protocol_lock_v2_DRAFT.md` |
| 6 | `configs/test_eval_manifest.json`, `scripts/eval_test_once.py`, `deploy/onnx/*_int8.onnx`, `docs/protocol_lock.md` v2.0 (sau xác nhận) |
| 7 | `scripts/collect_results.py`, `scripts/audit_leakage.py`, `scripts/audit_v10.py`, `reports/leakage_audit.md`, `README.md` (v10) |

### 12.3 Lệnh mở Test (chỉ bạn chạy, sau khi tick D6-H1)

```powershell
python scripts/eval_test_once.py --manifest configs/test_eval_manifest.json --dry-run   # xem danh sách
python scripts/eval_test_once.py --manifest configs/test_eval_manifest.json             # đúng 1 lần
```

### 12.4 Mẫu `reports/progress_log.md` (cuối ngày)

```markdown
## Ngày N — dd/mm
- Xong: <task ID>, <task ID>
- Lệch kế hoạch: <mô tả, lý do>
- Số liệu mới (nguồn file): <tên chỉ số = giá trị, reports/xxx.csv>
- Quyết định đã tick: <D?>
- Việc dồn sang mai / nhánh đã kích hoạt: <...>
```

### 12.5 Ánh xạ RealWaste → 3 nhóm (đề xuất, **cần kiểm tay 100 mẫu và xác minh danh sách lớp khi tải**)

| RealWaste | Nhóm đề xuất | Ghi chú |
|---|---|---|
| Food Organics, Vegetation | organic | |
| Cardboard, Glass, Metal, Paper | recyclable | trạng thái sạch/bẩn không có trong nhãn |
| Plastic | **chưa chốt** | có thể lẫn túi nilon/vật liệu mềm → kiểm tay |
| Miscellaneous Trash, Textile Trash | other_landfill | |

Lưu thành `data/mappings/realwaste_to_3.csv` có đánh số phiên bản và nhật ký kiểm tay.
