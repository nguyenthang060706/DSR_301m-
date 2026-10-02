# Knowledge Distillation for Lightweight Waste Classification in University Canteen Edge Devices

Nghiên cứu ứng dụng Knowledge Distillation (KD) nhằm tạo ra mô hình gọn nhẹ, chính xác và suy luận nhanh để phân loại rác canteen trường đại học thành 3 nhóm quyết định trên thiết bị biên.

Tài liệu kế hoạch nghiên cứu chính thức: **[`ke_hoach_nghien_cuu_MASTER_v9.5.md`](./ke_hoach_nghien_cuu_MASTER_v9.5.md)**.

---

## 1. Mục tiêu & 3 nhóm quyết định
Phân loại rác tại nguồn dựa trên quyết định bỏ rác thực tế:
1. **Hữu cơ (Organic)**: Rác thực phẩm, đồ ăn thừa, vỏ hoa quả.
2. **Tái chế khô (Recyclable)**: Chai nhựa sạch, lon nhôm, hộp giấy sạch.
3. **Còn lại / Khó tái chế (Other/Landfill)**: Hộp xốp bẩn, túi nilon dính dầu, giấy ăn bẩn.

---

## 2. Các câu hỏi nghiên cứu (RQ)
- **RQ1 (Cơ chế KD & Nhãn)**: So sánh các họ KD (Hinton, DKD, FitNet, Attention Transfer) dưới recipe nền hiện đại; so sánh hiệu quả giữa huấn luyện nhãn mịn ($10\text{--}15$ lớp rồi gộp 3 nhóm) so với huấn luyện trực tiếp 3 nhóm.
- **RQ2 (Domain Gap & Thích nghi)**: Đánh giá khoảng cách miền từ dữ liệu công khai sang ảnh canteen tự chụp; khảo sát đường cong thích nghi few-shot ($k \in \{0, 10, 25, 50, 100\}$).
- **Mục tiêu triển khai (Edge Goal)**: Lượng tử hóa INT8 (PTQ) và đo kiểm benchmark (latency, RAM, accuracy) trên phần cứng thiết bị biên thật (Android/Raspberry Pi).

---

## 3. Cấu trúc thư mục

```text
DSR_301m-/
├── ke_hoach_nghien_cuu_MASTER_v9.5.md   # Kế hoạch nghiên cứu chính thức (duy nhất)
├── requirements.txt                     # Danh mục thư viện phụ thuộc
├── data/
│   ├── raw/                             # Ảnh gốc (công khai, canteen, benchmark)
│   ├── mappings/                        # Bảng ánh xạ nhãn mịn -> 3 nhóm (label_map.csv)
│   └── splits/                          # Danh sách chia train/dev/test/few-shot
├── src/dsr/                             # Mã nguồn huấn luyện & KD
├── configs/                             # File cấu hình thí nghiệm (JSON/YAML)
├── scripts/                             # Script công cụ phân tích & đo kiểm
├── reports/                             # Báo cáo kết quả & lịch sử huấn luyện
└── checkpoints/                         # Trọng số mô hình sau huấn luyện
```

---

## 4. Bắt đầu (Tuần 1-2)
1. Kích hoạt môi trường ảo:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
2. Thực hiện quy ước gán nhãn `label_guide.md` và gán thử 100 ảnh canteen (Cohen's $\kappa \ge 0,7$).
3. Thiết lập dataset công khai nhãn mịn và bảng ánh xạ `label_map.csv`.
4. Chạy pilot recipe và pilot epoch (40 vs 80 epoch) để chốt protocol-lock.
