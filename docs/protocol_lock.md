# Protocol-Lock: Quy chuẩn Thực nghiệm Khóa cứng (Tuần 1 & 2)

**Phiên bản:** 1.0 (Khóa theo Kế hoạch MASTER v9.5)  
**Ngày phê duyệt:** 02/10/2026  
**Mục đích:** Đóng băng toàn bộ tham số kỹ thuật, quy trình chia dữ liệu và các mốc kiểm định cổng để đảm bảo tính minh bạch, nhất quán và có thể tái lập trước Hội đồng bảo vệ.

---

## 1. Bộ dữ liệu Công khai & Phân chia Dữ liệu (Dataset & Splits)

* **Tên bộ dữ liệu:** *Recyclable and Household Waste Classification* (Alistair King, Kaggle).
* **Đường dẫn dữ liệu cục bộ:** `data/raw/household_waste_30/`
* **Quy mô:** Đúng chính xác **15.000 ảnh nguyên bản** (không qua offline augmentation, mỗi lớp đúng 500 ảnh).
* **Số lớp nhãn mịn ($G$):** **30 lớp** (bao gồm cả ảnh studio `default` và ảnh đời thực `real_world`).
* **Bảng ánh xạ nhãn:** `data/mappings/label_map.csv` (Phiên bản 2.0).
  * **0 - Hữu cơ (`organic`):** 4 lớp (`food_waste`, `coffee_grounds`, `eggshells`, `tea_bags`) — 2.000 ảnh (13.3%).
  * **1 - Tái chế khô (`recyclable`):** 16 lớp (`plastic_water_bottles`, `aluminum_soda_cans`, `cardboard_boxes`, `newspaper`, `glass_beverage_bottles`...) — 8.000 ảnh (53.3%).
  * **2 - Còn lại / Khó tái chế (`other_landfill`):** 10 lớp (`styrofoam_food_containers`, `disposable_plastic_cutlery`, `paper_cups`, `plastic_straws`, `plastic_shopping_bags`...) — 5.000 ảnh (33.3%).
* **Chiến lược phân chia (Stratified Split):**
  * Seed cố định: `20261001`
  * Phân tầng theo cả [Class] và [Domain: default / real_world].
  * **Tập Train (85%):** **12.750 ảnh** $\rightarrow$ `data/splits/public_train.csv`
  * **Tập Dev/Holdout (15%):** **2.250 ảnh** $\rightarrow$ `data/splits/public_dev.csv`
  * **Data Leakage:** 0 ảnh trùng lặp (đã kiểm định bằng `tests/test_data.py`).

---

## 2. Recipe Huấn luyện Nền Khóa cứng (Locked Training Recipe §5.0)

Áp dụng đồng nhất cho mọi mô hình Tier A, Tier B, Teacher và các biến thể Knowledge Distillation:

| Thành phần | Tham số khóa cứng | Ghi chú & Căn cứ khoa học |
|---|---|---|
| **Optimizer** | **SGD (momentum = 0.9, weight_decay = 1e-4)** | Vượt trội áp đảo so với AdamW trong thực nghiệm Pilot (+8.6% F1). |
| **Learning Rate** | $\text{lr} = 0.01$ | Khởi tạo với Cosine Annealing decay về $10^{-5}$. |
| **Scheduler** | `CosineAnnealingLR(T_max=epochs, eta_min=1e-5)` | Suy giảm theo hàm cosine mượt mà, không warmup nhân tạo. |
| **Ngân sách Epoch** | **40 epochs** | Tốc độ đo thực tế: ~55s/epoch cho ResNet18; 40 epoch tốn ~39 phút. |
| **Batch Size** | **32** | Phù hợp dung lượng bộ nhớ VRAM 6GB của laptop RTX 4050. |
| **Augmentation** | `RandomResizedCrop(224, scale=(0.8, 1.0))` + `RandomHorizontalFlip(p=0.5)` + `TrivialAugmentWide()` | Online augmentation chuẩn hiện đại, bảo toàn tỷ lệ khung hình. |
| **Validation Transform** | `Resize(256)` + `CenterCrop(224)` + `Normalize` | Chuẩn ImageNet, không làm méo hình ảnh lúc đánh giá. |
| **Định dạng bộ nhớ** | `memory_format = torch.channels_last` | Tối ưu hóa tính toán Tensor Core trên phần cứng Ampere/Ada Lovelace. |
| **Mixed Precision** | **PyTorch AMP FP16 (`torch.amp.autocast('cuda')`)** | Tăng tốc độ gấp 2.5 lần, tiết kiệm VRAM. |
| **Label Smoothing** | **Teacher = 0.0** \| **Student = 0.1** | Teacher không làm phẳng soft target (Müller et al., 2019); Student có regularization. |

---

## 3. Nghiệm thu Cổng Tuần 1 (Gates Validation)

### 3.1 Cổng Teacher (Teacher Gate §4.1)
* **Yêu cầu:** Mức Giảm lỗi tương đối (Relative Error Reduction) trên Dev nhãn mịn:
  $$\text{RER} = \frac{\text{Err}_{\text{M0}} - \text{Err}_{\text{Teacher}}}{\text{Err}_{\text{M0}}} \ge 20\%$$
* **Số liệu thực nghiệm chính thức:**
  * **Student M0 (ResNet18, 40 ep):** Macro-F1 nhãn mịn = **88.33%** (Err = 11.67%) \| Canteen 3-Class F1 = **96.69%**
  * **Teacher (ResNet50, 40 ep):** Macro-F1 nhãn mịn = **90.85%** (Err = 9.15%) \| Canteen 3-Class F1 = **98.06%**
  * $$\text{RER} = \frac{11.67\% - 9.15\%}{11.67\%} = \mathbf{21.60\%} \ge 20.0\%$$
* **Kết luận:** **CHÍNH THỨC PASS CỔNG TEACHER.**
* **Checkpoint đã khóa:** `checkpoints/teacher_resnet50_clean/best.pt`

### 3.2 Cổng chọn Student Tier B (Edge Gate §4.1)
* Đã kiểm tra 3 ứng viên MobileNet trên ONNX Runtime:
  * `mobilenet_v3_small`: 1.53M params, 2.77ms latency (PASS)
  * `mobilenet_v3_large`: 4.22M params, 7.17ms latency (PASS)
  * `mobilenetv4_conv_small`: 2.51M params, 5.14ms latency (PASS)

---

## 4. Quy ước Đánh giá & Metric Chọn lọc

1. **Mức nhãn dùng để chọn lọc:** Toàn bộ quá trình chọn checkpoint tốt nhất (`best.pt`), tune siêu tham số và so sánh các phương pháp KD **bắt buộc dựa trên Macro-F1 nhãn mịn 30 lớp trên tập Dev**.
2. **Chỉ số 3 nhóm Canteen:** Dùng hàm ánh xạ từ 30 lớp $\rightarrow$ 3 nhóm quyết định để tính Macro-F1 Canteen báo cáo trong bảng tổng kết.
3. **Cầu chì phân kỳ (Divergence Fuse):** Nếu sau 20 epoch mà Dev Fine F1 $< 30\%$ hoặc loss NaN, dừng run để tránh lãng phí GPU.
