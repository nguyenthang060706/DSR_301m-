# KẾ HOẠCH TUẦN 3: KIẾN TRÚC SPATIAL ATTENTION & ATTENTION TRANSFER (Dòng 4)

## Mục tiêu
Triển khai phương pháp Attention Transfer (Zagoruyko & Komodakis) và tích hợp khối ECA (Efficient Channel Attention) vào ResNet18. Đo lường hiệu quả truyền đạt 'sự tập trung' từ Teacher sang Student so với Vanilla KD.

## Danh sách công việc

### Ngày 1: Chạy chính thức Dòng 3 (Vanilla KD) & Tích hợp ECA Module (Kiến trúc Student)
- **Dòng 3 (Chính thức):** Huấn luyện chính thức Vanilla KD (`train_kd.py`) với Teacher sạch (`teacher_resnet50_clean/best.pt`) để chốt số liệu Baseline KD chính thức cho Tuần 3 (thay thế cho số liệu pilot tuần 2).
- **Kiến trúc Student (ECA):** Thêm ECA (Efficient Channel Attention) module vào các block của ResNet18 (tạo ra model `resnet18_eca`).
- Cập nhật `src/dsr/models.py` để đăng ký model mới này.
- Unit Test: Khởi tạo thử `resnet18_eca` và feed-forward một batch ảnh.

### Ngày 2: Triển khai Attention Transfer Loss (L_Attention)
- Viết hàm trích xuất feature maps từ 4 stage cuối của ResNet50 (Teacher) và ResNet18 (Student).
- Cài đặt hàm tính Attention Map: Tính tổng bình phương theo kênh (sum of squared activations dọc trục channel).
- Cài đặt hàm L_Attention: Chuẩn hóa L2 các Attention Map và tính MSE loss giữa Teacher và Student.
- Cập nhật src/dsr/losses.py.

### Ngày 3: Viết Pipeline Huấn luyện (train_at.py)
- Clone 	rain_kd.py sang 	rain_at.py.
- Tích hợp Hook để trích xuất intermediate features trong quá trình forward pass.
- Cập nhật công thức Loss: L = L_CE + alpha * L_KD + gamma * L_Attention.

### Ngày 4-5: Chạy Thực nghiệm Ablation Dòng 4
- **Dòng 4a:** Student(ECA) + L_CE + L_KD (Kiểm tra sức mạnh riêng của kiến trúc ECA).
- **Dòng 4b:** Student(Thường) + L_CE + L_KD + L_Attention (Kiểm tra sức mạnh riêng của hàm Loss AT).
- **Dòng 4c:** Student(ECA) + L_CE + L_KD + L_Attention (Kết hợp toàn diện).

### Ngày 6-7: Tổng hợp & Phân tích
- Thu thập kết quả vào `reports/week3/week3_summary.md`.
- Đối chiếu điểm F1 của Dòng 4c với Student tự học (Dòng 1 - Macro-F1 95.04%) và Vanilla KD chính thức (Dòng 3 - chạy ở Ngày 1). Báo cáo mức độ vượt trội.
- Đánh giá vai trò độc lập của từng nhánh: Dòng 4a (Spatial Attention - ECA) vs Dòng 4b (Attention Transfer Loss) vs Dòng 4c (Full Attention).
