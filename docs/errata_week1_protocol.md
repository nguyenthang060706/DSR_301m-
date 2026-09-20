# ERRATA: Cập nhật Tiêu chuẩn Baseline Tuần 1

## Lý do cập nhật
Quá trình kiểm tra nghiêm ngặt ở cuối Tuần 2 đã phát hiện lỗi rò rỉ dữ liệu (Data Leakage) khi tập Validation nội bộ bị nhầm lẫn với tập Holdout (được Master Plan §6.4 quy định phải giữ sạch tuyệt đối). 

Để tuân thủ hoàn toàn Protocol-lock, hệ thống đã chuẩn hóa Validation Set về Fold 0 (432 ảnh) và tiến hành huấn luyện lại các mô hình nền tảng.

## Số liệu Baseline Mới (Thay thế số liệu cũ trong docs/week1_protocol.json/md)
- **Teacher (ResNet50):** 97.62% Macro-F1 (Tăng từ 94.45%)
- **Student (ResNet18):** 94.01% Macro-F1 (Tăng từ 91.37%)

## Trạng thái
Văn bản này đóng vai trò phụ lục (Addendum) đính kèm vĩnh viễn với week1_protocol.json. Toàn bộ các thực nghiệm KD/Attention từ Tuần 3 trở đi sẽ phải dùng mốc Baseline **94.01%** của Student để so sánh.
