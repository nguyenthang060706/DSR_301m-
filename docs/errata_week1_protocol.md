# ERRATA: Cập nhật Tiêu chuẩn Baseline Tuần 1

## Lý do cập nhật
Quá trình kiểm tra nghiêm ngặt ở cuối Tuần 2 đã phát hiện lỗi rò rỉ dữ liệu (Data Leakage): tập Dev/Corruption-Holdout bị dùng làm Validation Set hàng ngày, vi phạm Master Plan §6.4.

Để tuân thủ hoàn toàn Protocol-lock, hệ thống đã:
1. Chuẩn hóa Validation Set về Fold 0 (432 ảnh) từ `trashnet_cv_folds.csv`.
2. Huấn luyện lại toàn bộ mô hình nền tảng trên dữ liệu sạch.

## Số liệu Baseline Mới (Thay thế số liệu cũ trong docs/week1_protocol.md)

| Mô hình | Macro-F1 (cũ, bị rò rỉ) | Macro-F1 (mới, sạch) | Checkpoint |
|---------|--------------------------|----------------------|------------|
| Teacher (ResNet50) | 94.45% | **97.62%** | `checkpoints/teacher_resnet50_clean/best.pt` |
| Student (ResNet18) | 91.37% | **95.04%** | `checkpoints/student_resnet18_clean/best.pt` |

## Checkpoint bắt buộc sử dụng từ Tuần 3 trở đi

> [!CAUTION]
> Tuyệt đối KHÔNG sử dụng các checkpoint cũ sau đây (đã bị đánh dấu deprecated):
> - `checkpoints/teacher_resnet50_week1/best.pt`
> - `checkpoints/student_resnet18_week1/best.pt`
> - `checkpoints/week2_vanilla_kd_pilot/best.pt`
> - `checkpoints/week2_mobilenet_v3_large_baseline/best.pt`
> - `checkpoints/week2_efficientnet_b0_baseline/best.pt`

**Checkpoint hợp lệ (sạch):**
- Teacher: `checkpoints/teacher_resnet50_clean/best.pt`
- Student: `checkpoints/student_resnet18_clean/best.pt`
- Vanilla KD: `checkpoints/week2_vanilla_kd_clean/best.pt`
- MobileNetV3: `checkpoints/mobilenet_v3_clean/best.pt`
- EfficientNet-B0: `checkpoints/efficientnet_b0_clean/best.pt`
- EfficientFormer-L1: `checkpoints/efficientformer_l1_clean/best.pt`

## Trạng thái
Văn bản này đóng vai trò phụ lục (Addendum) đính kèm vĩnh viễn với `week1_protocol.json`. Toàn bộ thực nghiệm KD/Attention từ Tuần 3 trở đi **bắt buộc** sử dụng mốc Baseline **95.04%** (Student ResNet18 clean) để so sánh.
