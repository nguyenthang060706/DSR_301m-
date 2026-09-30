# VĂN BẢN KHÓA: Chốt Dòng 3 Chính thức (Vanilla Knowledge Distillation)

> **Căn cứ:** Kế hoạch Tuần 3 v2 §4.1, §4.2; MASTER v8 §5, §7.1, §8.4.1, §13; `reports/week3/week3_tuning_holdout.csv`.
> **Trạng thái:** **ĐÃ KHÓA (LOCKED)** — Làm mốc so sánh chính thức cho toàn bộ ma trận Ablation (Dòng 4a, 4b, 4c, Dòng 7).

---

## 1. Quyết định lựa chọn Dòng 3 Chính thức

Áp dụng quy tắc quyết định tiền nghiệm (§4.1 và §4.2), kết quả thực nghiệm trên tập **Dev/Corruption-Holdout (379 ảnh)**:

| Run | Cấu hình | Fold 0 Macro-F1 | **Holdout Macro-F1** | Chênh lệch so với $\tau=3$ | Quyết định |
|---|---|:---:|:---:|:---:|:---:|
| `w3_vkd_tau3_s42` (R1) | $\alpha=0.7, \tau=3$, seed 42 | 85.18% | 84.79% | 0.00% | Mốc tham chiếu |
| `w3_vkd_tau2_s42` (R2) | $\alpha=0.7, \tau=2$, seed 42 | 90.63% | 85.78% | +0.99% | Không chọn (chênh $< m = 1.0\%$) |
| **`w3_vkd_tau6_s42` (R3)** | **$\alpha=0.7, \tau=6$, seed 42** | **92.04%** | **88.67%** | **+3.88%** | **CHỌN CHÍNH THỨC ($> m = 1.0\%$)** |

- **Biên nhiễu:** $m = \max(1.0\%, |84.98\% - 84.79\%|) = 1.0\%$.
- **Kết luận:** $\tau = 6$ vượt trội hơn $\tau = 3$ một khoảng $3.88\% > m$. Theo quy tắc §4.2:
  👉 **`w3_vkd_tau6_s42` được chọn làm DÒNG 3 CHÍNH THỨC.**

---

## 2. Thông số Dòng 3 Chính thức

- **Mô hình Student:** ResNet18 thuần (không ECA), ImageNet-pretrained.
- **Mô hình Teacher:** ResNet50 sạch (`checkpoints/teacher_resnet50_clean/best.pt`).
- **Siêu tham số đã khóa:**
  - $\alpha = 0.7$ (cố định theo MASTER §8.4.1)
  - $\tau = 6.0$ (chọn qua lưới Holdout)
  - Optimizer: SGD momentum 0.9, weight decay 1e-4
  - Batch size: 32, Epochs: 100, LR schedule: Warmup 5 + Cosine Annealing (0.01 $\to$ 1e-6)
  - Seed: `42`
- **Kết quả nghiệm thu:**
  - **Best Epoch:** 47 / 100
  - **Val Accuracy (Fold 0):** 93.06%
  - **Macro-F1 (Fold 0):** **92.04%**
  - **Trash F1 (Fold 0):** 84.44%
  - **Holdout Accuracy:** 90.50%
  - **Holdout Macro-F1:** **88.67%** (vượt Student Dòng 1 trên Holdout là 88.55%)
- **Đường dẫn Checkpoint chính thức:**
  `checkpoints/w3_vkd_tau6_s42/best.pt`

---

## 3. Điều chỉnh so với Báo cáo Tuần 2 (`week2_summary.md`)

- Trong Báo cáo Tuần 2, kết quả `week2_vanilla_kd_clean` (Macro-F1 92.43% @ Epoch 61) chỉ là **kết quả Pilot sơ bộ** do chưa kiểm soát seed và chưa tối ưu nhiệt độ $\tau$.
- Kết quả Tuần 3 với seed 42 và $\tau = 6.0$ đạt **92.04% trên Fold 0** và **88.67% trên Holdout** chính là **kết quả chính thức duy nhất** được đưa vào bảng so sánh tổng thể và luận văn.
- Bản pilot cũ giữ nguyên trong báo cáo với nhãn `[PILOT, chưa seed]` làm mốc đối chứng lịch sử.
