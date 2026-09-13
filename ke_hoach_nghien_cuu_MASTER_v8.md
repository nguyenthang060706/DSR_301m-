# Kế hoạch nghiên cứu MASTER — Robustness-Aware Attention Distillation cho Phân loại Rác Nhẹ trên Biên

*Bản hợp nhất cuối cùng: khung logic (câu chuyện, RQ, định vị novelty) + đặc tả kỹ thuật đầy đủ (công thức, kiến trúc, ablation, protocol, timeline 14 tuần). Đây là bản duy nhất cần dùng để triển khai — không cần giữ song song các bản trước.*

> **Lịch sử review trước khi triển khai (3 vòng):**
> - **Vòng 1:** (1) tuần 5 không còn 5-fold cho dòng 4c — chỉ pilot/tune γ, 5-fold chính thức dồn sang tuần 8; (2) đổi tên "AugMix" thành "self-consistency" (§5, §7.2) vì augmentation thực tế là bộ nhiễu §9.2, không phải thuật toán AugMix gốc; (3) go/no-go TACO/RealWaste ở tuần 9 đổi từ ngưỡng đạt/rớt thành câu hỏi đánh giá cho RQ3.
> - **Vòng 2:** (4) chốt dứt điểm công thức L_Attention theo đúng Zagoruyko & Komodakis gốc (§5, §6.2) — không dùng hint-matching kiểu FitNet, tránh trùng baseline; (5) mở rộng protocol-lock (§8.1) từ chỗ chỉ áp cho cặp 4c-vs-7 sang toàn bộ ma trận ablation; (6) bắt buộc chốt epoch budget/LR schedule ngay tuần 1 (§8.2) để tránh domino rework.
> - **Vòng 3:** (7) thêm kiến trúc chia dữ liệu Dev/Corruption-holdout (§6.4) — tách riêng khỏi 5-fold từ tuần 1 để vừa tránh rò rỉ khi tune hyperparameter (§8.4), vừa tránh rò rỉ khi dựng TrashNet-C (§9.2); (8) yêu cầu StratifiedKFold + cùng phân chia fold cho mọi cấu hình (§8.3); (9) mở rộng phạm vi TrashNet-C sang {1,3,4c,7} vì §11.1/§11.2 cần cả dòng 1, đồng thời dùng lại 5 checkpoint của vòng 5-fold để robustness cũng có mean±std/CI; (10) bổ sung phương án dự phòng cho hai kết quả trung tâm nhất (dòng 7 vs 4c, và 4c vs max(4a,4b)) vào bảng rủi ro §14, vốn trước đó chỉ có phương án cho các so sánh phụ hơn.
>
> → **READY TO START IMPLEMENTATION** sau khi tuần 1 hoàn tất đúng 3 đầu ra bắt buộc: (a) văn bản chốt epoch budget/LR schedule, (b) văn bản chốt Dev/Corruption-holdout split, (c) class-mapping TACO→6 lớp.

> **Đã sửa 1 lỗi từ bản nháp trước:** thứ tự Gap↔RQ được thống nhất lại theo đúng 1-1 (Gap1→RQ1, Gap2→RQ2, Gap3→RQ3) thay vì thứ tự bị đảo (Gap1→RQ3, Gap2→RQ1, Gap3→RQ2) ở bản nháp FINAL trước đó.

---

## 1. Tiêu đề & Câu chuyện cốt lõi

**Câu hỏi nghiên cứu trung tâm (phát biểu đầy đủ):** Liệu việc kết hợp đồng thời (1) module attention gắn trong kiến trúc student, (2) loss ép attention map của student khớp teacher, và (3) loss ép dự đoán của student ổn định dưới nhiễu chụp thực tế (được neo bởi chính teacher) — có giúp một mô hình nhẹ giữ được độ chính xác và độ bền vững gần với model lớn tốt hơn so với dùng từng kỹ thuật riêng lẻ, khi áp dụng cho phân loại rác trên thiết bị biên?

**Câu chuyện rút gọn (một câu):** Richer knowledge distillation → robustness tốt hơn → cải thiện đó có generalize ra ngoài training distribution hay không?

Ba gap dưới đây không phải ba contribution rời rạc — chúng là **ba tầng bằng chứng liên tiếp** cho đúng một câu chuyện này.

---

## 2. Research Questions (đã đồng bộ numbering)

- **RQ1** — *Does richer knowledge transfer improve over conventional logit KD?*
- **RQ2** — *Does teacher-anchored consistency add robustness beyond attention distillation?*
- **RQ3** — *Do robustness gains transfer to real-world domain shift?*

**Khung tuyên bố (bắt buộc, tránh overclaim):**

| Không nói | Nên nói |
|---|---|
| "We solve domain shift." | "We evaluate whether robustness-aware distillation improves generalization under real-world domain shift." |
| "KD có tốt hơn không?" (mơ hồ) | "Does richer knowledge transfer improve over conventional logit KD?" |
| "ECA tốt hơn CA" | Chỉ nêu lý do chọn ECA (overhead thấp), không phải kết quả so sánh thực nghiệm — vì không chạy ablation ECA vs CA |
| "a novel PCGrad algorithm" | "gradient-conflict-aware extension", "conditional application of PCGrad (Yu et al., 2020)" |
| "ultra-lightweight" / "siêu nhẹ" cho ResNet18 | "lightweight so với teacher" / "edge-deployable" |

---

## 3. Gap → RQ → Bằng chứng

| Gap | Mô tả | RQ | Thực nghiệm | Dữ liệu | Bằng chứng kỳ vọng |
|---|---|---|---|---|---|
| **Gap 1** | Vanilla/conventional logit KD không cải thiện nhất quán cho lightweight waste model (đối chứng: Frontiers 2026 vs MDPI Sustainability 2026 kết luận trái chiều nhau) | **RQ1** | Dòng 3 → 4a/4b → 4c (5-fold cho 3, 4c; 1-2 seed cho 4a, 4b) | TrashNet 6 lớp (in-distribution) | 4c > max(4a,4b) > dòng 3 rõ rệt → richer KD tốt hơn conventional logit KD |
| **Gap 2** | Attention alignment (kiến trúc + loss) một mình có thể không đảm bảo robustness dưới nhiễu thực tế | **RQ2** | So sánh sạch: dòng 6 vs 5 (teacher vs self, cô lập) **và** dòng 7 vs 4c (so sánh trung tâm, protocol khóa) | TrashNet-C | 7 > 4c, chênh lệch nhất quán qua fold, không do randomness |
| **Gap 3** | Robustness đo trên corruption tổng hợp (TrashNet-C) có thể không transfer sang real-world domain shift | **RQ3** | Đánh giá {3, 4c, 7} trên external set (không chỉ dòng 7) | TACO (primary) → RealWaste zero-shot (secondary) | Robustness-gain của 7 (so với 4c, 3) vẫn giữ được khi transfer ra ngoài distribution |

**Nguyên tắc bắt buộc, không đổi:** không thêm loss mới, teacher mới, kiến trúc student mới, hay benchmark mới chỉ để "chứng minh KD" hay "chứng minh robustness" — mọi RQ trên đều trả lời được bằng ablation/benchmark đã có sẵn trong kế hoạch, không cần mở thêm nhánh kỹ thuật.

---

## 4. Định vị tính mới (Novelty / Positioning)

### 4.1 Ba lớp tính mới

| Lớp | Nội dung | Mức độ mới |
|---|---|---|
| Lớp 1 | Áp attention-transfer loss sang domain rác | Yếu — domain transfer thuần, không nên dùng làm luận điểm chính |
| Lớp 2 | Kết hợp attention kiến trúc (ECA trong student) + attention-transfer loss | Đã giảm — một paper nông nghiệp 2025 (Swin→MobileNetV3) đã làm gần giống, chỉ khác domain |
| Lớp 3 | Thêm L_Consistency dạng **teacher-anchored** dưới nhiễu tự nhiên, kết hợp cùng lúc với Lớp 2 | **Còn trống** — chưa tìm thấy tiền lệ kết hợp cả 3 tầng này trong bất kỳ domain nào sau nhiều lượt search |

### 4.2 L_Consistency không phải cơ chế mới — định vị trung thực

Cơ chế "teacher dự đoán trên ảnh sạch, student dự đoán trên ảnh nhiễu, tối thiểu KL divergence giữa hai bên" đã tồn tại trong họ **Adversarially Robust Distillation** (Goldblum et al., 2019) — chỉ khác threat model là *đối kháng (PGD/FGSM)*, không phải *nhiễu tự nhiên (mờ, thiếu sáng, che khuất, nghiêng)*. Một số công trình khác (CMKD, Mixup-based distillation) đã benchmark corruption robustness (CIFAR-100-C) như hệ quả của KD.

**Kết luận novelty: đây là tính mới Loại C — tổ hợp + ứng dụng, không phải Loại A (phát minh cơ chế).** Cụ thể: mở rộng cơ chế robust-distillation từ threat model đối kháng sang corruption tự nhiên, kết hợp tường minh với attention-transfer ở tầng kiến trúc, áp dụng cho bài toán lightweight waste classification. Phù hợp với tạp chí ứng dụng, hội nghị khu vực, luận văn — không đủ cho venue top-tier nếu không có thêm lý giải lý thuyết (§10 — gradient analysis là nỗ lực bổ sung phần lý giải này).

**Ghi chú bắt buộc về bằng chứng số liệu cho Lớp 2:** câu "kết hợp attention kiến trúc (ECA) + attention-transfer loss là mới" chỉ đứng vững nếu có ablation tách riêng hai trục này — nếu gộp chung thành một cấu hình bật/tắt duy nhất, không có cách nào chứng minh bằng số liệu rằng cần *cả hai*, chỉ còn là tuyên bố suông. Ma trận ablation §7 (dòng 3/4a/4b/4c) dùng đúng cho mục đích này.

### 4.3 Tính thực tiễn & Motivation

Bộ augmentation mô phỏng nhiễu (§9) không phải nhiễu tự chế — nó ánh xạ vào các failure mode đã ghi nhận ở hệ thống phân loại rác thực tế đang vận hành:
- Một hệ thống ghi nhận độ chính xác giảm còn 82% ở điều kiện 200 lux so với điều kiện ánh sáng chuẩn.
- Nhiều review AI phân loại rác xác nhận ánh sáng, vật che khuất, bụi bẩn, và nền nhiễu là các yếu tố suy giảm hiệu năng phổ biến trong triển khai công nghiệp/ngoài trời.
- Xu hướng chuyển sang mô hình nhẹ triển khai trên thiết bị biên (thùng rác thông minh, ứng dụng di động) là nhu cầu thật, không phải mục tiêu phụ.

**Câu chuyện mở đầu Motivation (dùng phát hiện KD gần đây, đã kiểm chứng số liệu):** Echchidmi & Bouayad (2026, *Frontiers in Artificial Intelligence* 9:1804734, "Compact waste image classification with multi-student CNNs and edge-oriented model selection") chạy KD/no-KD cross-validation 5-fold trên TrashNet và kết luận KD logit chuẩn **không vượt trội có ý nghĩa thống kê** về macro-F1 dưới ngân sách huấn luyện họ dùng. Ngược lại, MDPI *Sustainability* 2026, 18(13):6392 ("KD-Garbage Framework") lại báo cáo KD cải thiện có ý nghĩa thống kê (p≈0.015) trên bộ dữ liệu rác 13 lớp của họ. Hai kết luận trái chiều này (cùng năm 2026, cùng bài toán phân loại rác) là điểm mở đầu Motivation mạnh hơn "kết hợp 3 kỹ thuật cho tốt hơn": câu hỏi đặt ra là liệu tín hiệu distillation phong phú hơn có làm KD "đáng dùng" một cách nhất quán hơn hay không (= chính RQ1).

**Lưu ý khi trích dẫn:** các con số cụ thể (94.18%, 41.04%, macro-F1 0.3648) đã được đối chiếu với bản tóm tắt/abstract công khai của bài Frontiers 2026 tại thời điểm soạn kế hoạch — vẫn nên tự tải PDF gốc và đọc lại đúng bảng số liệu trước khi trích vào bản thảo, vì abstract đôi khi làm tròn khác với bảng chi tiết. **τ=3 từ MDPI Sustainability 2026 chưa được xác nhận độc lập** — cần double-check trước khi dùng làm căn cứ chính thức (xem §15 checklist).

**Lưu ý khi viết Introduction:** tách rõ 2 phần — "vấn đề có thật, có bằng chứng" (động lực thực tiễn) và "đóng góp kỹ thuật cụ thể, có giới hạn rõ ràng" (khiêm tốn, đúng mức Loại C). Không để phần động lực mạnh che lấp đóng góp kỹ thuật mỏng. Đã đổi sang khung quốc tế — không dùng khung pháp lý Việt Nam (Luật BVMT 2020, 3 nhóm rác) làm động lực chính; chỉ có thể nhắc như ứng dụng phụ/tương lai nếu muốn.

### 4.4 Cấu trúc đóng góp theo Bậc (khác trục với "Lớp" ở §4.1)

**Phân biệt quan trọng:** "Lớp 1/2/3" đo *độ mạnh của tuyên bố novelty*. "Bậc 1/2/3" dưới đây đo *loại đóng góp* — hai trục độc lập, cố tình đặt tên khác nhau để tránh nhầm lẫn khi đọc song song.

| Bậc | Nội dung | Vai trò |
|---|---|---|
| Bậc 1 — Model improvement | ResNet18 → ResNet18 + ECA + KD (Logit) + Attention Transfer + teacher-anchored Consistency | Đóng góp chính (= Loại C) — tổ hợp kiến trúc + loss, có ablation số liệu chứng minh (§7) |
| Bậc 2 — Mathematical analysis | Đo cos(G_task, G_attn), cos(G_task, G_robust), cos(G_attn, G_robust) (§10) | Giải thích *tại sao* các thành phần ở Bậc 1 có thể hợp tác hoặc xung đột |
| Bậc 3 — Conditional optimization | Nếu Bậc 2 phát hiện xung đột đáng kể → gradient-conflict-aware extension bằng PCGrad (§10.6-10.7); nếu không → không cần can thiệp | Mở rộng có điều kiện, **không phải nhánh bắt buộc** |

**Cách dùng khi viết bài:** ba bậc tạo một chuỗi lập luận (Bậc 1 → Bậc 2 giải thích Bậc 1 → Bậc 3 xử lý nếu Bậc 2 phát hiện vấn đề) — trình bày tuần tự đúng thứ tự này trong Method/Discussion, không trộn Bậc 3 vào phần giới thiệu đóng góp chính ở Introduction.

---

## 5. Công thức toán học đầy đủ

Ký hiệu: `x` = ảnh gốc, `x̃ = T(x)` = ảnh sau augmentation mô phỏng nhiễu thực tế, `f_S/f_T` = student/teacher (teacher đóng băng, stop-gradient), `p^clean_S = softmax(f_S(x)/τ)`, `p^noisy_S = softmax(f_S(x̃)/τ)`, `p^clean_T = softmax(f_T(x)/τ)`.

| Thành phần | Công thức | Vai trò |
|---|---|---|
| L_CE | Cross-entropy(y, softmax(f_S(x))) | Học nhãn cứng |
| L_logit | τ²·KL(p^clean_T ‖ p^clean_S) | KD chuẩn (Hinton) — cả hai nhìn ảnh sạch |
| L_Attention | Σ_l ‖Â_S^l − Â_T^l‖² , trong đó Â^l = normalize( Σ_c |A_c^l|² ) — tổng bình phương activation theo chiều channel tại mỗi vị trí không gian, chuẩn hóa (L2-norm) rồi mới so sánh; chỉ bilinear interpolate nếu resolution lệch giữa teacher/student, KHÔNG có conv học tham số | Học "nhìn ở đâu" giống teacher (spatial attention map, đúng công thức Zagoruyko & Komodakis — không phải feature-hint kiểu FitNet) |
| L_Consistency | τ²·KL(p^clean_T ‖ p^noisy_S) | Student nhìn ảnh nhiễu nhưng phải khớp phán đoán teacher trên ảnh sạch (teacher-anchored, khác self-consistency dùng chung bộ augmentation §9.2) |
| L_Consistency-self | τ²·KL(sg[p^clean_S] ‖ p^noisy_S) — `sg` = stop-gradient | **Chỉ dùng làm biến thể kiểm soát** ở dòng ablation 2 và 5 (§7), để cô lập câu hỏi "cần teacher hay tự thân đã đủ". Không phải thành phần của L_total đề xuất |

**L_total = L_CE + α·L_logit + γ·L_Attention + δ·L_Consistency** *(L_Consistency ở đây luôn là bản teacher-anchored; bản self chỉ tồn tại trong các dòng ablation kiểm soát)*

Điểm kỹ thuật quan trọng: L_logit và L_Consistency dùng chung một lần forward của teacher (chỉ chạy trên ảnh sạch) — không tốn thêm chi phí tính teacher, chỉ thêm một forward pass của student trên x̃.

**⚠️ Cảnh báo cài đặt — chiều của `F.kl_div` (bug ngầm rất khó phát hiện vì loss vẫn giảm bình thường):** công thức L_logit = τ²·KL(p_T ‖ p_S) nghĩa là p_T là "target". Trong PyTorch, `F.kl_div(input, target)` tính KL(target ‖ exp(input)) — **không phải** KL(input ‖ target). Do đó `input` phải là log-prob của **student**, `target` phải là prob của **teacher**:

```python
student_log_prob = F.log_softmax(z_student / T, dim=1)
teacher_prob = F.softmax(z_teacher / T, dim=1).detach()  # teacher đã freeze
loss_logit = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * T**2
```

Đặt ngược lại sẽ tính ra KL(p_S ‖ p_T) — sai chiều, đổi gradient từ "mode-covering" sang "mode-seeking". Cùng cảnh báo áp dụng cho L_Consistency: `input` là log-prob student trên ảnh nhiễu, `target` là prob teacher trên ảnh sạch (đã detach).

**✅ Đã chốt (không còn là quyết định mở) — định nghĩa A^l dùng đúng công thức Zagoruyko gốc, không dùng hint-matching kiểu FitNet:** vì tổng bình phương activation theo chiều channel tự động rút feature map về còn 1 kênh không gian, phép so sánh giữa teacher (channel rộng, ResNet50) và student (channel hẹp, ResNet18) không cần conv học tham số để khớp channel — chỉ cần bilinear interpolate nếu lệch resolution (thường không lệch vì ResNet18/50 dùng chung lịch downsample per stage, chỉ khác channel width). Lý do chốt theo hướng này thay vì giữ conv-projector (từng để mở ở bản trước):
- Giữ đúng ý nghĩa "attention" (Zagoruyko & Komodakis) thay vì lẫn với FitNet — tránh dòng 4b (= baseline "Attention Transfer", §7.1) trên thực tế đo trùng cơ chế với baseline FitNet (train riêng ở tuần 3), điều sẽ làm mất một baseline có ý nghĩa.
- Không phát sinh tham số học được ngoài backbone → khớp đúng câu đã ghi ở cuối mục này ("L_Attention chỉ lan gradient tới backbone ≤ layer trích xuất, không có nhánh tham số nào khác cần tính vào vùng đo ở §10.3").
- Giữ rõ ràng **hai loại "attention" khác trục, bổ trợ chứ không trùng nhau** trong cùng bài: ECA (kiến trúc, channel-attention — "kênh nào quan trọng") và L_Attention (loss, spatial-attention-map — "vị trí nào trong ảnh quan trọng"). Nêu rõ sự phân biệt này ở Method để tránh reviewer hiểu nhầm là dùng 2 tên gọi cho cùng một cơ chế.

Vì teacher (ResNet50) không có ECA, A_T^l chỉ là attention map tính trực tiếp từ activation thô của teacher tại 4 vị trí cuối stage — không cần chèn module gì vào teacher. Ghi rõ điều này trong Methods để tránh hiểu nhầm là teacher cũng cần ECA.

**Điểm cấu trúc cần nêu trong bài:** L_Attention chỉ lan gradient tới các layer backbone ≤ layer trích xuất attention map — **không chạm tới lớp phân loại cuối (head)**. Ba loss còn lại (CE, logit, Consistency) lan gradient tới toàn bộ backbone + head. Đây là cơ sở cho việc chia vùng đo ở §10.

---

## 6. Kiến trúc & Dữ liệu

### 6.1 Chốt kiến trúc student

**Quyết định: student = ResNet18, cố định xuyên suốt toàn bộ pipeline (KD, ECA, L_Attention, L_Consistency, edge deployment).**

**Lý do không chọn MobileNetV3 hay EfficientNet-B0 (dù nhẹ hơn):** cả hai đã có sẵn khối Squeeze-and-Excitation (SE, một dạng channel attention) nhúng trong hầu hết các block. Nếu chọn một trong hai làm student, dòng 3 ("student thuần, không ECA") trên thực tế vẫn có attention (SE) chạy ngầm, phá vỡ đúng luận điểm mà phép so dòng 3 vs 4a cần chứng minh. ResNet18 là kiến trúc duy nhất trong ba lựa chọn không có channel/spatial attention tích hợp sẵn — lựa chọn sạch duy nhất cho vai trò "student thuần".

**Đánh đổi cần ghi vào Hạn chế:** ResNet18 (~11.7M tham số) nặng hơn MobileNetV3 (~5.4M) hoặc EfficientNet-B0 (~5.3M) — hơi giảm sức nặng câu chuyện "siêu nhẹ" ở edge deployment (§12). Vẫn nhẹ hơn nhiều so với ResNet50 teacher (~25.6M) và triển khai bình thường trên Jetson Nano/Raspberry Pi. Không gọi ResNet18 là "ultra-lightweight" — dùng "lightweight so với teacher" hoặc "edge-deployable".

**Vị trí đặt ECA:** chèn ECA vào cuối mỗi trong 4 stage của ResNet18 (sau layer1–layer4). L_Attention trích attention map ở CÙNG 4 vị trí này — để dòng 4a (kiến trúc) và 4b (loss) so sánh trên cùng một tập vị trí feature map.

**Quyết định module attention: ECA (Wang et al. 2020), không dùng CA (Hou et al. 2021), không ghép cả hai.** Lý do: (1) ECA gần như không thêm tham số, hợp với việc "vẫn phải nhẹ"; (2) ghép ECA+CA biến "kiến trúc attention" thành thiết kế phức hợp mới, vượt phạm vi novelty Loại C đã chốt. Vì không chạy ablation ECA vs CA thực nghiệm, **không được viết "ECA tốt hơn CA"**.

**Hệ quả lên timeline:** ResNet18 (= student) train ngay tuần 1, cùng lúc với teacher — vì go/no-go tuần 1 cần con số này. MobileNetV3/EfficientNet-B0 chỉ còn vai trò baseline so sánh ngoài, train ở tuần 2.

### 6.2 Matching resolution cho L_Attention (không cần matching channel)

**Cập nhật (khớp §5 đã chốt):** vì Â^l đã được rút về 1 kênh không gian bằng tổng bình phương activation theo channel (không có conv học tham số), bước duy nhất còn cần là **bilinear interpolate về cùng kích thước không gian** nếu resolution của teacher/student lệch nhau ở cùng stage — thường không xảy ra vì ResNet18 và ResNet50 dùng chung lịch downsample theo stage (chỉ khác channel width, không khác spatial size), nhưng vẫn giữ bước resize làm phương án an toàn. Cơ chế này không phụ thuộc việc student có module ECA hay không — đây là cơ sở kỹ thuật để chạy được dòng 4b: áp L_Attention lên student KHÔNG có ECA, tách riêng khỏi hiệu ứng kiến trúc.

### 6.3 Chiến lược dữ liệu

- **Train:** TrashNet — nguyên **6 lớp gốc** (cardboard, glass, metal, paper, plastic, trash), **không relabel theo 3 nhóm Luật BVMT 2020** (khung quốc tế). Cho phép so sánh trực tiếp với 4 công trình đối chứng quốc tế mà không cần quy đổi nhãn. Kiểm tra class imbalance ngay tuần 1 (lớp "trash" vốn ít ảnh hơn) để quyết định weighted loss / stratified sampling.
- **Student:** ResNet18 (§6.1) — cố định xuyên suốt.
- **Teacher:** ResNet50 — mục tiêu tham khảo là accuracy hơn student baseline ≥8%, nhưng **không phải điều kiện tuyệt đối**: gap nhỏ hơn vẫn có thể đủ nếu KL(p_T‖p_S) trên **toàn bộ validation set** cho thấy soft-target đủ khác biệt (báo cáo mean/median/std/percentile). Nếu teacher không đủ mạnh, đổi sang **DenseNet121 hoặc ConvNeXt-Tiny** (không dùng EfficientNet-B0 vì đã giữ vai trò CNN baseline nhẹ) — chỉ cần train nhanh 1 seed ở tuần 1.

### 6.4 🔧 Kiến trúc chia dữ liệu tổng thể — chưa được định nghĩa rõ, có nguy cơ rò rỉ giữa 5-fold và TrashNet-C

**Vấn đề:** nếu TrashNet-C (§9.2) được dựng bằng cách làm nhiễu *toàn bộ* ảnh TrashNet, thì khi đánh giá checkpoint của fold-i trên TrashNet-C, một phần ảnh trong bộ test đó chính là ảnh đã nằm trong tập **train** của fold-i (chỉ khác là bị thêm nhiễu) — đây là rò rỉ dữ liệu thật, không phải rủi ro lý thuyết, và sẽ làm robustness accuracy bị thổi phồng. Vấn đề này độc lập nhưng cùng gốc với rò rỉ tuning đã nêu ở §8.4.4.

**Cách chia thống nhất (giải quyết cả hai rò rỉ cùng lúc):**
1. Ngay tuần 1, tách một **split cố định, stratified** (vd. ~15% TrashNet) khỏi toàn bộ pipeline CV — gọi là **Dev/Corruption-holdout**. Split này **không bao giờ** xuất hiện trong tập train của bất kỳ fold nào trong 5-fold.
2. Split này dùng cho **cả hai việc**: (a) toàn bộ hyperparameter tuning α/τ/γ/δ (§8.4.4), (b) là nguồn ảnh sạch để dựng TrashNet-C (§9.2) — áp augmentation nhiễu lên đúng các ảnh trong split này.
3. Phần còn lại (~85% TrashNet) mới đem chia StratifiedKFold thành 5 fold (§8.3) cho vòng CV chính thức của Table 1.
4. Vì Dev/Corruption-holdout không nằm trong train của fold nào, có thể an toàn đánh giá **cả 5 checkpoint** (từ 5 fold, cho mỗi model trong {1,3,4c,7}) trên cùng một TrashNet-C này mà không lo rò rỉ — cho ra đúng mean±std/CI cho robustness đã đề xuất ở §9.3, không cần matching riêng theo từng fold.

Đây là kiến trúc chia dữ liệu cần chốt bằng văn bản ở tuần 1 (cùng lúc với epoch budget, §8.2), không nên để ngầm định rồi phát hiện muộn khi đã train xong 5-fold ở tuần 8.

**⚠️ Vẫn còn một lớp mỏng hơn cần chốt — không phải leakage dữ liệu, mà là leakage tiêu chí tuning:** dùng chung Dev/Corruption-holdout cho cả (a) chọn hyperparameter và (b) dựng TrashNet-C không gây rò rỉ *huấn luyện* (ảnh này chưa từng vào gradient update), nhưng nếu **δ** (trọng số Consistency — mục đích duy nhất là robustness) được chọn bằng cách xem accuracy **dưới nhiễu** trên chính holdout này, thì δ đã được tối ưu để tốt trên đúng phép đo sau này dùng làm bằng chứng chính cho RQ2 — làm TrashNet-C hết còn là phép đo "chưa từng bị chạm tới". γ không bị vấn đề này (mục đích của γ không phải robustness, tiêu chí chọn tự nhiên là accuracy sạch).

**Cách xử lý (tái dùng đúng tường lửa severity đã có ở §9.2, không cần cắt thêm dữ liệu như phương án chia 3 phần Dev/Corruption-test/CV-pool):**
- Tiêu chí tuning cho δ **chỉ được phép dùng** accuracy sạch hoặc accuracy dưới nhiễu **severity 1-2** trên Dev/Corruption-holdout (đúng mức nhiễu mà training đã "thấy" qua L_Consistency) — **không bao giờ** dùng severity 3 làm tín hiệu chọn δ.
- Severity 3 trên holdout này giữ nguyên vai trò bằng chứng chính, hoàn toàn "chưa bị chạm tới" bởi cả training LẪN tuning — không chỉ chưa bị chạm bởi training như §9.2 đã nêu.
- Phương án chia 3 phần (Dev≈10% / Corruption-test≈10% / CV pool≈80%) là lựa chọn "sạch tuyệt đối" hợp lệ nếu muốn, nhưng không bắt buộc khi đã có tường lửa severity này — và đúng như lưu ý, nên tránh nếu làm giảm đáng kể dữ liệu cho 5-fold trên một dataset vốn đã nhỏ (~2500 ảnh, lớp "trash" vốn ít).

### 6.5 🆕 Dataset thứ 2 (Tier mở rộng, không bắt buộc) — Garbage Classification (Kaggle, ~12 lớp, ~15K ảnh)

**Vai trò:** không thay thế TrashNet — đây là một **tầng bằng chứng bổ sung** cho RQ1/RQ2 (RQ3 đã có TACO/RealWaste riêng): lặp lại đúng ma trận {1, 3, 4c, 7} trên một dataset train thứ hai, để kiểm tra kết luận chính có phải đặc thù của TrashNet (ảnh studio, 6 lớp, ~2500 ảnh) hay không. **Cần tự kiểm tra lại license/tính khả dụng của bộ dữ liệu cụ thể tại thời điểm triển khai trước khi chốt.**

**Vì sao chọn dataset thứ hai thay vì kiến trúc student thứ hai:** giữ đúng nguyên tắc effort-management đã có ở §4.1/§4.3 — "1 kiến trúc, 2 dataset" dễ được reviewer chấp nhận hơn "2 kiến trúc, 1 dataset nhỏ", vì generalize qua *dữ liệu* là câu hỏi bám sát RQ1/RQ2 hiện có, còn generalize qua *kiến trúc* là mở rộng khác, không bắt buộc.

**Phạm vi ma trận:** chỉ {1, 3, 4c, 7} — không lặp lại 4a/4b/2/5/6/8, vì mục đích là kiểm tra tính bền của kết luận chính (RQ1: 4c>3; RQ2: 7>4c), không phải tái tạo toàn bộ phân tích ablation chi tiết.

**🆕 Bước tiền đề bắt buộc — train lại teacher trên dataset 2:** vì dataset 2 có ~12 lớp, khác hẳn 6 lớp TrashNet, teacher ResNet50 đã train ở tuần 1 (§6.3) **không dùng lại được trực tiếp** (khác không gian đầu ra, khác domain). Cần train một teacher ResNet50 mới trên dataset 2 trước khi chạy được dòng 3/4c/7 (những dòng cần logit/attention map của teacher) — dòng 1 (student-only) không cần bước này. Đây là một go/no-go nhỏ tương tự tuần 1 gốc (kiểm tra teacher-student gap/KL(p_T‖p_S) trên dataset 2), cần tính vào effort tuần 15 bên dưới, không phải một khoản phát sinh ngoài kế hoạch.

**🆕 Feasibility check sớm (tuần 1, song song, không chặn đầu ra bắt buộc của tuần 1):** tải thử dataset 2, đo thời gian 1 epoch thật của ResNet18 trên phần cứng dự kiến dùng, và kiểm tra dung lượng ổ cứng còn trống — cho tín hiệu sớm về việc 3 tuần (15-17) có khả thi hay không, tránh phát hiện muộn ở tuần 15 (sau khi đã cam kết xong Tier chính 14 tuần) rằng phần cứng không đủ. Nếu ước lượng 1 epoch quá chậm (vd. hạ tầng free-tier/cluster chia sẻ, xem giả định compute budget ở §13), cân nhắc bỏ Tier mở rộng ngay từ đầu thay vì để đến tuần 15 mới quyết định.

**🔧 Ước lượng effort đã sửa (từ ~2 tuần lên ~3 tuần + buffer):** ước lượng "~2 tuần" ban đầu chỉ tính compute time thuần cho 4 cấu hình × 5-fold, chưa tính (a) debug pipeline trên dataset mới (mapping lớp, kiểm tra imbalance, có thể phải chỉnh lại augmentation config), (b) khả năng phải tune lại learning rate/schedule vì phân bố ảnh khác TrashNet, (c) 🆕 bước train teacher mới trên dataset 2 (xem trên), (d) GPU-hours thực tế cho 4 model × 5 fold cộng dồn thời gian hàng đợi nếu chạy trên hạ tầng chia sẻ (xem giả định compute budget ở §13). Chốt: **3 tuần** (tuần 15-17), có buffer riêng nằm trong tuần buffer mở rộng (tuần 19-20) nếu vượt.

**Go/no-go:** không phải điều kiện bắt buộc để nộp bài — đây là hạng mục **Tier mở rộng**, chỉ chạy nếu Tier chính (tuần 1-14) hoàn tất đúng hạn. Nếu không đủ thời gian, bỏ qua toàn bộ, không ảnh hưởng go/no-go nào của Tier chính.

**Thống kê:** dùng lại đúng khung ưu tiên bằng chứng đã chốt ở §11.2 (effect size + CI + tỷ lệ fold thắng trước, p-value hỗ trợ sau) — không phát sinh tiêu chuẩn kiểm định riêng cho dataset thứ 2.

---

## 7. Ma trận Ablation

### 7.1 Baseline bên ngoài (bắt buộc)

CNN thuần: ResNet18, MobileNetV3, EfficientNet-B0
KD kinh điển: Vanilla KD (Hinton), FitNet, Attention Transfer (Zagoruyko & Komodakis)
🆕 KD + augmentation mạnh: CutMix+KD (CutMix, Yun et al. 2019, ghép cùng KD chuẩn) — baseline gần nhất về mặt "dùng augmentation mạnh trong lúc distillation", cần để tách bạch với vai trò cụ thể của L_Consistency (khác CutMix ở chỗ neo bằng đúng dự đoán teacher trên ảnh nhiễu tự nhiên, không trộn nhãn/ảnh giữa hai mẫu)
🆕 Kiến trúc nhẹ có attention tích hợp sẵn: EfficientFormer-L1 — baseline ngoài bổ sung, đại diện nhóm "attention built-in ở tầng kiến trúc" (khác ECA gắn thêm vào ResNet18), giúp định vị điểm hiệu năng-kích thước của dòng 7 so với một kiến trúc vốn đã thiết kế cho attention thay vì gắn thêm. **Cài đặt qua `timm` (`timm.create_model('efficientformer_l1', pretrained=...)`).**

**🆕 Ghi chú optimizer cho EfficientFormer-L1:** kiến trúc hybrid CNN/attention (ViT-style) thường nhạy với learning rate/optimizer hơn CNN thuần và hội tụ ổn định hơn với **AdamW + cosine-annealing schedule** (đúng recipe gốc của EfficientFormer/họ ViT) thay vì SGD dùng cho ResNet18. Vì EfficientFormer-L1 chỉ là baseline ngoài (§7.1), không nằm trong ma trận ablation nội bộ dòng 1-8, dùng optimizer/schedule riêng của nó **không vi phạm protocol-lock** — §8.1 chỉ khóa protocol cho toàn bộ ma trận ablation nội bộ, không áp cho baseline ngoài (đã có tiền lệ với MobileNetV3/EfficientNet-B0). Vẫn cần huấn luyện đủ epoch để đạt hội tụ thật của chính kiến trúc này (không nhất thiết bằng đúng epoch budget của §8.2), tránh so sánh không công bằng do model chưa hội tụ.

**Quy ước bắt buộc (tránh train trùng lặp):** cả ba KD kinh điển trên cài trên CÙNG backbone student = ResNet18 — điều kiện để so sánh công bằng. Hệ quả: **Vanilla KD = dòng 3**, **Attention Transfer = dòng 4b** — dùng chéo cho cả hai bảng, không train hai lần. CutMix+KD cũng train trên cùng backbone ResNet18 (baseline ngoài, không tham gia ma trận ablation nội bộ §7.2). MobileNetV3, EfficientNet-B0, EfficientFormer-L1 chỉ là baseline so sánh ngoài (kiến trúc khác), không tham gia ablation nội bộ.

Focus-RCNet và paper nông nghiệp không nằm trong danh sách trên vì không tái lập được đầy đủ trên cùng backbone — xử lý bằng so sánh gián tiếp (§16).

### 7.2 Ablation nội bộ (bắt buộc — cô lập từng thành phần)

| # | Cấu hình | CE | Logit | ECA (kiến trúc) | L_Attention (loss) | Consistency | Vai trò / RQ |
|---|---|---|---|---|---|---|---|
| 1 | Student-only | ✓ | | | | | Sàn — dùng làm student baseline go/no-go tuần 1 |
| 2 | Student + self-consistency (không teacher) | ✓ | | | | self | Kiểm soát: chặn phản biện "chỉ là self-consistency đơn thuần trên augmentation nhiễu" |
| 3 | KD chuẩn — student thuần, không ECA | ✓ | ✓ | | | | ≈ Vanilla KD; nền của ma trận 4a/4b/4c — mốc dưới cho **RQ1** |
| 4a | KD + ECA (kiến trúc), không L_Attention | ✓ | ✓ | ✓ | | | Cô lập đóng góp riêng của kiến trúc ECA — **RQ1** |
| 4b | KD + L_Attention (loss), không ECA | ✓ | ✓ | | ✓ | | Cô lập đóng góp riêng của loss attention-transfer — **RQ1**. = baseline "Attention Transfer" ở §7.1, dùng lại số liệu, KHÔNG train lại |
| 4c | KD + ECA + L_Attention (cả hai) | ✓ | ✓ | ✓ | ✓ | | Richer KD đầy đủ — mốc trên cho **RQ1**, mốc dưới cho **RQ2** |
| 4c+ 🆕 | KD + ECA + L_Attention + augmentation-exposure (x̃ dùng CE với nhãn cứng, KHÔNG có KL Consistency) | ✓ (cả x, x̃) | ✓ | ✓ | ✓ | — (CE-aug, không KL) | Kiểm soát confound: dòng 7 khác 4c ở CÙNG LÚC hai biến (được thấy x̃ + có loss KL neo teacher) — 4c+ cô lập riêng biến "được thấy x̃", để So sánh #3-bis (RQ2) phân biệt được đóng góp đến từ việc tiếp xúc ảnh nhiễu hay từ chính cơ chế consistency |
| 5 | KD + self-consistency | ✓ | ✓ | | | self | Cô lập: self-consistency (không teacher) có đủ hay cần teacher-anchor — **RQ2** |
| 6 | KD + teacher-anchored consistency | ✓ | ✓ | | | teacher | Cô lập tác dụng riêng của teacher-anchoring — **RQ2** |
| 7 | Đầy đủ (đề xuất) | ✓ | ✓ | ✓ | ✓ | teacher | Đóng góp chính (Bậc 1) — mốc trên cho **RQ2**, model chính cho **RQ3** |
| 8 | Đầy đủ + PCGrad (nếu §10.6 phát hiện xung đột) | ✓ | ✓ | ✓ | ✓ | teacher | Gradient-conflict-aware extension, có điều kiện (Bậc 3) |

**So sánh #1 — bằng chứng cho RQ1:** dòng 4c so với 4a và 4b — nếu 4c > max(4a, 4b) > dòng 3 rõ rệt, đây là số liệu chứng minh trực tiếp rằng kết hợp kiến trúc + loss cho giá trị cộng thêm so với dùng riêng lẻ. Nếu chỉ một trục vượt trội, viết lại thành "trục đó là phần đóng góp chính, trục kia chỉ bổ trợ nhẹ" — vẫn hợp lệ, chỉ khiêm tốn hơn. **⚠️ Nếu γ bị điều chỉnh sau spot-check tuần 7 (§8.4.2), disclose rõ trong Methods rằng γ dùng để báo cáo 4c ở đây đã qua bước xác nhận bằng dòng 7 — không mô tả 4c như một cấu hình được tune hoàn toàn độc lập.**

**So sánh #2 — bằng chứng sạch cho RQ2 (teacher-anchoring):** dòng 6 vs 5 — CHỈ khác nhau ở self vs teacher, không lẫn ECA hay L_Attention.

**So sánh #3 — bằng chứng sạch cho câu hỏi nghiên cứu trung tâm (RQ2):** dòng 7 vs 4c — chỉ khác nhau ở việc có L_Consistency (teacher) hay không. Đây là phép so trực tiếp nhất — đưa vào phần Kết quả chính, không chỉ nằm im trong bảng ablation. **🔧 Bổ sung:** protocol (seed, epoch, LR schedule, augmentation) phải khóa giống nhau tuyệt đối giữa 4c và 7 (xem §8) — nếu không, reviewer có thể nói cải thiện đến từ training randomness thay vì từ consistency.

**So sánh #3-bis 🆕 — kiểm soát confound "tiếp xúc ảnh nhiễu" cho So sánh #3 (RQ2):** dòng 7 vs 4c+ vs 4c. Ba kịch bản đều được diễn giải trước, không có nhánh nào bị bỏ ngỏ:
- **7 > 4c+ > 4c:** cả hai đều đóng góp, nhưng cơ chế consistency-loss (neo teacher) mang lại thêm giá trị so với chỉ tiếp xúc ảnh nhiễu qua CE thường — bằng chứng mạnh nhất cho RQ2 đúng như kỳ vọng.
- **4c+ ≈ 7 (và cả hai > 4c):** phần lớn lợi ích đến từ việc tiếp xúc ảnh nhiễu trong training (tương đương một dạng augmentation mạnh), không nhất thiết cần cơ chế neo-teacher cụ thể — vẫn là kết luận hợp lệ, chỉ cần viết lại tuyên bố RQ2 khiêm tốn hơn: "phần lớn robustness-gain giải thích được bằng exposure to noisy inputs; teacher-anchoring đóng góp thêm không đáng kể trên setup này".
- **7 > 4c ≈ 4c+:** chỉ tiếp xúc ảnh nhiễu không đủ — cần đúng cơ chế KL-consistency neo teacher mới có tác dụng; đây là bằng chứng mạnh nhất phân biệt L_Consistency với một augmentation thông thường.

**Phạm vi chạy:** 4c+ mặc định chạy 1-2 seed (cùng nhóm với 4a/4b, §8.3) — chỉ nâng lên 5-fold nếu kết quả pilot nằm gần ranh giới giữa ba kịch bản trên, theo đúng nguyên tắc pilot-rồi-escalate đã áp dụng cho dòng 8.

**🆕 Định nghĩa kỹ thuật batch composition cho 4c+ (chốt trước khi code tuần 5, tránh ambiguity):**
- Mỗi iteration: giữ nguyên **B ảnh sạch x** (đúng batch size như 4c/7), tạo thêm **B ảnh nhiễu x̃ = T(x)** cùng nhãn cứng y — **không giảm** số ảnh sạch xuống B/2, để không làm suy yếu tín hiệu CE so với 4c/7.
- `L_CE(4c+) = ½·[L_CE(x) + L_CE(x̃)]` — tương đương về mặt toán học với việc gộp thành một batch 2B rồi lấy mean chuẩn; **không** cộng thẳng hai số hạng (sẽ vô tình tăng gấp đôi trọng số của L_CE so với α·L_logit và γ·L_Attention trong L_total, tự nó là một confound khác cần tránh).
- `L_logit` và `L_Attention` của 4c+ **chỉ tính trên x sạch** — giống hệt 4c, không đụng đến x̃ — giữ đúng nguyên tắc "chỉ CE mới có augmentation-exposure" để So sánh #3-bis cô lập sạch đúng một biến.
- Teacher chỉ forward **một lần trên x sạch** (giống 4c, không cần forward trên x̃) — vì 4c+ không có loss nào cần teacher nhìn ảnh nhiễu.
- Hệ quả compute: 4c+ có đúng 2 lần forward+backward của student mỗi iteration (1 trên x, 1 trên x̃) — khớp compute-parity với dòng 7 (dòng 7 cũng có 2 forward student mỗi iteration, chỉ khác là forward thứ hai dùng cho L_Consistency thay vì CE) — nhờ vậy chênh lệch 7 vs 4c+ (nếu có) không thể quy cho việc "train nhiều bước hơn", chỉ còn quy được cho đúng cơ chế KL-neo-teacher.
- Epoch budget (§8.2) giữ nguyên không đổi — thay đổi ở đây chỉ nằm trong cấu trúc nội bộ 1 iteration, không phải số epoch.

**Lưu ý dòng 7 vs dòng 5:** vẫn có thể báo cáo, nhưng đổi ý nghĩa — phép so này KHÔNG cô lập được riêng biến teacher-anchoring vì khác nhau cùng lúc ở 3 biến (ECA, L_Attention, self-vs-teacher). Chỉ dùng cho tuyên bố yếu hơn: "hệ thống đầy đủ tốt hơn một biến thể đơn giản" — KHÔNG dùng để suy ra riêng vai trò teacher-anchoring (việc đó thuộc So sánh #2).

---

## 8. Protocol thực nghiệm

### 8.1 Khóa protocol — nguyên tắc chung cho TOÀN BỘ ma trận ablation, không chỉ 4c vs 7

**⚠️ Sửa phạm vi (bản trước chỉ khóa protocol cho một cặp, gây xung đột ngầm với chính So sánh #1 và #2 ở §7.2):** nguyên tắc đúng phải là *mọi cấu hình trong ma trận §7.2 chỉ được phép khác nhau ở đúng (các) thành phần đang bị ablate — mọi yếu tố huấn luyện khác phải giống hệt nhau*, không riêng gì cặp 4c-vs-7. Nếu chỉ khóa protocol cho 4c-vs-7 mà bỏ ngỏ cho 3-vs-4a-vs-4b-vs-4c (So sánh #1, bằng chứng RQ1) và 6-vs-5 (So sánh #2, bằng chứng RQ2), thì đúng phản biện "chênh lệch có thể do randomness/khác protocol" mà §8.1 bản trước cố chặn cho 4c-vs-7 vẫn còn nguyên với hai so sánh còn lại — tự mâu thuẫn với mục đích của chính mục này.

Cần cố định xuyên suốt **toàn bộ dòng 1–8**:
- Cùng seed / cùng số lần lặp seed (trong nhóm cùng mức 1-2 seed hoặc cùng nhóm 5-fold)
- Cùng training budget (epoch, batch size) — xem §8.2, chốt từ tuần 1
- Cùng learning rate schedule
- Cùng augmentation pipeline (ngoại trừ chính augmentation nhiễu T(x) là biến đang bị ablate ở dòng 2/5/6/7/8)

So sánh #3 (4c vs 7) vẫn là nơi rủi ro bị soi kỹ nhất (vì là bằng chứng trung tâm cho RQ2/RQ chính) nên cần ghi thành bảng protocol tường minh riêng trong Methods, nhưng đây là *trường hợp nổi bật nhất của một nguyên tắc áp dụng chung*, không phải ngoại lệ duy nhất cần khóa.

→ Ghi thành bảng protocol cụ thể trong Methods cho cả ma trận, không chỉ nêu ý định.

### 8.2 Đồng bộ epoch budget (toàn bộ pipeline) — phải chốt từ tuần 1, không phải một quyết định ngầm định

Bảng ablation chính (§7.2) và vòng 5-fold chính thức (§8.3) phải dùng **cùng epoch budget** cho từng cấu hình. Bài học từ lỗi của Frontiers 2026 (con số teacher lệch nhau ~7 điểm % giữa hai bảng khác nhau trong cùng bài, do budget huấn luyện khác nhau).

**⚠️ Domino risk chưa nêu rõ ở bản trước:** timeline (§13) huấn luyện các dòng ablation trải dài tuần 1→7 (dòng 1 ở tuần 1, dòng 3 ở tuần 4, 4a/4b/4c ở tuần 5, 2/5/6 ở tuần 6, 7 ở tuần 7), trong khi γ/δ còn đang được tune dần trong cùng khoảng thời gian đó. Nếu epoch budget (số epoch, batch size, hình dạng LR schedule) KHÔNG được chốt ngay từ tuần 1 mà bị điều chỉnh giữa chừng (vd. sau khi thấy dòng 7 cần train lâu hơn để hội tụ ổn định ở tuần 6-7), toàn bộ số liệu dòng 1/3/4a/4b/4c đã train ở tuần 1-5 sẽ vi phạm chính yêu cầu "cùng epoch budget" ở trên — buộc phải train lại, tốn kém hơn nhiều so với domino risk đã nêu riêng cho dòng 4c/7 ở §8.3/§14.

**Yêu cầu cụ thể:** chốt epoch budget + batch size + hình dạng LR schedule **làm đầu ra bắt buộc của tuần 1**, dựa trên đường cong hội tụ (convergence curve) của teacher/student baseline — dùng con số này xuyên suốt mọi dòng ablation và cả vòng 5-fold tuần 8, không đổi lại sau đó trừ khi ghi rõ lý do + con số cụ thể trong Methods (không được để ẩn).

### 8.3 Phạm vi 5-fold (giới hạn để tránh bùng nổ compute)

Không phải mọi dòng ở §7 cần chạy đủ 5-fold — tổng số cấu hình duy nhất đã là ~13-14 (🆕 tính cả 4c+), nhân 5-fold sẽ ra 65-70 lần train, vượt quá khả năng của 14 tuần nếu không có cụm GPU mạnh.

**⚠️ Hai điều kiện kỹ thuật bắt buộc cho vòng 5-fold, chưa nêu rõ ở bản trước:**
- **Cùng một phân chia fold (identical fold indices) cho mọi cấu hình** trong nhóm bắt buộc (1, 3, 4c, 7, baseline) — paired t-test/Wilcoxon/tỷ lệ fold thắng chỉ có ý nghĩa khi so sánh "cùng fold" giữa các model. Cố định seed chia fold một lần, lưu lại danh sách index, tái sử dụng cho toàn bộ nhóm.
- **Dùng StratifiedKFold, không phải KFold thường**, chia trên phần TrashNet còn lại **sau khi đã tách Dev/Corruption-holdout (§6.4)** — lớp "trash" trong TrashNet vốn ít ảnh hơn hẳn 5 lớp còn lại; KFold ngẫu nhiên có thể tạo fold thiếu hụt nghiêm trọng lớp này, làm macro-F1 dao động mạnh giữa fold vì lý do chia dữ liệu chứ không phải vì model.

**Chạy đủ 5-fold + paired t-test (bắt buộc, số liệu cho Table 1 chính):**
- Dòng 1 (Student-only) — mốc sàn
- Dòng 3 (KD chuẩn) — mốc KD kinh điển
- Dòng 4c — mốc trước khi thêm Consistency. **Cùng nguyên tắc như dòng 7: tuần 5 chỉ pilot/tune γ (1-2 seed), KHÔNG chạy 5-fold ở bước này — 5-fold chính thức cho 4c dồn vào tuần 8, vì γ có thể còn đổi nhẹ sau khi thêm Consistency (xem bảng rủi ro §14)**
- Dòng 7 (Full method) — kết quả chính. **Bắt buộc có pilot 1 seed (cuối tuần 6) xác nhận hội tụ ổn định trước khi cam kết 5-fold**
- 1-2 baseline ngoài mạnh nhất (§7.1)

**Chỉ chạy 1-2 seed (không 5-fold):**
- Dòng 2, 4a, 4b, 5, 6 — mục đích là xem *chiều hướng* đóng góp, không cần độ chính xác thống kê chặt
- 🆕 Dòng 4c+ — kiểm soát confound cho So sánh #3-bis (§7.2); nâng 5-fold có điều kiện nếu kết quả nằm gần ranh giới giữa ba kịch bản diễn giải, cùng nguyên tắc escalate như dòng 8
- Dòng 8 (Full + PCGrad, nếu chạy) — chỉ cần đủ bằng chứng xem PCGrad có cải thiện dòng 7 hay không; nâng 5-fold nếu khả quan và còn thời gian

**Chuẩn hóa ngôn ngữ (bắt buộc khi viết Kết quả):** với các dòng chỉ chạy 1-2 seed, **cấm** dùng "statistically significant"/p-value như đã kiểm định chặt — dùng "cho thấy xu hướng cải thiện", "kết quả sơ bộ gợi ý...". Ngôn ngữ khẳng định thống kê đầy đủ chỉ dùng cho nhóm 5-fold bắt buộc.

### 8.4 Chiến lược tune hyperparameter

1. **α, τ:** điểm khởi đầu từ paper nông nghiệp (α≈0.7, τ∈{2,6}). MDPI Sustainability 2026 ghi nhận τ=3 ổn định/gần tối ưu — dùng làm điểm khởi đầu có căn cứ (⚠️ cần double-check τ=3 trên bảng gốc trước khi trích chính thức — nếu không khớp, quay lại τ∈{2,6}).
2. **γ (attention):** grid {0.1, 0.5, 1.0}, tune sớm ở tuần 5 khi chạy dòng 4c (trên Dev/Corruption-holdout, §6.4). Giá trị chốt carry-forward cho các dòng sau (6, 7, 8).

   **⚠️ Cần chốt trước, không để ngầm định — spot-check ở tuần 7 tạo ra sự lệch giữa "tune cho 4c" và "tune cho 4c rồi freeze":** nếu spot-check 1-2 giá trị γ lân cận trong bối cảnh dòng 7 (có Consistency) cho thấy một γ khác tốt hơn, việc "áp dụng ngược lại" giá trị đó cho 4c sẽ khiến con số chính thức của 4c (5-fold tuần 8) bị ảnh hưởng bởi thông tin từ một cấu hình mà 4c không có (Consistency) — không phải rò rỉ dữ liệu (dev split vẫn tách biệt khỏi 5-fold), nhưng làm So sánh #1 (RQ1: 4c đại diện cho "richer KD tối ưu độc lập") không còn hoàn toàn sạch. So sánh #3 (4c vs 7) không bị ảnh hưởng vì §8.1 vẫn đảm bảo cả hai dùng chung một γ dù giá trị là gì.

   **Chốt quy trình rẽ nhánh trước khi chạy tuần 7 (không để phát sinh giữa chừng):**
   - Nếu spot-check xác nhận γ ban đầu vẫn tốt nhất/gần nhất cho dòng 7 → giữ nguyên, không có gì cần disclose thêm.
   - Nếu spot-check tìm ra γ khác tốt hơn rõ rệt cho dòng 7 → áp dụng γ mới cho **toàn bộ** 4c/6/7/8 (bắt buộc, để giữ đúng §8.1), và ghi rõ trong Methods/Hạn chế: *"γ được chọn qua hai bước — grid search ban đầu trên dòng 4c, sau đó xác nhận/điều chỉnh khi có Consistency"* — không mô tả là "tune cho 4c rồi freeze". Nếu 4b (pilot, không thuộc nhóm 5-fold bắt buộc) đã chạy với γ cũ, chấp nhận như một xấp xỉ nhẹ khác (4b không dùng cho kiểm định thống kê chặt, chỉ nêu xu hướng — xem §8.3).
3. **δ (consistency):** grid {0.1, 0.5, 1.0}, tune ở tuần 6-7, **sau** γ đã cố định. **Tiêu chí chọn δ chỉ dùng accuracy sạch hoặc accuracy dưới nhiễu severity 1-2 trên Dev/Corruption-holdout (§6.4) — không được dùng severity 3, vì severity 3 trên chính holdout này là bằng chứng chính cho TrashNet-C/RQ2, cần giữ "chưa bị chạm tới" bởi cả tuning lẫn training.**
4. Toàn bộ tuning chạy trên Dev/Corruption-holdout split đã tách riêng ở §6.4 — **không dùng bất kỳ fold nào trong 5 fold chính thức để tune**, tránh rò rỉ (γ/δ "nhìn thấy" trước dữ liệu sẽ dùng để báo cáo paired t-test/Wilcoxon). Chỉ full 5-fold ở bước đánh giá cuối (§8.3).
5. Xuất bảng sensitivity (accuracy theo γ, δ) để trả lời câu hỏi phản biện "tại sao chọn giá trị này".

---

## 9. Robustness & External Validation (Gap 3 / RQ3)

### 9.1 Ba lớp benchmark

| Benchmark | Vai trò | Chi phí | Ghi chú |
|---|---|---|---|
| **TrashNet-C** (tự tạo) | Robustness nội bộ trên corruption tổng hợp | Đã có pipeline (§9.2) | RQ2 + RQ3 |
| **TACO** (external, primary) | Domain shift thực tế, gần deployment nhất | Crop từ box detection → classification; freeze mapping TACO→6 lớp từ tuần 1 | Không được bỏ nếu thiếu thời gian |
| **RealWaste zero-shot** (external, secondary) | Domain-shift ngoại lai thứ hai, sanity check | Rẻ (~0.5 ngày, chỉ infer, dùng mapping có sẵn từ Frontiers 2026) | Cắt trước Tier C nếu thiếu thời gian, nhưng rẻ nên nên giữ |

**Domain shift TrashNet vs TACO (cần nêu khi lý giải kết quả):** TrashNet là ảnh chụp studio, nền trơn/đồng nhất, vật thể chiếm gần hết khung hình, ánh sáng đều; TACO là ảnh chụp thực địa, nền lộn xộn, vật thể ở nhiều tỉ lệ/góc nhìn, ánh sáng không kiểm soát, phân bố lớp lệch khác TrashNet. Dùng các khác biệt này (background, scale, lighting, viewpoint, class distribution) để giải thích gap giữa performance nội bộ và ngoại lai — không coi là nhiễu ngẫu nhiên.

**Mốc tham chiếu RealWaste:** Frontiers 2026 công bố LCNet-0.5 (train thuần trên TrashNet) đạt 94.18% top-1 trên TrashNet nhưng chỉ **41.04% top-1 / macro-F1 0.3648** khi zero-shot trên RealWaste. Nếu Full method (dòng 7) giữ accuracy/macro-F1 cao hơn rõ rệt so với mốc này, đây là con số so sánh trực tiếp, thuyết phục. **Cần re-check số liệu trên PDF gốc trước khi trích chính thức.**

### 9.2 Tập augmentation T cho TrashNet-C (benchmark hóa theo tinh thần ImageNet-C)

**🆕 Mở rộng thành 8 loại "seen" + 4 loại "unseen" (thay vì 6 loại dùng chung cho cả train và test), theo đúng tinh thần ImageNet-C / ImageNet-C-bar (Hendrycks & Dietterich):**

- **Seen (8 loại — có thể xuất hiện khi train L_Consistency, severity 1-2):** Gaussian blur, gamma darkening, random occlusion patch, affine tilt ±15°, motion blur, brightness change, 🆕 Gaussian noise, 🆕 JPEG compression artifact.
- **Unseen (4 loại — CHỈ dùng ở test-time trên TrashNet-C, không bao giờ xuất hiện trong train hay trong tuning δ):** 🆕 elastic transform, 🆕 fog/haze overlay, 🆕 zoom blur (radial), 🆕 salt-and-pepper (impulsive) noise.

Mỗi loại (cả seen và unseen) có 3 mức độ severity.

**🆕 Biện minh ranh giới seen/unseen theo cơ chế sinh nhiễu (để tránh reviewer bắt bẻ ranh giới là tùy tiện):** 4 loại unseen được chọn vì khác *cơ chế biến dạng* so với các loại seen gần nhất về mặt hình thức, không chỉ khác tên gọi — elastic transform là biến dạng phi tuyến/không cứng (non-rigid warp), khác affine tilt (biến đổi cứng/rigid); fog/haze là suy giảm tương phản do tán xạ ánh sáng, có tính không đồng nhất theo không gian, khác gamma darkening (làm tối đều toàn ảnh theo một hàm số); zoom blur là mờ theo hướng tâm-tỏa (radial), khác motion blur (mờ theo một hướng tuyến tính) và khác Gaussian blur (mờ đẳng hướng/isotropic); salt-and-pepper là nhiễu xung dạng điểm ảnh rời rạc (impulsive, per-pixel), khác occlusion patch (che khuất theo vùng liền khối) và khác Gaussian noise (nhiễu liên tục có phân phối chuẩn). Nêu ngắn gọn đoạn này trong Methods khi mô tả TrashNet-C.

**⚠️ Cảnh báo leakage:** nếu dùng cùng loại nhiễu vừa để train L_Consistency vừa để dựng bộ test robustness mà không tách bạch, robustness claim sẽ yếu đi nhiều. Tách theo **cả loại nhiễu (seen/unseen) lẫn severity**:
- **Train L_Consistency:** chỉ sample trong 8 loại seen, severity 1-2 (nhẹ-vừa). 4 loại unseen không bao giờ vào training.
- **Tune δ (§8.4.3):** cũng chỉ được dùng severity 1-2 (hoặc accuracy sạch) trên 8 loại seen, trên Dev/Corruption-holdout làm tiêu chí chọn — cùng một tường lửa, mở rộng thêm một tầng từ train sang cả tuning (§6.4). 4 loại unseen không dùng để tune bất kỳ hyperparameter nào.
- **Test robustness (TrashNet-C):** đánh giá đủ cả 3 mức trên cả 12 loại, nhưng báo cáo **tách riêng hai cột**: (a) seen, severity 3 (chưa từng thấy lúc train và chưa từng dùng để tuning, nhưng cùng loại nhiễu) và (b) unseen, cả 3 severity (chưa từng thấy dưới bất kỳ hình thức nào, kể cả loại nhiễu) — cột (b) là bằng chứng generalization mạnh hơn cột (a), dùng làm bằng chứng chính cho RQ2/RQ3 về khả năng tổng quát hóa sang nhiễu hoàn toàn mới.
- **Phương án dự phòng nếu cần bằng chứng mạnh hơn:** train Consistency chỉ trên 6/8 loại seen, giữ thêm 2 loại seen còn lại hoàn toàn held-out (tương đương một lớp "unseen" thứ hai, mức độ nhẹ hơn).

### 9.3 🔧 Bổ sung quan trọng: đánh giá đa model, không chỉ dòng 7

Kế hoạch gốc (timeline tuần 9) chỉ đặt go/no-go trên "Full method (dòng 7) giữ accuracy tốt hơn baseline trên TACO/RealWaste". Để RQ3 thực sự tách bạch được *robustness-gain đến từ attention (4c) hay từ consistency (7)* khi transfer ra ngoài distribution, cần chạy TrashNet-C/TACO/RealWaste trên **cả {dòng 3, 4c, 7}**, không chỉ dòng 7 — rồi so sánh mức giữ-vững hiệu năng giữa ba dòng này trên từng benchmark ngoại lai. Đây là điều kiện để RQ3 trả lời đúng câu hỏi "cải thiện nào transfer" thay vì chỉ "model cuối có transfer hay không".

**⚠️ Xung đột cần sửa — thiếu dòng 1 trong phạm vi TrashNet-C:** phạm vi {3, 4c, 7} ở trên đủ cho RQ3 (so external), nhưng §11.1 ("Robustness benchmark: % suy giảm accuracy giữa **Student thường** vs Full method trên từng loại nhiễu") và §11.2 (Dual Pareto: "cùng tập model (**dòng 1**, 3, 4c, 7 + baseline mạnh)") đều cần số liệu robustness của **dòng 1** — nếu không chạy dòng 1 qua TrashNet-C thì không thể tạo ra hai bảng/biểu đồ này như đã cam kết. Sửa: **TrashNet-C chạy trên {1, 3, 4c, 7}** (rẻ, chỉ là inference thêm trên pipeline đã có sẵn — không cần train thêm gì vì dòng 1 đã có checkpoint từ tuần 1). TACO/RealWaste (đắt hơn, primary evidence cho RQ3) giữ nguyên phạm vi {3, 4c, 7} vì dòng 1 không cần thiết cho câu hỏi "cải thiện nào transfer" — chỉ thêm dòng 1 vào đó nếu dư thời gian.

**Dùng checkpoint nào để đánh giá robustness:** để giữ đúng mức độ chặt chẽ thống kê đã áp cho Table 1 (§11.2), robustness/external evaluation cho {1,3,4c,7} nên dùng lại **chính 5 checkpoint đã train ở vòng 5-fold tuần 8** (không train/infer từ một checkpoint đơn lẻ mới), báo cáo mean±std/CI cho robustness accuracy tương tự Table 1 — tránh tình trạng phần accuracy có đầy đủ 5-fold rigor còn phần robustness chỉ dựa trên 1 run.

### 9.4 🆕 mCE và Relative Robustness (tính lại từ số liệu sẵn có — gần như miễn phí, ~0.5 ngày)

Bổ sung hai chỉ số chuẩn của literature robustness (theo tinh thần Hendrycks & Dietterich, 2019), tính trực tiếp từ accuracy đã có trên TrashNet-C cho {1,3,4c,7} + baseline mạnh nhất — không cần train/infer thêm, chỉ tổng hợp lại số liệu sẵn có:

- **CE_c (Corruption Error) theo loại nhiễu c:** lỗi trung bình qua các severity của model đang xét trên loại nhiễu c, chuẩn hóa theo lỗi tương ứng của một mô hình tham chiếu — ở đây dùng **dòng 1 (Student-only)** làm mô hình tham chiếu (thay cho vai trò AlexNet trong ImageNet-C gốc, vì không có mốc cộng đồng sẵn có cho bài toán rác). **mCE** = trung bình CE_c qua toàn bộ loại nhiễu.
- **Relative Robustness (Relative mCE):** so sánh **mức suy giảm** từ accuracy sạch (không phải accuracy tuyệt đối) giữa model đang xét và mô hình tham chiếu — cô lập câu hỏi "model có giữ vững hiệu năng tốt hơn tương đối" khỏi việc model đó vốn đã có accuracy sạch cao hơn.
- **Báo cáo tách riêng theo seen (8 loại, §9.2) và unseen (4 loại, §9.2):** hai bộ mCE/Relative-Robustness riêng biệt — mCE trên seen đo khả năng chống nhiễu đã huấn luyện, mCE trên unseen đo khả năng tổng quát hóa sang nhiễu hoàn toàn mới, đúng tinh thần phân tách ImageNet-C vs ImageNet-C-bar.
- Đây là chỉ số **bổ sung**, không thay thế accuracy/macro-F1 đã có ở Table 1 — đưa vào bảng robustness riêng (§11.1).

### 9.5 🆕 (Tùy chọn, Tier mở rộng, không go/no-go) Few-shot fine-tuning trên TACO

Ngoài đánh giá zero-shot chính ở §9.1-9.3 (bằng chứng chính cho RQ3), thử nghiệm bổ sung nếu còn dư thời gian: fine-tune nhẹ (few-shot, vd. 5-10 ảnh/lớp lấy từ TACO) cho {4c, 7} rồi đánh giá lại trên phần TACO còn lại — câu hỏi phụ trợ "nếu cho phép một lượng nhỏ dữ liệu domain đích, khoảng cách domain-shift có thu hẹp nhanh hơn ở dòng 7 hay 4c hay không". Không thay đổi kết luận chính của RQ3 (vốn dựa trên zero-shot), chỉ là bằng chứng phụ. Effort ước tính ~1 ngày (dùng lại checkpoint và pipeline TACO đã có ở tuần 9). Đặt ở buffer tuần 18 (§13), bỏ qua ngay nếu hết thời gian.

---

## 10. Gradient Interference Analysis (Bậc 2 → Bậc 3)

### 10.1 Định vị trung thực

Đo cosine similarity giữa gradient các loss **không phải phát minh mới** — có tiền lệ trực tiếp: một nghiên cứu về distillation từ VLM phát hiện cosine similarity giữa gradient CE và KD thường âm (xung đột); NetDistiller phát hiện tới 50% gradient student xung đột với gradient teacher, áp PCGrad; cơ chế gốc PCGrad (Yu et al., 2020).

**Vai trò trong nghiên cứu này:** dùng công cụ đã kiểm chứng để tạo bằng chứng cho tổ hợp cụ thể (Attention + Consistency + CE, cấu trúc bất đối xứng riêng — §10.3) — điều chưa ai làm, không phải công cụ đo.

**Phân biệt phạm vi bằng chứng:** phân tích gradient đo xung đột giữa các **số hạng loss** trên một checkpoint đã có ECA cố định — KHÔNG trả lời "bản thân kiến trúc ECA có đóng góp gì, tách biệt với loss attention-transfer". Câu hỏi đó chỉ trả lời được bằng ablation §7.2 (dòng 3/4a/4b/4c). §7.2 chứng minh Lớp 2 (§4.1); §10 hỗ trợ Lớp 3.

### 10.2 Gộp nhóm gradient

L_CE và L_logit dùng chung forward trên ảnh sạch, gần như luôn hợp tác — gộp thành một nhóm:

```
G_task   = ∇θ L_CE + α·∇θ L_logit     (giám sát nhiệm vụ chính, nhánh sạch)
G_attn   = ∇θ L_Attention              (prior không gian từ teacher)
G_robust = ∇θ L_Consistency            (prior phân phối, nhánh nhiễu)
```

Ba câu hỏi cần đo: cos(G_task, G_attn), cos(G_task, G_robust), cos(G_attn, G_robust).

### 10.3 Vùng đo (do cấu trúc bất đối xứng ở §5)

| Vùng | Gradient tồn tại | Ghi chú |
|---|---|---|
| Backbone (layers ≤ l, nơi trích attention map) | G_task, G_attn, G_robust — đủ cả 3 | Đo đầy đủ 3 cặp |
| Head (layers > l) | Chỉ G_task, G_robust | G_attn = 0 tại đây |

### 10.4 Chỉ số đo

- Cosine similarity **theo từng block** (không đo toàn cục)
- Tỷ lệ xung đột = % block có cos < 0, theo dõi qua các giai đoạn train (đầu/giữa/cuối)
- Tỷ lệ độ lớn ‖G_i‖/‖G_j‖ — kiểm tra một loss có "thắng thế" chỉ vì norm lớn hơn không

### 10.5 Cách trích gradient

```python
g_task   = torch.autograd.grad(L_CE + alpha*L_logit, shared_params, retain_graph=True)
g_attn   = torch.autograd.grad(L_Attention,          shared_params, retain_graph=True)
g_robust = torch.autograd.grad(L_Consistency,        shared_params, retain_graph=True)
```

Không instrument toàn bộ training chính (tốn ~2x backward mỗi bước). Chạy **run chẩn đoán riêng, ngắn** trên checkpoint đã train ở cấu hình đầy đủ.

### 10.5-bis 🆕 Gradient dynamics theo epoch (mở rộng từ snapshot tĩnh)

**Vấn đề của snapshot đơn:** đo cosine similarity chỉ trên MỘT checkpoint đã hội tụ (§10.5) cho biết trạng thái *cuối* của mối quan hệ giữa các gradient, nhưng không đủ để phát hiện việc mối quan hệ này thay đổi qua các giai đoạn train — đúng như cây quyết định ở §10.6 đã dự tính một khả năng cụ thể ("cos(G_task, G_robust) âm ở đầu train, dương ở cuối") mà một snapshot cuối duy nhất không thể quan sát được.

**Thiết kế đo lại:** log cos(G_task, G_attn), cos(G_task, G_robust), cos(G_attn, G_robust) và tỷ lệ xung đột mỗi vài epoch (vd. mỗi 2-3 epoch) xuyên suốt quá trình train của dòng 7, thay vì chỉ tại checkpoint cuối — vẫn dùng đúng cơ chế trích gradient và vùng đo bất đối xứng đã có ở §10.2-10.3, chỉ khác là log lặp lại theo thời gian.

**Chi phí:** tận dụng lại đúng lần **pilot 1-seed dòng 7 cuối tuần 6** (§13) làm nguồn log — không cần thêm run mới, chỉ cần bật instrument gradient trong lần pilot này.

**Đầu ra:** 2 biểu đồ đường theo epoch — (a) cosine similarity của 3 cặp, (b) % block xung đột — biến §10 từ một con số tĩnh thành một narrative training dynamics, dùng trực tiếp làm căn cứ cho quyết định có/không cần PCGrad (dòng 8) ở §10.6, thay vì chỉ dựa vào một điểm đo cuối.

### 10.6 Cây quyết định

| Kết quả | Hành động |
|---|---|
| cos(G_task, G_attn) ≥ 0 phần lớn | Bằng chứng toán học cho narrative "prior không gian bổ trợ" |
| cos(G_task, G_robust) âm ở đầu train, dương ở cuối | Phát hiện thú vị: network tự điều chỉnh khi hội tụ — kể được câu chuyện training dynamics |
| Có xung đột thật (cos < 0 rõ, tỷ lệ cao) | Áp dụng **gradient-conflict-aware extension** bằng PCGrad (dòng 8) — dùng đúng khung diễn đạt §10.7 |
| Không xung đột ở đâu cả | Kết luận hợp lệ: tổ hợp tuyến tính đơn giản đã đủ tốt |

### 10.7 Khung diễn đạt cho PCGrad (tránh overclaim)

**Chuỗi lập luận đúng:** phát hiện (§10.4-10.5) → giải thích (§10.2-10.3) → can thiệp có điều kiện (PCGrad, chỉ nếu bước phát hiện đòi hỏi). Không đảo ngược chuỗi này.

**Cách gọi tên (bắt buộc):**
- ĐƯỢC dùng: "gradient-conflict-aware extension", "gradient-surgery-based optimization extension", "conditional application of PCGrad (Yu et al., 2020)"
- KHÔNG dùng: "a novel PCGrad algorithm", "our PCGrad" — đóng góp ở đây là *việc áp dụng có điều kiện, dựa trên bằng chứng gradient đo được cho đúng tổ hợp 3 loss này* (Bậc 3), không phải bản thân thuật toán.
- Cả hai kết quả (xung đột / không xung đột) đều là kết luận hợp lệ — không coi "không xung đột" là thất bại.

---

## 11. Bộ chỉ số đánh giá & Kiểm định thống kê

### 11.1 Bảng kết quả chính (Table 1)

Cột: `Method | Accuracy | Macro-F1 | Balanced Acc | Params (M) | FLOPs | Size (MB) | Latency (ms) | ECE`

- **Class imbalance:** Macro F1, Balanced Accuracy, per-class recall — bắt buộc.
- **Error analysis:** Confusion matrix + phân tích định tính 2-3 case điển hình.
- **Feature visualization:** t-SNE/UMAP + **silhouette score** định lượng (trên embedding gốc trước khi giảm chiều).
- **Calibration:** Expected Calibration Error (ECE).
- **Grad-CAM++:** minh họa vùng nhìn, đối chiếu attention map teacher-student — **chỉ minh họa, không phải bằng chứng novelty định lượng**.
- **Robustness benchmark:** % suy giảm accuracy giữa Student thường vs Full method trên từng loại nhiễu, kèm **mCE/Relative Robustness** (§9.4), tách riêng seen (8 loại) / unseen (4 loại).
- **🆕 Per-class × per-corruption heatmap:** ma trận nhiệt (hàng = 6 lớp TrashNet, cột = 12 loại nhiễu §9.2, giá trị = % suy giảm accuracy so với clean) cho dòng 7 (và tùy chọn 4c để so sánh) — bổ sung cho con số robustness benchmark tổng hợp ở trên, giúp phát hiện cặp lớp/loại-nhiễu nào là điểm yếu cụ thể thay vì chỉ một con số trung bình che khuất chi tiết. Effort thấp — tổng hợp lại số liệu suy luận đã có từ TrashNet-C theo 2 chiều thay vì 1 chiều, không cần infer thêm.

### 11.2 Thứ tự ưu tiên bằng chứng thống kê (§11-bis)

**Bằng chứng chính:** effect size (Cohen's d) + mean±std qua 5-fold + 95% CI của khác biệt + **tỷ lệ fold thắng** (VD "5/5" hay "4/5" fold mà dòng 7 vượt baseline) + xu hướng ablation (§7.2) nhất quán theo đúng hướng kỳ vọng.

**Bằng chứng hỗ trợ (không phải căn cứ quyết định duy nhất):** paired t-test và Wilcoxon signed-rank test.

**Lý do đảo thứ tự ưu tiên:**
1. n=5 fold là mẫu rất nhỏ — power thấp, p-value "đẹp" ở n=5 dễ là kết quả của 1-2 fold lệch mạnh.
2. **Cảnh báo giả định độc lập của paired t-test trên k-fold (Dietterich, 1998):** các fold chia sẻ phần lớn dữ liệu train với nhau — không độc lập thật, nên p-value có xu hướng lạc quan hơn thực tế một cách có hệ thống. Không dùng ngôn ngữ "kiểm định nghiêm ngặt" trong Kết quả; nêu rõ giới hạn này ở Hạn chế. Tùy chọn nếu còn thời gian: corrected resampled paired t-test (Nadeau & Bengio, 2003).
3. Tỷ lệ fold thắng là bằng chứng rẻ (không tốn thêm run), trực quan hơn p-value cho người đọc không chuyên thống kê.
4. Ablation (§7.2) là trục bằng chứng độc lập với thống kê k-fold — nếu hướng ablation nhất quán với gradient analysis (§10) và kết quả 5-fold, đó là bằng chứng hội tụ từ nhiều nguồn (multi-source convergent evidence).

**Wilcoxon signed-rank test:** chạy song song paired t-test cho cùng cặp so sánh (dòng 7 vs baseline mạnh nhất, So sánh #2/#3 nếu đủ dữ liệu) — 0 ngày thêm (dùng lại dữ liệu 5-fold), tăng độ tin cậy khi n nhỏ, cho phép so sánh trực tiếp phương pháp luận thống kê với Frontiers 2026 (họ cũng dùng Wilcoxon).

**Dual Pareto framing (~0.5 ngày, dùng số liệu sẵn có):** ngoài biểu đồ accuracy-vs-size/FLOPs tiêu chuẩn, vẽ thêm biểu đồ thứ hai: **robustness (accuracy trung bình dưới corruption, severity 3) vs size/FLOPs**, cùng tập model (dòng 1, 3, 4c, 7 + baseline mạnh). Góc trình bày chưa đối thủ nào làm.

**Về Focus-RCNet:** vì không tái lập được trên cùng backbone/dataset, so sánh ở dạng **gián tiếp qua số liệu đã công bố**, nêu rõ khác biệt dataset/backbone như một giới hạn.

---

## 12. Edge Deployment

Pipeline: PyTorch → ONNX → TensorRT/TFLite → Raspberry Pi hoặc Jetson Nano. Đo FPS, latency, memory thực tế cho model tốt nhất + 1-2 baseline.

**Phương án dự phòng nếu không có phần cứng:** báo cáo latency ước tính qua FLOPs + benchmark CPU thường (laptop), nêu rõ trong Hạn chế. Không làm sập bài, chỉ giảm sức nặng phần này.

**Đo quantization thật (nâng ưu tiên, ~1 ngày, làm nếu khả thi):** cả Frontiers 2026 và MDPI Sustainability 2026 chỉ báo cáo kích thước INT8 **ước lượng**, không đo accuracy sau lượng tử hóa thật — tự nhận đây là hạn chế của họ. Nếu đo được accuracy thực tế sau PTQ/QAT cho dòng 7 trên phần cứng thật, đây là điểm vượt trội cụ thể so với cả 2 đối thủ gần nhất. Nếu không kịp, giữ phương án ước lượng qua FLOPs — không phải go/no-go bắt buộc.

---

## 13. Timeline chi tiết — 14 tuần (Tier chính) + 6 tuần mở rộng tùy chọn (Tier mở rộng, tuần 15-20)

**🆕 Giả định compute budget (đọc trước khi dùng bảng dưới đây):** mọi ước lượng "ngày"/"tuần" trong timeline này giả định có sẵn quyền truy cập GPU tương đối liên tục, không có hàng đợi job dài (vd. GPU riêng hoặc cụm ít người dùng chung). Nếu chạy trên cluster chia sẻ hoặc dịch vụ free-tier (Colab/Kaggle notebook), các con số này nên được hiểu là **compute time thuần**, không phải wall-clock time thực tế — cần cộng thêm buffer cho thời gian chờ hàng đợi, giới hạn phiên chạy liên tục (vd. Colab free ngắt sau ~12h), và khả năng phải resume/checkpoint lại giữa chừng. Kế hoạch không đưa ra con số buffer cụ thể cho phần này vì phụ thuộc hạ tầng thực tế của người thực hiện — chỉ nêu rõ giả định để tránh nhầm lẫn compute time với thời gian thực tế đến hạn nộp bài.

| Tuần | Nội dung | Đầu ra | Go/No-go |
|---|---|---|---|
| 1 | Chuẩn bị dữ liệu (TrashNet 6 lớp gốc), kiểm tra class imbalance, bắt đầu crop TACO song song **và freeze class-mapping protocol TACO→6 lớp**. **🔧 Tách Dev/Corruption-holdout stratified (~15%) khỏi toàn bộ pipeline CV, giữ lại phần còn lại cho 5-fold (§6.4)**. Train teacher ResNet50 **và student thuần ResNet18** (dòng 1). **🔧 Chốt epoch budget/batch size/LR schedule dùng xuyên suốt toàn bộ ablation + 5-fold (§8.2), dựa trên convergence curve của cặp teacher/student này**. **🆕 (Song song, không chặn các đầu ra trên) Feasibility check dataset 2 (§6.5): tải thử Garbage Classification, đo thời gian 1 epoch thật của ResNet18 trên phần cứng dự kiến dùng, kiểm tra dung lượng ổ cứng** | Teacher/student baseline accuracy, thống kê phân bố lớp, văn bản chốt mapping, **văn bản chốt cách chia dữ liệu (§6.4) và epoch budget/LR schedule**, **🆕 ước lượng thời gian/tính khả thi thực tế cho Tier mở rộng tuần 15-17** | Teacher nên hơn student ≥8% (tham khảo, không tuyệt đối); nếu thấp hơn nhưng KL(p_T‖p_S) vẫn khác biệt rõ, có thể tiếp tục. Nếu teacher không đủ mạnh, thử 1 seed DenseNet121/ConvNeXt-Tiny |
| 2 | Train 2 baseline CNN thuần còn lại — MobileNetV3, EfficientNet-B0. **🆕 + EfficientFormer-L1** (baseline kiến trúc có attention tích hợp sẵn, §7.1) | Bảng baseline sàn (4 kiến trúc) | — |
| 3 | Cài đặt 3 baseline KD kinh điển (Vanilla KD, FitNet, Attention Transfer). **🆕 + CutMix+KD** (cùng backbone ResNet18, §7.1) | Bảng baseline KD (4 baseline) | — |
| 4 | Student thuần (không ECA) + L_CE + α·L_logit — tái lập KD chuẩn | Ablation dòng 3, phải ≈ số Vanilla KD tuần 3 | Lệch >2% thì dừng debug pipeline |
| 5 | Train dòng 4a (1-2 seed). Dòng 4b tái dùng số liệu Attention Transfer tuần 3. Ghép thành dòng 4c (**pilot/tune 1-2 seed, KHÔNG 5-fold ở bước này**) + tune γ, chốt dùng cho các dòng sau. **🆕 + pilot dòng 4c+ (1-2 seed, cùng γ vừa chốt) — kiểm soát confound cho So sánh #3-bis (§7.2)** | Ablation 4a, 4b, 4c, **4c+** (pilot); γ chốt (tạm) | Dòng 4a > 3 VÀ 4b > 3; nếu một trục không vượt, xem lại vị trí module/siêu tham số trước khi kết luận |
| 6 | Thêm L_Consistency (dòng 2/5/6, 1-2 seed). Viết code PCGrad song song. **Cuối tuần: pilot 1 seed dòng 7** sanity-check hội tụ trước khi tune+freeze tuần 7. **🆕 Bật instrument log gradient liên tục theo epoch (§10.5-bis) ngay trong lần pilot này — không cần run riêng thêm** | Ablation 2,5,6; code PCGrad sẵn sàng; kết quả pilot dòng 7; **🆕 log gradient theo epoch** | Dòng 6 > 5; pilot dòng 7 hội tụ bình thường — nếu bug, debug ngay, KHÔNG chuyển sang tune+freeze |
| 7 | Ghép đầy đủ, tune δ (1-2 seed/giá trị, không 5-fold) + gradient interference analysis (1-2 seed) + **FREEZE tổng thể cuối tuần**: kiến trúc, toàn bộ loss, mọi hyperparameter | Ablation dòng 7 (1-2 seed), bảng sensitivity δ, biểu đồ cosine similarity, văn bản freeze | Dòng 7 (1-2 seed) vượt rõ 4c và 6; 5-fold chính thức dời sang tuần 8, chỉ sau khi freeze |
| 7b | (Nếu phát hiện xung đột) chạy dòng 8 — Full + PCGrad, 1-2 seed trước, nâng 5-fold nếu khả quan | Ablation dòng 8 | — |
| 8 | 5-fold chính thức cho nhóm bắt buộc: dòng 1, 3, **4c (lần đầu 5-fold)**, 7 + 1-2 baseline ngoài mạnh nhất + tính effect size, CI, tỷ lệ fold thắng, paired t-test. **🆕 + 4c+ nếu escalate từ pilot tuần 5 (§7.2, §8.3)** | Bảng kết quả chính | — |
| 9 | Xây TrashNet-C mở rộng **(🆕 8 loại seen + 4 loại unseen, §9.2)**, chạy toàn bộ method. **Chạy TrashNet-C trên {1, 3, 4c, 7}** (cần cho §11.1/§11.2). Test TACO **+ RealWaste zero-shot (~0.5 ngày) trên {3, 4c, 7}** (§9.3). **🆕 + tính mCE/Relative Robustness tách seen/unseen (§9.4, ~0.5 ngày) + per-class×per-corruption heatmap (§11.1)** | Bảng robustness nội bộ + ngoại lai, **🆕 bảng mCE/RR, heatmap** | **Không phải go/no-go pass/fail.** Đây là câu hỏi đánh giá cho RQ3: so sánh mức giữ-vững hiệu năng của {3, 4c, 7} trên TACO/RealWaste để mô tả *cải thiện nào transfer, ở mức nào* — kết quả "transfer một phần" hoặc "không transfer rõ" vẫn là kết luận hợp lệ, báo cáo trung thực thay vì coi là thất bại. Mốc 41.04%/0.3648 (sau khi đã re-check §4.3) chỉ dùng làm điểm tham chiếu, không phải ngưỡng đạt/rớt |
| 10 | Class imbalance metrics, error analysis, calibration (ECE) | 3 bảng/phân tích bổ sung | — |
| 11 | t-SNE/UMAP, Grad-CAM++ | Bộ hình minh họa | — |
| 12 | Edge deployment: ONNX→TensorRT/TFLite, đo trên RPi/Jetson Nano. **+ Dual Pareto chart (~0.5 ngày). + Quantization thật PTQ/QAT nếu khả thi (~1 ngày)** | Bảng deploy, Pareto accuracy-size & robustness-size, bảng quantization (nếu kịp) | Có phương án dự phòng nếu thiếu phần cứng; quantization không phải go/no-go |
| 13 | Tổng hợp bảng/hình, viết Kết quả + Thảo luận, biểu đồ Pareto. **+ Wilcoxon (0 ngày)** | Bản thảo kết quả hoàn chỉnh | — |
| 14 | Viết Related Work (§16) **+ Table 1 so sánh (§16.1, ~0.5 ngày) + Motivation quốc tế (§4.3)**, hoàn thiện bản thảo, buffer sửa lỗi. **Nếu còn thời gian dư:** exploratory extension adaptive δ (§17, tùy chọn) | Bản thảo cuối (Tier chính) | — |
| 🆕 15-17 | **(Tier mở rộng, không bắt buộc)** Dataset thứ 2 (§6.5): dựng pipeline, **train teacher ResNet50 mới trên dataset 2** (go/no-go nhỏ, giống tuần 1 gốc) + train {1, 3, 4c, 7} × 5-fold trên Garbage Classification | Bảng kết quả (Table 1-bis) cho dataset thứ 2 | Không go/no-go bắt buộc cho Tier chính — nhưng cần teacher mới đạt gap tối thiểu tương tự tuần 1 (hoặc KL(p_T‖p_S) đủ khác biệt) trước khi chạy dòng 3/4c/7; dừng giữa chừng nếu hết thời gian, không ảnh hưởng Tier chính đã hoàn tất ở tuần 14 |
| 🆕 18 | **(Tier mở rộng)** Phân tích thống kê dataset thứ 2 (dùng lại khung §11.2, không phát sinh tiêu chuẩn riêng) + few-shot TACO tùy chọn nếu còn dư thời gian (§9.5) | Bảng so sánh RQ1/RQ2 giữa 2 dataset; kết quả few-shot TACO (nếu chạy) | — |
| 🆕 19-20 | **(Tier mở rộng)** Buffer: tích hợp kết quả dataset thứ 2 vào Kết quả/Thảo luận, xử lý phát sinh compute-queue (giả định §13), sửa bản thảo cuối bao gồm cả Tier mở rộng | Bản thảo cuối cùng (Tier chính + mở rộng) | — |

**Lưu ý domino risk (tuần 8-12):** toàn bộ tuần 8-12 build trên checkpoint đóng băng cuối tuần 7 — nếu dòng 7 chưa ổn ở thời điểm đó, mọi phân tích sau đều kế thừa rủi ro đó. Pilot 1-seed cuối tuần 6 là lớp bảo vệ đầu tiên.

**Chuỗi ưu tiên tổng quát (tham chiếu nhanh, khớp với timeline ở trên):** Teacher/Student baseline (tuần 1) → Vanilla KD/baseline KD kinh điển (tuần 2-4) → 4a/4b (tuần 5) → 4c pilot/tune γ (tuần 5) → pilot Full dòng 7 (cuối tuần 6) → gradient interference analysis (tuần 7, **trước** freeze — quyết định có cần PCGrad/dòng 8 hay không) → freeze tổng thể (cuối tuần 7) → 5-fold chính thức cho {1, 3, 4c, 7} (tuần 8) → external robustness TrashNet-C/TACO/RealWaste trên {3, 4c, 7} (tuần 9) → error analysis/t-SNE/calibration (tuần 10-11) → edge deployment (tuần 12). Lưu ý: gradient analysis đứng **trước** 5-fold và external robustness, không phải sau — vì nó quyết định nhánh PCGrad phải chốt trước khi freeze.

---

## 14. Bảng rủi ro & Phương án dự phòng

| Rủi ro | Mức độ | Biện pháp |
|---|---|---|
| L_Consistency bị coi là biến thể của Adversarially Robust Distillation, không phải ý tưởng mới | Đã xử lý | Chủ động trích dẫn + định vị lại novelty ở mức tổ hợp (§4.2) |
| Teacher-student gap không đủ lớn (< 8%) | Thấp-Trung bình | Không tự động dừng — kiểm tra KL(p_T‖p_S); nếu vẫn yếu, thử DenseNet121/ConvNeXt-Tiny |
| 5-fold áp cho toàn bộ ~13 cấu hình gây bùng nổ compute (60-65 run) | Cao | Giới hạn 5-fold cho nhóm bắt buộc (§8.3), còn lại 1-2 seed |
| Dòng 7 không vượt rõ dòng 6 | Trung bình | Vẫn hợp lệ — chuyển trọng tâm sang consistency là đóng góp chính |
| **🔧 Dòng 7 không vượt rõ dòng 4c** (So sánh #3 — bằng chứng trung tâm cho RQ2/câu hỏi nghiên cứu trung tâm, khác với dòng 7 vs 6 ở trên) | Trung bình-Cao | Đây là kết quả quan trọng nhất của cả bài; nếu không đạt, đó vẫn là kết luận khoa học hợp lệ ("teacher-anchored consistency không cho robustness thêm ngoài attention distillation trên setup này") — chuyển trọng tâm bài viết sang lý giải TẠI SAO bằng gradient analysis (§10, vd. cos(G_attn, G_robust) gần 0/âm là bằng chứng con cho việc hai tín hiệu không hợp tác). Không cố ép số liệu; báo cáo trung thực kèm phân tích |
| Dòng 4a hoặc 4b không vượt rõ dòng 3 | Trung bình | Vẫn hợp lệ — viết lại novelty thành "trục còn lại là đóng góp chính", khiêm tốn hóa tuyên bố |
| **🔧 Dòng 4c không vượt rõ max(4a, 4b) dù cả hai đều vượt dòng 3** (So sánh #1 — bằng chứng trung tâm cho RQ1, khác với trường hợp một trục lẻ không vượt dòng 3 ở trên) | Trung bình | Đổi tuyên bố So sánh #1 sang "kết hợp không cho lợi ích cộng thêm rõ rệt so với dùng trục mạnh nhất riêng lẻ" — vẫn là câu trả lời hợp lệ cho RQ1 (richer KD không nhất thiết nghĩa là phải kết hợp cả hai trục), không diễn giải lại RQ1 để che giấu kết quả âm |
| γ chốt sớm ở tuần 5 có thể không tối ưu khi thêm Consistency ở dòng 7 | Thấp | Chấp nhận như xấp xỉ hợp lý; spot-check 1-2 giá trị γ lân cận ở tuần 7 |
| **🔧 Spot-check γ ở tuần 7 (nếu đổi giá trị) phá vỡ khung "tune cho 4c rồi freeze" — γ chính thức của 4c bị ảnh hưởng bởi thông tin từ dòng 7 (có Consistency)** | Trung bình | Không phải leakage vào 5-fold (dev split vẫn tách biệt) — chỉ ảnh hưởng độ "sạch" của So sánh #1 (RQ1), không ảnh hưởng So sánh #3 (4c vs 7 vẫn dùng chung γ). Chốt trước quy trình rẽ nhánh (§8.4.2): nếu γ đổi, áp dụng cho toàn bộ 4c/6/7/8 và disclose "tune 2 bước" trong Methods/Hạn chế thay vì mô tả là "tune rồi freeze" |
| Focus-RCNet không tái lập được trên cùng backbone/dataset | Thấp | So sánh gián tiếp qua số liệu công bố, nêu rõ giới hạn |
| TrashNet quá sạch, robustness claim yếu | Đã xử lý | Test ngoại lai TACO/RealWaste |
| Baseline mở rộng tốn thời gian nhất | Cao | Đã tính riêng 2 tuần (2-3), không dồn chung với ablation chính |
| Edge deployment phụ thuộc phần cứng | Trung bình | Phương án dự phòng: ước tính qua FLOPs |
| Chọn ResNet18 nặng hơn MobileNetV3/EfficientNet-B0 | Thấp | Đánh đổi có chủ đích: vẫn nhẹ hơn teacher nhiều, giữ dòng 3 là baseline "zero built-in attention" thật sự |
| Số liệu trích từ Frontiers 2026/MDPI Sustainability 2026 bị trích sai/lệch, hoặc bài bị rút/đính chính | Thấp-Trung bình | Đã đối chiếu qua tìm kiếm độc lập; riêng τ=3 **chưa** xác nhận độc lập — bắt buộc tự tải PDF gốc đối chiếu trước khi đưa vào bản thảo |
| Đổi từ 3 nhóm rác (luật VN) sang 6 lớp TrashNet gốc làm mất câu chuyện động lực thực tiễn VN | Thấp | Chấp nhận có chủ đích để so sánh trực tiếp với 4 bài đối chứng quốc tế; có thể giữ 1 đoạn ngắn Discussion/Future Work nhắc hướng áp dụng 3-nhóm cho ngữ cảnh VN |
| Gradient analysis không cho kết quả rõ ràng (cos gần 0) | Thấp | Vẫn báo cáo trung thực — "không phát hiện xung đột đáng kể" là kết luận hợp lệ |
| Trùng lặp với công trình chưa search ra | Thấp | Tự kiểm tra thêm trên Google Scholar/Semantic Scholar trước khi submit |
| Cam kết 5-fold cho dòng 7 trước khi biết method hội tụ tốt/không bug | Đã xử lý | Pilot 1 seed cuối tuần 6; 5-fold chính thức dời sang tuần 8 |
| Cam kết 5-fold cho dòng 4c ở tuần 5 trước khi γ ổn định (γ có thể đổi khi thêm Consistency ở tuần 7) | Đã xử lý | Tuần 5 chỉ pilot/tune γ (1-2 seed); 5-fold chính thức cho 4c dồn sang tuần 8 cùng nhóm bắt buộc |
| Epoch budget không chốt từ tuần 1, bị đổi giữa chừng (vd. tuần 6-7 khi tune δ) → toàn bộ dòng 1/3/4a/4b/4c đã train tuần 1-5 vi phạm yêu cầu "cùng epoch budget" (§8.2), phải train lại | Cao nếu bỏ sót | Bắt buộc chốt epoch budget/batch size/LR schedule ngay ở đầu ra tuần 1 (§8.2), dùng xuyên suốt, không đổi ngầm |
| Protocol-lock (§8.1) trước đây chỉ áp cho cặp 4c-vs-7, bỏ ngỏ cho So sánh #1 (3/4a/4b/4c) và #2 (6 vs 5) — cùng một phản biện "khác biệt do randomness/protocol" vẫn áp dụng được cho hai so sánh này | Đã xử lý | Mở rộng §8.1 thành nguyên tắc chung cho toàn bộ ma trận ablation, không riêng một cặp |
| TrashNet-C dựng từ toàn bộ TrashNet (không tách riêng) → đánh giá checkpoint fold-i trên ảnh nhiễu vốn thuộc train của chính fold-i đó — rò rỉ dữ liệu thật | Cao nếu bỏ sót | Tách Dev/Corruption-holdout cố định từ tuần 1 (§6.4), không nằm trong train của fold nào; TrashNet-C và toàn bộ hyperparameter tuning đều dùng chung split này |
| **🔧 Dùng chung Dev/Corruption-holdout cho cả tuning và TrashNet-C không gây leakage dữ liệu, nhưng nếu δ được chọn bằng accuracy dưới nhiễu severity 3 trên chính holdout đó, δ đã được tối ưu để tốt trên đúng bằng chứng dùng để chứng minh RQ2 — TrashNet-C hết còn "chưa bị chạm tới"** | Trung bình | Mở rộng tường lửa severity đã có ở §9.2: δ chỉ được tune bằng severity 1-2/accuracy sạch, không bao giờ severity 3; giữ severity 3 trên holdout "sạch" với cả training lẫn tuning (§6.4, §8.4.3). Không cần chia 3 phần dữ liệu (Dev/Corruption-test/CV pool riêng) trừ khi muốn cực sạch về phương pháp và chấp nhận giảm dữ liệu 5-fold |
| §9.3 chỉ mở rộng robustness sang {3,4c,7} nhưng §11.1 (% suy giảm so với "Student thường") và §11.2 (Dual Pareto) đều cần số liệu robustness của dòng 1 | Đã xử lý | TrashNet-C chạy trên {1,3,4c,7}; TACO/RealWaste (đắt hơn) giữ {3,4c,7} |
| Go/no-go TACO/RealWaste ở tuần 9 bị đặt thành ngưỡng đạt/rớt, mâu thuẫn với khung "we evaluate whether..." ở §2 | Đã xử lý | Đổi thành câu hỏi đánh giá (RQ3): báo cáo mức transfer của {3,4c,7}, không coi kết quả yếu là "fail" |
| Paired t-test trên 5-fold không thỏa giả định độc lập (Dietterich, 1998) | Trung bình | Đảo thứ tự ưu tiên bằng chứng (§11.2) — effect size/CI/tỷ lệ fold thắng là bằng chứng chính, p-value chỉ hỗ trợ |
| PCGrad bị viết như đóng góp thuật toán mới | Đã xử lý | Dùng đúng khung §10.7 |
| Exploratory extension adaptive δ (§17) âm thầm phình phạm vi | Đã xử lý | Neo chặt vào buffer tuần 14, không go/no-go riêng, bỏ ngay nếu hết thời gian |
| 🔧 Đánh giá external chỉ chạy dòng 7, không tách bạch được gain từ attention hay consistency | Mới nhận diện | Mở rộng §9.3: chạy {3, 4c, 7} trên TACO/RealWaste, không chỉ dòng 7 |
| 🆕 Dòng 7 khác 4c ở CÙNG LÚC hai biến (tiếp xúc ảnh nhiễu + có KL-consistency neo teacher) — không tách được đóng góp riêng của từng biến | Trung bình | Thêm dòng 4c+ (§7.2) cô lập riêng biến "tiếp xúc ảnh nhiễu" (CE-aug, không KL); ba kịch bản kết quả (7>4c+>4c / 4c+≈7 / 7>4c≈4c+) đều đã có diễn giải sẵn ở So sánh #3-bis |
| 🆕 Ước lượng effort dataset thứ 2 (§6.5, "~2 tuần") chỉ tính compute time thuần, chưa tính debug pipeline mới, tune lại LR/schedule, và GPU-hours thực tế cho 4×5 model | Trung bình | Nâng lên 3 tuần (tuần 15-17) + đặt vào Tier mở rộng không bắt buộc, có buffer riêng ở tuần 19-20; không go/no-go nào của Tier chính (tuần 1-14) phụ thuộc vào hạng mục này |
| 🆕 Ranh giới seen/unseen corruption (§9.2) có thể bị xem là tùy tiện — vd. elastic transform có thể tương quan phần nào với affine tilt, fog với gamma darkening, về mặt "ảnh bị biến dạng nói chung" | Trung bình | Đã viết đoạn biện minh theo cơ chế sinh nhiễu (rigid vs non-rigid warp, suy giảm tương phản không đồng nhất vs làm tối đều, radial vs linear vs isotropic blur, impulsive vs vùng liền khối) ở §9.2 — đưa nguyên đoạn này vào Methods |
| 🆕 Giả định compute budget (GPU liên tục, không hàng đợi) ngầm định trong mọi ước lượng ngày/tuần ở §13 — có thể sai nếu chạy trên cluster chia sẻ hoặc free-tier (Colab/Kaggle) | Thấp-Trung bình | Đã nêu rõ giả định ở đầu §13; coi các con số "ngày" là compute time, cộng thêm buffer thực tế tùy hạ tầng — không có con số buffer cụ thể vì phụ thuộc người thực hiện |

---

## 15. Checklist trước khi nộp bài

- [ ] Ký hiệu toán học nhất quán xuyên suốt (α=logit, γ=Attention, δ=Consistency)
- [ ] Phần PCGrad (nếu dòng 8 kích hoạt) dùng đúng khung §10.7 — không "novel PCGrad algorithm"
- [ ] Nếu làm exploratory extension adaptive δ (§17): chỉ ở Discussion dạng phụ lục, không ở Abstract/Introduction
- [ ] Related Work trích đủ danh mục §16, đặc biệt ARD/PCGrad/DHO/NetDistiller
- [ ] Bảng kết quả chính có đủ: accuracy, macro-F1, balanced accuracy, efficiency metrics, ECE
- [ ] Ablation có dòng self-consistency control (dòng 2, 5)
- [ ] Ablation tách riêng ECA (kiến trúc) và L_Attention (loss) thành ma trận 3/4a/4b/4c — KHÔNG gộp chung
- [ ] Dùng tên module cụ thể "ECA" thay vì "ECA/CA" mơ hồ
- [ ] Công thức L_Consistency-self định nghĩa tường minh trong §5
- [ ] Kết quả nêu bật So sánh #2 (6 vs 5) và #3 (7 vs 4c) — không dùng "7 vs 5" để suy ra riêng vai trò teacher-anchoring
- [ ] Focus-RCNet ghi rõ là so sánh gián tiếp, không phải tái lập nội bộ
- [ ] Robustness test có cả nội bộ (TrashNet-C) và ngoại lai (TACO/RealWaste), **🔧 chạy trên cả {3, 4c, 7}**
- [ ] Gradient interference analysis có ít nhất biểu đồ cosine similarity theo giai đoạn train
- [ ] Introduction tách rõ động lực thực tiễn (mạnh) và đóng góp kỹ thuật (khiêm tốn)
- [ ] Hạn chế nêu rõ nếu thiếu edge deployment thực tế hoặc hạng mục bị cắt do thời gian
- [ ] Hạn chế nêu rõ dòng 2/4a/4b/5/6 chỉ 1-2 seed — chỉ dòng 1/3/4c/7 + baseline mạnh nhất có bằng chứng thống kê đầy đủ
- [ ] Bảng kết quả chính có 95% CI và effect size (Cohen's d) bên cạnh paired t-test
- [ ] Bảng kết quả chính báo cáo tỷ lệ "số fold thắng" cho mỗi so sánh 5-fold chính
- [ ] Kết quả trình bày effect size + mean/std + CI + tỷ lệ fold thắng + xu hướng ablation TRƯỚC, p-value chỉ nêu SAU
- [ ] Nếu teacher gap < 8%, Hạn chế nêu rõ lý do vẫn tiếp tục (bằng chứng KL(p_T‖p_S))
- [ ] Class-mapping protocol TACO→6 lớp đã chốt văn bản từ tuần 1, không đổi giữa chừng
- [ ] Không dùng khung pháp lý Luật BVMT 2020/3 nhóm rác làm động lực chính ở Introduction
- [ ] Số liệu trích từ Frontiers 2026/MDPI Sustainability 2026 (94.18%, 41.04%, macro-F1 0.3648, τ=3) đã đối chiếu lại trực tiếp trên PDF gốc
- [ ] Bảng Related Work (§16.1) có đủ 5 dòng: Focus-RCNet, paper nông nghiệp, Frontiers 2026, MDPI Sustainability 2026, và bài này
- [ ] Epoch budget của bảng ablation chính và vòng 5-fold giống nhau, hoặc nêu rõ lý do nếu khác
- [ ] Kết quả có cả paired t-test và Wilcoxon signed-rank test cho so sánh chính
- [ ] Có biểu đồ robustness-vs-size (Dual Pareto) bên cạnh accuracy-vs-size tiêu chuẩn
- [ ] Hạn chế nêu rõ phạm vi robustness benchmark giới hạn ở TrashNet-C + TACO + RealWaste zero-shot, chưa bao phủ mọi điều kiện triển khai thực tế
- [ ] Code KL divergence (L_logit, L_Consistency) đúng chiều `F.kl_div(student_log_prob, teacher_prob)` — không đảo input/target
- [ ] KL(p_T‖p_S) ở go/no-go tuần 1 tính trên toàn bộ validation set, không phải vài batch
- [ ] Nhiễu train L_Consistency (severity 1-2) tách khỏi nhiễu đánh giá robustness chính (severity 3) — không leakage
- [ ] Kết quả/Thảo luận nêu cụ thể domain shift TrashNet vs TACO khi lý giải chênh lệch performance
- [ ] Không dùng "ultra-lightweight"/"siêu nhẹ" cho ResNet18
- [ ] Không claim "ECA tốt hơn CA"
- [ ] Các dòng 1-2 seed dùng ngôn ngữ "xu hướng", không dùng "statistically significant"
- [ ] t-SNE/UMAP có kèm silhouette score định lượng
- [ ] Văn bản freeze tổng thể cuối tuần 7 đã chốt trước khi bắt đầu tuần 8
- [ ] Pilot 1-seed dòng 7 đã chạy và hội tụ ổn định cuối tuần 6, trước khi tune δ/freeze ở tuần 7
- [ ] Hạn chế nêu rõ paired t-test trên 5-fold không thỏa giả định độc lập (Dietterich, 1998) — không dùng "kiểm định nghiêm ngặt"
- [ ] Công thức L_Attention dùng đúng attention map kiểu Zagoruyko (tổng bình phương activation theo channel, không conv học tham số) — đã code đúng, không lẫn với FitNet hint-matching
- [ ] 5-fold cho nhóm bắt buộc (1,3,4c,7,baseline) dùng CÙNG một phân chia fold (identical indices), và dùng StratifiedKFold do lớp "trash" ít ảnh
- [ ] Fold/split dùng để tune α/τ/γ/δ tách biệt khỏi 5 fold dùng báo cáo Table 1 chính; nếu không tách được, Hạn chế nêu rõ rò rỉ nhẹ
- [ ] Epoch budget/batch size/LR schedule đã chốt thành văn bản ở tuần 1, dùng giống hệt cho mọi dòng ablation (1-8) và vòng 5-fold — không có dòng nào âm thầm dùng budget khác
- [ ] Protocol-lock (seed nhóm, epoch, LR schedule, augmentation) áp dụng cho toàn bộ ma trận ablation §7.2, không chỉ riêng cặp 4c-vs-7
- [ ] Dev/Corruption-holdout (§6.4) đã tách trước tuần 1, không nằm trong train của bất kỳ fold nào trong 5-fold — dùng cho cả tuning và làm nguồn ảnh sạch của TrashNet-C
- [ ] TrashNet-C chạy trên {1,3,4c,7} (không chỉ {3,4c,7}) vì §11.1/§11.2 cần số liệu robustness của dòng 1
- [ ] Robustness/external evaluation dùng lại 5 checkpoint đã train ở vòng 5-fold tuần 8 (báo cáo mean±std/CI), không phải 1 checkpoint đơn lẻ mới
- [ ] Tiêu chí chọn δ chỉ dùng accuracy sạch hoặc severity 1-2 trên Dev/Corruption-holdout — không bao giờ dùng severity 3 làm tín hiệu tuning, để severity 3 trên TrashNet-C giữ vai trò bằng chứng "chưa bị chạm tới" cả training lẫn tuning
- [ ] Bảng rủi ro/Thảo luận có phương án dự phòng cho cả hai trường hợp kết quả trung tâm không đạt: dòng 7 không vượt 4c (So sánh #3), và 4c không vượt max(4a,4b) (So sánh #1) — không chỉ các so sánh phụ hơn
- [ ] Nếu γ đổi sau spot-check tuần 7 (§8.4.2), Methods disclose rõ quy trình tune 2 bước cho γ, không mô tả 4c như tune độc lập; nếu γ không đổi, không cần disclose thêm
- [ ] 🆕 Ablation có dòng 4c+ cô lập "tiếp xúc ảnh nhiễu" (CE-aug) khỏi "có KL-consistency neo teacher" — So sánh #3 (RQ2) không tự nhận cả hai biến là một
- [ ] 🆕 Kết quả trình bày đủ ba kịch bản diễn giải cho So sánh #3-bis (7>4c+>4c / 4c+≈7 / 7>4c≈4c+), không chỉ nêu kịch bản đẹp nhất
- [ ] 🆕 Corruption set TrashNet-C đã mở rộng thành 8 seen + 4 unseen; Methods có đoạn ngắn biện minh vì sao 4 loại unseen khác cơ chế so với 8 loại seen (§9.2)
- [ ] 🆕 Bảng robustness có mCE và Relative Robustness, tách riêng seen/unseen (§9.4)
- [ ] 🆕 Có per-class × per-corruption heatmap bên cạnh con số robustness benchmark tổng hợp (§11.1)
- [ ] 🆕 Gradient interference analysis có log liên tục theo epoch (§10.5-bis), không chỉ một checkpoint cuối — ít nhất 2 biểu đồ theo thời gian
- [ ] 🆕 Nếu chạy dataset thứ 2 (§6.5, Tier mở rộng): dùng đúng khung ưu tiên bằng chứng đã có ở §11.2, không phát sinh tiêu chuẩn kiểm định riêng; nếu không chạy kịp, ghi rõ đây là hạng mục Tier mở rộng bị cắt do thời gian, không phải thiếu sót của Tier chính
- [ ] 🆕 Giả định compute budget (GPU liên tục, không hàng đợi) đã nêu rõ trong Methods/Kế hoạch nếu timeline thực tế bị ảnh hưởng — không lẫn compute time với wall-clock time khi báo cáo tiến độ
- [ ] 🆕 Batch composition của 4c+ đúng công thức đã chốt (§7.2): B ảnh sạch + B ảnh nhiễu, `L_CE = ½[L_CE(x)+L_CE(x̃)]`, L_logit/L_Attention chỉ trên x, teacher chỉ 1 forward trên x sạch
- [ ] 🆕 Nếu chạy dataset 2 (§6.5): đã train teacher ResNet50 mới trên dataset 2 trước dòng 3/4c/7 — không tái dùng teacher đã train trên TrashNet (khác class count/domain)
- [ ] 🆕 EfficientFormer-L1 dùng AdamW + cosine-annealing (không SGD), Methods ghi rõ lý do miễn trừ protocol-lock vì đây là baseline ngoài (§7.1), không thuộc ma trận ablation nội bộ

---

## 16. Related Work — Danh mục trích dẫn bắt buộc

- **Adversarially Robust Distillation** (Goldblum et al., 2019) — cơ chế gốc của L_Consistency; nêu rõ đây là mở rộng sang corruption tự nhiên, không phát minh lại
- **CMKD** — corruption robustness qua correlation matching, test CIFAR-100-C
- **Mixup-based distillation robustness transfer** — test CIFAR-100-C
- **Cross-View Consistency Regularisation for KD (CRLD)**
- **PCGrad** (Yu et al., 2020) — cơ chế gradient surgery gốc
- **Dual-Head Optimization (DHO)** — phát hiện xung đột gradient CE-KD trong distillation
- **NetDistiller** — phát hiện xung đột gradient student-teacher, áp PCGrad
- **Focus-RCNet** và **paper nông nghiệp** (Hybrid Attention+Logit Distillation, Swin→MobileNetV3) — hai đối chứng gần domain nhất, bắt buộc so sánh trực tiếp
- **Echchidmi & Bouayad (2026)**, "Compact waste image classification with multi-student CNNs and edge-oriented model selection", *Frontiers in Artificial Intelligence* 9:1804734 — mở đầu Motivation, nguồn số liệu zero-shot RealWaste, nguồn tham chiếu lỗi đồng bộ epoch budget
- **KD-Garbage Framework** (MDPI *Sustainability* 2026, 18(13):6392) — đối chứng thứ hai, kết luận trái chiều với Frontiers 2026 về ý nghĩa thống kê của KD; nguồn tham khảo τ (cần double-check)

### 16.1 Bảng so sánh trực tiếp Related Work (Table 1 — bắt buộc)

Dựng bảng kiểu Table 1 của Frontiers 2026, cột: `Paper | Dataset | Backbone (Teacher→Student) | Attention (kiến trúc/loss) | Robustness test | Gradient-conflict analysis | Kiểm định thống kê`, với các dòng: Focus-RCNet, paper nông nghiệp (Swin→MobileNetV3), Frontiers 2026, MDPI Sustainability 2026, và bài này. Bằng chứng trực quan cho novelty Loại C (§4.2) — làm rõ cột "Gradient-conflict analysis" và "Robustness test ngoại lai" trống ở cả 4 đối chứng, chỉ có ở bài này. Soạn tuần 13-14, ~0.5 ngày, không ảnh hưởng thực nghiệm.

---

## 17. Phụ lục — Exploratory extension tùy chọn: adaptive δ theo corruption severity

**Phạm vi và ràng buộc:** KHÔNG phải một phần của RQ chính, KHÔNG phải điều kiện go/no-go, KHÔNG được đụng vào core pipeline đã freeze cuối tuần 7. Chỉ thực hiện nếu còn thời gian dư ở buffer tuần 14.

**Lý do giới hạn chặt:** adaptive weighting (α_t, γ_t, δ_t theo thời gian) sẽ phá vỡ khái niệm "freeze" cốt lõi — nếu ảnh hưởng core method, toàn bộ chuỗi go/no-go từ tuần 7 trở đi mất hiệu lực. Bản thân ý tưởng cũng không mới (GradNorm — Chen et al. 2018; uncertainty weighting — Kendall et al. 2018), không dùng làm luận điểm novelty chính.

**Thiết kế tối giản nếu thực hiện:** heuristic đơn giản (không học f(...) bằng gradient) — scale δ theo ước lượng độ nặng corruption của batch hiện tại. Chạy 1 seed, không 5-fold, không đưa vào bảng kết quả chính.

**Cách viết nếu có kết quả:** chỉ đặt trong Discussion, dạng exploratory, không đưa vào Abstract/Introduction, không dùng để thay đổi kết luận chính của §7/§8.

**Nếu không còn thời gian:** bỏ qua hoàn toàn — không go/no-go nào bị ảnh hưởng.
