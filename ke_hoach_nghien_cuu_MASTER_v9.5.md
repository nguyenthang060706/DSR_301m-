# Kế hoạch nghiên cứu MASTER v9.5 — Knowledge Distillation cho phân loại rác canteen 3 nhóm trên thiết bị biên

*Bản v9.5 là tài liệu hoàn chỉnh, tự đứng được (thay thế toàn bộ v8, v9.0–v9.4). Lịch sử sửa đổi chi tiết từ v8 đến v9.5 được chuyển xuống Phụ lục A ở cuối file.*

---

## 1. Câu hỏi nghiên cứu & Mục tiêu

**Câu hỏi trung tâm:** Làm thế nào dùng Knowledge Distillation (KD) để tạo ra mô hình nhỏ, chính xác và đủ nhanh để phân loại rác canteen trường đại học thành 3 nhóm quyết định trên thiết bị biên?

| Hạng mục | Phát biểu | Bằng chứng chính |
|---|---|---|
| **RQ1 (Cơ chế KD & Nhãn)** | Trong cùng ngân sách tính toán và ngân sách tune, tín hiệu distillation phong phú hơn (DKD, FitNet, Attention Transfer) có vượt trội logit KD cổ điển không; và việc huấn luyện bằng nhãn mịn (10–15 lớp) rồi gộp sang 3 nhóm có lợi thế gì so với huấn luyện trực tiếp bằng 3 nhóm? | So sánh M0, M1, M1b, M2, M3, M4 (§4.2); Thí nghiệm độ mịn nhãn (§3.1) |
| **RQ2 (Domain Gap & Thích nghi)** | Khoảng cách miền từ dữ liệu công khai sang ảnh canteen đại học lớn đến mức nào; các mô hình KD có thu hẹp được khoảng cách này tốt hơn mô hình chuẩn không; và cần bao nhiêu ảnh canteen (few-shot) để thích nghi hiệu quả? | Zero-shot trên test canteen; Đường cong few-shot (§7.3); Đánh giá ngoại lai TACO/RealWaste (§6.2) |
| **Mục tiêu triển khai (Edge Goal)** | Đạt mô hình nén INT8 hoạt động ổn định trên thiết bị biên thật (Android/Raspberry Pi) với sự đánh đổi tối ưu giữa Macro-F1, độ trễ (latency), kích thước (file size) và bộ nhớ (RAM). | Báo cáo kiểm chuẩn trên phần cứng thật (§8) |

*Mở rộng tùy chọn (RQ2-bis, chỉ làm nếu dư thời gian trước tuần 5):* Đánh giá độ bền (robustness) dưới bộ nhiễu tổng hợp Canteen-C và lợi ích của hàm mất mát Consistency neo theo Teacher (nhóm M7).

**Khung tuyên bố khoa học (chống overclaim):** Không dùng từ "giải quyết triệt để domain shift", chỉ dùng "đánh giá mức độ thu hẹp khoảng cách miền". Không gọi ResNet18 là "siêu nhẹ". Kết luận âm (KD không thắng baseline mạnh, hoặc các kỹ thuật KD phức tạp không hơn logit KD) vẫn là kết luận khoa học có giá trị cao và được báo cáo trung thực.

---

## 2. Định vị tính mới & Đóng góp

Đề tài thuộc loại **Nghiên cứu thực nghiệm có kiểm soát + Ứng dụng bối cảnh hẹp + Dữ liệu chuyên biệt**. Không phát minh thuật toán mới; 3 đóng góp cụ thể:

- **C1 — Dữ liệu và quy trình gán nhãn canteen (Chắc chắn):** Bộ ảnh chụp thực tế tại canteen trường đại học (metadata chi tiết: ca chụp, vật phẩm, tình trạng bẩn/sạch, cờ mơ hồ); quy ước nhãn bằng văn bản; kiểm định độ đồng thuận Cohen's kappa (κ ≥ 0,7) giữa hai người gán độc lập.
- **C2 — So sánh KD công bằng & Toàn diện (Chắc chắn):** Đánh giá có kiểm soát giữa Logit cổ điển (Hinton), Logit hiện đại (DKD), Feature (FitNet), và Attention (AT) dưới cùng recipe huấn luyện hiện đại và cùng ngân sách tune. Kiểm chứng trên cả CNN chuẩn (Tier A: ResNet18) và mô hình triển khai biên nhẹ (Tier B: MobileNet).
- **C3 — Khảo sát đường cong thích nghi miền (Chắc chắn):** Báo cáo thực nghiệm số lượng mẫu canteen tối thiểu (k-shot) để đưa mô hình huấn luyện từ dữ liệu công khai đạt ngưỡng sử dụng thực tế.

---

## 3. Dữ liệu & Thiết kế nhãn

### 3.1 Thiết kế nhãn: Huấn luyện mịn, quyết định thô
- **Chiến lược:** Huấn luyện trên $G \approx 10\text{--}15$ lớp nhãn mịn (gộp từ dữ liệu công khai), sau đó dùng bảng ánh xạ cố định (`label_map.csv`) gom thành 3 nhóm quyết định khi suy luận:
  1. *Hữu cơ (Organic)*: Rác thực phẩm, đồ ăn thừa, vỏ hoa quả,...
  2. *Tái chế khô (Recyclable)*: Chai nhựa sạch, lon nhôm, hộp giấy sạch,...
  3. *Còn lại / Khó tái chế (Other/Landfill)*: Hộp xốp bẩn, túi nilon dính dầu, giấy ăn bẩn,...
- **Thí nghiệm độ mịn nhãn (RQ1):** Huấn luyện M0 (CE) và M1 (Hinton KD) ở 2 nhánh:
  - (a) Huấn luyện trên nhãn mịn $G$ lớp $\rightarrow$ ánh xạ sang 3 nhóm.
  - (b) Huấn luyện trực tiếp trên 3 nhóm (sử dụng 1 Teacher 3 nhóm riêng biệt).
  So sánh kết quả để kiểm chứng giả thuyết "3 nhóm quá ít dark knowledge khiến Logit KD kém hiệu quả".

### 3.2 Nguồn dữ liệu công khai
- **Bộ chính (Train/Dev/Holdout):** *Recyclable and Household Waste Classification* (Kaggle, ~15.000 ảnh, 30 lớp gốc) hoặc *TrashBusters/combined* (HuggingFace). Lọc bỏ ảnh trùng lặp bằng perceptual hash (pHash/MD5). Tách 15% làm tập Dev/Holdout cố định ngay tuần 1 (stratified split).
- **Tập ngoại lai (Outlier/Zero-shot):** TACO (crop từ bbox) và RealWaste.

### 3.3 Quy ước nhãn & Quy trình gán nhãn canteen
- Viết tài liệu `docs/label_guide.md` trước khi chụp. Quy định rõ các ca ranh giới (ví dụ: ly nhựa dính trà sữa, hộp xốp dính dầu mỡ $\rightarrow$ xếp vào "Còn lại").
- Gán thử 100 ảnh độc lập bởi 2 người ở Tuần 1. Tính Cohen's kappa: yêu cầu $\kappa \ge 0,70$. Nếu không đạt, sửa quy ước và gán lại trước khi thu thập diện rộng.
- Ảnh khó quyết định được gắn cờ `ambiguous=True`. Báo cáo riêng biệt trên tập sạch và tập mơ hồ (tập mơ hồ không dùng để tune mô hình).

### 3.4 Thu thập ảnh canteen (Tập Test & Pool Few-shot)
- Cố định camera góc nhìn từ trên xuống (top-down), nền khay/bàn đồng nhất, 1 vật chính/ảnh. Đa dạng hóa có kiểm soát: độ chiếu sáng, góc xoay, trạng thái méo mó/dơ bẩn.
- **Đơn vị thu thập theo Cụm (Cluster):** Ghi rõ `cluster_id` (gồm: ngày chụp + ca ăn + vật phẩm cụ thể).
- **Quy mô dự kiến:**
  - Test canteen cố định: $\ge 200$ ảnh/nhóm ($\approx 600$ ảnh tổng).
  - Pool few-shot: $\ge 100$ ảnh/nhóm.
  - Validation canteen: $\ge 50$ ảnh/nhóm.
- **Nguyên tắc phân chia:** Chia Train/Val/Test/Few-shot theo **Cụm vật phẩm/Ngày**, tuyệt đối không chia ngẫu nhiên từng ảnh để chống rò rỉ hình thái.

---

## 4. Mô hình & Ma trận thực nghiệm

### 4.1 Kiến trúc mô hình
- **Teacher (mặc định):** ResNet50 (pretrain ImageNet). Huấn luyện bằng recipe §5.0, **không dùng label smoothing** (Müller et al., 2019).
  - *Cổng kiểm tra Teacher (Tuần 2):* Đánh giá trên tập Dev nhãn mịn. Teacher phải đạt mức **Giảm lỗi tương đối (Relative Error Reduction)** so với Student M0 tối thiểu 20%:
    $$\text{RER} = \frac{\text{Err}_{\text{M0}} - \text{Err}_{\text{Teacher}}}{\text{Err}_{\text{M0}}} \ge 20\% \quad (\text{với } \text{Err} = 1 - \text{Macro-F1}_{\text{fine}})$$
    Nếu không đạt, thử 1 lần DenseNet121 hoặc ConvNeXt-Tiny trong tuần 2. Sau tuần 2, khóa cứng Teacher.
- **Tier A — Nghiên cứu cơ chế (Bắt buộc):** Student ResNet18 (không tích hợp sẵn attention, dùng làm sàn so sánh chuẩn mực).
- **Tier B — Triển khai biên nhẹ (Bắt buộc):** Chọn **đúng một** kiến trúc student triển khai ở cuối Tuần 1 qua 3 bước kiểm tra (Gate):
  1. *Cổng xuất:* Xuất thành công sang ONNX và TFLite FP32/INT8 không lỗi toán tử.
  2. *INT8 Smoke Test:* Mức tụt Macro-F1 Dev của M0 sau PTQ INT8 $\le 2,0$ điểm.
  3. *Latency:* Ứng viên đạt latency mô phỏng thấp nhất giữa MobileNetV3-Small, MobileNetV3-Large, và MobileNetV4-Conv-Small (`timm`).

### 4.2 Ma trận thực nghiệm Tier A (ResNet18) & Tier B (MobileNet)

| ID | Cấu hình | Vai trò trong RQ | Mức ưu tiên |
|---|---|---|---|
| **M0** | Student thuần (Cross-Entropy), recipe §5.0 | Baseline sàn (phải tối ưu mạnh) | **Bắt buộc** |
| **M1** | Hinton Logit KD (Hinton et al., 2015) | Mốc KD logit kinh điển | **Bắt buộc** |
| **M1b** | Decoupled KD - DKD (Zhao et al., CVPR 2022) | Mốc KD logit hiện đại (tách TCKD/NCKD) | **Bắt buộc** |
| **M2** | Feature Hint - FitNet (Romero et al., 2015) | Đại diện họ Feature-based KD | **Bắt buộc** |
| **M3** | Attention Transfer - AT (Zagoruyko et al., 2017) | Đại diện họ Attention-based KD | **Bắt buộc** |
| **M4** | AT + Hinton Logit KD | Đại diện Richer KD kết hợp | **Bắt buộc** |
| **Tier B** | MobileNet đã chọn với {M0, M1, KD tốt nhất} | Đánh giá chuyển giao sang Edge Student | **Bắt buộc** |
| *M5* | CutMix + KD | Augmentation-distillation kết hợp | *Mở rộng (nếu dư)* |
| *M7s/c/7* | Consistency distillation (Bộ nhiễu Canteen-C) | Đánh giá tính bền vững nâng cao | *Mở rộng (nếu dư)* |

### 4.3 Định dạng hàm mất mát (Loss Functions)
Tổng quát: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \alpha \mathcal{L}_{\text{Logit}} + \beta \mathcal{L}_{\text{Feature}} + \gamma \mathcal{L}_{\text{Attention}}$
1. **CE Loss:** Có label smoothing 0,1 cho student; kiểm tra áp dụng Class weights ở Tuần 1.
2. **Logit KD (M1):** $T^2 \cdot \text{KL}(\text{Softmax}(z_S/T) \parallel \text{Softmax}(z_T/T))$. Chú ý: input là log-prob của Student, target là prob của Teacher đã `.detach()`.
3. **DKD Loss (M1b):** $\alpha_{\text{TCKD}} \cdot \text{TCKD} + \beta_{\text{NCKD}} \cdot \text{NCKD}$. Test đơn vị bắt buộc: khi $\alpha=1$ và $\beta=(1-p_t^T)$, loss phải khớp chính xác với Hinton KD.
4. **Attention Transfer (M3):** Sử dụng chuẩn hóa **mean theo spatial** giữa activation maps của Student và Teacher ở các stage tương ứng (tránh lệch trọng số giữa các stage có độ phân giải khác nhau):
   $$\mathcal{A}(F) = \frac{\sum_{c=1}^C |F_c|^2}{\| \sum_{c=1}^C |F_c|^2 \|_2}, \quad \mathcal{L}_{\text{AT}} = \sum_{j} \left\| \frac{\mathcal{A}(F_S^j)}{\|\mathcal{A}(F_S^j)\|_2} - \frac{\mathcal{A}(F_T^j)}{\|\mathcal{A}(F_T^j)\|_2} \right\|_2$$

---

## 5. Protocol thực nghiệm & Recipe nền

### 5.0 Recipe huấn luyện nền hiện đại (Khóa ở Tuần 1)
Áp dụng đồng nhất cho **tất cả** các mô hình trong ma trận:
- **Optimizer:** SGD momentum 0,9 (lr 0,01, weight decay 1e-4) hoặc AdamW (lr 1e-3, weight decay 0,01). Chốt bằng pilot M0 trên Dev ở Tuần 1.
- **LR Schedule:** Cosine Annealing, warmup 5 epoch.
- **Augmentation nền:** RandomResizedCrop (scale 0,8–1,0), RandomHorizontalFlip, **TrivialAugmentWide**.
- **Label smoothing:** 0,1 trên CE của Student. Teacher tuyệt đối không dùng label smoothing.
- **EMA:** Exponential Moving Average decay 0,999 cho trọng số Student.
- **Online Teacher View (Beyer et al., CVPR 2022):** Teacher nhận **cùng một ảnh đã augment** trong batch với Student tại mỗi bước (chạy online forward, không dùng cache logit tĩnh).
- **Epoch Budget:** Chốt qua pilot ở Tuần 1: chạy thử M0 và M1 ở 40 epoch và 80 epoch. Nếu chênh lệch Macro-F1 nhãn mịn trên Dev $\le 0,5$ điểm, chọn mức **40 epoch** để tiết kiệm tính toán; nếu $> 0,5$, chọn **80 epoch**.

### 5.1 Protocol-lock & Quy trình Tune công bằng
- **Ngân sách Tune bằng nhau ($N = 5$ cấu hình/phương pháp):**
  - Thực hiện trên Dev nhãn mịn bằng 1 seed.
  - **Lịch tune rút gọn (Successive Halving / Median Pruning):** Chạy 50% số epoch của budget chính. Tại mốc giữa, cấu hình có Macro-F1 nhãn mịn thấp hơn trung vị (median) của các lần thử cùng phương pháp sẽ bị dừng sớm.
  - Cấu hình tốt nhất được chạy lại với $\ge 3$ seed ở đầy đủ epoch budget.
- **Mức nhãn dùng để ra quyết định:** Mọi quyết định lựa chọn mô hình, chọn siêu tham số tune, chọn checkpoint và pilot epoch **bắt buộc dùng Macro-F1 nhãn mịn (hoặc Cross-Entropy loss trên Dev)**. Chỉ số 3 nhóm chỉ tính khi báo cáo kết quả cuối cùng.
- **Cầu chì phân kỳ (Divergence Fuse):** Nếu sau 20 epoch mà Macro-F1 nhãn mịn Dev dưới 30% hoặc loss NaN, tự động hủy run và kiểm tra lại LR/loss scale.
- **Không Early Stopping ở Main runs:** Chạy đủ epoch budget đã chốt, lấy checkpoint có kết quả Dev nhãn mịn tốt nhất và báo cáo thêm trung bình 10 epoch cuối.

---

## 6. Đánh giá ngoại lai & Mở rộng (Tùy chọn)

### 6.1 Đánh giá ngoại lai (Out-of-Distribution - Bắt buộc)
Đánh giá zero-shot trên **TACO** (crop vật phẩm) và **RealWaste** cho các mô hình $\{M0, M1, M1b, \text{KD tốt nhất}\}$. Báo cáo mức độ sụt giảm hiệu năng tương đối để đo lường độ bền vững ngoài miền dữ liệu phân phối.

### 6.2 Canteen-C & Nhóm M7 (Mở rộng nếu dư thời gian)
- Dựng 14 loại nhiễu tổng hợp đặc thù canteen (9 seen, 5 unseen). Đối chiếu với TrivialAugment để loại các phép trùng lặp.
- Chỉ thực hiện nếu vượt qua cổng kiểm tra Go/No-go ở Tuần 5.

---

## 7. Đánh giá & Thống kê

### 7.1 Chỉ số đo lường
- **Chính:** Macro-F1 trên 3 nhóm canteen; Macro-F1 và Top-1 Accuracy trên nhãn mịn; Confusion Matrix.
- **Lỗi nghiêm trọng (Critical Error):** Tỷ lệ nhầm lẫn giữa *Hữu cơ $\leftrightarrow$ Tái chế khô* (làm nhiễm bẩn rác tái chế).
- **Độ tin cậy:** ECE (Expected Calibration Error).
- **Chẩn đoán Teacher:** Đo entropy và xác suất trung bình $p_t$ của Teacher trên nhãn đúng để giải thích hành vi của Hinton KD so với DKD.

### 7.2 Phương pháp thống kê: Cluster Bootstrap
- Do các ảnh chụp trong cùng một ca/vật thể có tương quan nội tại, **không dùng bootstrap từng ảnh độc lập**.
- **Cluster Bootstrap (theo Cụm vật phẩm / Ngày):** Lấy mẫu lại có hoàn lại ở cấp độ `cluster_id` với $B = 2.000$ lần lặp để tính khoảng tin cậy 95% (95% CI) cho Macro-F1 và Recall từng lớp.
- Kiểm định McNemar theo cặp trên cùng tập Test cố định (kèm hiệu chỉnh Holm cho đa giả thuyết).

### 7.3 Phương án Tinh chỉnh Thích nghi Ít mẫu bằng Ảnh Cốt lõi Canteen (RQ2 - Few-Shot Adaptation)
- **Định nghĩa Ảnh Cốt lõi (Core Canteen Images):** Là tập hợp các ảnh rác đại diện thực tế tại canteen trường (hộp xốp dính dầu mỡ, ly nhựa dính đồ uống, thức ăn thừa trên khay ăn, túi nilon mềm, chai lọ/lon canteen) được chụp ở góc độ camera thực tế (top-down khay ăn / miệng thùng rác).
- **Các mức thí nghiệm $k$-shot:** $k \in \{0, 5, 10, 20, 50\}$ ảnh cốt lõi / nhóm quyết định ($3 \times k$ ảnh tổng).
  - $k = 0$: Đánh giá Zero-shot (đo mức độ sụt giảm do Domain Gap thuần túy).
  - $k \in \{5, 10, 20\}$: Đánh giá thích nghi cực hạn (Extreme Few-shot adaptation) — phản ánh năng lực triển khai thực tế khi canteen chỉ chụp mẫu tối thiểu.
  - $k = 50$: Đánh giá ngưỡng bão hòa thích nghi.
  - Mỗi mức $k$ được lấy mẫu ngẫu nhiên theo cụm (`cluster_id`) và lặp lại 3 seed để tính giá trị trung bình kèm độ lệch chuẩn.
- **Quy trình Tinh chỉnh (Fine-tuning Protocol):**
  1. *Khởi tạo:* Nạp checkpoint tốt nhất của mô hình đã huấn luyện trên tập công khai.
  2. *Đóng băng có chọn lọc (Layer Freezing):* Thử nghiệm 2 chiến lược:
     - **Head Adaptation (Linear Probing):** Đóng băng toàn bộ backbone, chỉ cập nhật classification head với LR $10^{-3}$, 10 epoch.
     - **Full Fine-tuning nhẹ:** Mở khóa toàn bộ mô hình, huấn luyện với LR rất nhỏ ($10^{-4}$ với Cosine decay), weight decay $10^{-4}$, 15 epoch, tránh hiện tượng quên thảm họa (catastrophic forgetting).
- **Mục tiêu so sánh khoa học:**
  - So sánh tốc độ phục hồi độ chính xác giữa: $\{M0, M1, M1b, \text{KD tốt nhất}\}$ trên cả Tier A (ResNet18) và Tier B (MobileNet).
  - Kiểm chứng giả thuyết: *Mô hình học qua Knowledge Distillation sở hữu không gian đặc trưng khái quát hơn, giúp thích nghi nhanh hơn với độ dốc F1 cao hơn so với mô hình Student tự học (M0) khi chỉ có lượng ảnh cốt lõi rất nhỏ.*
- **Tập đánh giá độc lập (Frozen Test Set):** Đánh giá tuyệt đối trên tập Test Canteen cố định ($\ge 600$ ảnh, hoàn toàn cách ly với pool ảnh cốt lõi).


---

## 8. Triển khai & Đo kiểm trên Thiết bị biên

1. **Quy trình xuất:** PyTorch $\rightarrow$ ONNX $\rightarrow$ TFLite / ONNX Runtime.
2. **Lượng tử hóa Post-Training (PTQ INT8):**
   - Tập calibration gồm 200 ảnh phân tầng từ Dev công khai + Validation canteen (tuyệt đối không dùng Test).
   - Đo lường độ sụt giảm Macro-F1 thật sau INT8 trên Test canteen.
3. **Quy trình đo độ trễ chuẩn hóa:**
   - Thiết bị: Điện thoại Android (Snapdragon/MediaTek) hoặc Raspberry Pi / Qualcomm AI Hub.
   - Chạy tối thiểu **50 lần warm-up** trước khi bấm giờ. Đo lặp lại 200 lần suy luận: ghi nhận Trung vị (Median), P95 và Độ lệch chuẩn (Std).
   - Ghi nhận nhiệt độ ban đầu, thiết lập độ sáng cố định, tắt chế độ tiết kiệm pin, ghi nhận tình trạng thermal throttling.
   - Báo cáo bộ ba thông số Pareto: **Macro-F1 vs Latency vs Model Size**.

---

## 9. Timeline 14 tuần, Ngân sách GPU & Mốc Go/No-go

### 9.1 Ngân sách tính toán (RTX 4050 Laptop)
- *Tốc độ đo thực tế:* Pipeline KD (ResNet50 Teacher + ResNet18 Student) thuần GPU mất $\approx 70\text{s} / 1.000$ ảnh.
- Khi có DataLoader (bật AMP FP16, `channels_last`, `num_workers=4`): Ước tính $\approx 110\text{--}130\text{s}$ / epoch cho tập train ~12.500 ảnh.
- Với budget **40 epoch**: Mỗi lần chạy chính tốn $\approx 1,3\text{--}1,5$ giờ; lịch tune rút gọn (20 epoch) tốn $\approx 0,7$ giờ.
- **Tổng ngân sách:**
  - Pilot & Teacher (Tuần 1–2): ~12 giờ.
  - Tune Tier A ($N=5$, 5 phương pháp, cắt tỉa): ~15 giờ.
  - Main Tier A (6 cấu hình × 3 seed = 18 runs): ~25 giờ.
  - Thí nghiệm độ mịn nhãn (M0, M1 3 lớp + Teacher 3 lớp = 7 runs): ~10 giờ.
  - Tier B (3 cấu hình × 3 seed = 9 runs): ~8 giờ.
  - Few-shot canteen (45 runs ngắn): ~6 giờ.
  - **Tổng cộng phần lõi: ~76–85 giờ GPU** ($\approx 10\text{--}12$ ngày chạy 7h/ngày, hoàn toàn khả thi trên laptop).

### 9.2 Timeline thực hiện theo tuần

```
Tuần 1-2:   Setup, Pilot Recipe, Khóa Teacher & Student Tier B; Chụp ảnh cốt lõi Canteen đợt 1 (100 ảnh, Kappa >= 0.70)
Tuần 3-4:   Tune M1, M1b, M2, M3, M4; Chạy Main runs Tier A; Thu thập mở rộng Canteen pool & Test set
Tuần 5:     [GO/NO-GO] Hoàn tất Main runs Tier A; ĐÓNG BĂNG Test Canteen (>= 600 ảnh); Khóa pool Ảnh Cốt lõi
Tuần 6-7:   Huấn luyện Tier B (MobileNet); Thí nghiệm độ mịn nhãn; Viết chương Methods
Tuần 8-9:   [THỰC HIỆN FINETUNING RQ2] Chạy thực nghiệm Thích nghi ít mẫu bằng Ảnh Cốt lõi (k-shot in {0, 5, 10, 20, 50})
Tuần 10-11: Xuất INT8, đo kiểm phần cứng thật (Edge Deploy); Phân tích lỗi (Confusion Matrix, Error Cases)
Tuần 12-14: Thống kê Cluster Bootstrap, viết Results & Discussion, bảo vệ luận văn
```


### 9.3 Điểm Go/No-go quyết định (Cuối Tuần 5)
Tại ngày cuối cùng của Tuần 5, kiểm tra 2 điều kiện:
1. Tập Test canteen đã hoàn tất thu thập, gán nhãn 2 người đạt $\kappa \ge 0,7$ và đã được **đóng băng**.
2. Toàn bộ các lần chạy Tune và ít nhất 1 seed của Main runs M0–M4/M1b đã hoàn thành bình thường.

**Xử lý:**
- **Nếu ĐẠT:** Tiếp tục kế hoạch chuẩn. Nếu còn dư nhiều thời gian, xem xét chạy pilot nhóm M7 (Consistency) vào Tuần 6.
- **Nếu TRỄ:** Kích hoạt phương án cắt giảm: **Loại bỏ vĩnh viễn toàn bộ nhóm M7 và Canteen-C**. Toàn bộ thời gian còn lại dành trọn cho Tier B, Few-shot và Tối ưu đo đạc thiết bị biên.

---

## 10. Bảng quản trị rủi ro

| Rủi ro | Mức độ | Biện pháp xử lý chủ động |
|---|---|---|
| Canteen từ chối cho chụp ảnh | Rất cao | Xin phép ngay tuần 1; nếu khó khăn, mua đồ ăn canteen về phòng thực nghiệm dựng góc chụp mô phỏng với khay và bát đĩa chuẩn của trường. |
| Teacher không qua cổng RER $\ge 20\%$ | Trung bình | Đổi sang ConvNeXt-Tiny trong tuần 2; nếu vẫn không đạt, giữ ResNet50 và ghi rõ trong bài: "Khoảng cách Teacher-Student hẹp là đặc tính của dữ liệu, làm giảm biên độ cải thiện của KD". |
| DKD không hơn Hinton KD | Thấp | Là phát hiện thực nghiệm hợp lệ (chứng minh với số lượng lớp nhỏ, dark knowledge ít, việc tách TCKD/NCKD không tạo khác biệt lớn). |
| Nghẽn cổ chai DataLoader trên laptop | Cao | Bật PyTorch AMP (`torch.cuda.amp`), chuyển định dạng `memory_format=torch.channels_last`, resize ảnh sẵn về $256\times256$ lưu trên SSD. |
| Test canteen quá nhỏ gây CI rộng | Trung bình | Dùng Cluster Bootstrap theo cụm vật thể để ước lượng trung thực; báo cáo effect size thay vì chỉ nhìn $p$-value. |

---

## 11. Checklist nghiệm thu trước khi nộp bài

- [ ] Văn bản `docs/label_guide.md` và chỉ số Cohen's kappa đạt $\ge 0,70$ được ghi rõ trong báo cáo.
- [ ] Bảng ánh xạ nhãn `label_map.csv` đính kèm có đánh số phiên bản.
- [ ] Bảng Protocol-lock chi tiết: Optimizer, LR schedule, Augmentation, Seed set, Ngân sách epoch budget.
- [ ] Kết quả kiểm tra Cổng Teacher (RER $\ge 20\%$) và Cổng chọn Student Tier B (Export + INT8 drop $\le 2\%$) có số liệu minh chứng.
- [ ] Mọi chỉ số chọn lọc mô hình và tune siêu tham số đều dựa trên **Macro-F1 nhãn mịn Dev**.
- [ ] Thống kê sử dụng **Cluster Bootstrap theo cụm vật phẩm** (95% CI) và kiểm định McNemar có hiệu chỉnh Holm.
- [ ] Kết quả đo trên thiết bị thật có đầy đủ: Tên SoC, phiên bản runtime, thông số đo lặp (Median, P95), nhiệt độ và mức độ tụt sau INT8.
- [ ] Nếu kích hoạt điểm dừng Tuần 5: Báo cáo trung thực lý do không thực hiện nhóm M7; tuyệt đối không báo cáo kết quả dở dang.

---

## Phụ lục A. Lịch sử thay đổi (Changelog)

- **v8 $\rightarrow$ v9.0:** Chuyển trọng tâm duy nhất sang rác Canteen trường đại học; 3 nhóm quyết định; tách Tier A (ResNet18) và Tier B (MobileNet); đưa ra cơ chế nhãn mịn.
- **v9.0 $\rightarrow$ v9.1:** Hạ ngân sách tune $N=10 \rightarrow 5$; M5 chuyển sang tùy chọn; M7 chuyển thành giả thuyết kiểm chứng; cố định danh sách seen/unseen.
- **v9.1 $\rightarrow$ v9.2:** Bổ sung quy trình đo latency chuẩn (warm-up 50 lần, thermal throttling); quy định tập calibration INT8 phân tầng; viết tường lửa nội tuyến.
- **v9.2 $\rightarrow$ v9.3:** Bổ sung Recipe nền hiện đại (§5.0: TrivialAugment, Cosine, EMA); bổ sung DKD (M1b); Teacher không dùng label smoothing; thiết lập cổng chọn Student Tier B.
- **v9.3 $\rightarrow$ v9.4:** Phân tích khả thi tính toán trên RTX 4050; đưa ra bảng giờ GPU ước tính; pilot epoch 40/80 thay vì chốt cứng; quy tắc cắt tỉa tune median pruning.
- **v9.4 $\rightarrow$ v9.5 (Bản hiện tại):**
  1. *Giải quyết mâu thuẫn RQ:* Cắt M7/Canteen-C khỏi RQ chính, chuyển thành mở rộng; tập trung vào RQ1 (KD + Nhãn) và RQ2 (Domain gap + Few-shot).
  2. *Nâng cấp Tier B:* Chuyển MobileNet lên thành phần bắt buộc.
  3. *Tái cấu trúc timeline:* Đẩy Main runs lên Tuần 4–5; dời điểm Go/No-go về cuối Tuần 5 với tiêu chí định lượng rõ ràng.
  4. *Chuẩn hóa chỉ số chọn:* Toàn bộ quy trình tune/pilot/cổng teacher dùng Macro-F1 nhãn mịn; cổng Teacher dùng Giảm lỗi tương đối (RER $\ge 20\%$).
  5. *Đổi mới phương pháp thống kê:* Áp dụng Cluster Bootstrap theo cụm vật phẩm/ngày chụp.
  6. *Hoàn thiện chi tiết kỹ thuật:* Bổ sung Teacher 3 lớp cho ablation; chuẩn hóa Attention map bằng spatial mean; chuyển changelog xuống phụ lục.
