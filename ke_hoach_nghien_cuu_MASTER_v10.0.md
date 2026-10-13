# Kế hoạch nghiên cứu MASTER v10.0 — Phát hiện và phân loại đa vật thể rác canteen 3 nhóm trên thiết bị biên: Pipeline Detector + Knowledge Distillation Classifier

*Bản v10.0 là tài liệu hoàn chỉnh, tự đứng được (thay thế v8, v9.0–v9.5). Thay đổi lớn so với v9.5: bài toán chuyển từ phân loại ảnh 1 vật sang phát hiện và phân loại nhiều vật thể trên một khay/bàn. Lịch sử sửa đổi nằm ở Phụ lục A.*

---

## 0. Quyết định hướng đi (đọc trước)

**Bài toán mới:** Đầu vào là 1 ảnh khay/bàn canteen chứa nhiều vật rác. Đầu ra là danh sách bounding box, mỗi box có nhóm (Hữu cơ / Tái chế khô / Còn lại) và độ tin cậy.

**Hướng được chọn: Pipeline 2 giai đoạn (Detector class-agnostic → KD Classifier trên crop).** Lý do:

1. Giữ nguyên lõi nghiên cứu đã xây dựng (RQ1: so sánh KD; nhãn mịn → 3 nhóm; ma trận M0–M4; 15.000 ảnh Kaggle), vì classifier vẫn học từ ảnh đơn lẻ.
2. KD cho detection (feature-based KD trên YOLO) cần viết lại phần lớn mã huấn luyện và kế hoạch. Với 14 tuần và GPU laptop thì rủi ro quá cao.
3. Đóng góp mới của hướng đa vật thể (dữ liệu có box, lỗi lan truyền detector → classifier, latency theo số vật) vẫn đo được rõ ràng.

**Đối chứng bắt buộc (rẻ, không KD):** một mô hình YOLO end-to-end 3 lớp (D2) huấn luyện trên cùng dữ liệu, để trả lời "pipeline 2 giai đoạn có đáng không so với 1 mô hình".

**Hướng 2 (KD cho detection, ví dụ FGD hoặc Localization Distillation) chỉ là mở rộng tùy chọn**, chỉ xét nếu qua Go/No-go tuần 5 với dư thời gian rõ ràng. Không nằm trong cam kết cốt lõi.

---

## 1. Câu hỏi nghiên cứu & Mục tiêu

**Câu hỏi trung tâm:** Làm thế nào kết hợp một detector nhẹ với classifier được chưng cất tri thức (KD) để phát hiện và phân loại nhiều vật rác trên khay canteen thành 3 nhóm, chính xác và đủ nhanh trên thiết bị biên?

| Hạng mục | Phát biểu | Bằng chứng chính |
|---|---|---|
| **RQ1 (Cơ chế KD & Nhãn)** | Trong cùng ngân sách tính toán và tune, tín hiệu distillation phong phú hơn (DKD, FitNet, AT) có vượt logit KD cổ điển không; huấn luyện nhãn mịn rồi gộp sang 3 nhóm có lợi gì so với huấn luyện trực tiếp 3 nhóm? (áp dụng cho classifier trên crop) | M0, M1, M1b, M2, M3, M4 (§4.4); thí nghiệm độ mịn nhãn (§3.1) |
| **RQ2 (Domain Gap & Thích nghi)** | Khoảng cách miền từ dữ liệu công khai (ảnh đơn lẻ, ảnh TACO, ảnh ghép) sang ảnh khay canteen thật lớn đến mức nào; KD có thu hẹp tốt hơn không; cần bao nhiêu dữ liệu canteen thật (classifier: số crop; detector: số khay) để thích nghi? | Zero-shot trên test canteen; đường cong few-shot (§7.4); đánh giá ngoại lai (§6) |
| **RQ3 (Pipeline & Lỗi lan truyền)** | Lỗi của detector làm giảm hiệu năng classifier bao nhiêu (box thật so với box detector sinh ra)? Pipeline 2 giai đoạn so với YOLO end-to-end 3 lớp đánh đổi thế nào giữa độ chính xác và độ trễ? | Thí nghiệm oracle-box và phân rã lỗi (§7.2); so sánh D1+classifier với D2 (§4.4) |
| **Mục tiêu triển khai (Edge Goal)** | Cả pipeline INT8 chạy ổn định trên thiết bị biên thật, báo cáo đánh đổi Macro-F1/mAP, latency theo số vật trong khung, kích thước, RAM | Kiểm chuẩn phần cứng thật (§8) |

*Mở rộng tùy chọn (chỉ làm nếu qua Go/No-go tuần 5 và còn dư thời gian):* (i) KD cho detector (FGD hoặc Localization Distillation, student YOLO nhỏ, teacher YOLO lớn hơn); (ii) CutMix + KD (M5); (iii) Canteen-C và Consistency (M7).

**Khung tuyên bố khoa học (chống overclaim):** Không dùng từ "giải quyết triệt để domain shift", chỉ dùng "đánh giá mức độ thu hẹp khoảng cách miền". Không gọi ResNet18 là "siêu nhẹ". Kết luận âm (KD không thắng baseline mạnh; pipeline không hơn end-to-end; kỹ thuật KD phức tạp không hơn logit KD) vẫn là kết luận khoa học hợp lệ và được báo cáo trung thực. Kết quả trên ảnh ghép tổng hợp không được dùng để tuyên bố hiệu năng thực tế; mọi tuyên bố đều dựa trên Test canteen thật.

---

## 2. Định vị tính mới & Đóng góp

Đề tài thuộc loại **nghiên cứu thực nghiệm có kiểm soát + ứng dụng bối cảnh hẹp + dữ liệu chuyên biệt**. Không phát minh thuật toán mới. Bốn đóng góp cụ thể:

- **C1 — Dữ liệu & quy trình gán nhãn canteen đa vật thể:** Ảnh khay canteen thật có bounding box và nhóm, metadata (ca chụp, điều kiện sáng, mức chồng lấp, cờ mơ hồ), quy ước gán nhãn bằng văn bản, kiểm định đồng thuận hai người ở mức box.
- **C2 — So sánh KD công bằng & toàn diện cho classifier:** Logit cổ điển (Hinton), logit hiện đại (DKD), Feature (FitNet), Attention (AT), dưới cùng recipe và ngân sách tune; kiểm chứng trên ResNet18 (Tier A) và MobileNet (Tier B).
- **C3 — Đường cong thích nghi miền:** Số crop/khay canteen thật tối thiểu để đưa mô hình huấn luyện từ dữ liệu công khai đạt ngưỡng dùng được.
- **C4 — Phân tích lỗi lan truyền và đánh đổi pipeline:** Oracle-box so với detected-box, phân rã lỗi (bỏ sót / định vị / nhầm nhóm / dương tính giả), và độ trễ end-to-end theo số vật trên khung.

---

## 3. Dữ liệu & Thiết kế nhãn

### 3.1 Thiết kế nhãn: huấn luyện mịn, quyết định thô (cho classifier)

- Huấn luyện trên $G \approx 10\text{--}15$ lớp nhãn mịn (gộp từ dữ liệu công khai) **cộng 1 lớp `background`** (xem §4.2), sau đó dùng bảng ánh xạ cố định `label_map.csv` (phiên bản 2) gom thành 3 nhóm khi suy luận; lớp `background` bị loại khỏi đầu ra:
  1. *Hữu cơ (Organic)*: rác thực phẩm, đồ ăn thừa, vỏ hoa quả...
  2. *Tái chế khô (Recyclable)*: chai nhựa sạch, lon nhôm, hộp giấy sạch...
  3. *Còn lại / Khó tái chế (Other/Landfill)*: hộp xốp bẩn, túi nilon dính dầu, giấy ăn bẩn...
- **Thí nghiệm độ mịn nhãn (RQ1):** Huấn luyện M0 (CE) và M1 (Hinton KD) ở 2 nhánh: (a) nhãn mịn $G$ lớp rồi ánh xạ sang 3 nhóm; (b) trực tiếp 3 nhóm (dùng 1 Teacher 3 nhóm riêng). So sánh để kiểm chứng giả thuyết "3 nhóm quá ít dark knowledge khiến logit KD kém hiệu quả".
- Detector D1 là **class-agnostic (1 lớp `waste`)** nên không phụ thuộc bảng ánh xạ nhóm. Detector D2 (đối chứng) dùng 3 lớp.

### 3.2 Nguồn dữ liệu công khai

| Nguồn | Dùng cho | Ghi chú |
|---|---|---|
| *Recyclable and Household Waste Classification* (Kaggle, ~15.000 ảnh, 30 lớp gốc) hoặc *TrashBusters/combined* | **Classifier** (Train/Dev/Holdout); nguồn crop cho ảnh ghép | Lọc trùng bằng pHash/MD5. Tách 15% Dev/Holdout cố định ngay tuần 1 (stratified). |
| **TACO** (ảnh có bbox/mask) | **Detector** (pretrain class-agnostic: gộp mọi lớp thành `waste`) | Chia cố định theo ảnh: 80% train / 20% **TACO-heldout** (tập ngoại lai cho detector). Không dùng TACO-heldout để train hay tune. |
| **RealWaste** | Ngoại lai cho classifier (zero-shot) | Không dùng để train. |
| Ảnh ghép tổng hợp (§3.5) | Detector (pretrain/bổ sung) | Chỉ là dữ liệu huấn luyện, không bao giờ vào Val/Test. |

*Lưu ý (theo hiểu biết của mình, cần xác minh khi tải):* TACO là ảnh rác ngoài trời với rất ít thức ăn thừa, và nhãn TACO không phản ánh trạng thái sạch/bẩn. Vì vậy TACO chỉ dùng để detector học khái niệm "vật rác", không dùng để suy ra nhóm Canteen. Với D2 (3 lớp), bảng `taco_to_3.csv` chỉ là nguồn phụ và phải kiểm tra tay một mẫu 100 box trước khi dùng.

### 3.3 Thu thập ảnh khay canteen đa vật thể (Test, Val, Pool few-shot, Train detector)

- **Thiết lập chụp:** camera top-down cố định chiều cao và góc, độ phân giải ≥ 1080p; khay/bàn nền đồng nhất (ít nhất 2 loại nền); 3 điều kiện sáng; **2–8 vật/khay**, đa dạng mức chồng lấp (không chồng / chồng nhẹ / chồng nhiều); khoảng 20% ảnh "khay chỉ có thức ăn hoặc khay trống" làm mẫu âm.
- **Chụp thêm 30–50 ảnh nền trống** (khay/bàn không vật) để làm nền cho ảnh ghép và làm mẫu `background` cho classifier.
- **Cụm (Cluster):** `cluster_id` = ngày chụp + ca ăn (+ khay). Một khay thuộc đúng một cụm.
- **Quy mô tối thiểu (tính theo số khay và số instance, theo tiến độ dự kiến):**

| Tập | Số khay | Instance tối thiểu | Vai trò |
|---|---|---|---|
| Test canteen (đóng băng) | ≥ 150 | ≥ 200 instance/nhóm (≈ 600 box) | Đánh giá cuối, không dùng để tune |
| Validation canteen | ≥ 50 | ≥ 50 instance/nhóm | Chọn ngưỡng confidence, chọn checkpoint detector |
| Pool few-shot / Train detector thật | ≥ 250 | — | Thích nghi ít mẫu (§7.4) và huấn luyện detector |

  Lên kế hoạch thành phần khay để đủ instance mỗi nhóm và theo dõi số instance từng nhóm hàng tuần, vì thường có nhóm bị thiếu.
- **Nguyên tắc phân chia:** chia theo **cụm (ngày/ca)**, tuyệt đối không chia ngẫu nhiên từng ảnh/box. Các khay cùng ngày không được nằm ở hai tập khác nhau. Test nên gồm các ngày chưa xuất hiện ở Train/Pool.
- **Ước lượng công gán nhãn (cần đo lại ở pilot tuần 2):** khoảng 1–2 phút/khay với CVAT/Roboflow, tức ≈ 8–15 giờ cho ~450 khay, chưa tính gán nhãn lần 2 cho kappa.

### 3.4 Quy ước gán nhãn box & Kiểm định đồng thuận

- Viết `docs/label_guide.md` (phiên bản có phần box) **trước khi chụp diện rộng**. Quy định tối thiểu:
  - Box ôm sát phần nhìn thấy của vật; vật bị che thì box chỉ phần thấy.
  - Cụm đồng chất không tách được (khăn giấy vo viên, một bát thức ăn thừa) = **1 box**; khay/bát/đĩa của canteen **không** được gán box.
  - Vật nhỏ hơn ngưỡng kích thước (ví dụ < 16 px ở ảnh gốc) bị bỏ qua và ghi rõ trong tài liệu.
  - Ca ranh giới (ly nhựa dính trà sữa, hộp xốp dính dầu) xếp vào "Còn lại"; ca khó gắn cờ `ambiguous=True`.
- **Kappa mức box (Tuần 2):** hai người gán độc lập ~40 khay (≥ 100 box). Hai chỉ số: (1) độ đồng thuận vị trí: ghép box theo IoU ≥ 0,5, tính F1 giữa hai người, yêu cầu ≥ 0,80; (2) Cohen's kappa trên nhãn nhóm của các box đã ghép, yêu cầu ≥ 0,70. Không đạt thì sửa quy ước và gán lại trước khi thu thập diện rộng.
- Tập mơ hồ báo cáo riêng, không dùng để tune.
- Định dạng nhãn: YOLO txt (`class_id x_center y_center width height`, chuẩn hóa) kèm file metadata CSV (`cluster_id`, điều kiện sáng, mức chồng lấp, `ambiguous`).

### 3.5 Ảnh ghép tổng hợp (Copy-Paste) cho detector

Mục tiêu: tạo nhiều ảnh nhiều vật có box tự động từ ảnh đơn lẻ Kaggle, để pretrain detector trước khi có nhiều ảnh thật.

1. Tách nền một **tập con** ảnh Kaggle (khoảng 3.000 vật, cân bằng nhóm) bằng công cụ tách nền (ví dụ `rembg`); loại ảnh tách lỗi bằng kiểm tra diện tích mask.
2. Dán 3–6 vật khác loại lên ảnh nền trống canteen (§3.3) với xoay ngẫu nhiên, thay đổi tỉ lệ, làm mờ viền (feather) và thêm bóng đổ/hiệu chỉnh màu để giảm khoảng cách miền.
3. Box sinh tự động từ mask; có chồng lấp có kiểm soát (IoU giữa các vật ≤ 0,3).
4. Quy mô ban đầu 3.000–5.000 ảnh; tỉ lệ ảnh ghép trong tổng train detector được ghi nhận và là một biến của thí nghiệm (§4.4, E3).
5. **Rủi ro đã biết:** ảnh ghép lệch miền so với ảnh thật (ánh sáng, bóng, kết cấu). Vì vậy ảnh ghép chỉ để pretrain; đánh giá luôn trên ảnh canteen thật.

---

## 4. Mô hình & Ma trận thực nghiệm

### 4.1 Kiến trúc pipeline

```
Ảnh khay → [Detector D1, class-agnostic] → các box (NMS, ngưỡng conf)
        → crop (padding 10–20%) → [Classifier KD] → nhóm (hoặc background → loại)
        → danh sách {box, nhóm, điểm tin cậy}
```

- Giới hạn tối đa **N_max = 8 crop/khung** (xếp theo confidence) để độ trễ có chặn trên; ghi rõ trong báo cáo.
- Điểm tin cậy cuối = tích hoặc trung bình có quy định cố định của confidence detector và xác suất classifier (chốt một công thức trên Val ở tuần 5, không đổi sau đó).

### 4.2 Classifier (Teacher & Student)

- **Teacher (mặc định):** ResNet50 (pretrain ImageNet). Huấn luyện bằng recipe §5.0, **không dùng label smoothing** (Müller et al., 2019).
  - *Cổng kiểm tra Teacher (Tuần 2):* trên Dev nhãn mịn, Teacher phải đạt **Giảm lỗi tương đối** so với Student M0 tối thiểu 20%:
  $$\text{RER} = \frac{\text{Err}_{\text{M0}} - \text{Err}_{\text{Teacher}}}{\text{Err}_{\text{M0}}} \ge 20\% \quad (\text{với } \text{Err} = 1 - \text{Macro-F1}_{\text{fine}})$$
  Không đạt thì thử 1 lần DenseNet121 hoặc ConvNeXt-Tiny trong tuần 2. Sau tuần 2, khóa cứng Teacher.
- **Tier A (bắt buộc):** Student ResNet18, sàn so sánh chuẩn mực cho cơ chế KD.
- **Tier B (bắt buộc):** chọn **đúng một** student biên cuối Tuần 2 qua 3 cổng: (1) xuất ONNX và TFLite FP32/INT8 không lỗi toán tử; (2) tụt Macro-F1 Dev của M0 sau PTQ INT8 ≤ 2,0 điểm; (3) latency mô phỏng thấp nhất trong {MobileNetV3-Small, MobileNetV3-Large, MobileNetV4-Conv-Small} (`timm`).
- **Lớp `background` (mới):** thêm 1 lớp mịn gồm (i) crop ngẫu nhiên từ ảnh nền trống canteen, (ii) crop từ các box dương tính giả của D1 trên tập train (hard negatives, sinh sau khi D1 có bản đầu), với số mẫu xấp xỉ trung bình một lớp mịn. Áp dụng **thống nhất cho mọi mô hình** trong ma trận (Teacher, M0–M4).
- **Augmentation theo hình học box (mới):** thêm nhiễu khung crop khi train (dịch ±10% kích thước box, scale ±15%, padding 5–20%), để classifier quen với box không hoàn hảo của detector.

### 4.3 Detector

- **D1 (class-agnostic, 1 lớp `waste`):** họ YOLO nhỏ (YOLOv8n hoặc YOLO11n, `ultralytics`), pretrain COCO. Chọn **đúng một** cuối Tuần 2 qua 3 cổng: (1) xuất TFLite INT8 không lỗi; (2) tụt AP50 sau INT8 trên Val canteen ≤ 3,0 điểm; (3) latency mô phỏng thấp hơn ở imgsz 416. Sau đó khóa kiến trúc.
- **D2 (đối chứng end-to-end, 3 lớp):** cùng kiến trúc, cùng dữ liệu, nhãn 3 nhóm, **không KD**.
- *Giấy phép:* thư viện `ultralytics` dùng AGPL-3.0, cần ghi chú khi công bố mã.

### 4.4 Ma trận thực nghiệm

**Classifier (Tier A: ResNet18; Tier B: MobileNet đã chọn):**

| ID | Cấu hình | Vai trò trong RQ | Mức ưu tiên |
|---|---|---|---|
| **M0** | Student thuần (CE), recipe §5.0 | Baseline sàn (phải tối ưu mạnh) | **Bắt buộc** |
| **M1** | Hinton Logit KD | Mốc KD logit kinh điển | **Bắt buộc** |
| **M1b** | Decoupled KD, DKD (Zhao et al., CVPR 2022) | Mốc KD logit hiện đại | **Bắt buộc** |
| **M2** | FitNet (Romero et al., 2015) | Họ Feature-based KD | **Bắt buộc** |
| **M3** | Attention Transfer (Zagoruyko et al., 2017) | Họ Attention-based KD | **Bắt buộc** |
| **M4** | AT + Hinton Logit KD | Richer KD kết hợp | **Bắt buộc** |
| **Tier B** | MobileNet với {M0, M1, KD tốt nhất} | Chuyển giao sang Edge Student | **Bắt buộc** |
| *M5* | CutMix + KD | Augmentation-distillation | *Mở rộng (nếu dư)* |
| *M7* | Consistency distillation (Canteen-C) | Độ bền nâng cao | *Mở rộng (nếu dư)* |

**Detector & Pipeline (mới):**

| ID | Cấu hình | Vai trò | Mức ưu tiên |
|---|---|---|---|
| **E3a–d** | D1 huấn luyện với (a) TACO; (b) TACO + ảnh ghép; (c) (b) + khay thật; (d) chỉ khay thật. 1 seed cho (a)(b)(d), 3 seed cho (c) | Đo giá trị từng nguồn dữ liệu, nền cho RQ2 phía detector | **Bắt buộc** |
| **E3-curve** | D1 tinh chỉnh với số khay thật $n \in \{0, 25, 50, 100, 250\}$, 3 seed | Đường cong thích nghi detector | **Bắt buộc** |
| **E4** | Pipeline = D1 + {M0, M1, KD tốt nhất}; chạy với box thật (oracle) và box detector | Lỗi lan truyền (RQ3) | **Bắt buộc** |
| **D2** | YOLO 3 lớp end-to-end, không KD | Đối chứng pipeline (RQ3) | **Bắt buộc** (cắt cuối cùng nếu thiếu thời gian) |
| *E3-size* | D1 ở imgsz {320, 416, 640} (AP vs latency) | Đánh đổi kích thước đầu vào | *Mở rộng* |
| *KD-det* | KD cho detector (FGD/LD) | Hướng 2 | *Mở rộng (nếu dư)* |

### 4.5 Hàm mất mát (classifier)

Tổng quát: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \alpha \mathcal{L}_{\text{Logit}} + \beta \mathcal{L}_{\text{Feature}} + \gamma \mathcal{L}_{\text{Attention}}$

1. **CE:** label smoothing 0,1 cho student; kiểm tra class weights ở Tuần 1 (lớp `background` có thể cần trọng số riêng).
2. **Logit KD (M1):** $T^2 \cdot \text{KL}(\text{Softmax}(z_T/T) \,\|\, \text{Softmax}(z_S/T))$, trong đó input của `kl_div` là log-prob của Student, target là prob của Teacher đã `.detach()`.
3. **DKD (M1b):** $\alpha_{\text{TCKD}} \cdot \text{TCKD} + \beta_{\text{NCKD}} \cdot \text{NCKD}$. Test đơn vị bắt buộc: khi $\alpha=1$ và $\beta=(1-p_t^T)$, loss phải khớp chính xác với Hinton KD.
4. **Attention Transfer (M3):** theo Zagoruyko et al., với $\mathcal{A}(F)=\sum_{c=1}^{C}|F_c|^2$ (vector hóa theo không gian):
$$\mathcal{L}_{\text{AT}} = \sum_{j} \left\| \frac{\mathcal{A}(F_S^j)}{\|\mathcal{A}(F_S^j)\|_2} - \frac{\mathcal{A}(F_T^j)}{\|\mathcal{A}(F_T^j)\|_2} \right\|_2$$
*(Lưu ý: v9.5 mô tả bằng lời là "chuẩn hóa mean theo spatial" nhưng công thức lại dùng chuẩn L2; v10.0 thống nhất theo công thức L2 của bài gốc. Nếu muốn dùng chuẩn hóa mean thì phải sửa cả công thức và ghi vào Protocol-lock trước khi chạy.)*

---

## 5. Protocol thực nghiệm & Recipe nền

### 5.0 Recipe classifier (khóa ở Tuần 1, áp dụng đồng nhất cho mọi mô hình classifier)

- **Optimizer:** SGD momentum 0,9 (lr 0,01, wd 1e-4) hoặc AdamW (lr 1e-3, wd 0,01); chốt bằng pilot M0 trên Dev ở Tuần 1.
- **LR schedule:** Cosine Annealing, warmup 5 epoch.
- **Augmentation nền:** RandomResizedCrop (scale 0,8–1,0), RandomHorizontalFlip, TrivialAugmentWide, cộng nhiễu khung crop (§4.2).
- **Label smoothing:** 0,1 trên CE của Student; Teacher không dùng.
- **EMA:** decay 0,999 cho trọng số Student.
- **Online Teacher View (Beyer et al., CVPR 2022):** Teacher nhận **cùng ảnh đã augment** với Student mỗi bước (forward online, không cache logit tĩnh).
- **Epoch budget:** pilot M0 và M1 ở 40 và 80 epoch; nếu chênh Macro-F1 nhãn mịn Dev ≤ 0,5 điểm chọn **40 epoch**, ngược lại **80 epoch**.

### 5.0b Recipe detector (khóa ở Tuần 2)

- Khởi tạo từ trọng số COCO; imgsz 416 cho thí nghiệm chính (640 chỉ ở E3-size nếu làm); augmentation mặc định của `ultralytics` có Mosaic nhưng **tắt ở những epoch cuối**; không dùng thêm copy-paste nội bộ khi đã dùng ảnh ghép riêng (để biến "ảnh ghép" rõ ràng).
- Epoch và LR chốt bằng một pilot trên Val canteen (so 2 mức epoch, cùng quy tắc ngưỡng chênh như §5.0); sau tuần 2 khóa.
- Chọn checkpoint theo **AP50 trên Val canteen**; ngưỡng confidence/NMS chọn trên Val (tối đa hóa F1 end-to-end), **không** chọn trên Test.

### 5.1 Protocol-lock & Tune công bằng

- **Ngân sách tune bằng nhau (N = 5 cấu hình/phương pháp)** cho classifier, trên Dev nhãn mịn, 1 seed; cắt tỉa median ở 50% epoch; cấu hình tốt nhất chạy lại ≥ 3 seed ở đủ epoch.
- **Mức nhãn quyết định:** mọi lựa chọn mô hình/siêu tham số/checkpoint classifier dùng **Macro-F1 nhãn mịn Dev** (hoặc CE loss Dev). Chỉ số 3 nhóm chỉ dùng khi báo cáo cuối.
- **Cầu chì phân kỳ:** sau 20 epoch mà Macro-F1 nhãn mịn Dev < 30% hoặc loss NaN thì hủy run.
- **Không early stopping ở main runs:** chạy đủ budget, lấy checkpoint Dev tốt nhất và báo cáo thêm trung bình 10 epoch cuối.
- **Quy tắc chống rò rỉ giữa hai giai đoạn:** classifier và detector **không** dùng dữ liệu Test canteen ở bất kỳ bước nào (huấn luyện, tune, hard-negative mining, calibration INT8).

---

## 6. Đánh giá ngoại lai

- **Classifier (bắt buộc):** zero-shot trên **RealWaste** cho {M0, M1, M1b, KD tốt nhất}; báo cáo mức sụt tương đối.
- **Detector (bắt buộc):** zero-shot trên **TACO-heldout** (class-agnostic AP50, recall@0,5) cho D1 ở các cấu hình E3a–c; báo cáo riêng là ngoài miền khay canteen.
- **Mở rộng:** Canteen-C (14 nhiễu tổng hợp, 9 seen / 5 unseen) và nhóm M7, chỉ làm nếu qua Go/No-go tuần 5.

---

## 7. Đánh giá & Thống kê

### 7.1 Classifier (trên crop)

- **Chính:** Macro-F1 trên 3 nhóm canteen (crop từ box thật của Test); Macro-F1 và Top-1 trên nhãn mịn (Kaggle holdout); Confusion Matrix.
- **Lỗi nghiêm trọng:** tỉ lệ nhầm *Hữu cơ ↔ Tái chế khô* (làm nhiễm bẩn rác tái chế).
- **Độ tin cậy:** ECE.
- **Chẩn đoán Teacher:** entropy và $p_t$ trung bình trên nhãn đúng, để giải thích hành vi Hinton KD so với DKD.

### 7.2 Detector & End-to-end

- **Detector:** AP50, AP50–95, precision và **recall@IoU0,5 class-agnostic** (đây là trần của pipeline), trên Val và Test canteen thật.
- **End-to-end:** ghép box dự đoán với box thật (IoU ≥ 0,5, ghép tham lam theo confidence, mỗi box thật chỉ ghép một lần). Một dự đoán đúng khi ghép được **và** nhóm đúng. Báo cáo precision/recall/F1 theo nhóm, Macro-F1 end-to-end, mAP 3 lớp (cho D2 và pipeline), và Critical Error ở mức box.
- **Phân rã lỗi (mỗi box thật thuộc đúng một loại):** (1) bỏ sót (không có dự đoán IoU ≥ 0,1); (2) định vị kém (IoU 0,1–0,5); (3) ghép được nhưng nhầm nhóm; (4) đúng. Dương tính giả chia: nền, trùng lặp, nhóm sai. Báo cáo tỉ lệ từng loại theo mô hình.
- **Oracle-box (E4):** chạy cùng classifier trên crop từ **box thật** và trên **box detector**. Hiệu hai kết quả = phần mất do detector. Đây là phân tích trung tâm của RQ3.
- **Đếm vật:** sai số số vật trên khay (MAE) làm chỉ số phụ.

### 7.3 Phương pháp thống kê: Cluster Bootstrap

- Box trong cùng khay và khay cùng ca/ngày có tương quan nội tại, nên **không bootstrap từng ảnh/box độc lập**.
- **Cluster bootstrap theo `cluster_id` (ngày + ca)**, lấy mẫu lại có hoàn lại ở cấp cụm, $B = 2.000$, tính 95% CI cho Macro-F1, F1 từng nhóm, AP50 và hiệu giữa hai mô hình (**bootstrap ghép cặp** trên cùng tập cụm).
- McNemar ghép cặp cho so sánh classifier trên cùng tập crop cố định, hiệu chỉnh Holm cho đa giả thuyết; báo cáo effect size thay vì chỉ $p$-value. Với so sánh pipeline, dùng bootstrap ghép cặp theo cụm.
- Số cụm Test có thể ít; ghi rõ số cụm và báo cáo CI rộng một cách trung thực.

### 7.4 Thích nghi ít mẫu (RQ2)

- **Classifier:** $k \in \{0, 5, 10, 20, 50\}$ crop/nhóm lấy từ **Pool khay thật** (lấy mẫu theo cụm, 3 seed). $k=0$ đo domain gap thuần túy; $k \in \{5,10,20\}$ là thích nghi cực hạn; $k=50$ là ngưỡng bão hòa.
  - *Head Adaptation (linear probing):* đóng băng backbone, chỉ cập nhật head, LR $10^{-3}$, 10 epoch.
  - *Full fine-tuning nhẹ:* mở khóa toàn bộ, LR $10^{-4}$ cosine, wd $10^{-4}$, 15 epoch.
  - So sánh tốc độ phục hồi F1 giữa {M0, M1, M1b, KD tốt nhất} trên Tier A và Tier B. Giả thuyết: mô hình KD có không gian đặc trưng khái quát hơn nên thích nghi nhanh hơn M0. Kết quả có thể phủ định giả thuyết này và vẫn được báo cáo.
- **Detector:** đường cong E3-curve với $n \in \{0, 25, 50, 100, 250\}$ khay thật, tinh chỉnh từ bản pretrain TACO + ảnh ghép.
- Đánh giá trên **Test canteen đóng băng**, cách ly hoàn toàn với Pool.

---

## 8. Triển khai & Đo kiểm trên Thiết bị biên

1. **Xuất:** PyTorch → ONNX → TFLite / ONNX Runtime cho **cả detector và classifier**.
2. **PTQ INT8:** tập calibration 200 ảnh phân tầng từ Dev công khai + Val canteen (detector dùng ảnh khay Val, classifier dùng crop Val), **tuyệt đối không dùng Test**. Đo mức sụt thật trên Test canteen cho từng model và cho cả pipeline.
3. **Quy trình đo độ trễ chuẩn hóa:**
   - Thiết bị: Android (Snapdragon/MediaTek) hoặc Raspberry Pi / Qualcomm AI Hub.
   - ≥ 50 lần warm-up, đo 200 lần suy luận: Median, P95, Std; ghi nhiệt độ ban đầu, độ sáng cố định, tắt tiết kiệm pin, ghi nhận thermal throttling.
   - **Độ trễ end-to-end:** $T_{\text{pipeline}} = T_{\text{detector}} + N \cdot T_{\text{classifier}}$ (hoặc suy luận crop theo lô nếu runtime hỗ trợ). Đo với $N \in \{1, 3, 6\}$ vật/khung và báo cáo cả phần preprocessing/NMS/crop, không chỉ thời gian mô hình.
   - D2 (end-to-end) đo cùng điều kiện để so sánh công bằng.
4. **Báo cáo Pareto:** Macro-F1 end-to-end (hoặc mAP) vs latency (theo N) vs tổng kích thước mô hình vs RAM đỉnh.

---

## 9. Timeline 14 tuần, Ngân sách GPU & Mốc Go/No-go

### 9.1 Ngân sách tính toán (RTX 4050 Laptop)

**Phần classifier (giữ từ v9.5; tốc độ đo thực tế):** pipeline KD ResNet50 → ResNet18 thuần GPU ≈ 70 s/1.000 ảnh; có DataLoader (AMP FP16, `channels_last`, `num_workers=4`) ≈ 110–130 s/epoch cho ~12.500 ảnh. Budget 40 epoch ≈ 1,3–1,5 giờ/run; tune rút gọn (20 epoch) ≈ 0,7 giờ.

| Hạng mục | Giờ GPU ước tính | Ghi chú |
|---|---|---|
| Pilot & Teacher (Tuần 1–2) | ~12 | Có thêm lớp `background`, giữ ước lượng cũ |
| Tune Tier A (N=5, 5 phương pháp) | ~15 | |
| Main Tier A (6 cấu hình × 3 seed) | ~25 | |
| Độ mịn nhãn (7 runs) | ~10 | |
| Tier B (3 cấu hình × 3 seed) | ~8 | |
| Few-shot classifier (45 runs ngắn) | ~6 | |
| **Subtotal classifier** | **~76–85** | |
| Detector: E3a–d (~10 runs) | ~6.5 | **Đã đo thực tế Tuần 2 (RTX 4050)**: 45,2 s/epoch ở imgsz 416; ~0,63 giờ (38 phút)/run 50 epoch |
| Detector: E3-curve (5 mức × 3 seed) | ~4–5 | ~0,3 giờ/run (ít ảnh thật hơn) |
| D2 (3 seed) | ~2 | |
| E4 + phân rã lỗi (suy luận) | ~2 | |
| Sinh ảnh ghép + tách nền | ~3–5 | Chủ yếu CPU/GPU nhẹ, chạy song song |
| **Subtotal detector/pipeline** | **~18–20** | |
| **Tổng cộng** | **~94–105** | ≈ 13–15 ngày chạy 7 giờ/ngày |

*Tốc độ detector đã được đo kiểm thực nghiệm (pilot Tuần 2: 45,23 s/epoch, VRAM 1,2 GB, export ONNX 1,8 ms/ảnh).* Ngân sách GPU hoàn toàn khả thi trên laptop RTX 4050 mà không cần cắt giảm hạng mục.


### 9.2 Timeline theo tuần

```
Tuần 1:   Setup; xin phép chụp canteen; viết label_guide (có phần box); chia Dev/Holdout Kaggle + chia TACO 80/20;
          pilot recipe classifier (epoch 40/80); chốt Protocol-lock classifier
Tuần 2:   Cổng Teacher (RER ≥ 20%); cổng Student Tier B; cổng chọn D1 (YOLOv8n vs YOLO11n);
          pilot đo tốc độ detector; chụp đợt 1 (~60 khay + 30–50 ảnh nền); kappa box (~40 khay, 2 người)
Tuần 3-4: Tune + Main runs Tier A; xây pipeline ảnh ghép; huấn luyện D1 bản (a)(b);
          chụp & gán nhãn đợt 2
Tuần 5:   [GO/NO-GO] ĐÓNG BĂNG Test canteen; khóa Pool; hoàn tất tune và ≥ 1 seed Main Tier A;
          sinh hard-negative `background` từ D1
Tuần 6-7: Tier B; độ mịn nhãn; E3a–d (data ablation detector); D2; viết chương Methods
Tuần 8-9: Few-shot classifier (RQ2) + E3-curve (detector); các seed còn lại của main runs
Tuần 10:  E4 end-to-end: oracle-box vs detected-box; phân rã lỗi
Tuần 11-12: Xuất INT8 cả pipeline; đo kiểm phần cứng thật (latency theo N); Confusion Matrix, ca lỗi
Tuần 13-14: Cluster bootstrap, viết Results & Discussion, hoàn thiện, bảo vệ
```

### 9.3 Điểm Go/No-go (cuối Tuần 5)

Kiểm tra 3 điều kiện:

1. Test canteen hoàn tất, **đóng băng**: ≥ 150 khay, ≥ 200 instance/nhóm, kappa box đạt (§3.4).
2. Toàn bộ tune và ≥ 1 seed Main runs M0–M4/M1b của Tier A đã hoàn thành.
3. **Cổng detector:** D1 (TACO + ảnh ghép + ≥ 50 khay thật) đạt **recall@IoU0,5 class-agnostic ≥ 0,80 trên Val canteen**. *(Ngưỡng 0,80 là đề xuất ban đầu của mình; điều chỉnh sau pilot tuần 2 nhưng phải khóa trước tuần 5.)*

**Xử lý:**
- **ĐẠT:** tiếp tục kế hoạch chuẩn; nếu dư rõ rệt mới xét mở rộng (KD-det, E3-size, M5/M7).
- **TRỄ hoặc detector chưa đạt:** loại vĩnh viễn M7/Canteen-C, M5, KD-det và E3-size; ưu tiên cho E3a–d, E4, Tier B, few-shot và đo biên. Nếu recall vẫn thấp, tăng khay thật cho detector trước khi thêm bất kỳ thí nghiệm mới nào.

---

## 10. Bảng quản trị rủi ro

| Rủi ro | Mức độ | Biện pháp xử lý chủ động |
|---|---|---|
| Canteen từ chối cho chụp | Rất cao | Xin phép ngay tuần 1; nếu khó, dựng góc chụp mô phỏng với khay/bát đĩa chuẩn của trường và đồ ăn mua về |
| Chi phí gán nhãn box lớn hơn dự kiến | Cao | Đo thời gian/khay ở pilot tuần 2; dùng detector pretrain gợi ý box rồi sửa tay (pre-label); thu hẹp số khay Pool nếu cần nhưng giữ nguyên Test |
| Ảnh ghép lệch miền so với ảnh thật | Cao | Chỉ dùng để pretrain; đo giá trị qua E3a–d; luôn đánh giá trên khay thật |
| Thiếu dữ liệu nhóm Hữu cơ cho detector (TACO ít thức ăn) | Cao | Bổ sung bằng ảnh thật và ghép từ ảnh thức ăn Kaggle; theo dõi recall riêng nhóm Hữu cơ ở E4 |
| Lỗi detector dồn sang classifier | Trung bình | Đây chính là đối tượng của RQ3; thêm lớp `background`, nhiễu khung crop, hard-negative |
| Teacher không qua cổng RER ≥ 20% | Trung bình | ConvNeXt-Tiny ở tuần 2; nếu vẫn không đạt, giữ ResNet50 và ghi rõ khoảng cách Teacher–Student hẹp là đặc tính dữ liệu |
| YOLO INT8 sụt AP nhiều | Trung bình | Cổng chọn D1 có tiêu chí INT8; thử calibration lớn hơn; báo cáo FP16 làm phương án dự phòng |
| Latency pipeline tăng theo số vật | Trung bình | N_max = 8; suy luận crop theo lô; báo cáo theo N |
| Test canteen ít cụm gây CI rộng | Trung bình | Cluster bootstrap; báo cáo effect size; ghi rõ số cụm |
| Nghẽn DataLoader trên laptop | Cao | AMP, `channels_last`, resize sẵn 256×256 lưu SSD |
| DKD không hơn Hinton KD | Thấp | Là phát hiện hợp lệ (ít lớp, dark knowledge ít) |

---

## 11. Checklist nghiệm thu trước khi nộp bài

- [ ] `docs/label_guide.md` (có quy ước box) và kết quả đồng thuận mức box (F1 vị trí ≥ 0,80; kappa nhóm ≥ 0,70).
- [ ] `label_map.csv` (phiên bản 2, có lớp `background`) và `taco_to_3.csv` (nếu dùng) có đánh số phiên bản và nhật ký kiểm tra tay.
- [ ] Bảng Protocol-lock: optimizer, LR schedule, augmentation, seed, epoch budget — cho cả classifier và detector.
- [ ] Số liệu Cổng Teacher (RER ≥ 20%), cổng Student Tier B và cổng detector D1 (export + INT8).
- [ ] Mọi chọn lọc mô hình/siêu tham số classifier dựa trên **Macro-F1 nhãn mịn Dev**; ngưỡng detector chọn trên **Val canteen**, không phải Test.
- [ ] Xác nhận bằng nhật ký: không có dữ liệu Test canteen trong train/tune/hard-negative/calibration; chia theo cụm ngày/ca.
- [ ] Kết quả E3a–d, E3-curve, E4 (oracle vs detected) và phân rã lỗi có đủ.
- [ ] Cluster bootstrap (95% CI) và McNemar có hiệu chỉnh Holm; ghi rõ số cụm.
- [ ] Đo thiết bị thật: tên SoC, phiên bản runtime, Median/P95, nhiệt độ, latency theo N, mức sụt sau INT8 cho từng model và cả pipeline; có D2 so sánh.
- [ ] Nếu kích hoạt điểm dừng Tuần 5: báo cáo trung thực nội dung bị cắt; không báo cáo kết quả dở dang.

---

## Phụ lục A. Lịch sử thay đổi (Changelog)

- **v8 → v9.0:** Chuyển trọng tâm sang rác canteen; 3 nhóm quyết định; tách Tier A (ResNet18) và Tier B (MobileNet); cơ chế nhãn mịn.
- **v9.0 → v9.1:** Hạ ngân sách tune N=10 → 5; M5 tùy chọn; M7 thành giả thuyết kiểm chứng.
- **v9.1 → v9.2:** Quy trình đo latency chuẩn; calibration INT8 phân tầng.
- **v9.2 → v9.3:** Recipe nền hiện đại (TrivialAugment, Cosine, EMA); thêm DKD (M1b); Teacher không label smoothing; cổng chọn Student Tier B.
- **v9.3 → v9.4:** Khả thi tính toán RTX 4050; bảng giờ GPU; pilot epoch 40/80; median pruning.
- **v9.4 → v9.5:** Cắt M7/Canteen-C khỏi RQ chính; Tier B bắt buộc; dời Go/No-go về cuối tuần 5; chỉ số chọn là Macro-F1 nhãn mịn; RER cho cổng Teacher; Cluster Bootstrap; Teacher 3 lớp cho ablation.
- **v9.5 → v10.0 (bản hiện tại):**
  1. *Đổi bài toán:* phân loại ảnh 1 vật → phát hiện và phân loại đa vật thể; chọn pipeline Detector + KD Classifier; YOLO 3 lớp end-to-end làm đối chứng; KD cho detection chuyển thành mở rộng.
  2. *RQ:* giữ RQ1, RQ2 (mở rộng sang detector); thêm RQ3 (lỗi lan truyền, pipeline vs end-to-end); thêm C4.
  3. *Dữ liệu:* thêm ảnh khay đa vật thể có box; quy ước gán nhãn box và kappa mức box; ảnh ghép Copy-Paste (§3.5); chia dữ liệu theo cụm khay/ngày; quy mô tính theo instance.
  4. *TACO:* tách 80/20, 20% giữ làm ngoại lai cho detector (tránh mâu thuẫn dùng TACO vừa train vừa OOD); dùng class-agnostic.
  5. *Classifier:* thêm lớp `background` và nhiễu khung crop; `label_map.csv` lên phiên bản 2.
  6. *Đánh giá:* thêm AP50/AP50–95, recall class-agnostic, F1 end-to-end, oracle-box, phân rã lỗi; bootstrap theo cụm ngày/ca, ghép cặp cho so sánh pipeline.
  7. *Edge:* latency end-to-end theo số vật, export INT8 cho cả hai model.
  8. *Timeline/GPU:* thêm ~24–28 giờ GPU (ước tính, cần đo lại); thêm cổng detector vào Go/No-go; thứ tự cắt giảm khi thiếu thời gian.
  9. *Sửa nhất quán:* công thức Attention Transfer thống nhất theo chuẩn L2 của bài gốc (v9.5 mô tả "mean" nhưng công thức dùng L2); sửa chiều KL trong mô tả Logit KD cho đúng thứ tự tham số `kl_div`.
