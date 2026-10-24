# Kế hoạch tuần 2 — Baseline CNN và đo khả năng truyền tri thức

## 1. Mục tiêu và phạm vi

Tuần 2 chuyển từ việc khóa protocol sang thiết lập các mốc so sánh trước khi chạy ablation chính. Mục tiêu cuối tuần là có:

1. Baseline ResNet18 tuần 1 được tái lập bằng đúng protocol đã khóa.
2. Baseline CNN thuần MobileNetV3 và EfficientNet-B0; EfficientFormer-L1 chỉ chạy nếu môi trường và compute còn đủ.
3. Phân tích khả năng truyền tri thức của teacher ResNet50 sang student ResNet18 bằng phân phối soft target, không chỉ nhìn vào khoảng cách accuracy.
4. Pipeline Vanilla KD trên cùng backbone ResNet18, sẵn sàng để chạy chính thức ở tuần 4.
5. Báo cáo quyết định rõ ràng cho tuần 3: baseline KD kinh điển nào sẽ chạy và có cần thay teacher hay không.

Tuần này chưa chạy ECA, Attention Transfer, Consistency, PCGrad, 5-fold chính thức hoặc TrashNet-C. Không thay đổi split, class mapping, batch size, epoch budget, optimizer hay LR schedule đã khóa ở tuần 1.

## 2. Trạng thái đầu vào đã biết

- TrashNet: 2.527 ảnh, 6 lớp gốc; lớp `trash` ít nhất với 137 ảnh.
- Dev/Corruption-holdout: 379 ảnh; CV pool: 2.148 ảnh và 5 fold cố định.
- Teacher ResNet50: accuracy 94,45%, macro-F1 94,48%.
- Student ResNet18: accuracy 92,34%, macro-F1 91,37%.
- Accuracy gap chỉ 2,11 điểm phần trăm, nên tiêu chí tiếp tục không thể là “teacher phải hơn student 8%”. Cần đo KL trên toàn bộ holdout và kiểm tra độ khác biệt của soft target.
- Protocol khóa: batch size 32, SGD momentum 0,9, weight decay `1e-4`, weighted cross-entropy, 80 epoch, cosine LR từ `0,01` đến `1e-6`, warmup 5 epoch.

## 3. Việc phải xử lý trước khi chạy benchmark

Implementation hiện tại có hai điểm chặn cần giải quyết trong ngày đầu:

- `src/dsr/train.py` đang đọc `max_epochs_candidate` và `lr_schedule_candidate`, nhưng `configs/week1_protocol.json` đã dùng `max_epochs` và `lr_schedule`.
- `train.py` chưa thực thi warmup 5 epoch dù warmup đã là một phần của protocol khóa.

Sửa tối thiểu, không đổi public API: để entrypoint đọc đúng các khóa đã khóa, thêm scheduler warmup + cosine, và bổ sung test/config validation để lỗi tương tự không quay lại. Đồng thời kiểm tra checkpoint tuần 1 có thể load lại bằng đúng số lớp và thứ tự lớp.

## 4. Lịch làm việc 7 ngày

### Ngày 1 — Khóa khả năng tái lập của pipeline

- Chạy toàn bộ test bằng `pytest` và chạy dry-run cho từng model cần dùng.
- Sửa đường đọc protocol trong `train.py`; xác nhận batch size, epoch, LR, warmup và device được in ra/log lại.
- Kiểm tra checkpoint `teacher_resnet50_week1/best.pt` và `student_resnet18_week1/best.pt`; nếu chưa có checkpoint thật, ghi nhận rõ là blocker compute, không tạo số liệu giả.
- Tạo thư mục đầu ra thống nhất: `reports/week2/`, `checkpoints/week2/`, `reports/week2/logs/`.

**Đầu ra:** pipeline chạy được một epoch thật hoặc có log blocker; test hồi quy cho protocol.

### Ngày 2 — Tái lập student và đo baseline bổ sung

- Chạy lại ResNet18 với đúng cấu hình tuần 1 để kiểm tra sai khác do seed, preprocessing hoặc checkpoint.
- Chạy MobileNetV3 và EfficientNet-B0 trên cùng split holdout, cùng image size, class weights và budget 80 epoch.
- Dùng cùng nhóm chỉ số: accuracy, macro-F1, balanced accuracy, F1/recall từng lớp, loss và epoch đạt macro-F1 tốt nhất.
- Lưu history CSV và checkpoint tốt nhất theo macro-F1; không dùng early stopping.

**Đầu ra:** `week2_cnn_baselines.csv`, history/checkpoint cho ResNet18, MobileNetV3 và EfficientNet-B0.

### Ngày 3 — EfficientFormer-L1 và kiểm tra tính công bằng

- Chỉ chạy EfficientFormer-L1 nếu `timm` hoạt động ổn định và thời gian một epoch phù hợp với ngân sách tuần.
- Ghi riêng optimizer/schedule nếu kiến trúc này cần AdamW + cosine; đây là baseline ngoài, không được trộn vào protocol ablation ResNet18.
- Tính số tham số và FLOPs nếu công cụ sẵn có; nếu chưa có, ghi `pending` thay vì ước lượng không có nguồn.
- So sánh chất lượng, kích thước mô hình và thời gian train; không kết luận EfficientFormer tốt hơn chỉ từ một seed.

**Đầu ra:** baseline thứ tư hoặc biên bản loại khỏi tuần 2 với lý do compute/môi trường cụ thể.

### Ngày 4 — Đo soft-target và KL teacher–student

- Load teacher và student checkpoint tốt nhất, đặt cả hai ở `eval()` và dùng `no_grad()`.
- Trên toàn bộ 379 ảnh holdout, ghi logits/probability của teacher và student theo đúng thứ tự 6 lớp.
- Tính `KL(p_T || p_S)` ở `T ∈ {2, 3, 6}`; báo cáo mean, median, standard deviation, p90/p95, min/max và tỷ lệ mẫu có KL khác 0.
- Tính thêm entropy teacher, teacher–student top-1 agreement và tỷ lệ teacher đúng nhưng student sai.
- Kiểm tra hướng KL trong PyTorch: input là `student_log_prob`, target là `teacher_prob.detach()`.

**Đầu ra:** `reports/week2/teacher_student_kl.csv` và một bảng tóm tắt dùng được trong Methods/Results.

### Ngày 5 — Cài đặt Vanilla KD tối thiểu

- Thêm loss chuẩn:

  `L = L_CE + α · T² · KL(p_T || p_S)`

- Dùng `alpha=0.7` và chạy lần đầu với `T=3`; nếu bằng chứng tuần 2 hoặc tài liệu gốc không ủng hộ `T=3`, giữ danh sách ứng viên `{2, 3, 6}` cho spot-check thay vì tự chốt bằng một run.
- Teacher freeze hoàn toàn; teacher chỉ forward trên ảnh sạch. Student dùng cùng backbone ResNet18 và cùng augmentation sạch của baseline.
- Đảm bảo loss KD dùng ảnh sạch ở cả hai model, không thêm corruption hoặc ECA.
- Log riêng `ce_loss`, `kd_loss`, `total_loss`, accuracy, macro-F1 và per-class metrics.

**Đầu ra:** code/pipeline cho Vanilla KD và một pilot 1 seed trên holdout.

### Ngày 6 — Kiểm tra Vanilla KD và quyết định teacher

- Chạy lại Vanilla KD với cùng seed nếu ngày 5 có lỗi hoặc kết quả bất thường.
- So sánh Student-only với Vanilla KD trên accuracy, macro-F1, balanced accuracy, lớp `trash`, và KL đầu ra.
- Kiểm tra sanity: loss KD không được âm; khi student copy đúng teacher thì KL phải tiến gần 0; đổi thứ tự input/target phải bị test bắt lỗi.
- Nếu KL rất thấp ở mọi nhiệt độ và Vanilla KD không có tín hiệu cải thiện, chạy feasibility check 1 seed với teacher thay thế đã nêu trong kế hoạch tổng thể (DenseNet121 hoặc ConvNeXt-Tiny), không mở thêm nhánh model nếu compute không đủ.

**Đầu ra:** bảng so sánh Student-only/Vanilla KD và biên bản go/no-go teacher.

### Ngày 7 — Tổng hợp và chốt đầu vào tuần 3

- Chọn baseline CNN mạnh nhất theo macro-F1 và balanced accuracy, nhưng giữ ResNet18 làm backbone bắt buộc cho các baseline KD.
- Viết báo cáo tuần 2 gồm protocol, seed, checkpoint, kết quả, sai khác so với tuần 1, giới hạn và quyết định tuần 3.
- Chốt một cấu hình Vanilla KD tham chiếu để tuần 4 tái lập dòng 3; mọi thay đổi `alpha`/`T` sau tuần này phải ghi thành thí nghiệm, không sửa ngầm.
- Kiểm tra artifact: mỗi kết quả phải truy ngược được tới config, seed, checkpoint và history.

**Đầu ra bắt buộc:** `reports/week2/week2_summary.md`, bảng baseline CNN, bảng KL, checkpoint Vanilla KD, và quyết định teacher/temperature.

## 5. Lệnh kiểm tra và chạy dự kiến

Từ thư mục gốc repository:

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q
python src/dsr/train.py --config configs/week1_protocol.json --model resnet18 --run-name week2_resnet18_repro
```

Các lệnh MobileNetV3/EfficientNet-B0/EfficientFormer-L1 chỉ được thêm sau khi entrypoint hỗ trợ chúng và đã có test cho model registry. Không dùng tên run trùng với checkpoint tuần 1.

## 6. Tiêu chí go/no-go cuối tuần

### Go

- Protocol khóa được đọc đúng và warmup được thực thi hoặc có bằng chứng log tương ứng.
- Baseline ResNet18 tái lập được trong sai số nhỏ đã ghi rõ theo seed.
- Ít nhất MobileNetV3 và EfficientNet-B0 có metrics/checkpoint hợp lệ; EfficientFormer-L1 là tùy chọn có điều kiện.
- KL được tính đúng chiều trên toàn bộ holdout ở ba nhiệt độ và Vanilla KD chạy được end-to-end.
- Teacher được giữ lại nếu soft target có độ phân tán hữu ích, dù accuracy gap nhỏ.

### No-go hoặc cần điều chỉnh

- Không chạy benchmark nếu config vẫn trỏ vào khóa cũ hoặc warmup/protocol không xác minh được.
- Nếu Vanilla KD không tạo khác biệt nhưng KL vẫn có tín hiệu, tiếp tục sang tuần 3 và ghi nhận đó là baseline âm tính cần giải thích.
- Nếu teacher có entropy thấp, KL thấp và prediction gần như trùng student, chạy một teacher thay thế tối đa 1 seed; không mở rộng ablation trước khi có quyết định.
- Nếu thiếu GPU/thời gian, bỏ EfficientFormer-L1 trước; không bỏ đo KL, Vanilla KD hoặc hai baseline MobileNetV3/EfficientNet-B0.

## 7. Phân bổ thời gian và nguyên tắc báo cáo

- 20%: sửa và kiểm thử pipeline.
- 35%: baseline CNN.
- 20%: KL/soft-target analysis.
- 20%: Vanilla KD pilot và sanity checks.
- 5%: tổng hợp báo cáo và backup artifact.

Không dùng p-value để tuyên bố hiệu quả trong tuần 2 vì đây mới là baseline/pilot và chưa có 5-fold chính thức. Chỉ dùng các cụm “baseline”, “tín hiệu sơ bộ”, “không quan sát thấy cải thiện rõ” và giữ nguyên toàn bộ log để phục vụ tuần 3–4.