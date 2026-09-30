# KẾ HOẠCH TUẦN 3 (v2) — Baseline KD kinh điển: Vanilla KD, Attention Transfer, FitNet, CutMix+KD

> **Trạng thái:** thay thế bản `docs/week3_plan.md` cũ (bản cũ là việc của Tuần 5 — ECA + dòng 4a/4b/4c; đổi tên thành `docs/week5_plan_draft.md`, không dùng trong tuần này).
> **Căn cứ:** MASTER v8 §5, §6.1, §6.4, §7.1–7.2, §8.1–8.4, §13 (tuần 3); `docs/errata_week1_protocol.md`; `reports/week2/week2_summary.md`.

---

## 1. Mục tiêu và phạm vi

Cài đặt và huấn luyện **4 baseline KD** trên **cùng backbone ResNet18 thuần (không ECA)**, dùng teacher sạch `checkpoints/teacher_resnet50_clean/best.pt`. Đầu ra là **Bảng baseline KD** làm mốc so sánh cho toàn bộ ma trận ablation về sau.

| Baseline | Vai trò trong MASTER | Khóa protocol (§8.1)? |
|---|---|---|
| Vanilla KD | **Dòng 3** — nền của 4a/4b/4c, mốc dưới RQ1 | **Có** |
| Attention Transfer (Zagoruyko) | **Dòng 4b** — dùng lại số liệu ở Tuần 5 | **Có** |
| FitNet | Baseline ngoài (§7.1) | Không bắt buộc (giữ cùng backbone/budget nếu được) |
| CutMix+KD | Baseline ngoài (§7.1), tách bạch với L_Consistency | Không bắt buộc |

**Không làm trong tuần này:** ECA, dòng 4a/4c, L_Consistency, PCGrad, 5-fold chính thức, TrashNet-C, TACO/RealWaste. Không đổi split, batch size, epoch, optimizer, LR schedule đã khóa.

---

## 2. Đầu vào đã xác nhận

| Mục | Giá trị | Nguồn |
|---|---|---|
| Protocol khóa | SGD momentum 0.9, wd 1e-4; lr 0.01 cosine → 1e-6; warmup 5; batch 32; **100 epoch**; weighted CE (inverse-frequency); student ImageNet-pretrained; aug: resize, hflip, rotation ±10° | `configs/week1_protocol.json`, `train_kd.py`, `data.py` |
| Train / Val | Train = Fold 1–4 (≈1716 ảnh); **Val = Fold 0 (432 ảnh, lớp trash chỉ 24 ảnh)** | errata, `data.py` |
| Dev/Corruption-holdout | 379 ảnh, không nằm trong fold nào (trash ≈ 21 ảnh) | `week1_protocol.md` |
| Teacher sạch | Macro-F1 97.62% (epoch 58) | errata |
| Student-only sạch (dòng 1) | Macro-F1 **95.04%** (epoch 39) | errata |
| Vanilla KD pilot (α=0.7, τ=3) | Best Macro-F1 **92.43%** (epoch 61), trash F1 84.44%; trung bình 10 epoch cuối ≈ 91.7% | `week2_vanilla_kd_clean_history.csv` |

**Giả định cần xác nhận ở Ngày 1:** (a) seed của run pilot Vanilla KD (trong `train_kd.py`/`data.py` hiện không thấy đặt seed); (b) `image_size` (kỳ vọng 224 → stage 56/28/14/7 cho cả ResNet18 và ResNet50); (c) giây/epoch thực tế trên RTX 4050.

---

## 3. Nguyên tắc

1. **Dòng khóa (Vanilla KD, AT)** chỉ khác nhau ở thành phần đang ablate; mọi thứ khác giống hệt protocol dòng 1. Tuần này chỉ được phép chọn **τ** (dòng 3) và **γ** (dòng 4b), và chỉ theo quy tắc ở §4.
2. **Baseline ngoài (FitNet, CutMix+KD)** giữ SGD + cùng budget; mọi sai khác phải ghi vào manifest và báo cáo.
3. **Chọn siêu tham số chỉ bằng Dev/Corruption-holdout** (§8.4.4). Val = Fold 0 chỉ dùng để chọn epoch tốt nhất của từng run. Holdout **không bao giờ** dùng để huấn luyện hay chọn epoch.
4. **Mọi run có seed cố định** (`--seed 42`) và **manifest** (seed, α, τ, γ/β, optimizer, teacher ckpt, git commit, giây/epoch).
5. **Quy tắc quyết định được viết trước khi thấy kết quả** (§4) — không đổi sau khi xem số.
6. **Ngôn ngữ báo cáo:** 1 seed × 1 fold → chỉ dùng "xu hướng", "không phân biệt được"; cấm "significant", cấm p-value (§8.3).

---

## 4. Quy tắc quyết định (chốt trước)

### 4.1 Biên nhiễu `m`
Sau khi có holdout-eval của **pilot τ=3 (chưa rõ seed)** và **run τ=3 seed 42**:

`m = max(1.0 điểm Macro-F1, |F1_holdout(pilot) − F1_holdout(rerun τ=3)|)`

Chênh lệch nhỏ hơn `m` được coi là **không phân biệt được**.

### 4.2 Chọn τ cho dòng 3 (α = 0.7 cố định — điểm khởi đầu §8.4.1)
- Lưới τ ∈ {2, 3, 6}, cả ba run cùng seed 42.
- Chọn τ có **holdout Macro-F1** cao nhất. Nếu τ dẫn đầu hơn τ=3 chưa tới `m` → **giữ τ=3**.
- **Dòng 3 chính thức = run seed 42 của τ được chọn** (không phải run pilot). Pilot giữ nguyên làm mốc tham chiếu, gắn nhãn "PILOT, chưa seed".
- **Cổng sớm:** nếu |pilot − rerun τ=3| > 2 điểm Macro-F1 trên **Fold 0** → dừng, debug pipeline (bất tất định / bug) trước khi chạy tiếp.

### 4.3 Chọn γ cho dòng 4b (τ, α lấy từ §4.2)
- Lưới γ ∈ {0.1, 0.5, 1.0} (đúng §8.4.2).
- Chọn γ có holdout Macro-F1 cao nhất; nếu γ = 0.5 nằm trong nhóm cách đỉnh chưa tới `m` → **chọn 0.5**.
- γ này là **giá trị tạm** cho Tuần 5. Nếu Tuần 5 chọn γ khác cho dòng 4c → **chạy lại 4b** ở γ mới (1–2 seed, rẻ), không giữ số cũ.

### 4.4 FitNet và CutMix+KD
Không tune lưới. Dùng giá trị mặc định ở §5; chỉ **hiệu chỉnh độ lớn loss** bằng dry-run (không nhìn val):
- AT: nếu γ=1.0 cho `γ·L_AT < 1%` của `L_CE` ở epoch 1–5 → **lỗi cài đặt reduction** (sửa code, **không** nới lưới γ).
- FitNet: β mặc định = 1; chỉ đổi theo bậc 10 nếu `β·L_hint` < 1% hoặc > 200% của `L_CE + α·L_KD` ở epoch 1.

---

## 5. Đặc tả kỹ thuật

Tất cả dùng dạng cộng (additive) như MASTER §5: `L = L_CE(weighted) + α·L_KD + (thành phần riêng)`.

### 5.1 `src/dsr/losses.py` — KD dùng chung
```python
def kd_loss(student_logits, teacher_logits, tau):
    s = F.log_softmax(student_logits / tau, dim=1)          # input = log-prob student
    t = F.softmax(teacher_logits / tau, dim=1).detach()     # target = prob teacher
    return F.kl_div(s, t, reduction="batchmean") * tau ** 2
```
`train_kd.py` và mọi script mới **import hàm này**, không copy công thức.

### 5.2 Attention Transfer (= dòng 4b), đúng công thức Zagoruyko (§5)
`L = L_CE + α·L_KD + γ·L_AT`, **không** có conv/projector học tham số.
```python
def attention_map(feat, size=None, eps=1e-6):
    a = feat.pow(2).sum(dim=1, keepdim=True)                 # Σ_c |A_c|²  → (B,1,H,W)
    if size is not None and a.shape[-2:] != size:            # chỉ resize khi lệch
        a = F.interpolate(a, size=size, mode="bilinear", align_corners=False)
    return F.normalize(a.flatten(1), p=2, dim=1, eps=eps)    # (B,H·W), ‖·‖₂ = 1

def attention_transfer_loss(student_feats, teacher_feats):
    loss = 0.0
    for fs, ft in zip(student_feats, teacher_feats):
        qs = attention_map(fs)
        qt = attention_map(ft, size=fs.shape[-2:]).detach()
        loss = loss + (qs - qt).pow(2).sum(dim=1).mean()     # ∈ [0,4] mỗi layer
    return loss                                              # tổng 4 layer
```
- **Reduction cố định:** tổng bình phương theo vị trí không gian, **trung bình theo batch**, tổng qua 4 layer. **Không** dùng mean theo không gian (mã tham chiếu gốc dùng β≈1000 để bù; với γ ∈ {0.1..1.0} loss sẽ quá nhỏ và 4b ≈ dòng 3 chỉ vì lý do đó).
- Hook `layer1..layer4` của student (có grad) và teacher (`no_grad`, `eval`); giải phóng feature sau mỗi iteration.
- Teacher không có ECA; attention map lấy trực tiếp từ activation thô.

### 5.3 FitNet (baseline ngoài) — phiên bản một giai đoạn
`L = L_CE + α·L_KD + β·L_hint`, với α, τ **giống dòng 3**.
- Hint layer: **layer3** (student 256 kênh → teacher 1024 kênh, cùng độ phân giải).
- Regressor: `Conv2d(256, 1024, kernel_size=1)`; `L_hint = MSE(regressor(f_S), f_T.detach())` (mean theo phần tử và batch).
- Regressor nằm trong **cùng optimizer SGD** (param group riêng, cùng lr/wd/schedule); **bỏ khi suy luận và không tính vào số tham số student**.
- **Ghi rõ là biến thể một giai đoạn**, khác bản gốc hai giai đoạn (hint pretraining rồi KD). Nếu còn buffer ở Ngày 6–7, có thể chạy thêm bản hai giai đoạn như phụ lục.
- Tiêu chí chuyển optimizer đặt sẵn: nếu loss NaN hoặc **val Macro-F1 < 0.5 sau epoch 20** (Vanilla KD đạt ≈ 0.75–0.85 ở mốc này) → giảm lr nhóm regressor ×0.1 trước, rồi mới xét đổi optimizer; ghi cả hai vào log.

### 5.4 CutMix+KD (baseline ngoài)
- Yun et al. 2019: xác suất áp dụng mỗi batch **p = 0.5**, `λ ~ Beta(1,1)`, hoán vị ngẫu nhiên trong batch, hộp cắt diện tích ≈ `(1−λ)`, **cắt (clip) vào biên ảnh rồi tính lại `λ_adj = 1 − diện_tích_thực/(H·W)`**.
- Augmentation gốc (flip, rotation ±10°) giữ nguyên, áp trước CutMix.
- **Teacher forward trên ảnh đã trộn** `x_mix`:
  `L = λ_adj·CE_w(z_S, y_a) + (1−λ_adj)·CE_w(z_S, y_b) + α·kd_loss(z_S, z_T(x_mix), τ)`; batch không được CutMix thì dùng loss dòng 3.
- Dùng đúng `criterion_ce` có class weight của protocol.

---

## 6. Hàng đợi run (một GPU, chạy tuần tự)

| # | Run | Nội dung | Ghi chú |
|---|---|---|---|
| R1 | `w3_vkd_tau3_s42` | Vanilla KD, τ=3 | Chạy lại có seed; cũng dùng ước lượng nhiễu (§4.1) |
| R2 | `w3_vkd_tau2_s42` | Vanilla KD, τ=2 | |
| R3 | `w3_vkd_tau6_s42` | Vanilla KD, τ=6 | |
| R4 | `w3_at_g0p1_s42` | AT, γ=0.1 | Cần τ đã chốt |
| R5 | `w3_at_g0p5_s42` | AT, γ=0.5 | |
| R6 | `w3_at_g1p0_s42` | AT, γ=1.0 | |
| R7 | `w3_fitnet_s42` | FitNet một giai đoạn | |
| R8 | `w3_cutmixkd_s42` | CutMix+KD | |

**Ngân sách:** đo giây/epoch ở Ngày 1 → `t_run ≈ 100 × giây/epoch`; tổng 8 run phải vừa GPU-giờ của tuần. Nếu không vừa, **cắt theo thứ tự**: (1) bỏ R1, dùng pilot cho điểm τ=3 và ghi rõ "chưa seed"; (2) không chạy FitNet hai giai đoạn; (3) cuối cùng mới bỏ γ=0.1 và ghi rõ trong báo cáo. Không cắt R2/R3/R5/R6 vì đó là các dòng khóa.

Lệnh mẫu (PowerShell, từ thư mục gốc repo):
```powershell
$env:PYTHONPATH = "src"
python src/dsr/train_kd.py --config configs/week1_protocol.json `
  --teacher-ckpt checkpoints/teacher_resnet50_clean/best.pt `
  --run-name w3_vkd_tau3_s42 --alpha 0.7 --temperature 3 --seed 42 --out-dir reports/week3
```

---

## 7. Lịch 7 ngày

### Ngày 1 — Hạ tầng và luật chơi
- Đổi tên `docs/week3_plan.md` cũ → `docs/week5_plan_draft.md`; đặt bản này làm `docs/week3_plan.md`.
- `src/dsr/kd_common.py`: `set_seed` (random/numpy/torch/cuda, `cudnn.deterministic=True`, `benchmark=False`, `generator` cho DataLoader), `evaluate`, `write_kd_history`, `build_scheduler`, `write_manifest`, `make_holdout_loader`.
- `src/dsr/losses.py` với `kd_loss`; `train_kd.py` gọi hàm này và nhận `--seed`.
- **Test:** (a) `kd_loss` khớp công thức inline cũ; (b) KL ≥ 0, bằng 0 khi logit trùng; (c) đảo input/target bị test bắt; (d) hồi quy 20 batch seed cố định trước/sau refactor trong dung sai; (e) **assert giao tập** holdout ∩ train = ∅, holdout ∩ val = ∅.
- `scripts/eval_holdout.py` → `reports/week3/week3_holdout_eval.csv`; chấm: teacher sạch, student sạch, pilot Vanilla KD.
- Xác nhận 3 giả định ở §2; đo giây/epoch; cập nhật ngân sách §6.
- Khởi chạy R1–R3 (nền, tuần tự).
- **Cổng:** test xanh, hồi quy khớp, không có rò rỉ split.

### Ngày 2 — Attention Transfer loss (trong lúc R1–R3 chạy) và chốt dòng 3
- `losses.attention_transfer_loss` + `src/dsr/features.py` (hook helper).
- **Test AT:** hai map giống nhau → 0; bất biến theo hệ số nhân; giá trị mỗi layer ∈ [0,4]; nhánh bilinear khi lệch kích thước; teacher không nhận gradient; hình dạng 4 layer khớp (56/28/14/7 nếu 224).
- Khi R1–R3 xong: holdout-eval → tính `m` (§4.1) → áp quy tắc §4.2 → kiểm tra cổng sớm.
- Viết `docs/vanilla_kd_official.md`: dòng 3 chính thức = run được chọn; ghi rõ đây **điều chỉnh** so với week2_summary (vốn hứa "chạy chính thức ở Tuần 3") và lý do (seed + tune τ); pilot giữ làm mốc "PILOT".
- Ghi `configs/week3_kd_hparams.json` (α, τ **khóa**). Không sửa `week1_protocol.json`.
- **Cổng:** α, τ đã khóa bằng văn bản trước khi chạy bất kỳ R4–R8.

### Ngày 3 — `train_at.py` và khởi chạy γ-sweep
- Tạo `train_at.py` (import từ `kd_common`/`losses`/`features`, không copy-paste vòng lặp KD).
- Dry-run 1 epoch: log `L_CE`, `α·L_KD`, `γ·L_AT` (γ=1.0) → kiểm tra quy tắc độ lớn §4.4.
- Khởi chạy R4–R6.
- **Cổng:** `γ·L_AT` không nằm dưới ngưỡng 1% của `L_CE`.

### Ngày 4 — FitNet (trong lúc R4–R6 chạy)
- `losses.hint_loss`, `FitNetRegressor`, `train_fitnet.py`.
- **Test:** shape sau regressor = shape teacher layer3; gradient chạm regressor và backbone student; regressor không có trong `state_dict` dùng để đếm tham số.
- Dry-run 1 epoch để hiệu chỉnh độ lớn β (§4.4).

### Ngày 5 — CutMix+KD và chốt γ
- `train_cutmixkd.py`.
- **Test:** 1000 hộp ngẫu nhiên đều nằm trong ảnh; `λ_adj` bằng diện tích thực; teacher nhận `x_mix`; lưu lưới 8 ảnh trộn để kiểm tra bằng mắt.
- Khi R6 xong: holdout-eval R4–R6 → áp quy tắc §4.3 → khóa γ tạm → ghi vào `configs/week3_kd_hparams.json`.
- Khởi chạy R7, R8.

### Ngày 6 — Tổng hợp
- Khi R7–R8 xong: holdout-eval (chỉ để đối chiếu, không dùng để tune).
- Dựng `reports/week3/week3_kd_baselines.csv` (§8) và bảng tuning `reports/week3/week3_tuning_holdout.csv`.
- Buffer cho run lỗi/chạy lại.

### Ngày 7 — Báo cáo và bàn giao
- Viết `reports/week3/week3_summary.md` (§9).
- **Chẩn đoán tùy chọn (chỉ suy luận, để giải thích chứ không để kết luận):** entropy teacher và top-1 của teacher trên **ảnh train (không augment)** so với Fold 0. Nếu teacher gần như ghi nhớ tập train thì soft target trên train chứa ít "dark knowledge", đây là giả thuyết cho việc KD không vượt Student-only.
- Chuẩn bị bàn giao (§11).

---

## 8. Đầu ra

| Tệp | Nội dung |
|---|---|
| `docs/vanilla_kd_official.md` | Chốt dòng 3, α, τ, quy tắc đã áp dụng |
| `configs/week3_kd_hparams.json` | α, τ, γ, β, tham số CutMix |
| `reports/week3/week3_holdout_eval.csv` | Holdout của teacher, student, pilot và các run |
| `reports/week3/week3_tuning_holdout.csv` | Lưới τ và γ kèm quyết định |
| `reports/week3/week3_kd_baselines.csv` | Bảng chính (cột dưới đây) |
| `reports/week3/*_history.csv`, `checkpoints/<run>/best.pt`, manifest JSON | Truy vết đầy đủ |
| `reports/week3/week3_summary.md` | Báo cáo tuần |

Cột `week3_kd_baselines.csv`: `row_label` (3 / 4b / FitNet / CutMix+KD), `run_name`, `protocol_locked`, `seed`, `optimizer`, `alpha`, `tau`, `gamma_or_beta`, `best_epoch`, `val_accuracy`, `val_macro_f1`, `val_macro_f1_last10_mean`, `val_balanced_accuracy`, `val_f1_trash`, `holdout_macro_f1`, `params_M_student`, `sec_per_epoch`. Kèm hai dòng tham chiếu: Student-only (95.04%) và Teacher (97.62%), và dòng pilot Vanilla KD gắn nhãn "PILOT".

---

## 9. Cách diễn giải

- So từng baseline với **dòng 1 (95.04%)** và **dòng 3**. Chênh lệch nhỏ hơn `m` → "không phân biệt được".
- Báo cả **best-epoch** và **trung bình 10 epoch cuối**, kèm F1 lớp `trash` (chỉ 24 ảnh: đổi 1–2 ảnh làm F1 dao động vài điểm).
- Nếu cả 4 baseline không vượt dòng 1: kết luận hợp lệ, phù hợp động cơ RQ1 (MASTER §4.3), không ép số. Nhưng **1 seed × 1 fold không đủ để nói "KD không nhất quán"** — bằng chứng thật nằm ở 5-fold Tuần 8.
- Không viết "capacity gap" như một kết luận; đó mới là giả thuyết (teacher chỉ hơn student 2.6 điểm).
- Báo cáo nêu rõ: FitNet là biến thể một giai đoạn; FitNet/CutMix+KD dùng mặc định, không tune, trong khi dòng 3/4b có tune τ/γ.

---

## 10. Rủi ro và dự phòng

| Rủi ro | Dự phòng |
|---|---|
| Không vừa GPU-giờ | Thứ tự cắt ở §6 |
| R1 vs pilot lệch > 2 điểm (Fold 0) | Dừng, kiểm tra seed/bất tất định/bug trước khi tiếp tục |
| AT loss quá nhỏ hoặc quá lớn | Sửa reduction theo §5.2; không đổi lưới γ |
| FitNet hội tụ kém với SGD | Tiêu chí ở §5.3 (đặt sẵn), ghi log cả hai lần |
| Kết quả nằm trong biên nhiễu `m` | Báo "không phân biệt được"; không chọn thắng thua |
| Refactor làm hỏng pipeline dòng 3 | Test hồi quy Ngày 1; giữ nguyên `train_kd.py` cũ ở nhánh git riêng đến hết tuần |

---

## 11. Bàn giao cho Tuần 4 và Tuần 5

**Tuần 4 (kiểm tra determinism, MASTER §13):** chạy lại dòng 3 với (α, τ) đã khóa.
- Cùng seed 42 → kiểm tra tất định; go/no-go: **lệch > 2 điểm Macro-F1 trên Fold 0 so với dòng 3 chính thức thì dừng và debug** trước Tuần 5.
- (Đề xuất) thêm 1 seed khác để ước lượng phương sai seed cho nhóm chạy 1–2 seed.

**Tuần 5:** dòng 4b lấy số liệu R4–R6 (γ đã chốt tạm). Nếu grid γ trên 4c chọn giá trị khác → chạy lại 4b ở γ mới. Dòng 3, 4b dùng nguyên α, τ đã khóa ở `configs/week3_kd_hparams.json`.

---

## 12. Checklist hoàn thành

- [ ] `docs/week3_plan.md` cũ đã đổi tên thành `week5_plan_draft.md`
- [ ] `kd_loss` dùng chung; test hồi quy và test hướng KL xanh
- [ ] Không có giao tập giữa holdout với train/val
- [ ] Mọi run có seed và manifest
- [ ] α, τ khóa bằng văn bản **trước** khi chạy R4–R8
- [ ] AT: reduction đúng, test độ lớn xanh, không có projector
- [ ] FitNet: ghi rõ biến thể một giai đoạn; regressor không tính vào params
- [ ] CutMix+KD: teacher nhận ảnh trộn; hộp và `λ_adj` hợp lệ
- [ ] γ chốt theo quy tắc §4.3, ghi là giá trị tạm cho Tuần 5
- [ ] Bảng `week3_kd_baselines.csv` đủ cột, có dòng tham chiếu và dòng PILOT
- [ ] Báo cáo dùng ngôn ngữ "xu hướng", không có p-value

---

## Phụ lục — Thay đổi so với bản kế hoạch trước

1. Không còn nâng pilot Vanilla KD thành chính thức: chạy lại có seed và tune τ ∈ {2,3,6}; pilot giữ làm mốc và ước lượng nhiễu.
2. Thêm quy tắc chọn τ, γ trên Dev/Corruption-holdout, viết trước khi thấy kết quả (§8.4.4).
3. Chốt reduction của L_AT và kiểm tra độ lớn để tránh "4b ≈ dòng 3" do loss quá nhỏ.
4. Bổ sung seed, manifest, cổng nhiễu sớm 2 điểm ngay trong tuần 3.
5. Cụ thể hóa FitNet (một giai đoạn, layer3, regressor) và CutMix+KD (teacher nhìn ảnh trộn, `λ_adj`).
6. Đưa KD loss vào `losses.py` dùng chung thay vì clone `train_kd.py`.
7. Bảng kết quả thêm balanced accuracy, trung bình 10 epoch cuối, optimizer, tham số; bỏ Go/No-Go tự đặt cho tuần 3 (MASTER không có).
8. 4b có thể chạy lại ở Tuần 5 nếu γ thay đổi.
