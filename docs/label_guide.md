# Hướng dẫn Quy ước Gán nhãn Rác Canteen Đại học (Label Guide v0)

Tài liệu này xác lập quy chuẩn gán nhãn chính thức cho bộ ảnh thu thập tại canteen trường đại học, phục vụ phân loại rác tại nguồn dựa trên **quyết định bỏ rác thực tế**, không phải phân loại vật liệu lý thuyết.

---

## 1. Ba nhóm quyết định bỏ rác

| Nhóm nhãn | Tên tiếng Anh | Định nghĩa & Ý nghĩa xử lý |
|---|---|---|
| **0 - Hữu cơ** | `organic` | Rác phân hủy sinh học, thức ăn thừa, phụ phẩm nông nghiệp / thực phẩm. Đưa vào ủ compost hoặc xử lý thức ăn chăn nuôi. |
| **1 - Tái chế khô** | `recyclable` | Vật liệu có giá trị tái chế cao và **ở trạng thái sạch / ráo**, không gây nhiễm bẩn dây chuyền ép kiện (nhựa cứng sạch, kim loại, giấy bìa khô). |
| **2 - Còn lại / Khó tái chế** | `other_landfill` | Rác thải dơ bẩn, dính dầu mỡ, vật liệu phức hợp, rác không có giá trị thu hồi hoặc chi phí tái chế quá cao. Đưa đi chôn lấp hoặc đốt rác phát điện. |

---

## 2. Danh mục vật phẩm canteen đại học điển hình

### 2.1 Nhóm Hữu cơ (`organic`)
- Cơm thừa, mì, bún, phở (đã ráo nước sốt chính).
- Vỏ trái cây (cam, chuối, dưa hấu, táo...).
- Xương gà, xương cá, thịt thừa, rau củ thừa.
- Vỏ trứng, bã trà, bã cà phê.
- Đồ ăn vặt thừa (bánh tráng, bánh mì ăn dở không kèm bao bì).

### 2.2 Nhóm Tái chế khô (`recyclable`)
- **Chai nhựa PET**: Chai nước khoáng (Lavie, Aquafina...), chai trà xanh/nước ngọt đã uống cạn, ráo nước.
- **Lon kim loại**: Lon nhôm nước ngọt (Coca, Pepsi...), lon bia, lon trà đã uống cạn.
- **Hộp giấy / Bìa carton khô**: Hộp sữa tiệt trùng (đã uống cạn, dẹp phẳng), bìa carton sạch không dính mỡ.
- **Ly nhựa sạch**: Ly nhựa PP/PET (trà tắc, nước mía) đã đổ sạch đá/nước, không bám cặn thức ăn.

### 2.3 Nhóm Còn lại / Khó tái chế (`other_landfill`)
- **Hộp xốp thực phẩm**: Hộp xốp (EPS) đựng cơm tấm, xôi, bánh bao (thường dính mỡ, dầu hành, nước mắm).
- **Đồ dùng ăn uống 1 lần**: Muỗng, nĩa, dao nhựa dùng 1 lần; que khuấy; ống hút nhựa.
- **Giấy bẩn**: Khăn giấy lau miệng, khăn ướt, giấy ăn dính dầu mỡ/nước sốt, tăm tre.
- **Bao bì mềm & Túi nilon**: Túi nilon đựng đồ ăn, màng bọc thực phẩm, túi bánh kẹo tráng màng nhôm.
- **Ly giấy tráng màng PE bẩn**: Ly cà phê giấy, tô mì giấy đã qua sử dụng dính nước dùng.

---

## 3. Quy tắc xử lý các ca ranh giới (Edge Cases)

Nguyên tắc cốt lõi: **"Nếu nghi ngờ gây nhiễm bẩn thùng tái chế $\rightarrow$ Xếp vào Còn lại (`other_landfill`)"**.

1. **Ly trà sữa / nước ngọt còn đá hoặc trân châu:**
   - Nếu vật phẩm chụp ở trạng thái nguyên vẹn còn chất lỏng/thức ăn bên trong $\rightarrow$ Gán nhãn `other_landfill` (vì nếu ném cả ly vào thùng tái chế sẽ làm hỏng toàn bộ giấy/nhựa khác).
   - Ghi chú: Gắn cờ `ambiguous = True`.
2. **Hộp xốp đựng đồ ăn:**
   - Dù sạch hay bẩn $\rightarrow$ Mặc định gán nhãn `other_landfill` (xốp EPS trong canteen trường học tại VN hầu như không có cơ sở thu gom tái chế hiệu quả).
3. **Chai nhựa còn nắp và nhãn:**
   - Vẫn gán `recyclable` nếu chai đã cạn nước.
4. **Hộp bã mía / Bao bì sinh học dính thức ăn:**
   - Nếu canteen có hệ thống phân hủy riêng: Hữu cơ.
   - Tại điều kiện phân loại rác thông thường tại trường học hiện nay $\rightarrow$ Gán `other_landfill`.
5. **Vật thể lạ / Rác văn phòng rơi vào khay ăn (bút bi, kẹp bấm, khẩu trang):**
   - Mặc định xếp vào `other_landfill`.

---

## 4. Cấu trúc Metadata của từng ảnh

Mỗi ảnh canteen khi thu thập phải có một bản ghi tương ứng trong file `data/canteen_metadata.csv` với các trường:

```csv
image_id,cluster_id,date,shift,device,fine_label,coarse_label,is_clean,is_distorted,ambiguous,annotator_1,annotator_2,final_label
```

- `image_id`: Tên file ảnh (ví dụ: `canteen_20261002_001.jpg`).
- `cluster_id`: Mã cụm định danh (Định dạng: `YYYYMMDD_Ca_LoaiVat`, ví dụ: `20261002_trua_chaipet`).
- `fine_label`: Nhãn mịn (ví dụ: `pet_bottle`, `foam_box`, `leftover_food`, `plastic_cup`...).
- `coarse_label`: 1 trong 3 nhóm (`organic`, `recyclable`, `other_landfill`).
- `ambiguous`: `True` nếu thuộc ca ranh giới hoặc 2 người gán không thống nhất ban đầu.
- `annotator_1`, `annotator_2`: Nhãn ban đầu của người 1 và người 2.

---

## 5. Quy trình Kiểm định Độ đồng thuận (Cohen's Kappa)

1. Lấy mẫu ngẫu nhiên **100 ảnh thử nghiệm đợt đầu** (chụp tại canteen trong ngày 1-2).
2. Người 1 và Người 2 tiến hành gán nhãn độc lập hoàn toàn, không trao đổi.
3. Tính hệ số Cohen's Kappa:
   $$\kappa = \frac{p_o - p_e}{1 - p_e}$$
   - $p_o$: Tỷ lệ quan sát đồng thuận.
   - $p_e$: Tỷ lệ đồng thuận kỳ vọng ngẫu nhiên.
4. **Tiêu chuẩn chấp thuận:**
   - $\kappa \ge 0,70$: Quy ước rõ ràng, đạt chuẩn, cho phép tiến hành chụp và gán nhãn diện rộng.
   - $\kappa < 0,70$: Bắt buộc dừng lại, mở phiên họp đối chiếu toàn bộ các ca bất đồng, bổ sung quy tắc cụ thể vào tài liệu này và gán thử nghiệm lại 100 ảnh mới.
