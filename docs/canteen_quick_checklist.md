# Cẩm nang Bỏ túi: Quy tắc Chụp & Gán nhãn Khay Canteen (v10.0)

Tài liệu hướng dẫn nhanh cho nhóm nghiên cứu khi chụp ảnh khay ăn và gán nhãn bounding box tại canteen.

---

## 1. Góc chụp và Khung hình chuẩn
* **Góc chụp:** Chụp từ trên xuống (**Top-down 60° đến 90°**) giống góc nhìn camera gắn trên trần hoặc nắp thùng rác thông minh.
* **Khoảng cách:** Cách khay khoảng **50 – 80 cm** (đảm bảo thấy trọn vẹn toàn bộ khay ăn và các vật phẩm trên khay).
* **Nền:** Đặt trên khay ăn (inox, nhựa vàng), mặt bàn canteen, hoặc khu vực trả khay.
* **Đa vật thể (Multi-object):** Khay thực tế gồm **nhiều vật phẩm rác cùng lúc** (thức ăn thừa, hộp xốp, ly nhựa, muỗng nĩa, vỏ hộp...). Mỗi vật phẩm được gắn 1 bounding box riêng.
* **Định danh cụm (`cluster_id`):** Bắt buộc theo định dạng `YYYYMMDD_Ca` (regex: `^\d{8}_[a-z]+$`), ví dụ `20261024_trua`, `20261024_chieu`. **TUYỆT ĐỐI KHÔNG** thêm hậu tố tên vật phẩm (như `_hopxop`).

---

## 2. Quy tắc phân loại 3 nhóm rác cho từng Bounding Box

### Nhóm 0: Hữu cơ (`organic` / coarse_id = 0)
* Cơm thừa, mì, bún, xôi, thịt, xương, rau củ thừa trên đĩa/khay.
* Vỏ trái cây (chuối, cam, dưa hấu...).
* Bã trà, túi lọc trà, bã cà phê.

### Nhóm 1: Tái chế khô (`recyclable` / coarse_id = 1)
* **Chai nhựa PET:** Chai nước khoáng, chai nước ngọt rỗng sạch ráo.
* **Lon kim loại:** Lon nhôm Coca, Pepsi, bia đã uống cạn sạch.
* **Giấy / Bìa carton khô:** Hộp sữa tươi sạch, giấy văn phòng khô.
* *Lưu ý:* Phải ở trạng thái sạch hoặc ráo nước, không dính dầu mỡ thức ăn thừa.

### Nhóm 2: Còn lại / Khó tái chế (`other_landfill` / coarse_id = 2)
* **Hộp xốp thực phẩm:** Dù dính dầu mỡ hay sạch đều xếp vào nhóm này.
* **Đồ nhựa dùng 1 lần:** Muỗng nĩa nhựa, que khuấy, ống hút nhựa, nắp ly.
* **Khăn giấy bẩn:** Giấy ăn lau miệng, khăn ướt, tăm tre.
* **Bao bì màng mỏng:** Túi nilon, màng bọc thực phẩm, gói gia vị.
* **Ly giấy tráng PE / Ly xốp:** Ly cà phê mang đi, ly mì ăn liền.

---

## 3. Xử lý ca ranh giới (Edge Cases)
> **Nguyên tắc vàng:** *"Nếu nghi ngờ làm nhiễm bẩn rác tái chế → Gán vào nhóm Còn lại (`other_landfill`)"*.

1. **Ly nhựa còn đá hoặc trân châu:**
   * Gán `other_landfill` (vì còn chất lỏng và thức ăn bên trong).
2. **Hộp xốp dính dầu mỡ / nước mắm:**
   * Gán `other_landfill`.
3. **Chai nước suối còn nước:**
   * Gán `other_landfill` nếu còn nước bên trong; nếu đã ráo gán `recyclable`.
4. **Vật phẩm chồng lấp:**
   * Vẽ bounding box riêng cho từng vật nhìn thấy được; nếu vật bị che khuất > 85% không thể nhận diện thì bỏ qua hoặc đánh dấu `ambiguous = 1`.

---

## 4. Quy trình gán nhãn và kiểm định Kappa (v10.0)
1. **Pilot Kappa (40 khay):**
   - 2 người gán nhãn độc lập hoàn toàn, không trao đổi hoặc nhìn bài của nhau.
   - Nhãn lưu dưới dạng YOLO box format + category id (0, 1, 2).
2. **Tiêu chuẩn nghiệm thu:**
   - Box F1 (IoU >= 0.5) $\ge 0.80$
   - Cohen's Kappa mức phân loại $\kappa \ge 0.70$
3. Đạt chuẩn mới tiến hành gán nhãn diện rộng cho các ngày tiếp theo.
