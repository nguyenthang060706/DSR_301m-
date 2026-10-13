# Cẩm nang Bỏ túi: 5 Quy tắc Vàng Chụp & Gán nhãn Canteen

Tài liệu hướng dẫn nhanh để bạn và người bạn cùng gán nhãn mang theo khi chụp ảnh tại canteen trường.

---

## 1. Góc chụp và Khung hình chuẩn
* **Góc chụp:** Chụp từ trên xuống (**Top-down 60° đến 90°**) giống góc nhìn camera gắn trên nắp thùng rác thông minh.
* **Khoảng cách:** Cách vật thể khoảng **40 – 60 cm** (đảm bảo vật phẩm nằm trọn vẹn trong khung hình).
* **Nền:** Đặt trên khay ăn inox, mặt bàn canteen, hoặc miệng thùng rác.
* **Quy tắc:** Mỗi bức ảnh chỉ có **1 vật phẩm chính** làm tâm điểm (có thể có thức ăn thừa dính kèm).

---

## 2. Quy tắc "Kim chỉ nam" phân loại 3 nhóm

### Nhóm 0: Hữu cơ (`organic`)
* Cơm thừa, mì, bún, xôi, thịt, xương gà, xương cá, rau củ ăn dở.
* Vỏ trái cây (cam, chuối, dưa hấu...).
* Bã trà, túi lọc trà, bã cà phê.

### Nhóm 1: Tái chế khô (`recyclable`)
* **Chai nhựa PET:** Chai nước suối (Aquafina, Lavie...), chai C2, Trà xanh 0 độ đã uống cạn nước.
* **Lon kim loại:** Lon nhôm Coca, Pepsi, RedBull, bia đã uống cạn.
* **Hộp giấy / Bìa carton khô:** Hộp sữa tươi tiệt trùng sạch, bìa carton sạch.
* *Lưu ý:* Phải ở trạng thái sạch hoặc ráo nước, không chứa dầu mỡ thức ăn.

### Nhóm 2: Còn lại / Khó tái chế (`other_landfill`)
* **Hộp xốp thực phẩm:** Dù dính dầu mỡ hay sạch đều xếp vào nhóm này.
* **Đồ dùng 1 lần:** Muỗng nhựa, nĩa nhựa, que khuấy, ống hút nhựa.
* **Khăn giấy bẩn:** Giấy ăn lau miệng, khăn ướt, tăm tre.
* **Bao bì mềm:** Túi nilon đựng đồ ăn, màng bọc thực phẩm, túi bánh tráng.
* **Ly giấy tráng PE / Ly xốp:** Ly cà phê mang đi, ly mì tôm ăn liền.

---

## 3. Xử lý các ca ranh giới "Gây tranh cãi" (Edge Cases)
> **Nguyên tắc vàng:** *"Nếu nghi ngờ làm nhiễm bẩn thùng rác tái chế $\rightarrow$ Hạ xuống nhóm Còn lại (`other_landfill`)"*.

1. **Ly nhựa trà tắc / trà sữa còn đá hoặc trân châu:**
   * Xếp vào `other_landfill` (vì còn chất lỏng và thức ăn bên trong, nếu ném vào thùng tái chế sẽ làm hỏng toàn bộ giấy và chai sạch).
2. **Hộp xốp dính mỡ hành / nước mắm:**
   * Xếp vào `other_landfill`.
3. **Chai nước suối còn 1/3 chai nước:**
   * Xếp vào `other_landfill` (chưa đổ hết nước).

---

## 4. Quy trình gán nhãn 2 người độc lập:
1. Chụp khoảng **50 – 100 ảnh đợt 1**, lưu vào thư mục `data/raw/canteen_raw/`.
2. Tạo 1 bản copy của file `data/canteen_metadata_template.csv`.
3. **Người 1** điền cột `annotator_1`.
4. **Người 2** điền cột `annotator_2` (tuyệt đối không nhìn kết quả của Người 1).
5. Chạy lệnh:
   ```powershell
   python scripts/calc_kappa.py --csv data/canteen_metadata.csv
   ```
6. Nếu **$\kappa \ge 0.70$ $\rightarrow$ ĐẠT CHUẨN!**
