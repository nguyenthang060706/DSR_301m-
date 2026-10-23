"""scripts/build_presentation_deck_v2.py

Tạo Slide Deck thuyết trình cập nhật theo phản hồi và Kế hoạch MASTER v10.0:
- Định nghĩa rõ bài toán: Phân loại rác khay canteen ĐA VẬT THỂ (Multi-Object Canteen Waste Sorting),
  từ bỏ giả định phi thực tế chụp đơn lẻ 1-2 vật thể.
- Trình bày rõ Quy trình Xử lý Dữ liệu: Lọc trùng 2 tầng (MD5, pHash), chuẩn hóa RGB 256x256,
  định dạng YOLO class-agnostic, tổng hợp Copy-Paste.
- Cấu trúc Pipeline 2 giai đoạn: Detector D1 (Class-Agnostic) -> KD Classifier (3 nhóm + background).
- Quy chuẩn Bounding Box, kiểm định đồng thuận 2 tầng (Box F1 >= 0.80, Kappa >= 0.70) và chống rò rỉ theo cụm.
"""

import os
import sys
import io
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def create_deck(output_pptx_path):
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(5.625)
    blank_layout = prs.slide_layouts[6]

    # Bảng màu chuẩn Corporate / Academic
    COLOR_PRIMARY = RGBColor(0x1B, 0x36, 0x5D)    # Deep Navy Blue
    COLOR_ACCENT = RGBColor(0x1B, 0x8A, 0x5A)     # Emerald Green (Tái chế / Sinh thái)
    COLOR_BG_CARD = RGBColor(0xF0, 0xF4, 0xF8)    # Nền Card xám xanh nhạt
    COLOR_CARD_BORDER = RGBColor(0xD0, 0xDE, 0xE9)# Viền Card
    COLOR_TEXT_DARK = RGBColor(0x22, 0x28, 0x31)  # Chữ đen than
    COLOR_MUTED = RGBColor(0x5A, 0x6E, 0x7F)      # Xám phụ đề
    COLOR_WHITE = RGBColor(0xFF, 0xFF, 0xFF)      # Trắng
    COLOR_HIGHLIGHT = RGBColor(0xC0, 0x39, 0x2B)  # Đỏ nhấn mạnh (Cảnh báo / Ca ranh giới)

    def add_header(slide, title_text, slide_num, subtitle_text=None):
        # Dải màu xanh Emerald bên trái
        accent_bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(0.18), Inches(5.625))
        accent_bar.fill.solid()
        accent_bar.fill.fore_color.rgb = COLOR_ACCENT
        accent_bar.line.fill.background()

        # Tiêu đề slide
        title_box = slide.shapes.add_textbox(Inches(0.55), Inches(0.25), Inches(8.8), Inches(0.75))
        tf = title_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.text = title_text
        p.font.name = "Calibri"
        p.font.size = Pt(19)
        p.font.bold = True
        p.font.color.rgb = COLOR_PRIMARY

        if subtitle_text:
            p2 = tf.add_paragraph()
            p2.text = subtitle_text
            p2.font.name = "Calibri"
            p2.font.size = Pt(10.5)
            p2.font.color.rgb = COLOR_MUTED
            p2.space_before = Pt(2)

        # Số trang
        num_box = slide.shapes.add_textbox(Inches(9.1), Inches(5.2), Inches(0.6), Inches(0.3))
        tf_num = num_box.text_frame
        tf_num.margin_left = tf_num.margin_top = tf_num.margin_right = tf_num.margin_bottom = 0
        p_num = tf_num.paragraphs[0]
        p_num.text = f"{slide_num}/8"
        p_num.font.name = "Calibri"
        p_num.font.size = Pt(9.5)
        p_num.font.color.rgb = COLOR_MUTED

    # ==========================================
    # SLIDE 1: Title & Đặt lại Bài toán
    # ==========================================
    s1 = prs.slides.add_slide(blank_layout)
    bar1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(0.22), Inches(5.625))
    bar1.fill.solid()
    bar1.fill.fore_color.rgb = COLOR_ACCENT
    bar1.line.fill.background()

    tbox1 = s1.shapes.add_textbox(Inches(0.7), Inches(0.95), Inches(8.8), Inches(1.5))
    tf1 = tbox1.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "Phân Loại Rác Thải Đa Vật Thể Trên Khay Ăn Canteen"
    p1.font.name = "Calibri"
    p1.font.size = Pt(24)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_PRIMARY

    p1_sub = tf1.add_paragraph()
    p1_sub.text = "Quy Trình Xử Lý Dữ Liệu, Kiến Trúc Pipeline 2 Giai Đoạn & Giao Thức Đánh Giá (MASTER v10.0)"
    p1_sub.font.name = "Calibri"
    p1_sub.font.size = Pt(13)
    p1_sub.font.bold = True
    p1_sub.font.color.rgb = COLOR_ACCENT
    p1_sub.space_before = Pt(4)

    sbox1 = s1.shapes.add_textbox(Inches(0.7), Inches(2.6), Inches(8.8), Inches(2.5))
    stf1 = sbox1.text_frame
    stf1.word_wrap = True

    items_s1 = [
        ("Chuyển dịch trọng tâm bài toán:", "Từ bỏ giả định chụp đơn lẻ 1-2 vật thể trên nền trắng. Chuyển sang bài toán thực chiến: Nhận diện và phân loại nhiều loại rác đan xen trên khay ăn canteen."),
        ("Phương pháp chủ đạo:", "Pipeline 2 giai đoạn: Detector D1 (Class-Agnostic) định vị vật rác -> KD Classifier (Student ResNet18/MobileNetV3) phân loại vào 3 nhóm + lọc background."),
        ("Xử lý dữ liệu chuyên sâu:", "Lọc trùng lặp 2 tầng (MD5, pHash), chuẩn hóa kích thước 256x256 SSD, chuyển đổi TACO 80/20, tích hợp khay ăn Padova và sinh ảnh ghép Copy-Paste."),
        ("Kiểm soát chất lượng gán nhãn:", "Ban hành label_guide.md v2.0 có quy chuẩn Bounding Box; kiểm định độ đồng thuận 2 tầng (Box F1 >= 0.80 và Cohen's kappa >= 0.70) trên khay canteen thật.")
    ]
    for i, (hd, body) in enumerate(items_s1):
        p = stf1.paragraphs[0] if i == 0 else stf1.add_paragraph()
        p.text = f"• {hd} {body}"
        p.font.name = "Calibri"
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_TEXT_DARK
        p.space_before = Pt(4)
        p.runs[0].font.bold = True
        p.runs[0].font.color.rgb = COLOR_PRIMARY

    # ==========================================
    # SLIDE 2: Định nghĩa Bài toán & Kiến trúc Pipeline 2-Stage
    # ==========================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Định nghĩa Bài toán: Phân loại Rác Khay Canteen & Pipeline 2-Stage", 2,
               "Tại sao không chụp đơn lẻ? Khay ăn hỗn hợp thực tế đòi hỏi kết hợp Phát hiện BBox và Chưng cất Tri thức")

    col_w = Inches(4.25)
    # Box trái: Bài toán thực tế
    bg_l = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.55), Inches(1.15), col_w, Inches(2.6))
    bg_l.fill.solid()
    bg_l.fill.fore_color.rgb = COLOR_BG_CARD
    bg_l.line.color.rgb = COLOR_CARD_BORDER

    tb_l = s2.shapes.add_textbox(Inches(0.7), Inches(1.25), col_w - Inches(0.3), Inches(2.4))
    tf_l = tb_l.text_frame
    tf_l.word_wrap = True
    p = tf_l.paragraphs[0]
    p.text = "Thách Thức Thực Tế Trên Khay Ăn"
    p.font.name = "Calibri"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    challenges = [
        "Đa vật thể hỗn hợp: Một khay gồm đĩa cơm thừa, hộp xốp dính dầu, ly trà sữa, muỗng nhựa, khăn giấy.",
        "Che khuất & Chồng lấn: Các vật thể đè lên nhau (occlusion), dính bẩn lẫn nhau.",
        "Phân định Rác vs Dụng cụ: Bát, đĩa, khay và thìa kim loại là tài sản của trường (không phải rác), cần loại trừ.",
        "Hạn chế của mô hình đơn lẻ: Chụp 1 ảnh rồi phân loại cả ảnh sẽ thất bại vì ảnh chứa nhiều loại rác cùng lúc."
    ]
    for ch in challenges:
        p_c = tf_l.add_paragraph()
        p_c.text = f"• {ch}"
        p_c.font.name = "Calibri"
        p_c.font.size = Pt(9.5)
        p_c.font.color.rgb = COLOR_TEXT_DARK
        p_c.space_before = Pt(3)

    # Box phải: Kiến trúc Pipeline 2-Stage
    bg_r = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(5.15), Inches(1.15), col_w, Inches(2.6))
    bg_r.fill.solid()
    bg_r.fill.fore_color.rgb = COLOR_BG_CARD
    bg_r.line.color.rgb = COLOR_CARD_BORDER

    tb_r = s2.shapes.add_textbox(Inches(5.3), Inches(1.25), col_w - Inches(0.3), Inches(2.4))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True
    p = tf_r.paragraphs[0]
    p.text = "Giải Pháp: Pipeline 2 Giai Đoạn"
    p.font.name = "Calibri"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT

    pipeline_steps = [
        "Giai đoạn 1 (Detector D1 Class-Agnostic): YOLOv8n quét toàn bộ khay, phát hiện mọi vị trí có vật rác (1 lớp 'waste') -> Xuất Bounding Boxes (N <= 8).",
        "Giai đoạn 2 (KD Classifier): Crop từng vùng box rác, đưa qua Student Classifier (ResNet18 / MobileNetV3) để phân vào 3 nhóm rác canteen.",
        "Lớp Background đào thải lỗi: Nếu Detector crop nhầm mép bàn/khay rỗng, Classifier gán nhãn 'background' để tự động loại bỏ.",
        "Bảo toàn thành quả KD: Tận dụng trọn vẹn 15.000 ảnh Kaggle và các mô hình chưng cất tri thức Teacher-Student đã dày công tối ưu."
    ]
    for ps in pipeline_steps:
        p_s = tf_r.add_paragraph()
        p_s.text = f"• {ps}"
        p_s.font.name = "Calibri"
        p_s.font.size = Pt(9.5)
        p_s.font.color.rgb = COLOR_TEXT_DARK
        p_s.space_before = Pt(3)

    # Bottom summary box
    bot2 = s2.shapes.add_textbox(Inches(0.55), Inches(3.9), Inches(8.85), Inches(1.3))
    btf2 = bot2.text_frame
    btf2.word_wrap = True
    p_b = btf2.paragraphs[0]
    p_b.text = "• Nguyên lý vận hành: Ảnh khay canteen -> [Detector D1] -> N Bounding Boxes -> Crop từng vùng -> [KD Classifier] -> Danh sách {Box, Nhóm Rác, Độ tin cậy}. Mô hình đạt chuẩn thời gian thực trên chip biên."
    p_b.font.name = "Calibri"
    p_b.font.size = Pt(10.5)
    p_b.font.color.rgb = COLOR_PRIMARY
    p_b.font.bold = True

    # ==========================================
    # SLIDE 3: 3 Nhóm Quyết định bỏ rác Canteen
    # ==========================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Mục tiêu Phân loại: 3 Nhóm Quyết định Rác Canteen & Quy tắc Nhiễm bẩn", 3,
               "Phân loại dựa trên quy trình thu gom xử lý thực tế tại nguồn thay vì vật liệu lý thuyết")

    cards_data = [
        ("0 - Hữu cơ (Organic)", "Thức ăn thừa, cơm, mì, bún, rau củ, vỏ trái cây, xương thịt thừa, bã trà/cà phê",
         "Quy trình: Đưa đi ủ phân compost hoặc làm thức ăn chăn nuôi.\nYêu cầu: Tuyệt đối không lẫn bao bì nhựa, muỗng nĩa 1 lần hay túi nilon."),
        ("1 - Tái chế khô (Recyclable)", "Chai nhựa PET ráo nước, lon nhôm, hộp sữa tiệt trùng, thùng bìa carton sạch khô",
         "Quy trình: Thu gom ép kiện bán tái chế công nghiệp.\nYêu cầu: Bắt buộc sạch, đã uống cạn, ráo nước, không dính dầu mỡ thực phẩm."),
        ("2 - Còn lại (Other / Landfill)", "Hộp xốp dính dầu, ly nhựa trà sữa dính đá, muỗng nĩa nhựa 1 lần, khăn giấy bẩn",
         "Quy trình: Đưa đi chôn lấp hoặc đốt rác phát điện.\nYêu cầu: Rác phức hợp, bám dầu mỡ thức ăn, chi phí thu hồi tái chế quá cao.")
    ]
    for i, (title, items, purpose) in enumerate(cards_data):
        left = Inches(0.55 + i * 2.98)
        top = Inches(1.15)
        width = Inches(2.88)
        height = Inches(2.35)

        bg = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = COLOR_BG_CARD
        bg.line.color.rgb = COLOR_CARD_BORDER

        tb = s3.shapes.add_textbox(left + Inches(0.12), top + Inches(0.1), width - Inches(0.24), height - Inches(0.2))
        tf = tb.text_frame
        tf.word_wrap = True

        p = tf.paragraphs[0]
        p.text = title
        p.font.name = "Calibri"
        p.font.size = Pt(12.5)
        p.font.bold = True
        p.font.color.rgb = COLOR_PRIMARY

        p_items = tf.add_paragraph()
        p_items.text = items
        p_items.font.name = "Calibri"
        p_items.font.size = Pt(9.5)
        p_items.font.color.rgb = COLOR_TEXT_DARK
        p_items.space_before = Pt(3)

        p_purp = tf.add_paragraph()
        p_purp.text = purpose
        p_purp.font.name = "Calibri"
        p_purp.font.size = Pt(9)
        p_purp.font.color.rgb = COLOR_MUTED
        p_purp.space_before = Pt(4)

    bot3 = s3.shapes.add_textbox(Inches(0.55), Inches(3.65), Inches(8.85), Inches(1.5))
    btf3 = bot3.text_frame
    btf3.word_wrap = True

    bot3_points = [
        ("Nguyên tắc cốt lõi về nhiễm bẩn (Contamination Rule):", "'Nếu nghi ngờ vật phẩm gây nhiễm bẩn dây chuyền tái chế -> Bắt buộc xếp vào nhóm Còn lại (other_landfill)'. Tránh làm hỏng toàn bộ mẻ rác tái chế."),
        ("Xử lý ca ranh giới điển hình:", "Ly trà sữa/nước ngọt còn đá/trân châu -> Còn lại (chất lỏng làm mốc giấy/nhựa sạch); Hộp xốp thực phẩm dính mỡ -> Còn lại; Chai PET cạn nước còn nhãn -> Tái chế."),
        ("Lớp bổ sung (Lớp 3: Background):", "Đại diện cho mặt bàn, mép khay rỗng, dụng cụ ăn uống tái sử dụng của trường học (thìa dĩa inox). Classifier nhận diện lớp này để không đếm nhầm vào rác.")
    ]
    for i, (hd, pt) in enumerate(bot3_points):
        p = btf3.paragraphs[0] if i == 0 else btf3.add_paragraph()
        p.text = f"• {hd} {pt}"
        p.font.name = "Calibri"
        p.font.size = Pt(10)
        p.font.color.rgb = COLOR_TEXT_DARK
        p.space_before = Pt(3)
        p.runs[0].font.bold = True
        p.runs[0].font.color.rgb = COLOR_PRIMARY

    # ==========================================
    # SLIDE 4: Quy trình Xử lý & Làm sạch Dữ liệu
    # ==========================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Quy trình Tiền xử lý, Làm sạch & Chuẩn hóa Dữ liệu (Data Pipeline)", 4,
               "Từ dữ liệu ảnh thô đến tập dữ liệu sẵn sàng huấn luyện: Lọc trùng 2 tầng và chuẩn hóa định dạng")

    steps_data = [
        ("1. Lọc trùng lặp 2 tầng", "Deduplication Pipeline",
         "• MD5 Checksum: Quét xóa 100% các file ảnh bị trùng lặp vật lý bit-by-bit khi cào hoặc gom về.\n• Perceptual Hash (pHash): Phát hiện các ảnh 'gần như giống hệt' (ngưỡng Hamming <= 5) do chụp liên tiếp ở góc máy cách nhau vài milimet hoặc crop nhẹ."),
        ("2. Chuẩn hóa Định dạng", "Format & Resolution",
         "• Chuyển đổi toàn bộ ảnh sang không gian màu chuẩn RGB.\n• Đưa về kích thước cơ bản 256x256 lưu trữ trực tiếp trên SSD tốc độ cao để triệt tiêu nghẽn cổ chai I/O khi nạp DataLoader GPU.\n• Ảnh crop huấn luyện CNN chuẩn hóa 224x224."),
        ("3. Chuẩn hóa Bounding Box", "YOLO Format [0, 1]",
         "• Toàn bộ tọa độ vật thể được chuẩn hóa về định dạng YOLO txt: <class_id> <x_center> <y_center> <w> <h> theo tỉ lệ [0.0, 1.0].\n• Với Detector D1: Toàn bộ nhãn chi tiết được gộp thành 1 lớp duy nhất (class 0: waste, class-agnostic).")
    ]
    for i, (stitle, ssub, scontent) in enumerate(steps_data):
        left = Inches(0.55 + i * 2.98)
        top = Inches(1.15)
        width = Inches(2.88)
        height = Inches(2.35)

        bg = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = COLOR_BG_CARD
        bg.line.color.rgb = COLOR_CARD_BORDER

        tb = s4.shapes.add_textbox(left + Inches(0.12), top + Inches(0.1), width - Inches(0.24), height - Inches(0.2))
        tf = tb.text_frame
        tf.word_wrap = True

        p = tf.paragraphs[0]
        p.text = stitle
        p.font.name = "Calibri"
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = COLOR_PRIMARY

        p_sub = tf.add_paragraph()
        p_sub.text = ssub
        p_sub.font.name = "Calibri"
        p_sub.font.size = Pt(9.5)
        p_sub.font.bold = True
        p_sub.font.color.rgb = COLOR_ACCENT
        p_sub.space_before = Pt(2)

        p_c = tf.add_paragraph()
        p_c.text = scontent
        p_c.font.name = "Calibri"
        p_c.font.size = Pt(9)
        p_c.font.color.rgb = COLOR_TEXT_DARK
        p_c.space_before = Pt(4)

    bot4 = s4.shapes.add_textbox(Inches(0.55), Inches(3.65), Inches(8.85), Inches(1.5))
    btf4 = bot4.text_frame
    btf4.word_wrap = True

    bot4_points = [
        ("Nhiễu hình học khi Crop (Crop Jitter Augmentation):", "Khi train Classifier, áp dụng dịch tâm ngẫu nhiên (+-10%) và thay đổi tỉ lệ (+-15%) của Bounding Box để Classifier thích nghi tốt với sai số phát hiện của Detector."),
        ("Tối ưu hóa DataLoader trên Laptop RTX 4050:", "Bật tính toán dấu phẩy động hỗn hợp (AMP FP16), bộ nhớ định dạng channels_last và 2-4 dataloader workers; giúp tốc độ huấn luyện đạt ~38 giây/epoch trên 1.200 ảnh."),
        ("Kiểm soát rò rỉ rạch ròi:", "Không để bất kỳ ảnh/vật thể trùng lặp nào xuất hiện đồng thời ở cả tập Train và tập Test.")
    ]
    for i, (hd, pt) in enumerate(bot4_points):
        p = btf4.paragraphs[0] if i == 0 else btf4.add_paragraph()
        p.text = f"• {hd} {pt}"
        p.font.name = "Calibri"
        p.font.size = Pt(10)
        p.font.color.rgb = COLOR_TEXT_DARK
        p.space_before = Pt(3)
        p.runs[0].font.bold = True
        p.runs[0].font.color.rgb = COLOR_PRIMARY

    # ==========================================
    # SLIDE 5: Chiến lược Đa nguồn Dữ liệu
    # ==========================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Chiến lược Đa nguồn Dữ liệu Phục vụ Pipeline 2 Giai Đoạn", 5,
               "Kết hợp dữ liệu công khai quy mô lớn, dữ liệu khay ăn thực tế và dữ liệu tổng hợp Copy-Paste")

    rows, cols = 5, 4
    table_shape = s5.shapes.add_table(rows, cols, Inches(0.55), Inches(1.15), Inches(8.85), Inches(2.2))
    table = table_shape.table
    table.columns[0].width = Inches(2.2)
    table.columns[1].width = Inches(1.5)
    table.columns[2].width = Inches(2.2)
    table.columns[3].width = Inches(2.95)

    headers = ["Nguồn Dữ liệu", "Quy mô", "Đặc điểm & Xử lý", "Vai trò trong Pipeline"]
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_PRIMARY
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.name = "Calibri"
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = COLOR_WHITE
        p.alignment = PP_ALIGN.CENTER

    sources_data = [
        ("1. Kaggle 30-class\n(Household Waste)", "15.000 ảnh\n(30 lớp mịn)", "Rác đơn lẻ chụp studio & đời thực, 500 ảnh/lớp, chia 85/15", "Huấn luyện Teacher & KD Classifier (M0-M4) theo cơ chế Fine-to-Coarse"),
        ("2. TACO Dataset\n(Trash Annotations)", "1.500 ảnh\n(4.784 BBox)", "Rác đa vật thể ngoài thực tế, gộp về 1 lớp waste, chia 80/20", "Pretrain Detector D1(a) học khái niệm 'vật rác' (đã hoàn thành 50 epoch, mAP50=51.1%)"),
        ("3. Padova Canteen\n(Food Recognition)", "74 ảnh khay\n(14 món ăn)", "Khay ăn canteen thật, có BBox thức ăn thừa (leftovers)", "Bổ sung dữ liệu đồ ăn thừa (Hữu cơ) cho Detector, bù đắp chỗ thiếu của TACO"),
        ("4. Copy-Paste Trays\n(Ảnh ghép tổng hợp)", "1.000+ ảnh\n(tự sinh)", "Ghép 3-6 vật Kaggle lên ảnh nền khay ăn canteen, IoU <= 0.3", "Huấn luyện trung gian D1(b) mô phỏng khay rác trước khi có đủ ảnh thật")
    ]
    for i, row in enumerate(sources_data):
        for j, val in enumerate(row):
            cell = table.cell(i + 1, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = COLOR_BG_CARD if i % 2 == 0 else COLOR_WHITE
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.name = "Calibri"
            p.font.size = Pt(9)
            p.font.color.rgb = COLOR_TEXT_DARK
            if j == 1:
                p.alignment = PP_ALIGN.CENTER

    bot5 = s5.shapes.add_textbox(Inches(0.55), Inches(3.6), Inches(8.85), Inches(1.5))
    btf5 = bot5.text_frame
    btf5.word_wrap = True

    points_s5 = [
        ("Giải quyết triệt để rủi ro thiếu dữ liệu Hữu cơ:", "TACO có điểm yếu là ít đồ ăn thừa. Nhóm đã bổ sung bộ khay canteen Padova và các vật thể thực phẩm Kaggle để Detector nhận diện tốt cả cơm/canh/thịt cá thừa."),
        ("Dữ liệu Canteen Việt Nam thật (50-60 khay):", "Được bảo lưu độc lập làm tập Kiểm thử cuối cùng (Test Canteen đóng băng) và Thích nghi ít mẫu Few-shot (RQ2), đảm bảo tính thực tiễn tại trường."),
        ("Không lãng phí tài nguyên:", "Mỗi nguồn dữ liệu đều giữ một vai trò không thể thay thế trong chuỗi Pipeline.")
    ]
    for i, (hd, pt) in enumerate(points_s5):
        p = btf5.paragraphs[0] if i == 0 else btf5.add_paragraph()
        p.text = f"• {hd} {pt}"
        p.font.name = "Calibri"
        p.font.size = Pt(10)
        p.font.color.rgb = COLOR_TEXT_DARK
        p.space_before = Pt(3)
        p.runs[0].font.bold = True
        p.runs[0].font.color.rgb = COLOR_PRIMARY

    # ==========================================
    # SLIDE 6: Quy chuẩn Gán nhãn Bounding Box & Kiểm định 2 Tầng
    # ==========================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Quy chuẩn Gán nhãn Bounding Box & Kiểm định Độ Đồng Thuận 2 Tầng", 6,
               "Tài liệu hóa quy ước docs/label_guide.md v2.0 và quy trình kiểm định khoa học nghiêm ngặt")

    rules_cards = [
        ("Quy ước Vẽ Bounding Box", "Rác vs Dụng cụ Canteen",
         "• Khay ăn, đĩa, bát, thìa inox: KHÔNG gán box (đây là tài sản trường học gom rửa).\n• Thức ăn thừa trong đĩa/bát: Vẽ 1 box bọc trọn vùng thức ăn thừa (gán Hữu cơ).\n• Hộp xốp / ly nhựa: Vẽ box bọc trọn hộp/ly kể cả có dính thức ăn bên trong.\n• Muỗng nhựa, ống hút, khăn giấy: Vẽ box riêng từng vật; cụm dính liền vẽ 1 box."),
        ("Xử lý Che khuất & Kích thước", "Tightness & Occlusion",
         "• Ôm sát viền ngoài (tight box): Sai số viền <= 5% kích thước vật thể.\n• Che khuất một phần (< 80%): Vẽ box bọc vùng nhìn thấy được (visible box).\n• Che khuất nghiêm trọng (>= 80%): Bỏ qua không gán box để tránh gây nhiễu.\n• Ngưỡng kích thước tối thiểu: Bỏ qua mẩu vụn li ti < 16x16 px trên ảnh gốc."),
        ("Kiểm định Độ đồng thuận 2 Tầng", "Chỉ số Nghiệm thu bắt buộc",
         "• Pilot 40 khay mẫu (>= 100 boxes) do 2 người gán nhãn độc lập hoàn toàn:\n  1. Tầng 1 (Vị trí Box): Khớp box IoU >= 0.5, yêu cầu Box F1 >= 0.80.\n  2. Tầng 2 (Nhãn Phân loại): Trên các cặp box đã khớp, tính Cohen's kappa >= 0.70.\n• Nếu không đạt: Bắt buộc dừng lại, họp đối chiếu ca sai và gán lại 40 khay mới.")
    ]
    for i, (rtitle, rsub, rcontent) in enumerate(rules_cards):
        left = Inches(0.55 + i * 2.98)
        top = Inches(1.15)
        width = Inches(2.88)
        height = Inches(2.45)

        bg = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = COLOR_BG_CARD
        bg.line.color.rgb = COLOR_CARD_BORDER

        tb = s6.shapes.add_textbox(left + Inches(0.12), top + Inches(0.1), width - Inches(0.24), height - Inches(0.2))
        tf = tb.text_frame
        tf.word_wrap = True

        p = tf.paragraphs[0]
        p.text = rtitle
        p.font.name = "Calibri"
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = COLOR_PRIMARY

        p_sub = tf.add_paragraph()
        p_sub.text = rsub
        p_sub.font.name = "Calibri"
        p_sub.font.size = Pt(9.5)
        p_sub.font.bold = True
        p_sub.font.color.rgb = COLOR_ACCENT
        p_sub.space_before = Pt(2)

        p_c = tf.add_paragraph()
        p_c.text = rcontent
        p_c.font.name = "Calibri"
        p_c.font.size = Pt(8.8)
        p_c.font.color.rgb = COLOR_TEXT_DARK
        p_c.space_before = Pt(3)

    bot6 = s6.shapes.add_textbox(Inches(0.55), Inches(3.75), Inches(8.85), Inches(1.4))
    btf6 = bot6.text_frame
    btf6.word_wrap = True

    bot6_points = [
        ("Công cụ tự động hóa kiểm định:", "Đã xây dựng script scripts/calc_agreement_multiobject.py để tự động đo Box F1 và Cohen's Kappa chỉ trong 1 giây sau khi 2 người nộp nhãn."),
        ("Quản lý ảnh mơ hồ (Ambiguous):", "Các trường hợp tranh cãi được đánh dấu ambiguous=True trong canteen_metadata.csv; dùng để báo cáo độ bền mô hình, không dùng để tune.")
    ]
    for i, (hd, pt) in enumerate(bot6_points):
        p = btf6.paragraphs[0] if i == 0 else btf6.add_paragraph()
        p.text = f"• {hd} {pt}"
        p.font.name = "Calibri"
        p.font.size = Pt(10)
        p.font.color.rgb = COLOR_TEXT_DARK
        p.space_before = Pt(3)
        p.runs[0].font.bold = True
        p.runs[0].font.color.rgb = COLOR_PRIMARY

    # ==========================================
    # SLIDE 7: Thu thập Ảnh Canteen & Chống Rò rỉ Dữ liệu
    # ==========================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Thu thập Thực tế Canteen & Giao thức Chống Rò rỉ Dữ liệu", 7,
               "Phân cụm theo ngày/ca chụp (cluster_id) và công cụ Pre-labeling tự động tiết kiệm 70% thời gian")

    tbox7 = s7.shapes.add_textbox(Inches(0.55), Inches(1.15), Inches(8.85), Inches(4.0))
    tf7 = tbox7.text_frame
    tf7.word_wrap = True

    points_s7 = [
        ("Thiết lập Góc Chụp Thực tế (Acquisition Protocol):",
         "Chụp góc 45 - 90 độ (top-down) từ trên xuống, mô phỏng camera gắn cố định trên thùng rác thông minh. Chụp trong điều kiện ánh sáng tự nhiên và đèn huỳnh quang canteen. Đợt 1 thu thập 50-60 khay rác thật + 20-30 ảnh khay trống làm ảnh nền."),
        ("Quy trình Gán Nhãn Bán Tự Động (Model-Assisted Pre-labeling):",
         "Nhóm không phải vẽ tay từng Bounding Box từ con số 0. Đã viết script auto_prelabel_canteen.py sử dụng mô hình TACO vừa huấn luyện để vẽ sẵn các khung BBox gợi ý -> Hai người gán nhãn chỉ cần mở Roboflow/CVAT chỉnh nhẹ viền và chọn 1 trong 3 nhóm rác (tiết kiệm hơn 70% công sức)."),
        ("Kiểm soát Rò rỉ Bằng Phân Cụm (Cluster-based Splitting):",
         "Mỗi ảnh được gán mã cụm định danh: cluster_id = YYYYMMDD_Ca (Ví dụ: 20261015_trua). Đã có script split_canteen_by_cluster.py tự động phân chia theo cụm ngày/ca. Toàn bộ ảnh cùng buổi chụp BẮT BUỘC chỉ nằm trọn trong 1 tập duy nhất (Train, Val, hoặc Test)."),
        ("Tập Test Canteen Đóng Băng Tuyệt Đối:",
         "Tập Test Canteen gồm các ngày chụp hoàn toàn độc lập, tuyệt đối KHÔNG xuất hiện trong tập Train hay Val. Đóng băng cố định và chỉ dùng để chấm điểm cuối cùng; không dùng để tune siêu tham số."),
        ("Quản lý Metadata Đầy đủ (canteen_metadata.csv):",
         "Lưu vết chi tiết: image_id, cluster_id, date, shift, device, total_boxes, organic_count, recyclable_count, other_count, ambiguous, annotator_1, annotator_2.")
    ]
    for i, (head, body) in enumerate(points_s7):
        p = tf7.paragraphs[0] if i == 0 else tf7.add_paragraph()
        p.text = f"• {head} {body}"
        p.font.name = "Calibri"
        p.font.size = Pt(10.5)
        p.font.color.rgb = COLOR_TEXT_DARK
        p.space_before = Pt(4)
        p.runs[0].font.bold = True
        p.runs[0].font.color.rgb = COLOR_PRIMARY

    # ==========================================
    # SLIDE 8: Khung Đánh giá, Quản trị Rủi ro & Mốc Go/No-go
    # ==========================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Khung Đánh giá Khoa học, Quản trị Rủi ro & Cột mốc Go/No-go", 8,
               "Ba câu hỏi nghiên cứu, đo lường độ trễ trên phần cứng thật và điểm kiểm soát rủi ro Tuần 5")

    tbox8 = s8.shapes.add_textbox(Inches(0.55), Inches(1.15), Inches(8.85), Inches(4.0))
    tf8 = tbox8.text_frame
    tf8.word_wrap = True

    points_s8 = [
        ("Khung 3 Câu hỏi Nghiên cứu (RQ1 - RQ3):",
         "• RQ1: So sánh hiệu quả giữa các cơ chế KD (Logit KD, DKD, Attention Transfer) và chứng minh lợi thế của cơ chế Fine-to-Coarse.\n• RQ2: Đo lường độ sụt giảm Domain Shift và đường cong thích nghi ít mẫu k-shot ∈ {0, 5, 10, 20, 50} trên khay canteen thật.\n• RQ3 (Mới): Đánh giá phân rã lỗi E4 (Oracle BBox vs Detected BBox) và so sánh Pipeline 2-Stage vs Mô hình YOLO 3 lớp trực tiếp (D2)."),
        ("Đo lường Độ trễ Thực tế trên Thiết bị Biên (Edge Goal):",
         "Cả Detector D1 và KD Classifier đều được xuất sang ONNX và lượng tử hóa INT8 PTQ. Đo độ trễ end-to-end: T_pipeline = T_detector + N * T_classifier trên điện thoại/Raspberry Pi với N ∈ {1, 3, 6} vật rác/khay. Yêu cầu sụt giảm Macro-F1 sau INT8 <= 2.0%."),
        ("Ngân sách Tính toán đã được Kiểm chứng trên RTX 4050 Laptop:",
         "Đã đo pilot thực nghiệm: Detector D1 train đạt 38.45 giây/epoch (1 run 50 epoch chỉ mất 32 phút, VRAM 1.2 GB). Tổng ngân sách tính toán toàn bộ đề tài chỉ ~94-105 giờ GPU, hoàn toàn khả thi trên laptop của nhóm."),
        ("Điểm Kiểm Soát Quyết Định (Go / No-go Milestone cuối Tuần 5):",
         "Nghiệm thu 3 điều kiện tiên quyết: (1) Test Canteen đóng băng >= 150 khay đạt chuẩn Kappa; (2) >= 1 seed Main runs Tier A hoàn tất; (3) Detector D1 đạt Recall@0.5 class-agnostic >= 0.80 trên Val canteen. Nếu không đạt: Cắt giảm các nhánh mở rộng (M5, KD-det) để dồn lực hoàn thiện bài thi chính.")
    ]
    for i, (head, body) in enumerate(points_s8):
        p = tf8.paragraphs[0] if i == 0 else tf8.add_paragraph()
        p.text = f"• {head} {body}"
        p.font.name = "Calibri"
        p.font.size = Pt(10)
        p.font.color.rgb = COLOR_TEXT_DARK
        p.space_before = Pt(3.5)
        p.runs[0].font.bold = True
        p.runs[0].font.color.rgb = COLOR_PRIMARY

    prs.save(output_pptx_path)
    print(f"[THÀNH CÔNG] Đã tạo bài thuyết trình mới tại: {output_pptx_path}")

if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else "Team_Meeting_Data_Collection_v2.pptx"
    create_deck(out_path)
