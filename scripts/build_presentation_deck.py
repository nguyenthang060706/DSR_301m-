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
    blank_layout = prs.slide_layouts[6] # blank layout

    # Color Palette
    COLOR_PRIMARY = RGBColor(0x1F, 0x3A, 0x5F)   # Deep Navy
    COLOR_ACCENT = RGBColor(0x2E, 0x8B, 0x57)    # Sea Green
    COLOR_BG_CARD = RGBColor(0xEA, 0xF1, 0xF6)   # Light Blue/Gray Card
    COLOR_CARD_BORDER = RGBColor(0xD0, 0xDE, 0xE9) # Subtle Card Border
    COLOR_TEXT_DARK = RGBColor(0x22, 0x2B, 0x33) # Charcoal Text
    COLOR_MUTED = RGBColor(0x60, 0x72, 0x82)     # Muted Slate
    COLOR_WHITE = RGBColor(0xFF, 0xFF, 0xFF)     # White

    def add_header(slide, title_text, slide_num, subtitle_text=None):
        # Left Accent Bar
        accent_bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(0.18), Inches(5.625))
        accent_bar.fill.solid()
        accent_bar.fill.fore_color.rgb = COLOR_ACCENT
        accent_bar.line.fill.background()

        # Top Title Box
        title_box = slide.shapes.add_textbox(Inches(0.6), Inches(0.28), Inches(8.6), Inches(0.75))
        tf = title_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.text = title_text
        p.font.name = "Calibri"
        p.font.size = Pt(20)
        p.font.bold = True
        p.font.color.rgb = COLOR_PRIMARY

        if subtitle_text:
            p2 = tf.add_paragraph()
            p2.text = subtitle_text
            p2.font.name = "Calibri"
            p2.font.size = Pt(11)
            p2.font.color.rgb = COLOR_MUTED

        # Slide Number
        num_box = slide.shapes.add_textbox(Inches(9.1), Inches(5.2), Inches(0.6), Inches(0.3))
        tf_num = num_box.text_frame
        tf_num.margin_left = tf_num.margin_top = tf_num.margin_right = tf_num.margin_bottom = 0
        p_num = tf_num.paragraphs[0]
        p_num.text = str(slide_num)
        p_num.font.name = "Calibri"
        p_num.font.size = Pt(10)
        p_num.font.color.rgb = COLOR_MUTED

    # ==========================================
    # SLIDE 1: Title & Overview
    # ==========================================
    s1 = prs.slides.add_slide(blank_layout)
    bar1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(0.25), Inches(5.625))
    bar1.fill.solid()
    bar1.fill.fore_color.rgb = COLOR_ACCENT
    bar1.line.fill.background()

    tbox1 = s1.shapes.add_textbox(Inches(0.8), Inches(1.15), Inches(8.5), Inches(1.3))
    tf1 = tbox1.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "Team Meeting: Data Collection & Protocol"
    p1.font.name = "Calibri"
    p1.font.size = Pt(26)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_PRIMARY

    p1_sub = tf1.add_paragraph()
    p1_sub.text = "Chuẩn hóa nguồn dữ liệu, quy trình gán nhãn và giao thức chia tập thực nghiệm"
    p1_sub.font.name = "Calibri"
    p1_sub.font.size = Pt(14)
    p1_sub.font.color.rgb = COLOR_ACCENT
    p1_sub.space_before = Pt(4)

    sbox1 = s1.shapes.add_textbox(Inches(0.8), Inches(2.65), Inches(8.5), Inches(2.4))
    stf1 = sbox1.text_frame
    stf1.word_wrap = True

    items_s1 = [
        "Đề tài: Knowledge Distillation cho phân loại rác Canteen trên thiết bị biên (Kế hoạch MASTER v9.5).",
        "Trọng tâm báo cáo: Quyết định nguồn dữ liệu, phương pháp chuyển đổi nhãn, phân bổ các tập độc lập và kiểm soát rò rỉ.",
        "Hiện trạng tuần 1-2: Đã đóng băng 15.000 ảnh công khai, tạo bảng mapping 30 lớp sang 3 nhóm và nghiệm thu Cổng Teacher M0/ResNet50 (PASS, RER = 21.60%).",
        "Kế hoạch triển khai: Ban hành docs/label_guide.md, đo Cohen's kappa (κ ≥ 0.70) trên 100 ảnh pilot và thu thập 1.050 ảnh Canteen thực tế."
    ]
    for i, item in enumerate(items_s1):
        p = stf1.paragraphs[0] if i == 0 else stf1.add_paragraph()
        p.text = f"• {item}"
        p.font.name = "Calibri"
        p.font.size = Pt(11.5)
        p.font.color.rgb = COLOR_TEXT_DARK
        p.space_before = Pt(4)

    # ==========================================
    # SLIDE 2: 3 Nhóm quyết định bỏ rác Canteen
    # ==========================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Mục tiêu dữ liệu: 3 nhóm quyết định bỏ rác Canteen", 2,
               "Phân loại dựa trên quyết định bỏ rác thực tế tại Canteen thay vì phân loại vật liệu lý thuyết")

    cards_data = [
        ("0 - Hữu cơ (Organic)", "Thức ăn thừa, cơm mì, bún, vỏ trái cây, bã cà phê, bã trà, vỏ trứng",
         "Mục đích: Ủ compost / chăn nuôi\nYêu cầu: Tuyệt đối không lẫn bao bì nhựa, kim loại hay hóa chất."),
        ("1 - Tái chế khô (Recyclable)", "Chai nhựa PET ráo nước, lon nhôm, hộp sữa tiệt trùng, thùng carton sạch",
         "Mục đích: Ép kiện bán phế liệu\nYêu cầu: Phải ở trạng thái sạch, ráo nước, không dính dầu mỡ thực phẩm."),
        ("2 - Còn lại (Other / Landfill)", "Hộp xốp dính dầu, ly nhựa trà sữa dính đá, muỗng nĩa nhựa, giấy ăn bẩn",
         "Mục đích: Chôn lấp hoặc đốt rác\nYêu cầu: Rác dơ bẩn, vật liệu phức hợp, rác không có giá trị thu hồi.")
    ]
    for i, (title, items, purpose) in enumerate(cards_data):
        left = Inches(0.6 + i * 2.95)
        top = Inches(1.2)
        width = Inches(2.85)
        height = Inches(2.25)

        bg = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = COLOR_BG_CARD
        bg.line.color.rgb = COLOR_CARD_BORDER

        tb = s2.shapes.add_textbox(left + Inches(0.12), top + Inches(0.1), width - Inches(0.24), height - Inches(0.2))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_top = tf.margin_left = tf.margin_right = tf.margin_bottom = 0

        p = tf.paragraphs[0]
        p.text = title
        p.font.name = "Calibri"
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = COLOR_PRIMARY

        p_items = tf.add_paragraph()
        p_items.text = items
        p_items.font.name = "Calibri"
        p_items.font.size = Pt(10)
        p_items.font.color.rgb = COLOR_TEXT_DARK
        p_items.space_before = Pt(4)

        p_purp = tf.add_paragraph()
        p_purp.text = purpose
        p_purp.font.name = "Calibri"
        p_purp.font.size = Pt(9.5)
        p_purp.font.color.rgb = COLOR_MUTED
        p_purp.space_before = Pt(5)

    bot_box = s2.shapes.add_textbox(Inches(0.6), Inches(3.6), Inches(8.8), Inches(1.5))
    btf = bot_box.text_frame
    btf.word_wrap = True
    btf.margin_top = btf.margin_left = btf.margin_right = btf.margin_bottom = 0

    bot_points = [
        "Nguyên tắc cốt lõi: 'Nếu nghi ngờ vật phẩm gây nhiễm bẩn dây chuyền tái chế -> Mặc định xếp vào nhóm Còn lại (other_landfill)'.",
        "Tính thiết thực: Hệ thống thùng rác Canteen tại trường ĐH thiết kế 3 ngăn; người dùng chỉ quyết định bỏ vào 1 trong 3 ngăn.",
        "Yêu cầu dữ liệu: Cần 2 nguồn dữ liệu độc lập: (1) Ảnh công khai quy mô lớn để học biểu diễn đặc trưng, (2) Ảnh Canteen thật để kiểm tra độ khái quát hóa."
    ]
    for i, pt in enumerate(bot_points):
        p = btf.paragraphs[0] if i == 0 else btf.add_paragraph()
        p.text = f"• {pt}"
        p.font.name = "Calibri"
        p.font.size = Pt(10.5)
        p.font.color.rgb = COLOR_TEXT_DARK
        p.space_before = Pt(3)

    # ==========================================
    # SLIDE 3: Chốt Nguồn Dữ liệu & Bản chất Chuyển đổi Nhãn
    # ==========================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Chốt Nguồn dữ liệu & Bản chất Chuyển đổi Nhãn", 3,
               "Đóng băng nguồn huấn luyện công khai, giải thích chuyển đổi từ TrashNet và chiến lược nhãn mịn")

    tbox3 = s3.shapes.add_textbox(Inches(0.6), Inches(1.15), Inches(8.8), Inches(4.0))
    tf3 = tbox3.text_frame
    tf3.word_wrap = True
    tf3.margin_top = tf3.margin_left = tf3.margin_right = tf3.margin_bottom = 0

    points_s3 = [
        ("Chốt Bộ dữ liệu Công khai Chính thức (Khóa cứng):",
         "Chọn Recyclable and Household Waste Classification (Alistair King, Kaggle). Chính xác 15.000 ảnh, 30 lớp mịn (500 ảnh/lớp, gồm cả ảnh studio và đời thực). Bỏ hoàn toàn phương án dự phòng 'TrashBusters' để đảm bảo tính nhất quán."),
        ("Dữ liệu Ngoại lai (External / Zero-shot OOD):",
         "TACO (crop theo bounding box) và RealWaste chỉ dùng để kiểm tra độ bền vững ngoài phân phối (out-of-distribution), tuyệt đối KHÔNG đưa vào tập huấn luyện."),
        ("Lý do Chuyển từ TrashNet 6 lớp (Tuần 1) sang 3 nhóm Canteen:",
         "TrashNet (2.527 ảnh) chỉ phân loại vật liệu lý thuyết trên phông trắng, hoàn toàn không có thức ăn thừa/hữu cơ và không phản ánh độ bẩn thực tế. Canteen thực tế vận hành theo 3 luồng xử lý."),
        ("Chiến lược 'Huấn luyện Nhãn mịn 30 lớp -> Gộp 3 nhóm quyết định':",
         "Mô hình học đặc trưng phân biệt sắc nét của 30 lớp mịn; khi suy luận, cộng dồn xác suất Softmax theo label_map.csv v2.0. Cách này bảo toàn tri thức tối ('Dark Knowledge') cho Knowledge Distillation, vượt trội so với học thẳng 3 lớp (entropy quá hẹp)."),
        ("Xác nhận Baseline M0 & Teacher:",
         "Baseline M0 (ResNet18) và Teacher (ResNet50) đã được huấn luyện lại hoàn toàn trên giao thức 30 lớp -> 3 nhóm mới (MASTER v9.5). Kết quả: Teacher Canteen F1 = 98.06%, Student M0 = 96.69%, đạt RER = 21.60% >= 20% (PASS cổng Teacher).")
    ]
    for i, (head, body) in enumerate(points_s3):
        p = tf3.paragraphs[0] if i == 0 else tf3.add_paragraph()
        p.text = f"• {head} {body}"
        p.font.name = "Calibri"
        p.font.size = Pt(10.5)
        p.font.color.rgb = COLOR_TEXT_DARK
        p.space_before = Pt(4)
        p.runs[0].font.bold = True
        p.runs[0].font.color.rgb = COLOR_PRIMARY

    # ==========================================
    # SLIDE 4: Quy trình Ánh xạ Nhãn & Ca Ranh giới
    # ==========================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Quy trình Ánh xạ Nhãn & Xử lý Ca Ranh giới (Edge Cases)", 4,
               "Bảng ánh xạ chuẩn hóa label_map.csv v2.0 và quy tắc phân loại rác phức tạp tại Canteen")

    rows, cols = 4, 4
    table_shape = s4.shapes.add_table(rows, cols, Inches(0.6), Inches(1.15), Inches(8.8), Inches(1.6))
    table = table_shape.table
    table.columns[0].width = Inches(1.8)
    table.columns[1].width = Inches(2.7)
    table.columns[2].width = Inches(1.5)
    table.columns[3].width = Inches(2.8)

    headers = ["Nhóm Quyết định (3)", "Các lớp mịn đại diện (30 lớp)", "Số lượng ảnh", "Quy tắc Canteen"]
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

    table_data = [
        ("0 - Hữu cơ (organic)", "food_waste, coffee_grounds, eggshells, tea_bags (4 lớp)", "2.000 (13.3%)", "Thực phẩm, đồ ăn thừa, hữu cơ tự nhiên"),
        ("1 - Tái chế khô (recyclable)", "plastic_water_bottles, aluminum_soda_cans, cardboard_boxes, newspaper, glass_bottles (16 lớp)", "8.000 (53.3%)", "Vật liệu khô, ráo nước, có giá trị thu hồi"),
        ("2 - Còn lại (other_landfill)", "styrofoam_containers, paper_cups, cutlery, plastic_straws, shopping_bags (10 lớp)", "5.000 (33.3%)", "Hộp xốp bẩn, ly giấy dính sáp, nhựa 1 lần")
    ]
    for i, row in enumerate(table_data):
        for j, val in enumerate(row):
            cell = table.cell(i + 1, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = COLOR_BG_CARD if i % 2 == 0 else COLOR_WHITE
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.name = "Calibri"
            p.font.size = Pt(9.5)
            p.font.color.rgb = COLOR_TEXT_DARK
            if j == 2:
                p.alignment = PP_ALIGN.CENTER

    edge_box = s4.shapes.add_textbox(Inches(0.6), Inches(2.95), Inches(8.8), Inches(2.2))
    etf = edge_box.text_frame
    etf.word_wrap = True
    etf.margin_top = etf.margin_left = etf.margin_right = etf.margin_bottom = 0

    edge_points = [
        ("Ly trà sữa / nước ngọt còn đá hoặc chất lỏng:", "Mặc định gán Còn lại (other_landfill) vì chất lỏng làm hỏng toàn bộ giấy/nhựa sạch trong thùng tái chế. Đồng thời gắn cờ ambiguous = True."),
        ("Hộp xốp thực phẩm (cơm tấm, xôi):", "Dù sạch hay dính mỡ đều gán Còn lại (canteen trường không có đơn vị thu gom tái chế hạt xốp EPS)."),
        ("Chai nhựa có nắp và nhãn mác:", "Vẫn gán Tái chế khô nếu chai đã uống cạn và ráo nước."),
        ("Đồ dùng ăn uống 1 lần (muỗng nĩa nhựa, que khuấy, ống hút):", "Gán Còn lại (nhựa giòn, giá trị tái chế thấp, dính dầu mỡ)."),
        ("Tính cố định:", "Bảng label_map.csv v2.0 được đóng băng cố định trước mọi thử nghiệm, không tùy tiện thay đổi giữa chừng.")
    ]
    for i, (case, rule) in enumerate(edge_points):
        p = etf.paragraphs[0] if i == 0 else etf.add_paragraph()
        p.text = f"• {case} {rule}"
        p.font.name = "Calibri"
        p.font.size = Pt(10)
        p.font.color.rgb = COLOR_TEXT_DARK
        p.space_before = Pt(3)
        p.runs[0].font.bold = True
        p.runs[0].font.color.rgb = COLOR_PRIMARY

    # ==========================================
    # SLIDE 5: Phân bố & Thiết kế Chia tập Dữ liệu
    # ==========================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Phân bố Dữ liệu & Thiết kế Chia tập Độc lập", 5,
               "Số liệu thực tế được kiểm chứng, phân biệt rạch ròi vai trò từng tập dữ liệu")

    stat_cards = [
        ("Dữ liệu Công khai (Kaggle)", "15.000 ảnh (30 lớp)",
         "• Train (85%): 12.750 ảnh (cập nhật trọng số)\n• Dev/Holdout (15%): 2.250 ảnh (chọn model, tune)\n• Tỷ lệ lớp: Hữu cơ 13.3% (2.000) | Tái chế khô 53.3% (8.000) | Còn lại 33.3% (5.000)\n• Tỷ lệ 1:4:2.5 phản ánh đúng phân bố 4 : 16 : 10 lớp gốc"),
        ("Dữ liệu Canteen Thật", "Mục tiêu 1.050 ảnh",
         "• Test Canteen (Cố định): 600 ảnh (200/nhóm)\n• Few-shot Pool: 300 ảnh (100/nhóm, k-shot)\n• Validation Canteen: 150 ảnh (50/nhóm, INT8)\n• Chụp tại Canteen trường ĐH, cân bằng 3 nhóm, bảo đảm tính độc lập hoàn toàn giữa 3 tập"),
        ("Phân định Rạch ròi các Tập", "Nguyên tắc Khoa học",
         "• Train: Cập nhật trọng số mô hình\n• Dev/Holdout: Chọn mô hình & tune tham số\n• Test Canteen: Đánh giá domain transfer cuối cùng\n• Few-shot Pool: Thích nghi ít mẫu k-shot\n• TACO/RealWaste: Đo lường độ bền OOD")
    ]
    for i, (title, subtitle, content) in enumerate(stat_cards):
        left = Inches(0.6 + i * 2.95)
        top = Inches(1.15)
        width = Inches(2.85)
        height = Inches(2.4)

        bg = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = COLOR_BG_CARD
        bg.line.color.rgb = COLOR_CARD_BORDER

        tb = s5.shapes.add_textbox(left + Inches(0.12), top + Inches(0.1), width - Inches(0.24), height - Inches(0.2))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_top = tf.margin_left = tf.margin_right = tf.margin_bottom = 0

        p = tf.paragraphs[0]
        p.text = title
        p.font.name = "Calibri"
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = COLOR_PRIMARY

        p_sub = tf.add_paragraph()
        p_sub.text = subtitle
        p_sub.font.name = "Calibri"
        p_sub.font.size = Pt(10)
        p_sub.font.bold = True
        p_sub.font.color.rgb = COLOR_ACCENT
        p_sub.space_before = Pt(2)

        p_c = tf.add_paragraph()
        p_c.text = content
        p_c.font.name = "Calibri"
        p_c.font.size = Pt(9.5)
        p_c.font.color.rgb = COLOR_TEXT_DARK
        p_c.space_before = Pt(4)

    bot5 = s5.shapes.add_textbox(Inches(0.6), Inches(3.7), Inches(8.8), Inches(1.5))
    btf5 = bot5.text_frame
    btf5.word_wrap = True
    btf5.margin_top = btf5.margin_left = btf5.margin_right = btf5.margin_bottom = 0

    points_s5 = [
        ("Làm rõ con số 13.3% / 53.3% / 33.3%:", "Đây là tỷ lệ thực tế tính từ 30 lớp x 500 ảnh của dataset công khai (4 lớp Hữu cơ, 16 lớp Tái chế, 10 lớp Còn lại), không phải ước lượng cảm tính."),
        ("Xử lý mất cân bằng lớp:", "Do tập công khai mất cân bằng 1 : 4 : 2.5, chỉ số đánh giá bắt buộc dùng Macro-F1 và Balanced Accuracy (tránh bị chi phối bởi lớp Tái chế khô)."),
        ("Tách biệt hoàn toàn:", "Tập Test Canteen (600 ảnh) được đóng băng riêng biệt, tuyệt đối không dùng để tune siêu tham số hay huấn luyện.")
    ]
    for i, (hd, pt) in enumerate(points_s5):
        p = btf5.paragraphs[0] if i == 0 else btf5.add_paragraph()
        p.text = f"• {hd} {pt}"
        p.font.name = "Calibri"
        p.font.size = Pt(10.5)
        p.font.color.rgb = COLOR_TEXT_DARK
        p.space_before = Pt(3)
        p.runs[0].font.bold = True
        p.runs[0].font.color.rgb = COLOR_PRIMARY

    # ==========================================
    # SLIDE 6: Quy ước Gán nhãn & Đảm bảo Chất lượng
    # ==========================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Quy ước Gán nhãn & Kiểm định Chất lượng (Quality Assurance)", 6,
               "Giao thức văn bản docs/label_guide.md, kiểm định Cohen's kappa và quản lý ảnh mơ hồ")

    qa_metrics = [
        ("κ ≥ 0,70", "Cohen's Kappa mục tiêu", "Độ đồng thuận giữa 2 annotator độc lập trên 100 ảnh pilot đầu tiên."),
        ("100 ảnh", "Mẫu thử nghiệm Tuần 1-2", "Gán nhãn độc lập hoàn toàn, không thảo luận trước khi tính toán."),
        ("ambiguous", "Cờ cho ảnh ranh giới", "Đánh giá riêng độ bền mô hình; KHÔNG dùng để train/tune mô hình.")
    ]
    for i, (m_val, m_label, m_desc) in enumerate(qa_metrics):
        left = Inches(0.6 + i * 2.95)
        top = Inches(1.15)
        width = Inches(2.85)
        height = Inches(1.3)

        bg = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = COLOR_BG_CARD
        bg.line.color.rgb = COLOR_CARD_BORDER

        tb = s6.shapes.add_textbox(left + Inches(0.12), top + Inches(0.08), width - Inches(0.24), height - Inches(0.16))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_top = tf.margin_left = tf.margin_right = tf.margin_bottom = 0

        p = tf.paragraphs[0]
        p.text = m_val
        p.font.name = "Calibri"
        p.font.size = Pt(20)
        p.font.bold = True
        p.font.color.rgb = COLOR_PRIMARY
        p.alignment = PP_ALIGN.CENTER

        p_lbl = tf.add_paragraph()
        p_lbl.text = m_label
        p_lbl.font.name = "Calibri"
        p_lbl.font.size = Pt(10.5)
        p_lbl.font.bold = True
        p_lbl.font.color.rgb = COLOR_ACCENT
        p_lbl.alignment = PP_ALIGN.CENTER
        p_lbl.space_before = Pt(1)

        p_desc = tf.add_paragraph()
        p_desc.text = m_desc
        p_desc.font.name = "Calibri"
        p_desc.font.size = Pt(9)
        p_desc.font.color.rgb = COLOR_TEXT_DARK
        p_desc.alignment = PP_ALIGN.CENTER
        p_desc.space_before = Pt(2)

    bot6 = s6.shapes.add_textbox(Inches(0.6), Inches(2.6), Inches(8.8), Inches(2.55))
    btf6 = bot6.text_frame
    btf6.word_wrap = True
    btf6.margin_top = btf6.margin_left = btf6.margin_right = btf6.margin_bottom = 0

    qa_protocol = [
        ("Văn bản Quy chuẩn:", "Đã ban hành docs/label_guide.md xác lập quy tắc gán nhãn, định nghĩa vật phẩm và phân loại ca ranh giới."),
        ("Quy trình Kiểm định Độ đồng thuận:", "Lấy mẫu ngẫu nhiên 100 ảnh canteen đầu tiên. Hai người gán nhãn độc lập hoàn toàn. Tính hệ số Cohen's kappa: κ = (p_o - p_e) / (1 - p_e)."),
        ("Xử lý khi κ < 0.70:", "Nếu không đạt, dừng chụp diện rộng; mở họp đối chiếu ma trận nhầm lẫn, sửa label_guide.md và gán lại mẫu mới."),
        ("Trọng tài Giải quyết Bất đồng:", "Các trường hợp tranh chấp sau thảo luận sẽ do Lead nghiên cứu quyết định và ghi rõ lý do vào metadata."),
        ("Quản lý Ảnh Mơ hồ (Ambiguous):", "Ảnh ranh giới được gắn cờ ambiguous = True; chỉ dùng để kiểm tra độ nhạy mô hình, tuyệt đối KHÔNG dùng để train hay tune.")
    ]
    for i, (title, desc) in enumerate(qa_protocol):
        p = btf6.paragraphs[0] if i == 0 else btf6.add_paragraph()
        p.text = f"• {title} {desc}"
        p.font.name = "Calibri"
        p.font.size = Pt(9.5)
        p.font.color.rgb = COLOR_TEXT_DARK
        p.space_before = Pt(3)
        p.runs[0].font.bold = True
        p.runs[0].font.color.rgb = COLOR_PRIMARY

    # ==========================================
    # SLIDE 7: Thu thập Ảnh Canteen & Chống Rò rỉ Dữ liệu
    # ==========================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Thu thập Ảnh Canteen & Giao thức Chống Rò rỉ Dữ liệu", 7,
               "Thiết lập chụp thực tế chuẩn hóa, phân cụm cluster_id và phát hiện trùng lặp đa tầng")

    tbox7 = s7.shapes.add_textbox(Inches(0.6), Inches(1.15), Inches(8.8), Inches(4.0))
    tf7 = tbox7.text_frame
    tf7.word_wrap = True
    tf7.margin_top = tf7.margin_left = tf7.margin_right = tf7.margin_bottom = 0

    points_s7 = [
        ("Giao thức Thu thập Thực tế (Acquisition Setup):",
         "Góc máy top-down vuông góc (90 độ) mô phỏng camera gắn trên nắp thùng rác thông minh. Đặt trên nền khay ăn hoặc mặt bàn Canteen đồng nhất, 1 vật phẩm chính mỗi ảnh."),
        ("Biến thiên có Kiểm soát:",
         "Chụp với 3 điều kiện ánh sáng (sáng tự nhiên, đèn canteen, góc khuất), 4 góc xoay vật thể (0, 90, 180, 270 độ), và 2 trạng thái hình thái (nguyên vẹn, bẹp méo/dính bẩn)."),
        ("Kiểm soát Rò rỉ Bằng Cụm (Cluster-based Splitting):",
         "Mỗi ảnh được gán mã cụm định danh: cluster_id = YYYYMMDD_Ca_LoaiVat (Ví dụ: 20261002_trua_chaipet). Toàn bộ ảnh cùng vật phẩm/buổi chụp BẮT BUỘC chỉ nằm trọn trong 1 tập (Val, Test, hoặc Few-shot), không bao giờ xuất hiện ở 2 tập khác nhau."),
        ("Lọc Trùng Lặp Hai Tầng (Deduplication Pipeline):",
         "(1) MD5 Checksum: Loại bỏ 100% ảnh trùng lặp file vật lý. (2) Perceptual Hash (pHash): Bắt các ảnh gần như giống hệt nhau (khoảng cách Hamming <= 5) do chụp liên tiếp hoặc crop nhẹ."),
        ("Cấu trúc Metadata Đầy đủ (canteen_metadata.csv):",
         "Ghi nhận chính xác: image_id, cluster_id, date, shift, device, fine_label, coarse_label, is_clean, is_distorted, ambiguous, annotator_1, annotator_2, final_label.")
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
    # SLIDE 8: Mục tiêu Đánh giá, Quản trị Rủi ro & Kế hoạch
    # ==========================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Mục tiêu Đánh giá, Quản trị Rủi ro & Kế hoạch Hành động", 8,
               "Khung câu hỏi nghiên cứu, tiêu chuẩn đo lường biên và mốc Go/No-go Tuần 5")

    tbox8 = s8.shapes.add_textbox(Inches(0.6), Inches(1.15), Inches(8.8), Inches(4.0))
    tf8 = tbox8.text_frame
    tf8.word_wrap = True
    tf8.margin_top = tf8.margin_left = tf8.margin_right = tf8.margin_bottom = 0

    points_s8 = [
        ("Mục tiêu Khoa học & Câu hỏi Nghiên cứu:",
         "RQ1: So sánh hiệu quả giữa các cơ chế KD (Hinton logit, DKD, Attention Transfer) và chứng minh lợi thế của Fine-to-Coarse (30 lớp -> 3 nhóm). RQ2: Đo lường mức độ suy giảm do Domain Shift và xây dựng đường cong thích nghi ít mẫu k-shot ∈ {0, 5, 10, 20, 50}."),
        ("Hệ Chỉ số Đánh giá Toàn diện:",
         "Macro-F1 (chỉ số chính chống lệch lớp), Balanced Accuracy, Recall từng nhóm và Critical Error (tỷ lệ nhầm lẫn nghiêm trọng giữa Hữu cơ <-> Tái chế khô làm hỏng mẻ tái chế)."),
        ("Tiêu chuẩn Đánh giá Biên (Edge Goal):",
         "Mô hình nén INT8 PTQ qua ONNX Runtime / TFLite chạy trên điện thoại Android / Raspberry Pi; yêu cầu độ tụt Macro-F1 <= 2.0%, đo lặp 200 lần với 50 lần warm-up, báo cáo Median / P95 Latency."),
        ("Quản trị Rủi ro Thực tế:",
         "Nếu Canteen từ chối cho chụp: Xin phép chính thức ngay tuần 1; dự phòng bằng cách mua đồ ăn Canteen về phòng lab dựng góc chụp mô phỏng. Nếu kích thước mẫu nhỏ: Dùng Cluster Bootstrap theo cụm (B = 2.000) để tính khoảng tin cậy 95% CI trung thực."),
        ("Kế hoạch & Mốc Go/No-go Tuần 5:",
         "Tuần 1-2: Hoàn tất pilot 100 ảnh, đạt κ >= 0.70. Tuần 3-4: Thu thập diện rộng. Cuối Tuần 5 [Go/No-go]: Đóng băng Test Canteen >= 600 ảnh, khóa pool ảnh cốt lõi. Nếu trễ, kích hoạt phương án cắt giảm nhóm mở rộng M7/Canteen-C.")
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

    # Save presentation
    prs.save(output_pptx_path)
    print(f"Presentation successfully created at: {output_pptx_path}")

if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else "Team_Meeting_Data_Collection.pptx"
    create_deck(out_path)
