"""scripts/auto_prelabel_canteen.py

Script hỗ trợ gán nhãn tự động (Model-Assisted Pre-labeling) cho ảnh khay canteen.
Sử dụng mô hình Detector D1 đã huấn luyện trên TACO để tự động phát hiện và sinh các
bounding box ban đầu (chuẩn YOLO txt) cho ảnh mới chụp tại canteen.

Thành viên gán nhãn sau đó chỉ cần mở công cụ (LabelImg / Roboflow / CVAT) để kiểm tra,
tinh chỉnh lại vị trí hoặc thêm bớt thay vì phải vẽ thủ công từ đầu (tiết kiệm ~70% thời gian).
"""

import sys
import io
import argparse
from pathlib import Path
from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description="Tự động sinh box gợi ý cho ảnh khay canteen")
    parser.add_argument("--model", type=str, default="runs/detect/runs/detect/pilot_d1_taco/weights/best.pt", help="Đường dẫn checkpoint detector")
    parser.add_argument("--source", type=str, default="data/raw/canteen_trays", help="Thư mục chứa ảnh khay mới chụp")
    parser.add_argument("--out_labels", type=str, default="data/raw/canteen_prelabels", help="Thư mục lưu nhãn gợi ý txt")
    parser.add_argument("--conf", type=float, default=0.25, help="Ngưỡng confidence tối thiểu")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[LỖI] Chưa cài đặt thư viện ultralytics!")
        sys.exit(1)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[LỖI] Không tìm thấy checkpoint: {model_path}")
        sys.exit(1)

    src_dir = Path(args.source)
    if not src_dir.exists():
        print(f"[LỖI] Không tìm thấy thư mục nguồn: {src_dir}")
        sys.exit(1)

    img_paths = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png")) + list(src_dir.glob("*.jpeg"))
    if not img_paths:
        print(f"[*] Thư mục {src_dir} hiện chưa có ảnh mới nào.")
        print(f"[*] Hướng dẫn: Khi đi chụp ảnh canteen về, copy ảnh vào {src_dir} rồi chạy lại script này.")
        return

    out_dir = Path(args.out_labels)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"PRE-LABELING HỖ TRỢ GÁN NHÃN KHAY CANTEEN")
    print(f"Mô hình sử dụng:  {model_path}")
    print(f"Số lượng ảnh nạp: {len(img_paths)}")
    print(f"Ngưỡng tin cậy:   {args.conf}")
    print(f"Thư mục nhãn xuất: {out_dir}")
    print("=" * 65)

    model = YOLO(str(model_path))
    results = model.predict(source=str(src_dir), conf=args.conf, save=False, verbose=False)

    total_boxes = 0
    for r in results:
        img_name = Path(r.path).stem
        lbl_file = out_dir / f"{img_name}.txt"
        
        boxes_txt = []
        for box in r.boxes:
            # YOLO normalized xywh
            cls_id = int(box.cls[0].item())
            # Luôn đưa về class 0 (waste) cho detector agnostic
            xywhn = box.xywhn[0].tolist()
            xc, yc, w, h = xywhn
            boxes_txt.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        with open(lbl_file, "w", encoding="utf-8") as f:
            if boxes_txt:
                f.write("\n".join(boxes_txt) + "\n")
            else:
                f.write("") # file rỗng nếu không có rác
        total_boxes += len(boxes_txt)

    print(f"[THÀNH CÔNG] Đã sinh xong gợi ý cho {len(img_paths)} ảnh (Tổng cộng {total_boxes} boxes).")
    print(f"[*] Các file nhãn .txt đã sẵn sàng tại: {out_dir}")

if __name__ == "__main__":
    main()
