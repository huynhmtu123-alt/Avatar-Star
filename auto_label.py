import os
import cv2
from pathlib import Path

def auto_label():
    print("==================================================")
    print("🤖 AUTO-LABELING BẰNG AI (ZERO-SHOT)")
    print("Dùng model YOLOv8 pre-trained để nhận diện nhân vật")
    print("==================================================")
    
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[LỖI] Cần cài đặt ultralytics: pip install ultralytics")
        return

    print("[*] Đang tải model YOLOv8n cơ bản...")
    model = YOLO('yolov8n.pt') 
    
    img_dir = Path("dataset/images/train")
    if not img_dir.exists():
        print(f"[LỖI] Không tìm thấy thư mục {img_dir}")
        return

    label_dir = Path("dataset/labels/train")
    label_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = list(img_dir.glob("*.jpg"))
    total = len(image_files)
    
    if total == 0:
        print("[!] Không có ảnh nào trong thư mục.")
        return
        
    print(f"[*] Bắt đầu quét {total} ảnh...")
    count_labeled = 0
    
    for i, img_path in enumerate(image_files):
        # Nhận diện, chỉ lấy class 0 (Person), conf 0.25 để lấy nhiều nhất có thể
        results = model(img_path, verbose=False, classes=[0], conf=0.25)
        
        # Lấy box (nếu có)
        boxes = results[0].boxes
        if len(boxes) > 0:
            label_path = label_dir / (img_path.stem + ".txt")
            with open(label_path, 'w') as f:
                for box in boxes:
                    # Lấy tọa độ chuẩn YOLO (Normalized center x, center y, width, height)
                    x_c, y_c, w, h = box.xywhn[0]
                    # Ép tất cả thành class 0 (kẻ địch)
                    f.write(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
            count_labeled += 1
            
        print(f"\r[*] Tiến độ: {i+1}/{total} ảnh | Đã sinh ra {count_labeled} file nhãn", end="")
        
    print(f"\n[✓] Hoàn tất! Đã tự động khoanh vùng được {count_labeled}/{total} ảnh.")
    print(f"[*] Dữ liệu nhãn được lưu tại: {label_dir}")

if __name__ == "__main__":
    auto_label()
