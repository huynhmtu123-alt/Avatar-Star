import os
import cv2
import numpy as np
from pathlib import Path

def auto_label():
    print("==================================================")
    print("🤖 AUTO-LABELING BẰNG AI (HYBRID MODE)")
    print("Dùng YOLOv8 + Color Check vùng tâm súng")
    print("==================================================")
    
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[LỖI] Cần cài đặt ultralytics: pip install ultralytics")
        return

    print("[*] Đang tải model YOLOv8n cơ bản...")
    model = YOLO('yolov8n.pt') 
    
    img_dir = Path("dataset/images/train")
    label_dir = Path("dataset/labels/train")
    label_dir.mkdir(parents=True, exist_ok=True)
    
    if not img_dir.exists():
        print(f"[LỖI] Không tìm thấy thư mục {img_dir}")
        return

    model_path = 'models/avatarstar.pt'
    if os.path.exists(model_path):
        print(f"[*] Kết hợp thêm model xịn {model_path} để bắt địch chuẩn hơn...")
        custom_model = YOLO(model_path)
    else:
        custom_model = None
    
    image_files = list(img_dir.glob("*.jpg"))
    total = len(image_files)
    
    if total == 0:
        print("[!] Không có ảnh nào trong thư mục.")
        return
        
    print(f"[*] Bắt đầu quét {total} ảnh...")
    count_labeled = 0
    
    for i, img_path in enumerate(image_files):
        # 1. Dùng model mặc định (Zero-shot)
        results = model(img_path, verbose=False, classes=[0], conf=0.25)
        boxes = results[0].boxes.xywhn.tolist()
        
        # 2. Dùng model custom (nếu có) để bổ trợ
        if custom_model:
            c_results = custom_model(img_path, verbose=False, conf=0.3)
            c_boxes = c_results[0].boxes.xywhn.tolist()
            boxes.extend(c_boxes)
            
        # 3. CƯỜNG HÓA TÂM (Center Reinforcement)
        img_cv = cv2.imread(str(img_path))
        if img_cv is not None:
            h_img, w_img = img_cv.shape[:2]
            center_x, center_y = w_img // 2, h_img // 2
            
            # Kiểm tra màu đỏ trong vùng tâm (tên địch, thanh máu)
            hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
            lower_red1 = np.array([0, 100, 100]); upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 100]); upper_red2 = np.array([180, 255, 255])
            mask_red = cv2.addWeighted(cv2.inRange(hsv, lower_red1, upper_red1), 1.0, 
                                       cv2.inRange(hsv, lower_red2, upper_red2), 1.0, 0.0)
            
            # Quét vùng tâm súng
            roi_center = mask_red[center_y-60:center_y+60, center_x-60:center_x+60]
            if np.sum(roi_center) > 400: # Có màu đỏ ở tâm
                has_center_box = any(abs(b[0]-0.5) < 0.12 and abs(b[1]-0.5) < 0.12 for b in boxes)
                if not has_center_box:
                    # Thêm một box vào chính xác vị trí có màu đỏ (hoặc đơn giản là tâm súng)
                    boxes.append([0.5, 0.5, 0.06, 0.12])
        
        if len(boxes) > 0:
            label_path = label_dir / (img_path.stem + ".txt")
            with open(label_path, 'w') as f:
                for box in boxes:
                    x_c, y_c, w, h = box
                    f.write(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
            count_labeled += 1
            
        print(f"\r[*] Tiến độ: {i+1}/{total} ảnh | Đã dán nhãn: {count_labeled}", end="")
        
    print(f"\n[✓] Hoàn tất! Đã dán nhãn được {count_labeled}/{total} ảnh.")

if __name__ == "__main__":
    auto_label()
