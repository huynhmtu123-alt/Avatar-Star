import cv2
import numpy as np
import os
from pathlib import Path

# HSV range màu xanh đồng minh
ALLY_GREEN_LOWER = np.array((40, 50, 50))  # Mở rộng dải màu một chút để bắt chính xác hơn
ALLY_GREEN_UPPER = np.array((90, 255, 255))

def refine_labels():
    print("==================================================")
    print("🛡️ REFINING DATASET: LOẠI BỎ ĐỒNG MINH (ADMIN THÓC...)")
    print("==================================================")
    
    img_dir = Path("dataset/images/train")
    lbl_dir = Path("dataset/labels/train")
    
    if not img_dir.exists() or not lbl_dir.exists():
        print("[LỖI] Không tìm thấy thư mục dataset")
        return

    lbl_files = list(lbl_dir.glob("*.txt"))
    removed_count = 0
    total_files = len(lbl_files)
    
    for i, lbl_path in enumerate(lbl_files):
        img_path = img_dir / (lbl_path.stem + ".jpg")
        if not img_path.exists():
            continue
            
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        h_img, w_img = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, ALLY_GREEN_LOWER, ALLY_GREEN_UPPER)
        
        with open(lbl_path, 'r') as f:
            lines = f.readlines()
            
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                _, x_c, y_c, w, h = map(float, parts)
                
                # Xác định vùng kiểm tra màu xanh (phía trên đầu nhân vật)
                # Head position in pixels
                cx, cy = int(x_c * w_img), int(y_c * h_img)
                bw, bh = int(w * w_img), int(h * h_img)
                
                # Kiểm tra vùng phía trên bounding box (nơi thường có tên/máu xanh)
                check_y1 = max(0, int((y_c - h/2 - 0.1) * h_img)) # Cao hơn đầu 10%
                check_y2 = int((y_c - h/4) * h_img)             # Đến giữa đầu
                check_x1 = max(0, int((x_c - w/2) * w_img))
                check_x2 = min(w_img, int((x_c + w/2) * w_img))
                
                roi_mask = mask[check_y1:check_y2, check_x1:check_x2]
                
                # Nếu có màu xanh trong vùng này -> Đồng đội
                if roi_mask.size > 0 and np.sum(roi_mask) > 100: # Có ít nhất vài chục pixel xanh
                    removed_count += 1
                else:
                    new_lines.append(line)
        
        # Ghi lại file label sạch
        with open(lbl_path, 'w') as f:
            f.writelines(new_lines)
            
        if (i+1) % 100 == 0:
            print(f"[*] Tiến độ: {i+1}/{total_files} file | Đã loại bỏ {removed_count} nhãn đồng đội")

    print(f"\n[✓] Hoàn tất! Đã loại bỏ tổng cộng {removed_count} nhãn đồng đội.")
    print("[*] Dataset của bạn hiện tại chỉ còn lại kẻ địch thực sự.")

if __name__ == "__main__":
    refine_labels()
