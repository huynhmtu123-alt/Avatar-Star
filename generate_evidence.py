import cv2
import os
from pathlib import Path
import random

def draw_more_evidence():
    print("[*] Drawing more refined labels as evidence...")
    img_dir = Path("dataset/images/train")
    lbl_dir = Path("dataset/labels/train")
    
    # Lấy danh sách các file label có dữ liệu (kẻ địch)
    lbl_files = [f for f in lbl_dir.glob("*.txt") if os.path.getsize(f) > 0]
    
    # Ưu tiên lấy từ các video mới
    new_lbl_files = [f for f in lbl_files if "Admin Thóc" in f.name or "Highlight Live" in f.name]
    
    if not new_lbl_files:
        selected_lbls = random.sample(lbl_files, min(5, len(lbl_files)))
    else:
        selected_lbls = random.sample(new_lbl_files, min(5, len(new_lbl_files)))
        
    for i, lbl_path in enumerate(selected_lbls):
        img_path = img_dir / (lbl_path.stem + ".jpg")
        if not img_path.exists():
            continue
            
        img = cv2.imread(str(img_path))
        h_img, w_img = img.shape[:2]
        
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    _, x_c, y_c, w, h = map(float, parts)
                    
                    x1 = int((x_c - w/2) * w_img)
                    y1 = int((y_c - h/2) * h_img)
                    x2 = int((x_c + w/2) * w_img)
                    y2 = int((y_c + h/2) * h_img)
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2) # Màu xanh lá cho nhãn đã refine
                    cv2.putText(img, "CLEAN_ENEMY", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        output_path = f"evidence_refined_{i}.jpg"
        cv2.imwrite(output_path, img)
        print(f"[✓] Saved refined evidence to {output_path}")

if __name__ == "__main__":
    draw_more_evidence()
