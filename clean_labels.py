import os
from pathlib import Path

def clean_labels():
    print("==================================================")
    print("🧹 DỌN DẸP DATASET: TỰ ĐỘNG XÓA NHÃN CỦA BẢN THÂN")
    print("==================================================")
    
    cleaned_count = 0
    removed_boxes = 0
    
    dirs_to_clean = [Path("dataset/labels/train"), Path("dataset/labels/val")]
    files = []
    for d in dirs_to_clean:
        if d.exists():
            files.extend(list(d.glob("*.txt")))
    
    for txt_file in files:
        with open(txt_file, "r") as f:
            lines = f.readlines()
            
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                cls, x_c, y_c, w, h = map(float, parts)
                
                # Logic nhận diện nhân vật của bản thân trong game TPS (Góc nhìn thứ 3):
                # 1. Nằm ở khoảng giữa màn hình theo chiều ngang (x từ 0.35 đến 0.65)
                # 2. Nằm ở nửa dưới màn hình (y > 0.45)
                # 3. Kích thước khá to vì đứng gần camera (width > 0.08, height > 0.15)
                is_player = (0.35 < x_c < 0.65) and (y_c > 0.45) and (w > 0.08) and (h > 0.15)
                
                if is_player:
                    removed_boxes += 1
                else:
                    new_lines.append(line)
                    
        # Ghi đè lại file label
        with open(txt_file, "w") as f:
            f.writelines(new_lines)
            
        if len(lines) != len(new_lines):
            cleaned_count += 1
            
    print(f"[✓] Xong! Đã quét lại {len(files)} file.")
    print(f"[*] Đã xóa thành công {removed_boxes} Bounding Box khoanh nhầm vào nhân vật của bạn.")
    print("[*] Số file được chỉnh sửa:", cleaned_count)

if __name__ == "__main__":
    clean_labels()
