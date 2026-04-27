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
                
                # Logic nhận diện nhân vật của bản thân và UI rác:
                # 1. Nhân vật bản thân (Góc nhìn thứ 3):
                is_player = (0.35 < x_c < 0.65) and (y_c > 0.45) and (w > 0.08) and (h > 0.15)
                
                # 2. UI rác (Skill bar, Minimap, Buffs):
                is_skill_bar = (y_c > 0.85) # Vùng dưới cùng màn hình
                is_minimap   = (x_c > 0.8) and (y_c < 0.3) # Góc trên bên phải
                is_buffs     = (x_c < 0.2) and (y_c < 0.2) # Góc trên bên trái
                
                # 3. Vật thể rác không phải hình người (Cái lu, tảng đá, hòm):
                # Kẻ địch thường cao gầy (ratio height/width > 1.2)
                aspect_ratio = h / w if w > 0 else 0
                is_not_humanoid = (aspect_ratio < 1.1) # Quá vuông hoặc nằm ngang
                
                # 4. Vật thể quá mỏng (Cửa, cột, thanh sắt):
                # Kẻ địch phải có độ rộng tối thiểu (width > 0.025)
                is_too_thin = (w < 0.028)
                
                if is_player or is_skill_bar or is_minimap or is_buffs or is_not_humanoid or is_too_thin:
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
