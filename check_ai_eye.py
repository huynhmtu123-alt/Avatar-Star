from ultralytics import YOLO
import os
from pathlib import Path

def generate_preview():
    # 1. Đường dẫn tới model mới nhất đang học
    model_path = "runs/detect/avatarstar-3/weights/last.pt"
    if not os.path.exists(model_path):
        print("[!] Không tìm thấy file last.pt. Có vẻ máy chưa học xong vòng nào.")
        return

    print(f"[*] Đang nạp model từ: {model_path}")
    model = YOLO(model_path)

    # 2. Tìm những ảnh có nhãn (chắc chắn có kẻ địch bên trong)
    img_dir = Path("dataset/images/train")
    label_dir = Path("dataset/labels/train")
    labeled_stems = [f.stem for f in label_dir.glob("*.txt") if os.path.getsize(f) > 0]
    
    if not labeled_stems:
        print("[!] Không tìm thấy ảnh nào có chứa kẻ địch trong dataset.")
        return
    
    # Chọn 10 tấm ảnh ngẫu nhiên có địch để xem độ bao quát
    import random
    if len(labeled_stems) > 10:
        test_stems = random.sample(labeled_stems, 10)
    else:
        test_stems = labeled_stems
    test_imgs = []
    for stem in test_stems:
        img_path = img_dir / (stem + ".jpg")
        if img_path.exists():
            test_imgs.append(img_path)

    if not test_imgs:
        print("[!] Không tìm thấy file ảnh tương ứng với nhãn.")
        return

    print(f"[*] Đang cho AI nhận diện thử trên {len(test_imgs)} ảnh có địch...")
    results = model(test_imgs, conf=0.05) # Hạ thấp conf xuống nữa để ép nó phải hiện ra các mục tiêu đang phân vân

    # 3. Lưu kết quả ra file ảnh
    for i, r in enumerate(results):
        out_name = f"preview_{i}.jpg"
        r.save(filename=out_name)
        print(f"[✓] Đã tạo: {out_name}")

if __name__ == "__main__":
    generate_preview()
