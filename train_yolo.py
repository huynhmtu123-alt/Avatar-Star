"""
train_yolo.py - Train YOLOv8 model cho Avatar Star enemy detection

Cách dùng:
  python tools/train_yolo.py

Yêu cầu:
  pip install ultralytics
  
Cấu trúc dataset cần có trước:
  dataset/
  ├── images/
  │   ├── train/   ← ảnh training (80%)
  │   └── val/     ← ảnh validation (20%)
  └── labels/
      ├── train/   ← file .txt YOLO format
      └── val/
"""

import os
import sys
import shutil
import random
from pathlib import Path


def check_dataset():
    """Kiểm tra dataset đã sẵn sàng chưa"""
    required = [
        "dataset/images/train",
        "dataset/images/val", 
        "dataset/labels/train",
        "dataset/labels/val",
    ]
    
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        print("[ERROR] Thiếu thư mục:")
        for m in missing:
            print(f"  - {m}")
        return False
    
    train_images = list(Path("dataset/images/train").glob("*.jpg"))
    train_labels = list(Path("dataset/labels/train").glob("*.txt"))
    
    print(f"[*] Train images: {len(train_images)}")
    print(f"[*] Train labels: {len(train_labels)}")
    
    if len(train_images) == 0:
        print("[ERROR] Không có ảnh trong dataset/images/train")
        return False
    
    if len(train_labels) == 0:
        print("[ERROR] Không có label trong dataset/labels/train")
        print("        Hãy label ảnh bằng LabelImg trước!")
        return False
    
    return True


def split_dataset(image_dir="dataset/images/train", ratio=0.8):
    """
    Tự động split dataset thành train/val
    Chỉ cần bỏ ảnh vào train/, script sẽ tự split
    """
    images = list(Path(image_dir).glob("*.jpg"))
    random.shuffle(images)
    
    split_idx = int(len(images) * ratio)
    train_imgs = images[:split_idx]
    val_imgs   = images[split_idx:]
    
    os.makedirs("dataset/images/val", exist_ok=True)
    os.makedirs("dataset/labels/val", exist_ok=True)
    
    # Move val images và labels
    moved = 0
    for img_path in val_imgs:
        label_path = Path("dataset/labels/train") / (img_path.stem + ".txt")
        
        if label_path.exists():
            shutil.move(str(img_path), f"dataset/images/val/{img_path.name}")
            shutil.move(str(label_path), f"dataset/labels/val/{label_path.name}")
            moved += 1
    
    print(f"[*] Split dataset: {len(train_imgs)} train / {moved} val")


def create_yaml():
    """Tạo file config YAML cho YOLO training"""
    yaml_content = """# Avatar Star - Enemy Detection Dataset
path: ./dataset
train: images/train
val: images/val

# Classes
nc: 1  # Chỉ có 1 class: enemy
names:
  0: enemy
"""
    with open("dataset/avatarstar.yaml", "w") as f:
        f.write(yaml_content)
    print("[*] Đã tạo dataset/avatarstar.yaml")


def train():
    """Train YOLOv8 model"""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Chưa cài ultralytics!")
        print("        Chạy: pip install ultralytics")
        sys.exit(1)
    
    print("[*] Bắt đầu training YOLOv8n...")
    print("[*] Model: yolov8n (nano - nhỏ nhất, nhanh nhất)")
    print()
    
    # Load model: Ưu tiên dùng model đang học dở để học tiếp
    last_model = "runs/detect/avatarstar-3/weights/last.pt"
    if os.path.exists(last_model):
        print(f"[*] Tìm thấy model đang học dở: {last_model}")
        print("[*] Sẽ học tiếp từ Vòng 73 để nâng cao trình độ...")
        model = YOLO(last_model)
    else:
        print("[*] Không tìm thấy model cũ, bắt đầu học từ đầu với yolov8n.pt")
        model = YOLO("yolov8n.pt")
    
    # Train
    results = model.train(
        data="dataset/avatarstar.yaml",
        epochs=100,           # Số epoch
        imgsz=640,            # Image size
        batch=16,             # Batch size (giảm xuống 8 nếu hết RAM)
        name="avatarstar",    # Tên run
        patience=20,          # Early stopping
        save=True,
        plots=True,
        workers=4,
    )
    
    print()
    print("=" * 50)
    print("✅ Training xong!")
    print(f"   Best model: runs/detect/avatarstar/weights/best.pt")
    print()
    print("Bước tiếp theo:")
    print("  Copy best.pt vào models/avatarstar.pt")
    print("  python start.py (tool sẽ tự dùng model mới)")
    print("=" * 50)


def main():
    print("""
╔══════════════════════════════════════════╗
║   Avatar Star - YOLO Trainer             ║
╚══════════════════════════════════════════╝
    """)
    
    # 1. Tạo YAML config
    create_yaml()
    
    # 2. Split dataset nếu chưa có val
    val_images = list(Path("dataset/images/val").glob("*.jpg")) if Path("dataset/images/val").exists() else []
    if len(val_images) == 0:
        print("[*] Chưa có val set, tự động split...")
        split_dataset()
    
    # 3. Kiểm tra dataset
    if not check_dataset():
        sys.exit(1)
    
    print()
    print("[*] Dataset OK! Bắt đầu training...")
    print()
    
    # 4. Train
    train()


if __name__ == "__main__":
    main()
