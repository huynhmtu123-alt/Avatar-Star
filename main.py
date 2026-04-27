import cv2
import numpy as np
import time
import keyboard
import win32api
import win32con
from mss import mss
from detect_yolo import YOLODetector
from config import FOV_SIZE, HEAD_OFFSET_Y

# ================= CẤU HÌNH AIMBOT SIÊU CẤP (RAGE MODE) =================
SMOOTH_FACTOR = 1.0     # 1.0 = Snap Aim (Giật ngay lập tức, không làm mượt)
AIM_KEY = 'shift'       # Đè phím này để tự động ngắm và khóa mục tiêu
QUICK_SCOPE_MODE = True # Tự động bật/tắt scope siêu tốc khi nhấn chuột trái để đạn bay thẳng
# ========================================================================

def move_mouse(x, y):
    """Di chuyển chuột tương đối"""
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(x), int(y), 0, 0)

def quick_scope_shot():
    """Thực hiện cú vẩy Scope và bắn trong chớp mắt"""
    # 1. Nhấn chuột phải (Bật Scope)
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
    time.sleep(0.01)
    
    # 2. Nhấn chuột trái (Bắn)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    time.sleep(0.01)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
    
    # 3. Thả chuột phải (Tắt Scope)
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)

def main():
    print("====================================================")
    print("🚀 AVATAR STAR AI AIMBOT - RAGE EDITION 🚀")
    print("====================================================")
    
    detector = YOLODetector()
    if not detector.is_ready():
        print("[LỖI] Không tìm thấy file models/avatarstar.pt")
        return

    sct = mss()
    screen_width = win32api.GetSystemMetrics(0)
    screen_height = win32api.GetSystemMetrics(1)
    
    # Mở rộng FOV để quét rộng hơn
    CURRENT_FOV = 400 
    
    monitor = {
        "top": (screen_height - CURRENT_FOV) // 2,
        "left": (screen_width - CURRENT_FOV) // 2,
        "width": CURRENT_FOV,
        "height": CURRENT_FOV
    }
    
    print(f"[STATUS] Đã nạp Model thành công!")
    print(f"[CONFIG] Snap Mode: ON (Smooth={SMOOTH_FACTOR})")
    print(f"[CONFIG] Quick Scope: {'BẬT' if QUICK_SCOPE_MODE else 'TẮT'}")
    print(f"[READY] Đè '{AIM_KEY}' để Khóa mục tiêu. Nhấn Chuột Trái để Bắn thẳng.")
    print("====================================================")

    last_shot_time = 0

    while True:
        if keyboard.is_pressed('q'):
            print("\n[EXIT] Đang thoát tool...")
            break
            
        # 1. Chụp màn hình
        img = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        # 2. Nhận diện địch
        targets = detector.find_targets(frame)
        
        # 3. Nếu đang đè phím Aim (Shift)
        if keyboard.is_pressed(AIM_KEY):
            if targets:
                # Lấy mục tiêu gần tâm nhất
                tx, ty, conf, _ = targets[0]
                cx, cy = CURRENT_FOV // 2, CURRENT_FOV // 2
                
                dx = tx - cx
                dy = ty - cy
                
                # Thực hiện Snap Aim
                move_mouse(dx * SMOOTH_FACTOR, dy * SMOOTH_FACTOR)
                
                # 4. Nếu nhấn chuột trái VÀ đang trong chế độ Quick Scope
                # (Sử dụng win32api để kiểm tra trạng thái chuột trái: 0x01)
                if QUICK_SCOPE_MODE and (win32api.GetAsyncKeyState(0x01) & 0x8000):
                    current_time = time.time()
                    # Giới hạn tốc độ bắn (tránh spam lệnh quá nhanh)
                    if current_time - last_shot_time > 0.3: 
                        quick_scope_shot()
                        last_shot_time = current_time

if __name__ == "__main__":
    main()
