"""
detect_yolo.py - YOLO-based enemy detection
Thay thế color detection khi đã có trained model

Ưu điểm so với color detection:
  - Detect enemy ngay cả khi chưa có thanh máu
  - Chính xác hơn trong mọi tình huống
  - Không bị ảnh hưởng bởi màu sắc map
"""

import cv2
import numpy as np
from pathlib import Path
from config import FOV_SIZE, HEAD_OFFSET_Y

# HSV range màu xanh đồng minh (vẫn dùng để filter)
ALLY_GREEN_LOWER = np.array((40, 100, 100))
ALLY_GREEN_UPPER = np.array((80, 255, 255))

MODEL_PATH = "models/avatarstar.pt"


class YOLODetector:
    def __init__(self):
        self.model      = None
        self.fov_center = (FOV_SIZE // 2, FOV_SIZE // 2)
        self._load_model()

    def _load_model(self):
        if not Path(MODEL_PATH).exists():
            print(f"[YOLO] Model chưa có tại {MODEL_PATH}")
            print(f"[YOLO] Fallback về color detection...")
            return

        try:
            from ultralytics import YOLO
            self.model = YOLO(MODEL_PATH)
            print(f"[YOLO] Model loaded: {MODEL_PATH}")
        except ImportError:
            print("[YOLO] ultralytics chưa được cài. Chạy: pip install ultralytics")

    def is_ready(self):
        return self.model is not None

    def find_targets(self, frame):
        """
        Detect enemy bằng YOLO
        Trả về list [(x, y, priority, type), ...]
        """
        if not self.is_ready():
            return []

        # Run YOLO inference
        results = self.model(
            frame,
            verbose=False,
            conf=0.45,       # Confidence threshold
            iou=0.5,         # NMS threshold
        )

        targets     = []
        ally_zones  = self._find_ally_zones(frame)

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                # Center của bounding box
                cx = (x1 + x2) // 2
                # Aim vào đầu (phía trên box)
                cy = y1 + HEAD_OFFSET_Y

                # Kiểm tra có phải đồng minh không
                if self._is_ally(cx, cy, ally_zones):
                    continue

                targets.append((cx, cy, conf, 'yolo'))

        # Sort theo khoảng cách đến tâm FOV
        targets.sort(key=lambda t: self._dist_to_center(t[0], t[1]))
        return targets

    def _find_ally_zones(self, frame):
        """Tìm vùng đồng minh bằng màu thanh máu xanh"""
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, ALLY_GREEN_LOWER, ALLY_GREEN_UPPER)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        zones = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 15 and h > 3 and w / max(h, 1) > 3:
                zones.append((x + w // 2, y + h // 2))

        return zones

    def _is_ally(self, x, y, ally_zones, threshold=70):
        for ax, ay in ally_zones:
            if ((x - ax) ** 2 + (y - ay) ** 2) ** 0.5 < threshold:
                return True
        return False

    def _dist_to_center(self, x, y):
        cx, cy = self.fov_center
        return ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5

    def draw_debug(self, frame, targets):
        debug = frame.copy()
        cx, cy = self.fov_center

        # FOV center
        cv2.circle(debug, (cx, cy), 5,  (255, 255, 0), -1)
        cv2.circle(debug, (cx, cy), FOV_SIZE // 2, (255, 255, 0), 1)

        for i, (x, y, conf, ttype) in enumerate(targets):
            color = (0, 0, 255)
            cv2.circle(debug, (x, y), 10, color, 2)
            cv2.putText(debug, f"{conf:.2f}", (x + 12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            if i == 0:
                cv2.line(debug, (cx, cy), (x, y), (0, 255, 255), 1)

        return debug
