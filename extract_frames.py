"""
extract_frames.py - Tool tự động download video YouTube và extract frame
Dùng để tạo dataset cho training YOLO model

Cách dùng:
  python tools/extract_frames.py --url "URL_YOUTUBE" --fps 2 --output dataset/images

Ví dụ:
  python tools/extract_frames.py --url "https://www.youtube.com/watch?v=mc4B2JExT6k" --fps 2
"""

import cv2
import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path


def download_video(url, output_path="temp_video.mp4"):
    """Download video từ YouTube bằng yt-dlp"""
    print(f"[*] Đang download video: {url}")
    
    cmd = [
        "yt-dlp",
        "-f", "best[height<=720]",   # Tối đa 720p để tiết kiệm
        "-o", output_path,
        "--no-playlist",
        url
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"[ERROR] Download thất bại:\n{result.stderr}")
        return None
    
    # yt-dlp có thể đổi extension
    for ext in ['.mp4', '.webm', '.mkv']:
        path = output_path.replace('.mp4', ext)
        if os.path.exists(path):
            return path
    
    if os.path.exists(output_path):
        return output_path
        
    print("[ERROR] Không tìm thấy file video sau download")
    return None


def extract_frames(video_path, output_dir, fps_extract=2, skip_similar=True, similarity_threshold=0.95):
    """
    Extract frame từ video
    
    Args:
        video_path: Đường dẫn file video
        output_dir: Thư mục lưu frame
        fps_extract: Số frame extract mỗi giây (2 = 2 frame/giây)
        skip_similar: Bỏ qua frame quá giống nhau (tránh duplicate)
        similarity_threshold: Ngưỡng giống nhau (0-1)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Không mở được video: {video_path}")
        return 0
    
    video_fps   = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration    = total_frames / video_fps
    
    print(f"[*] Video info:")
    print(f"    FPS: {video_fps:.1f}")
    print(f"    Total frames: {total_frames}")
    print(f"    Duration: {duration:.1f}s ({duration/60:.1f} phút)")
    print(f"    Extract: {fps_extract} frame/giây")
    print(f"    Ước tính: ~{int(duration * fps_extract)} frames")
    print()
    
    # Số frame cần skip giữa mỗi lần capture
    frame_interval = int(video_fps / fps_extract)
    
    frame_idx    = 0
    saved_count  = 0
    prev_frame   = None
    
    video_name = Path(video_path).stem
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Chỉ lấy frame theo interval
        if frame_idx % frame_interval == 0:
            
            # Skip frame quá giống frame trước (scene không thay đổi)
            if skip_similar and prev_frame is not None:
                # Resize nhỏ để so sánh nhanh
                small = cv2.resize(frame, (160, 90))
                small_prev = cv2.resize(prev_frame, (160, 90))
                
                # Tính similarity
                diff = cv2.absdiff(small, small_prev)
                similarity = 1 - (diff.mean() / 255)
                
                if similarity > similarity_threshold:
                    frame_idx += 1
                    continue
            
            # Lưu frame
            filename = f"{video_name}_frame_{saved_count:05d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            saved_count += 1
            prev_frame = frame.copy()
            
            # Progress
            progress = (frame_idx / total_frames) * 100
            print(f"\r[*] Progress: {progress:.1f}% | Saved: {saved_count} frames", end="", flush=True)
        
        frame_idx += 1
    
    cap.release()
    print(f"\n[✓] Xong! Đã extract {saved_count} frames → {output_dir}")
    return saved_count


def cleanup(temp_file):
    """Xóa file video tạm"""
    if os.path.exists(temp_file):
        os.remove(temp_file)
        print(f"[*] Đã xóa file tạm: {temp_file}")


def main():
    parser = argparse.ArgumentParser(description="Extract frames từ YouTube video cho YOLO dataset")
    parser.add_argument("--url",    required=False, help="YouTube URL")
    parser.add_argument("--video",  required=False, help="Đường dẫn file video local")
    parser.add_argument("--fps",    type=float, default=2, help="Frame per second extract (default: 2)")
    parser.add_argument("--output", default="dataset/images/train", help="Output directory")
    parser.add_argument("--keep-video", action="store_true", help="Giữ lại file video sau khi extract")
    args = parser.parse_args()
    
    temp_video = "temp_video.mp4"
    
    # 1. Lấy video path (từ local hoặc download)
    if args.video:
        if not os.path.exists(args.video):
            print(f"[ERROR] Không tìm thấy file video: {args.video}")
            sys.exit(1)
        video_path = args.video
        print(f"[✓] Dùng video local: {video_path}\n")
    elif args.url:
        video_path = download_video(args.url, temp_video)
        if not video_path:
            sys.exit(1)
        print(f"[✓] Download xong: {video_path}\n")
    else:
        print("[ERROR] Vui lòng cung cấp --url hoặc --video")
        sys.exit(1)
    
    # 2. Extract frames
    count = extract_frames(
        video_path=video_path,
        output_dir=args.output,
        fps_extract=args.fps
    )
    
    # 3. Cleanup (chỉ xóa nếu là video tải về, không xóa video local của user)
    if args.url and not args.keep_video:
        cleanup(video_path)
    
    print()
    print("=" * 50)
    print(f"✅ Hoàn tất! {count} frames saved vào: {args.output}")
    print()
    print("Bước tiếp theo:")
    print("  1. Cài LabelImg: pip install labelImg")
    print("  2. Chạy: labelImg")
    print("  3. Mở thư mục dataset/images/train")
    print("  4. Vẽ bounding box quanh ENEMY (không vẽ đồng minh)")
    print("  5. Save format: YOLO")
    print("=" * 50)


if __name__ == "__main__":
    main()
