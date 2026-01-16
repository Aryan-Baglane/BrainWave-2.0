#!/usr/bin/env python3
"""Convert MOV to MP4"""
import subprocess
import os

mov_file = r"D:\aadhar\247741.mov"
mp4_file = r"D:\aadhar\247741.mp4"

if os.path.exists(mov_file):
    print(f"Converting {mov_file} to MP4...")
    try:
        # Use ffmpeg with direct path
        cmd = [
            "ffmpeg",
            "-i", mov_file,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-y",
            mp4_file
        ]
        subprocess.run(cmd, check=True)
        print(f"✓ Conversion complete: {mp4_file}")
    except FileNotFoundError:
        print("FFmpeg not found. Trying alternative...")
        try:
            import cv2
            cap = cv2.VideoCapture(mov_file)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            out = cv2.VideoWriter(mp4_file, fourcc, fps, (width, height))
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
            cap.release()
            out.release()
            print(f"✓ Conversion complete: {mp4_file}")
        except Exception as e:
            print(f"Error: {e}")
else:
    print(f"File not found: {mov_file}")
