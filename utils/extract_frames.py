import cv2
import os

video_path = '../assets/ppe-3.mp4'
output_dir = '../frames'
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_idx % 5 == 0:
        cv2.imwrite(f"{output_dir}/frame_{frame_idx:04d}.jpg", frame)
    frame_idx += 1

cap.release()
