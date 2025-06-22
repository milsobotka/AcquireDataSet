import cv2
import os
import datetime
import numpy as np
import time
import json
import csv

camera_name = "logitech_carl_zeiss"
base_path = f"C:/Users/admin/Desktop/phd-workspace/input_data/{camera_name}"
timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
output_folder = os.path.join(base_path, timestamp_str)
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0.0:
    fps = 30.0
    print(f"FPS is not detected. Using default value: {fps}")
else:
    print(f"Camera FPS: {fps}")

print(f"Saving frames to folder: {output_folder}")

frames = []
timestamps = []

start_time = time.time()
duration = 30

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    timestamp_text = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    timestamps.append(timestamp_text)

    frames.append(frame)

    if time.time() - start_time > duration:
        print("Measurement time reached 30 seconds")
        break

cap.release()
cv2.destroyAllWindows()

frames = np.array(frames, dtype=np.uint8)
np.save(os.path.join(output_folder, 'video.npy'), frames)

print(f"Recording finished. Frames saved to: {output_folder}/video.npy")

if frames.size > 0:
    print(f"Frame size: {frames.shape[1]}x{frames.shape[2]}, Channels: {frames.shape[3]}")
else:
    print("No frames captured.")

csv_path = os.path.join(output_folder, 'timestamps.csv')
with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["frame_index", "timestamp"])
    for i, ts in enumerate(timestamps):
        writer.writerow([i, ts])

settings = {
    "camera_name": camera_name,
    "resolution": f"{frames.shape[2]}x{frames.shape[1]}" if frames.size > 0 else "Unknown",
    "fps": fps,
    "light_source": "lampa Newell 5000K 5%",
    "duration": duration
}

with open(os.path.join(output_folder, 'settings.json'), 'w') as f:
    json.dump(settings, f, indent=4)
