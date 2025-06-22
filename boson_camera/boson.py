import cv2
import os
import datetime
import numpy as np
import time
import json
import csv
import sys
from flirpy.camera.boson import Boson


def main():
    camera_name = "boson"

    root_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    base_path = os.path.join(root_dir, "dataset", camera_name)
    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
    output_folder = os.path.join(base_path, timestamp_str)

    os.makedirs(output_folder, exist_ok=True)

    print("Root dir   :", root_dir)
    print(f"Saving frames to folder: {output_folder}")
    print("Camera script using:", sys.executable)

    # duration in seconds
    duration = 30
    #duration = int(sys.argv[1]) if len(sys.argv) > 1 else 60

    with Boson(port="COM7") as camera:
        # Disable autofocus if possible (depends on your camera firmware/model)
        camera.set_ffc_manual()
        camera.do_ffc()

        # get 1st frame to check shape

        first_frame = camera.grab()
        if first_frame is None:
            raise Exception("No frames grabbed")

        h, w = first_frame.shape
        print(r"h = {h}, w = {w}".format(h=h, w=w))

        # set max frames
        # assume some max_fps * duration

        max_fps = 120
        max_frames = int(max_fps * duration + 4)  # add 4 seconds, late we will trim it

        frames = np.empty((max_frames, h, w), dtype=first_frame.dtype)
        timestamps = np.empty((max_frames,), dtype=np.float64)

        # fast loop, only saving frames and timestamps
        # convertion will be done after the loop

        idx = 0
        start_time = time.time()

        while idx < max_frames:
            current_time = time.time()
            if (current_time - start_time) > duration:
                print(f"Measurement time reached {duration} seconds")
                break

            frame = camera.grab()
            if frame is None:
                continue
            frames[idx] = frame
            timestamps[idx] = current_time
            idx += 1

        start_time = timestamps[0]
        end_time = timestamps[idx - 1]

    # end of fast loop
    frames = frames[:idx]
    timestamps = timestamps[:idx]

    # save frames
    np.save(os.path.join(output_folder, "video.npy"), frames)

    # save timestamps
    np.save(os.path.join(output_folder, "timestamps.npy"), timestamps)

    total_frames = len(frames)
    elapsed_time = end_time - start_time
    if elapsed_time > 0:
        fps_estimated = total_frames / elapsed_time
    else:
        fps_estimated = 0.0

    h, w = frames[0].shape

    print(f"Recording finished. Saved to: {output_folder}/video.npy")
    print(f"Total frames: {total_frames}, Estimated FPS: {fps_estimated:.2f}")
    if total_frames > 0:
        print(f"Frame size (height x width): {h} x {w}")
    else:
        print("No frames captured.")

    settings = {
        "camera_name": camera_name,
        "resolution": f"{h}x{w}" if frames.size > 0 else "Unknown",
        "fps": round(fps_estimated, 2),
        "light_source": "lampa Newell 5000K 40%",
        "duration_sec": elapsed_time,
        "frames_shape": list(frames.shape),
        "frames_description":
            "3-D ndarray: (N_frames, height, width). "
            "frames[i, y, x] to jasnosc piksela (y, x) w klatce i.",

        "timestamps_shape": list(timestamps.shape),
        "timestamps_unit": "seconds_since_unix_epoch (float64)",
        "timestamps_description":
            "1-D ndarray: timestamps[i] to czas (s) pobrania klatki i."
    }


    with open(os.path.join(output_folder, 'settings.json'), 'w') as f:
        json.dump(settings, f, indent=4)


if __name__ == "__main__":
    main()
