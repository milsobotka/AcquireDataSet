from __future__ import annotations

import sys
from flirpy.camera.boson import Boson
import os, time, json, numpy as np
from pathlib import Path
from threading import Event

__all__ = ["record_boson"]


def record_boson(start_evt: Event, stop_evt: Event, duration: int, out_dir: Path | str, com: str):
    camera_name = "boson"
    output_folder = out_dir
    os.makedirs(output_folder, exist_ok=True)

    print(f"[boson] Saving frames to folder: {output_folder}")
    print("[boson] Camera script using:", sys.executable)

    try:
        camera = Boson(port=com)
    except Exception as exc:
        raise RuntimeError(f"Could not open Boson on {com}: {exc}")

    with camera:
        # Disable autofocus if possible (depends on your camera firmware/model)
        try:
            camera.set_ffc_manual()
            camera.do_ffc()
        except Exception as exc:
            raise RuntimeError(f"FFC failed: {exc}")

        # get 1st frame to check shape

        first_frame = camera.grab()
        if first_frame is None:
            raise RuntimeError("Boson returned no frame on first grab")

        h, w = first_frame.shape
        print(r"[boson] h = {h}, w = {w}".format(h=h, w=w))

        # set max frames
        # assume some big max_fps * duration
        # Boson camera should capture with 60FPS

        max_fps = 80
        max_frames = int(max_fps * duration + 4)  # add 4, late we will trim it

        frames = np.empty((max_frames, h, w), dtype=first_frame.dtype)
        timestamps = np.empty((max_frames,), dtype=np.float64)

        # fast loop, only saving frames and timestamps
        # convertion will be done after the loop

        idx = 0

        print("[boson] Ready. Waiting for start event")
        start_evt.wait()  # set by GUI after 5s countdown
        print("[boson] Recording started")

        start_time = time.time()
        while idx < max_frames:
            current_time = time.time()
            if (current_time - start_time) > duration:
                print(f"[boson] Measurement time reached {duration} seconds")
                break

            frame = camera.grab()
            if frame is None:
                continue
            frames[idx] = frame
            timestamps[idx] = current_time
            idx += 1

        # end of fast loop
        np.save(os.path.join(output_folder, "video.npy"), frames[:idx])
        np.save(os.path.join(output_folder, "timestamps.npy"), timestamps[:idx])

        elapsed = timestamps[idx - 1] - timestamps[0] if idx > 1 else 0.0
        fps_est = idx / elapsed if elapsed else 0.0
        print(f"[boson] Done - {idx} frames, {fps_est:.2f} fps")

        settings = {
            "camera_name": camera_name,
            "resolution": f"{h}x{w}",
            "fps": round(fps_est, 2),
            "duration_sec": elapsed,
            "frames_shape": list(frames[:idx].shape),
            "timestamps_unit": "seconds_since_unix_epoch (float64)",
            "OutputFolder": str(output_folder).replace("\\", "/"),
        }

        with open(os.path.join(output_folder, "settings.json"), "w", encoding="utf-8") as fp:
            json.dump(settings, fp, indent=2)
        print(f"[boson] settings.json saved in {output_folder}")
