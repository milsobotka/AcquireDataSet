import os
import sys
import datetime
import time
import json
import gc
from typing import Optional
from queue import Queue, Full
import numpy as np
from vmbpy import *

import threading
import cv2  # cv2 only for pixel-format conversion

opencv_display_format = PixelFormat.Bgr8



class CameraConfig:
    # GenICam image-quality & link parameters
    exposure_auto: str = 'Off'
    balance_white_auto: str = 'Off'
    device_link_throughput_limit_mode: str = 'Off'
    device_link_throughput_limit: int = 450_000_000
    sensor_bit_depth: str = 'Bpp10'
    exposure_time: float = 15_002.343  # [microSeconds]
    gain_auto: str = 'Off'
    gain: int = 0

    # Geometry
    height: int = 608
    width: int = 968
    offset_x: int = 484
    offset_y: int = 304

    # Optional binning
    do_binning: bool = False
    binning_h: int = 1
    binning_v: int = 1
    binning_mode: str = 'Average'  # 0 = sum, 1 = average

    # recording
    capture_seconds = 120
    fps_target = 60
    lock_fps: bool = True
    max_frames = int(fps_target * capture_seconds)


def create_directories(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")


def abort(reason: str, return_code: int = 1):
    print(reason + '\n')
    sys.exit(return_code)


def parse_args() -> Optional[str]:
    args = sys.argv[1:]
    if len(args) > 1:
        abort("Invalid number of arguments. Abort.")
    return None if len(args) == 0 else args[0]


def get_camera(camera_id: Optional[str]) -> Camera:
    with VmbSystem.get_instance() as vmb:
        if camera_id:
            try:
                return vmb.get_camera_by_id(camera_id)
            except VmbCameraError:
                abort(f"Failed to access Camera '{camera_id}'. Abort.")
        else:
            cams = vmb.get_all_cameras()
            if not cams:
                abort('No Cameras accessible. Abort.')
            return cams[0]


def setup_camera(cam: Camera, cfg: CameraConfig):
    """ default settings
    exposure_auto_value = 'Off'
    balance_white_auto_value = 'Off'
    device_link_throughput_limit_mode_value = 'Off'
    device_link_throughput_limit_value = 450000000
    sensor_bit_depth_value = 'Bpp10'  # changed
    exposure_time_value = 15002.343
    gain_auto_value = 'Off'
    gain_value = 0
    height_value = 1216
    width_value = 1936
    offset_x = 0
    offset_y = 0
"""
    with cam:
        cam.ExposureAuto.set(cfg.exposure_auto)
        cam.BalanceWhiteAuto.set(cfg.balance_white_auto)
        cam.DeviceLinkThroughputLimitMode.set(cfg.device_link_throughput_limit_mode)
        cam.DeviceLinkThroughputLimit.set(cfg.device_link_throughput_limit)
        cam.SensorBitDepth.set(cfg.sensor_bit_depth)
        cam.ExposureTime.set(cfg.exposure_time)
        cam.GainAuto.set(cfg.gain_auto)
        cam.Gain.set(cfg.gain)

        # added
        if cfg.do_binning:
            cam.BinningHorizontal.set(cfg.binning_h)
            cam.BinningVertical.set(cfg.binning_v)
            cam.BinningHorizontalMode.set(cfg.binning_mode)
            cam.BinningVerticalMode.set(cfg.binning_mode)

        cam.Height.set(cfg.height)
        cam.Width.set(cfg.width)
        cam.OffsetX.set(cfg.offset_x)
        cam.OffsetY.set(cfg.offset_y)

        if cfg.lock_fps:
            cam.AcquisitionFrameRateEnable.set(True)
            cam.AcquisitionFrameRate.set(cfg.fps_target)
            cam.TriggerSelector.set('FrameStart')
            #cam.TriggerMode.set('Off')

        print(
            f"Camera settings: ExposureAuto={cfg.exposure_auto}, GainAuto={cfg.gain_auto}, ExposureTime={cfg.exposure_time}")

        try:
            stream = cam.get_streams()[0]
            stream.GVSPAdjustPacketSize.run()
            while not stream.GVSPAdjustPacketSize.is_done():
                pass
            print("Packet size adjusted")
        except (AttributeError, VmbFeatureError):
            print("Packet size adjustment not supported")


def setup_pixel_format(cam: Camera):
    cam_formats = cam.get_pixel_formats()
    cam_color_formats = intersect_pixel_formats(cam_formats, COLOR_PIXEL_FORMATS)
    cam_mono_formats = intersect_pixel_formats(cam_formats, MONO_PIXEL_FORMATS)

    if opencv_display_format in cam_formats:
        cam.set_pixel_format(opencv_display_format)
    elif cam_color_formats:
        cam.set_pixel_format(cam_color_formats[0])
    elif cam_mono_formats:
        cam.set_pixel_format(cam_mono_formats[0])
    else:
        abort('Camera does not support an OpenCV compatible format. Abort.')


class ChunkedNpyWriterCallback:

    def __init__(self, out_dir: str, shape: tuple[int, int, int],
                 chunk_size: int = 60, max_pending: int = 10,
                 max_frames: int = 3600):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir, self.chunk_size = out_dir, chunk_size
        H, W, C = shape

        self.chunk = np.empty((chunk_size, H, W, C), np.uint8)
        self.in_chunk = 0

        self.timestamps = np.empty(max_frames, np.float64)
        self.total_idx = 0
        self.block_idx = 0
        self.max_frames = max_frames

        self.q = Queue(max_pending)
        self.writer = threading.Thread(target=self._worker, daemon=True)
        self.writer.start()
        self.start_time = 0.0
        self.tick0 = 0

    # background writer
    def _worker(self):
        while True:
            item = self.q.get()
            if item is None: 
                self.q.task_done()
                break
            arr, idx = item
            np.save(os.path.join(self.out_dir,
                                 f"frameblock_{idx:05d}.npy"), arr)
            self.q.task_done()

    #  callback from VmbPy 
    def __call__(self, cam, stream, frame):
        if self.total_idx >= self.max_frames:
            cam.queue_frame(frame)
            return

        if frame.get_status() == FrameStatus.Complete:
            view = frame.as_opencv_image()  # Bgr8
            self.chunk[self.in_chunk] = view
            timestamp = frame.get_timestamp()
            frame_time_in_seconds = timestamp / 1_000_000_000.0

            if self.total_idx == 0:
                self.start_time = time.time()
                self.tick0 = frame.get_timestamp() / 1_000_000_000.0

            self.timestamps[self.total_idx] = self.start_time + frame_time_in_seconds - self.tick0
            self.in_chunk += 1
            self.total_idx += 1

            if self.in_chunk == self.chunk_size:  # full queue
                try:
                    self.q.put_nowait((self.chunk, self.block_idx))
                except Full:
                    print("Queue FULL -block dropped")
                self.block_idx += 1
                # new buffer
                self.chunk = np.empty_like(self.chunk)
                self.in_chunk = 0

        cam.queue_frame(frame)

    # shutdown 
    def close(self):
        if self.in_chunk:  # flush
            tail = self.chunk[:self.in_chunk].copy()
            self.q.put((tail, self.block_idx))

        self.q.put(None)  #
        self.q.join()  # wait
        self.writer.join()


def save_frames(frames, frame_count, output_folder):
    np.save(os.path.join(output_folder, 'video.npy'), frames[:frame_count])


def save_settings(settings, output_folder):
    with open(os.path.join(output_folder, 'settings.json'), 'w') as f:
        json.dump(settings, f, indent=4)


def main():
    cam_id = parse_args()
    camera_cfg = CameraConfig()
    camera_name = 'u235c'
    this_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(this_dir, os.pardir, os.pardir))
    base_dir = os.path.join(root_dir, "dataset")

    timestamp = get_timestamp()  # "YYYY-MM-DD_HH-MM-SS.ffffff"
    output_folder = os.path.join(base_dir, camera_name, timestamp)
    os.makedirs(output_folder, exist_ok=True)

    with VmbSystem.get_instance():
        with get_camera(cam_id) as cam:
            setup_camera(cam, camera_cfg)
            setup_pixel_format(cam)

            H, W, C = camera_cfg.height, camera_cfg.width, 3
            max_frames = camera_cfg.max_frames

            gc.disable()

            handler = ChunkedNpyWriterCallback(
                out_dir=output_folder,
                shape=(H, W, C),
                chunk_size=60,  # block size to write on disk = 1-second blocks at 60 fps
                max_pending=60,  # number of pre-allocated blocks
                max_frames=max_frames)  # frames expected to capture

            print("start recording")
            recording_time = time.time()
            cam.start_streaming(handler, buffer_count=600)
            time.sleep(camera_cfg.capture_seconds + 2)
            cam.stop_streaming()
            recording_time_stop = time.time()
            gc.enable()
            print("stop recording")
            handler.close()
            print("stop saving frames")
            # get n captured frames
            frame_count = handler.total_idx
            end_time = handler.timestamps[frame_count - 1]
            start_time = handler.timestamps[0]
            duration = (end_time - start_time) + 0.01666  # +0.01666 time for capturing 1st frame

            print(f'Frames captured {frame_count} valid frames')
            print(f'Frames max frames {handler.max_frames}')
            print(f'Frames start time {start_time}')
            print(f'Frames end time {end_time}')
            print(f'Frames duration time {duration} s')
            total_time = time.time() - start_time
            print(f'Total time {total_time}')
            fps = frame_count / duration
            print(f"Average FPS: {fps:.2f}, Total frames: {frame_count}")

            ts = handler.timestamps[:handler.total_idx]  # truncate to valid part
            np.save(os.path.join(output_folder, "timestamps.npy"), ts)
            settings = {
                "camera_name": camera_name,
                "resolution": f"{camera_cfg.width}x{camera_cfg.height}",
                "fps_target": camera_cfg.fps_target,
                "fps_measured": fps,
                "duration_recorded": duration,  # s
                "duration_requested": camera_cfg.capture_seconds,
                "light_source": "lampa Newell 5600K 70%",
                "ExposureAuto": camera_cfg.exposure_auto,
                "ExposureTime_us": camera_cfg.exposure_time,
                "SensorBitDepth": camera_cfg.sensor_bit_depth,
                "GainAuto": camera_cfg.gain_auto,
                "Gain": camera_cfg.gain,
                "Width": camera_cfg.width,
                "Height": camera_cfg.height,
                "OffsetX": camera_cfg.offset_x,
                "OffsetY": camera_cfg.offset_y,
                "BinningH": camera_cfg.binning_h if camera_cfg.do_binning else 1,
                "BinningV": camera_cfg.binning_v if camera_cfg.do_binning else 1,
                "BinningMode": camera_cfg.binning_mode if camera_cfg.do_binning else "None",
                "DeviceLinkThroughputLimitMode": camera_cfg.device_link_throughput_limit_mode,
                "DeviceLinkThroughputLimit": camera_cfg.device_link_throughput_limit,
                "LockFPS": camera_cfg.lock_fps,
                "RecordedFrames": frame_count,
                "OutputFolder": output_folder.replace("\\", "/")
            }

            save_settings(settings, output_folder)
            print(f"settings.json zapisany w  {output_folder}")


if __name__ == '__main__':
    main()
