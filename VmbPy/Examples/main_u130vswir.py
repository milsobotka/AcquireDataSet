import os
import sys
import datetime
import time
import json
from typing import Optional
from queue import Queue
import numpy as np
from vmbpy import *

opencv_display_format = PixelFormat.Mono8

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


def setup_camera(cam: Camera):
    exposure_auto_value = 'Off'
    device_link_throughput_limit_mode_value = 'Off'
    device_link_throughput_limit_value = 450000000
    sensor_bit_depth_value = 'Bpp12'
    exposure_time_value = 15002.908
    gain_auto_value = 'Off'
    gain_value = 0
    height_value = 516
    width_value = 648
    offset_x = 296
    offset_y = 344

    with cam:
        cam.ExposureAuto.set(exposure_auto_value)
        cam.DeviceLinkThroughputLimitMode.set(device_link_throughput_limit_mode_value)
        cam.DeviceLinkThroughputLimit.set(device_link_throughput_limit_value)
        cam.SensorBitDepth.set(sensor_bit_depth_value)
        cam.ExposureTime.set(exposure_time_value)
        cam.GainAuto.set(gain_auto_value)
        cam.Gain.set(gain_value)
        cam.Height.set(height_value)
        cam.Width.set(width_value)
        cam.OffsetX.set(offset_x)
        cam.OffsetY.set(offset_y)

        print(
            f"Camera settings: ExposureAuto={exposure_auto_value}, GainAuto={gain_auto_value}, ExposureTime={exposure_time_value}")

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


class FrameHandler:
    def __init__(self):

        self.display_queue = Queue()
        self.frame_count = 0
        self.start_time = None

    def get_image(self):
        return self.display_queue.get(True)

    def __call__(self, cam: Camera, stream: Stream, frame: Frame):
        if frame.get_status() == FrameStatus.Complete:
            display = frame.convert_pixel_format(
                opencv_display_format) if frame.get_pixel_format() != opencv_display_format else frame

            frame_data = display.as_opencv_image()
            #print(f"Processing frame {self.frame_count}")
            self.display_queue.put(frame_data.copy(), True)

            if self.frame_count == 0:
                self.start_time = time.time()
            self.frame_count += 1
        cam.queue_frame(frame)


def save_frames(frames, output_folder):
    frames_np = np.array(frames)
    np.save(os.path.join(output_folder, 'video.npy'), frames_np)


def save_settings(settings, output_folder):
    with open(os.path.join(output_folder, 'settings.json'), 'w') as f:
        json.dump(settings, f, indent=4)


def main():
    cam_id = parse_args()
    frames = []
    capture_duration = 60
    camera_name = 'u130vswir'
    base_dir = "C:/Users/admin/Desktop/phd-workspace/input_data_11_06"

    timestamp = get_timestamp()
    output_folder = os.path.join(base_dir, camera_name, timestamp)
    create_directories(output_folder)

    with VmbSystem.get_instance():
        with get_camera(cam_id) as cam:
            setup_camera(cam)
            setup_pixel_format(cam)

            handler = FrameHandler()

            try:
                cam.start_streaming(handler=handler, buffer_count=80)
                start_time = time.time()

                start_time = time.perf_counter()
                while time.perf_counter() - start_time < capture_duration:
                    display = handler.get_image()
                    frames.append(display.copy())


            finally:
                cam.stop_streaming()
                elapsed_time = time.time() - handler.start_time
                fps = handler.frame_count / elapsed_time if elapsed_time > 0 else 0

                print(f"Average FPS: {fps:.2f}, Total frames: {handler.frame_count}")

                save_frames(frames, output_folder)

                settings = {
                    "camera_name": camera_name,
                    "resolution": f"{frames[0].shape[1]}x{frames[0].shape[0]}" if frames else "Unknown",
                    "fps": fps,
                    "duration": capture_duration,
                    "light_source": "lampa Newell 5600K 70%",
                    "ExposureAuto": "Off",
                    "DeviceLinkThroughputLimitMode": "Off",
                    "DeviceLinkThroughputLimit": 450000000,
                    "SensorBitDepth": "Bpp12",
                    "ExposureTime": 15002.908,
                    "GainAuto": "Off",
                    "Gain": 0,
                    "Height": 516,
                    "Width": 648,
                    "OffsetX": 296,
                    "OffsetY": 344
                }

                save_settings(settings, output_folder)


if __name__ == '__main__':
    main()