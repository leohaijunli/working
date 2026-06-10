# -*- coding: utf-8 -*-
"""
video_source.py - selection of camera backends with auto-detection and fallback to local video files.
Designed for Raspberry Pi 5 with CSI cameras (libcamera/Picamera2).
"""

import time
import os
import logging
import threading
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


from picamera2 import Picamera2

import cv2



class UniversalVideoSource:
    def __init__(self, source=0, width=640, height=480, fps=25):
        self._source = source
        self.is_pi_camera = True
        self._width = width
        self._height = height
        self._fps = fps

        self._cv_cap = None               # cv2.VideoCapture
        self._backend = "picamera2"
        self._picam = None              # Picamera2 instance

        # frame and thread safety
        self._frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._frame_id: int = 0
        self._capture_thread: Optional[threading.Thread] = None
        self._running = False


        # statistics
        self._fps_actual: float = 0.0
        self._frame_count: int = 0
        self._fps_timer: float = 0.0

    # ════════════════════════════════════════════════════════
    # Properties
    # ════════════════════════════════════════════════════════

    @property
    def resolution(self) -> Tuple[int, int]:
        """Output resolution of the camera frames (width, height)."""
        return self._width, self._height

    @property
    def fps_actual(self) -> float:
        """Actual frames rate."""
        return self._fps_actual

    @property
    def frame_id(self) -> int:
        """Current frame ID."""
        return self._frame_id

    @property
    def is_opened(self) -> bool:
        """Camera is opened and capture stream is running."""
        return self._running

    # ════════════════════════════════════════════════════════
    # Initialization and opening
    # ════════════════════════════════════════════════════════

    def _is_rtsp_source(self) -> bool:
        """Check if source is an RTSP/RTMP/HTTP stream URL."""
        if not isinstance(self._source, str):
            return False
        return self._source.lower().startswith(
            ("rtsp://", "rtsps://", "rtmp://", "http://", "https://")
        )

    def open(self) -> bool:
        if isinstance(self._source, int):
            if self._open_picamera2():
                print(f"📸 Picamera2 is opening the camera index: {self._source}")
                return True
            logger.warning("Picamera2 is failed to open, trying open default video file...")
        else:
            if self._is_rtsp_source():
                if self._open_opencv(is_rtsp=True):
                    print(f"📡 OpenCV is opening RTSP stream: {self._source}")
                    return True
            elif os.path.exists(self._source):
                if self._open_opencv(is_rtsp=False):
                    print(f"🎞️ OpenCV is opening the video file: {self._source}")
                    return True
            else:
                print(f"❌ Error: source not found or invalid: {self._source}")
        return False


    def _open_picamera2(self) -> bool:
        """Picamera2 (libcamera, CSI)."""
        try:
            self._picam = Picamera2(camera_num=self._source)

            # picam2 configuration: BGR888
            config = self._picam.create_video_configuration(
                main={
                    "size": (self._width, self._height),
                    "format": "BGR888",  # BGR for yolo, OpenCV compability
                },
                controls={
                    "FrameRate": self._fps,
                },
                buffer_count=2,  # minimal buffering for low latency
            )
            self._picam.configure(config)
            self._picam.start()

            # time for camera to intialize ISP and auto-exposure to stabilize
            time.sleep(1.0)

            # actual sensor resolution
            camera_props = self._picam.camera_properties
            sensor_res = camera_props.get('PixelArraySize', (self._width, self._height))
            self._sensor_width = sensor_res[0]
            self._sensor_height = sensor_res[1]

            logger.info(
                f"Camera opened [Picamera2/libcamera CSI]: "
                f"{self._width}x{self._height}@{self._fps}, "
                f"sensor={self._sensor_width}x{self._sensor_height}"
            )

            # start a background thread for continuous capture
            self._running = True
            self._fps_timer = time.time()
            self._capture_thread = threading.Thread(
                target=self._capture_loop_picamera2,
                name="CameraCapture-Picamera2",
                daemon=True
            )
            self._capture_thread.start()

            logger.info("Camera capture thread started [Picamera2]")
            self._backend = "picamera2"

            return True

        except Exception as e:
            logger.error(f"Error opening Picamera2: {e}")
            if self._picam:
                try:
                    self._picam.close()
                except Exception:
                    pass
                self._picam = None
            return False

    def _open_opencv(self, is_rtsp: bool = False) -> bool:
        """Open local video file or RTSP stream using OpenCV."""
        try:
            import cv2 as cv
            self._is_rtsp = is_rtsp

            if is_rtsp:
                # Force TCP transport to reduce UDP packet loss
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
                self._cv_cap = cv.VideoCapture(self._source, cv.CAP_FFMPEG)
            else:
                self._cv_cap = cv.VideoCapture(self._source)

            if not self._cv_cap.isOpened():
                logger.error(f"OpenCV: failed to open source {self._source}")
                return False

            # RTSP streams do not always have a fixed FPS, so use the configured value; local files use the source FPS
            if is_rtsp:
                actual_fps = self._cv_cap.get(cv.CAP_PROP_FPS)
                self._fps = actual_fps if actual_fps > 0 else self._fps
                print(f"📡 RTSP stream FPS: {self._fps:.2f}")
            else:
                original_fps = self._cv_cap.get(cv2.CAP_PROP_FPS)
                self._fps = original_fps if original_fps > 0 else self._fps
                print(f"🎞️ OpenCV opened video file with original FPS: {original_fps:.2f}")

            self._cv_cap.set(cv.CAP_PROP_FRAME_WIDTH, self._width)
            self._cv_cap.set(cv.CAP_PROP_FRAME_HEIGHT, self._height)
            self._cv_cap.set(cv.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency

            actual_w = int(self._cv_cap.get(cv.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self._cv_cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self._cv_cap.get(cv.CAP_PROP_FPS)

            if actual_w != self._width or actual_h != self._height:
                logger.warning(f"Requested {self._width}x{self._height}, Received {actual_w}x{actual_h}")
                self._width = actual_w
                self._height = actual_h

            self._sensor_width = self._width
            self._sensor_height = self._height
            self._backend = "opencv"

            self._running = True
            self.is_pi_camera = False
            self._fps_timer = time.time()
            self._capture_thread = threading.Thread(
                target=self._capture_loop_opencv,
                name="RTSPCapture-OpenCV" if is_rtsp else "LocalVideoCapture-OpenCV",
                daemon=True
            )
            self._capture_thread.start()

            logger.info(f"Source opened [OpenCV{'(RTSP)' if is_rtsp else ''}]: {actual_w}x{actual_h}@{actual_fps:.0f}")
            return True

        except Exception as e:
            logger.error(f"Error opening source with OpenCV: {e}")
            return False
   
    # ════════════════════════════════════════════════════════
    # Finalization and cleanup
    # ════════════════════════════════════════════════════════

    def close(self):
        """Stop capture and close the camera."""
        self._running = False

        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)

        if self._picam:
            try:
                self._picam.stop()
                self._picam.close()
            except Exception:
                pass
            self._picam = None

        if self._cv_cap:
            self._cv_cap.release()
            self._cv_cap = None

        self._backend = "none"
        logger.info("Video source closed")

    # ════════════════════════════════════════════════════════
    # Frame capture
    # ════════════════════════════════════════════════════
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the last captured frame (thread-safe).

        Returns:
            numpy array of the BGR frame or None if no frame is available.
        """
        with self._frame_lock:
            if self._frame is not None:
                return True, self._frame.copy()
        return False, None

    def get_frame_no_copy(self) -> Optional[np.ndarray]:
        """
        Get a reference to the last captured frame without copying.
        fast but not thread-safe — only call from the capture thread or ensure external synchronization.

        Returns:
            numpy array of the BGR frame or None if no frame is available.
        """
        return True, self._frame

    def _capture_loop_picamera2(self):
        """Background thread for capturing frames with Picamera2."""
        logger.debug("Camera capture [Picamera2]: start")

        while self._running and self._picam:
            try:
                # capture_array returns a BGR888 numpy array
                frame = self._picam.capture_array("main")

                if frame is not None:
                    with self._frame_lock:
                        self._frame = frame
                        self._frame_id += 1
                    self._update_fps()

            except Exception as e:
                logger.error(f"capture error Picamera2: {e}")
                time.sleep(0.01)

        logger.debug("capture thread [Picamera2]: stop")

    def _capture_loop_opencv(self):
        """Background thread for capturing frames with OpenCV."""
        frame_interval = 1.0 / self._fps
        last_read_time = time.time()
        consecutive_fail = 0

        while self._running and self._cv_cap and self._cv_cap.isOpened():
            current_time = time.time()
            if current_time - last_read_time < frame_interval:
                time.sleep(0.005)
                continue

            ret, frame = self._cv_cap.read()

            if ret and frame is not None:
                with self._frame_lock:
                    self._frame = frame.copy()
                    self._frame_id += 1
                last_read_time = current_time
                consecutive_fail = 0
                self._update_fps()
            else:
                consecutive_fail += 1
                if consecutive_fail > 5:
                    if self._is_rtsp:
                        # RTSP disconnected → reconnect
                        logger.warning("RTSP stream lost, reconnecting in 2s...")
                        self._cv_cap.release()
                        self._cv_cap = None
                        time.sleep(2.0)
                        if self._running:
                            self._open_opencv(is_rtsp=True)
                        return  # A new capture thread has been started in _open_opencv
                    else:
                        # Local file ended → loop playback from the beginning
                        logger.info("Video file ended, looping from start [OpenCV]")
                        self._cv_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    consecutive_fail = 0
                else:
                    logger.warning(f"OpenCV capture failed (attempt {consecutive_fail}), retrying...")
                    time.sleep(0.01)

        logger.debug("capture thread [OpenCV]: stop")

    def _update_fps(self):
        """Update FPS counter."""
        self._frame_count += 1
        now = time.time()
        elapsed = now - self._fps_timer
        if elapsed >= 1.0:
            self._fps_actual = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_timer = now

    
    # ════════════════════════════════════════════════════════
    # pixel-angle conversion (for camera FOV calculations)
    # ════════════════════════════════════════════════════════

    def pixel_to_angle(self, px: float, py: float) -> Tuple[float, float]:
        """
        Convert pixel coordinates to angular offset
        from the center of the frame (in degrees).
        """
        cx = self._width / 2.0
        cy = self._height / 2.0
        dx = px - cx
        dy = py - cy
        angle_h = (dx / self._width) * self.fov_h
        angle_v = (dy / self._height) * self.fov_v
        return angle_h, angle_v

    def angle_to_pixel(self, angle_h: float, angle_v: float) -> Tuple[float, float]:
        """
        Convert angular offset (degrees) to pixel coordinates.
        """
        cx = self._width / 2.0
        cy = self._height / 2.0
        px = cx + (angle_h / self.fov_h) * self._width
        py = cy + (angle_v / self.fov_v) * self._height
        return px, py