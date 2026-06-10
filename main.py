import subprocess
import threading
import queue
import time
import json
import cv2
from datetime import datetime, timezone
import logging
from typing import Optional
from stream_controller import StreamController
from video_source import UniversalVideoSource
from ws_data_server import RerunVisualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ==================== Configuration Constants ====================
VIDEO_SRC = 0
RTSP_TARGET = "rtsp://127.0.0.1:8554/live/mystream"
TARGET_FPS = 30
FRAME_TIME = 1.0 / TARGET_FPS
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
VIDEO_FPS = 30
HEF_PATH = "yolov11n.hef"
CONF_THRESHOLD = 0.2


class FPSCounter:
    """FPS counter and frame rate controller."""

    def __init__(self, target_fps: float = 30, smoothing: float = 0.9):
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        self.smoothing = smoothing
        self.fps = target_fps
        self.last_time = time.time()

    def tick(self) -> float:
        current_time = time.time()
        elapsed = current_time - self.last_time
        if elapsed > 0:
            current_fps = 1.0 / elapsed
            self.fps = self.smoothing * self.fps + (1 - self.smoothing) * current_fps
        self.last_time = current_time
        return self.fps

    def sleep_to_maintain_fps(self):
        elapsed = time.time() - self.last_time
        sleep_time = self.frame_time - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


class MotionTerminalApp:
    """MX Motion Terminal main application."""

    def __init__(self):
        self.video_source: Optional[UniversalVideoSource] = None
        self.streamer: Optional[StreamController] = None
        self.fps_counter: Optional[FPSCounter] = None

    def initialize(self) -> bool:
        try:
            logging.info("📹 Initializing video source...")
            self.video_source = UniversalVideoSource(
                source=VIDEO_SRC,
                width=VIDEO_WIDTH,
                height=VIDEO_HEIGHT,
                fps=TARGET_FPS
            )
            if not self.video_source.open():
                logging.error("❌ Unable to open video source")
                return False

            logging.info("📡 Starting RTSP stream...")
            self.streamer = StreamController(
                rtsp_url=RTSP_TARGET,
                res=(VIDEO_WIDTH, VIDEO_HEIGHT),
                fps=TARGET_FPS
            )
            self.streamer.start()

            self.fps_counter = FPSCounter(target_fps=TARGET_FPS)

            logging.info("=" * 60)
            logging.info("🚀 System startup complete")
            logging.info(f"📹 Video source: {VIDEO_SRC}")
            logging.info(f"📡 RTSP stream: {RTSP_TARGET}")
            logging.info("=" * 60)
            logging.info("⏳ Waiting for data stream...")

            return True

        except Exception as e:
            logging.error(f"❌ Initialization failed: {e}", exc_info=True)
            return False

    def run(self):
        """Main loop."""
        retry_count = 0
        max_retries = 5
        frame_count = 0

        try:
            while self.video_source.is_opened:
                success, frame = self.video_source.get_frame()

                if not success or frame is None:
                    retry_count += 1
                    if retry_count < max_retries:
                        logging.warning(f"⏳ Waiting for video frames... ({retry_count}/{max_retries})")
                        time.sleep(0.1)
                        continue
                    else:
                        logging.error("⚠️ Unable to get video frame, exiting loop")
                        break

                retry_count = 0

                self.streamer.push_frame(frame)

                current_fps = self.fps_counter.tick()
                frame_count += 1

                if frame_count % 50 == 0:
                    logging.info(
                        f"✅ FPS: {current_fps:.1f}| "
                        f"UTC: {datetime.now(timezone.utc).strftime('%H:%M:%S.%f')[:-3]}"
                    )

                self.fps_counter.sleep_to_maintain_fps()

        except KeyboardInterrupt:
            logging.info("\n🛑 User interrupted, stopping...")
        except Exception as e:
            logging.error(f"❌ Error occurred: {e}", exc_info=True)

    def cleanup(self):
        """Clean up resources."""
        logging.info("🧹 Cleaning up resources...")

        if self.video_source:
            try:
                self.video_source.release()
            except Exception as e:
                logging.warning(f"⚠️ video_source.release() failed: {e}")

        if self.streamer:
            try:
                self.streamer.stop()  # 内部负责保存 avi + metadata
            except Exception as e:
                logging.error(f"❌ streamer.stop() failed: {e}")

        logging.info("👋 Program exiting")


def main():
    app = MotionTerminalApp()

    if not app.initialize():
        logging.error("Initialization failed, exiting program")
        return

    try:
        app.run()
    finally:
        app.cleanup()


if __name__ == "__main__":
    main()