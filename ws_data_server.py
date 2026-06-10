"""
Rerun visualization manager.
Provides real-time visualization of video streams and detection results.
"""

import cv2
import numpy as np
import rerun as rr
import socket
import threading
import logging
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class RerunVisualizer:
    """Optimized Rerun visualization manager."""
    
    def __init__(
        self, 
        app_name: str = "MX_Motion_Terminal",
        video_width: int = 640,
        video_height: int = 480
    ):
        """
        Initialize the Rerun visualizer.
        
        Args:
            app_name: Application name.
            video_width: Video width.
            video_height: Video height.
        """
        self.app_name = app_name
        self.video_width = video_width
        self.video_height = video_height
        self.frame_count = 0
        self._lock = threading.Lock()
        self._initialized = False
        
    def initialize(self) -> str:
        """
        Initialize the Rerun service and return the local IP.

        Returns:
            Local LAN IP address.
        """
        if self._initialized:
            logging.warning("Rerun is already initialized; skipping redundant initialization")
            return self._get_local_ip()
        
        rr.init(self.app_name, spawn=False)
        
        # Get the local IP address
        local_ip = self._get_local_ip()
        
        # Start the service
        server_uri = rr.serve_grpc()
        rr.serve_web_viewer(connect_to=server_uri)
        
        self._initialized = True
        
        logging.info("=" * 60)
        logging.info("🌐 Rerun service started")
        logging.info(f"🌐 Web viewer: http://{local_ip}:9090")
        logging.info("=" * 60)
        
        return local_ip
    
    @staticmethod
    def _get_local_ip() -> str:
        """
        Get the local LAN IP address.

        Returns:
            IP address string.
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception as e:
            logging.warning(f"Failed to get local IP: {e}, using default")
            return "127.0.0.1"
    
    def log_frame(
        self, 
        frame: np.ndarray, 
        detections: Optional[List[Dict]] = None,
        show_status: bool = True
    ):
        """
        Log a single frame to Rerun.
        
        Args:
            frame: Image in BGR format (OpenCV format).
            detections: List of detection results, in the format:
                [{"x1": float, "y1": float, "x2": float, "y2": float,
                  "class_name": str, "confidence": float}, ...]
            show_status: Whether to display the system status overlay.
        """
        if not self._initialized:
            logging.error("Rerun is not initialized; call initialize() first")
            return
        
        with self._lock:
            self.frame_count += 1
            
            # Set the timeline
            rr.set_time("frame", sequence=self.frame_count)
            
            # 1. Log image (convert to RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rr.log("camera/image", rr.Image(frame_rgb))
            
            # 2. Log detections
            if detections:
                self._log_detections(detections)
            else:
                # Clear previous detections
                rr.log("camera/image/detections", rr.Clear(recursive=False))
            
            # 3. System status indicator
            # if show_status:
            #     self._log_status()
            
            # Log confirmation for the first frame or periodically
            if self.frame_count == 1:
                logging.info("📤 First frame sent to Rerun")
            elif self.frame_count % 100 == 0:
                logging.debug(f"✅ Logged frame {self.frame_count}")
    
    def _log_detections(self, detections: List[Dict]):
        """
        Log detection boxes to Rerun.
        
        Args:
            detections: List of detection results.
        """
        bboxes = []
        labels = []
        colors = []
        
        for det in detections:
            # Extract coordinates
            x1 = float(det.get("x1", 0))
            y1 = float(det.get("y1", 0))
            x2 = float(det.get("x2", 0))
            y2 = float(det.get("y2", 0))
            
            # Convert normalized coordinates to pixel coordinates
            if x2 <= 1.0:
                x1 *= self.video_width
                x2 *= self.video_width
                y1 *= self.video_height
                y2 *= self.video_height
            
            bboxes.append([x1, y1, x2, y2])
            
            # Label
            class_name = det.get("class_name") or det.get("label") or "target"
            conf = det.get("confidence", 0.0)
            #labels.append(f"{class_name} {conf:.2f}")
            labels.append(class_name)  # Show only the class name
            
            # Color (red)
            colors.append([255, 0, 0])
        
        rr.log(
            "camera/image/detections",
            rr.Boxes2D(
                array=bboxes,
                array_format=rr.Box2DFormat.XYXY,
                labels=labels,
                colors=colors
            )
        )
        
        # Log info on first detections
        if self.frame_count <= 3:
            logging.info(f"   Detected {len(bboxes)} targets")
    
    def _log_status(self):
        """Log system status indicator."""
        rr.log(
            "camera/image/status",
            rr.Boxes2D(
                array=[[10, 10, 120, 40]],
                array_format=rr.Box2DFormat.XYXY,
                labels=["SYS_OK"],
                colors=[[0, 255, 0]]
            )
        )
    
    def get_frame_count(self) -> int:
        """Get the current frame count."""
        return self.frame_count
    
    def reset(self):
        """Reset the frame count."""
        with self._lock:
            self.frame_count = 0
            logging.info("Rerun frame count reset")