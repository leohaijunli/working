"""
Rerun 可视化管理器
提供视频流和检测结果的实时可视化功能
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
    """优化的 Rerun 可视化管理器"""
    
    def __init__(
        self, 
        app_name: str = "MX_Motion_Terminal",
        video_width: int = 640,
        video_height: int = 480
    ):
        """
        初始化 Rerun 可视化器
        
        Args:
            app_name: 应用名称
            video_width: 视频宽度
            video_height: 视频高度
        """
        self.app_name = app_name
        self.video_width = video_width
        self.video_height = video_height
        self.frame_count = 0
        self._lock = threading.Lock()
        self._initialized = False
        
    def initialize(self) -> str:
        """
        初始化 Rerun 服务并返回本机 IP
        
        Returns:
            本机局域网 IP 地址
        """
        if self._initialized:
            logging.warning("Rerun 已经初始化,跳过重复初始化")
            return self._get_local_ip()
        
        rr.init(self.app_name, spawn=False)
        
        # 获取本机 IP
        local_ip = self._get_local_ip()
        
        # 启动服务
        server_uri = rr.serve_grpc()
        rr.serve_web_viewer(connect_to=server_uri)
        
        self._initialized = True
        
        logging.info("=" * 60)
        logging.info("🌐 Rerun 服务已启动")
        logging.info(f"🌐 网页查看: http://{local_ip}:9090")
        logging.info("=" * 60)
        
        return local_ip
    
    @staticmethod
    def _get_local_ip() -> str:
        """
        获取本机局域网 IP
        
        Returns:
            IP 地址字符串
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception as e:
            logging.warning(f"获取本机 IP 失败: {e}, 使用默认值")
            return "127.0.0.1"
    
    def log_frame(
        self, 
        frame: np.ndarray, 
        detections: Optional[List[Dict]] = None,
        show_status: bool = True
    ):
        """
        记录单帧数据到 Rerun
        
        Args:
            frame: BGR 格式的图像 (OpenCV 格式)
            detections: 检测结果列表,格式为:
                [{"x1": float, "y1": float, "x2": float, "y2": float,
                  "class_name": str, "confidence": float}, ...]
            show_status: 是否显示系统状态框
        """
        if not self._initialized:
            logging.error("Rerun 未初始化,请先调用 initialize()")
            return
        
        with self._lock:
            self.frame_count += 1
            
            # 设置时间轴
            rr.set_time("frame", sequence=self.frame_count)
            
            # 1. 记录图像 (转为 RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rr.log("camera/image", rr.Image(frame_rgb))
            
            # 2. 记录检测框
            if detections:
                self._log_detections(detections)
            else:
                # 清除之前的检测框
                rr.log("camera/image/detections", rr.Clear(recursive=False))
            
            # 3. 系统状态指示器
            if show_status:
                self._log_status()
            
            # 首帧或定期打印确认信息
            if self.frame_count == 1:
                logging.info("📤 首帧数据已发送到 Rerun")
            elif self.frame_count % 100 == 0:
                logging.debug(f"✅ 已记录第 {self.frame_count} 帧")
    
    def _log_detections(self, detections: List[Dict]):
        """
        记录检测框到 Rerun
        
        Args:
            detections: 检测结果列表
        """
        bboxes = []
        labels = []
        colors = []
        
        for det in detections:
            # 提取坐标
            x1 = float(det.get("x1", 0))
            y1 = float(det.get("y1", 0))
            x2 = float(det.get("x2", 0))
            y2 = float(det.get("y2", 0))
            
            # 归一化坐标转换为像素坐标
            if x2 <= 1.0:
                x1 *= self.video_width
                x2 *= self.video_width
                y1 *= self.video_height
                y2 *= self.video_height
            
            bboxes.append([x1, y1, x2, y2])
            
            # 标签
            class_name = det.get("class_name") or det.get("label") or "target"
            conf = det.get("confidence", 0.0)
            #labels.append(f"{class_name} {conf:.2f}")
            labels.append(class_name)  # 仅显示类别名称
            
            # 颜色 (红色)
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
        
        # 首次检测时打印信息
        if self.frame_count <= 3:
            logging.info(f"   检测到 {len(bboxes)} 个目标")
    
    def _log_status(self):
        """记录系统状态指示器"""
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
        """获取当前帧计数"""
        return self.frame_count
    
    def reset(self):
        """重置帧计数"""
        with self._lock:
            self.frame_count = 0
            logging.info("Rerun 帧计数已重置")