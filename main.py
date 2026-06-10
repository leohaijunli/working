"""
主程序入口
整合视频源、检测器、RTSP 推流和 Rerun 可视化
"""

import time
import logging
from typing import Optional
from stream_controller import StreamController
from video_source import UniversalVideoSource
from ws_data_server import RerunVisualizer
import rerun as rr
import cv2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ==================== 配置常量 ====================
VIDEO_SRC = 0 # 使用摄像头,也可以替换为视频文件路径或 RTSP URL

#VIDEO_SRC = "7.mp4"
#VIDEO_SRC = "rtsp://192.168.1.91:8554/video"
#VIDEO_SRC = "rtsp://10.0.0.129:8554/interceptor"
RTSP_TARGET = "rtsp://127.0.0.1:8554/live/mystream"
TARGET_FPS = 30
FRAME_TIME = 1.0 / TARGET_FPS
VIDEO_WIDTH = 1456
VIDEO_HEIGHT = 1088
# VIDEO_WIDTH = 640
# VIDEO_HEIGHT = 480
VIDEO_FPS = 30

fourcc = cv2.VideoWriter_fourcc(*'MJPG')

# 确保 VIDEO_WIDTH 和 VIDEO_HEIGHT 是整数
output_video = cv2.VideoWriter(
    'output_video.avi', 
    fourcc, 
    float(VIDEO_FPS), 
    (int(VIDEO_WIDTH), int(VIDEO_HEIGHT))
)

HEF_PATH = "yolov11n.hef"
CONF_THRESHOLD = 0.2


class FPSCounter:
    """FPS 计数器和帧率控制"""
    
    def __init__(self, target_fps: float = 30, smoothing: float = 0.9):
        """
        初始化 FPS 计数器
        
        Args:
            target_fps: 目标帧率
            smoothing: 平滑系数 (0-1, 越大越平滑)
        """
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        self.smoothing = smoothing
        self.fps = target_fps
        self.last_time = time.time()
    
    def tick(self) -> float:
        """
        更新并返回当前 FPS
        
        Returns:
            平滑后的 FPS 值
        """
        current_time = time.time()
        elapsed = current_time - self.last_time
        
        if elapsed > 0:
            current_fps = 1.0 / elapsed
            self.fps = self.smoothing * self.fps + (1 - self.smoothing) * current_fps
        
        self.last_time = current_time
        return self.fps
    
    def sleep_to_maintain_fps(self):
        """睡眠以维持目标帧率"""
        elapsed = time.time() - self.last_time
        sleep_time = self.frame_time - elapsed
        
        if sleep_time > 0:
            time.sleep(sleep_time)


class MotionTerminalApp:
    """MX Motion Terminal 主应用"""
    
    def __init__(self):
        self.video_source: Optional[UniversalVideoSource] = None
        self.streamer: Optional[StreamController] = None
        self.detector: Optional[HailoDetector] = None
        self.visualizer: Optional[RerunVisualizer] = None
        self.fps_counter: Optional[FPSCounter] = None
        
    def initialize(self) -> bool:
        """
        初始化所有组件
        
        Returns:
            初始化是否成功
        """
        try:
            # 1. 初始化 Rerun 可视化
            logging.info("🎨 初始化 Rerun 可视化...")
            self.visualizer = RerunVisualizer(
                video_width=VIDEO_WIDTH,
                video_height=VIDEO_HEIGHT
            )
            local_ip = self.visualizer.initialize()
            
            # 2. 初始化视频源
            logging.info("📹 初始化视频源...")
            self.video_source = UniversalVideoSource(
                source=VIDEO_SRC,
                width=VIDEO_WIDTH,
                height=VIDEO_HEIGHT,
                fps=TARGET_FPS
            )
            if not self.video_source.open():
                logging.error("❌ 无法打开视频源")
                return False
            
            # 3. 启动 RTSP 推流
            logging.info("📡 启动 RTSP 推流...")
            self.streamer = StreamController(
                rtsp_url=RTSP_TARGET,
                res=(VIDEO_WIDTH, VIDEO_HEIGHT),
                fps=TARGET_FPS
            )
            self.streamer.start()
            
            # # 4. 初始化检测器
            # logging.info("🤖 初始化检测器...")
            # self.detector = HailoDetector(
            #     hef_path=HEF_PATH,
            #     conf_threshold=CONF_THRESHOLD,
            #     verbose=False
            # )
            
            # 5. FPS 计数器
            self.fps_counter = FPSCounter(target_fps=TARGET_FPS)
            
            # 打印启动信息
            logging.info("=" * 60)
            logging.info("🚀 系统启动完成")
            logging.info(f"📹 视频源: {VIDEO_SRC}")
            logging.info(f"📡 RTSP 推流: {RTSP_TARGET}")
            logging.info(f"🌐 Rerun 查看: http://{local_ip}:9090")
            logging.info("=" * 60)
            logging.info("⏳ 等待数据流...")
            
            return True
            
        except Exception as e:
            logging.error(f"❌ 初始化失败: {e}", exc_info=True)
            return False
    
    def run(self):
        """主循环"""
        retry_count = 0
        max_retries = 5
        frame_count = 0

        # specify the log file to save
        # spawn=False means no need to display on local screen.
    
        # save_path = "rpi5_video.rrd"
        # rr.init("UAV_Tracker", spawn=False)
        # rr.save(save_path) 
        # print(f"💾 Rerun log saved to: {save_path}")
        
        try:
            while self.video_source.is_opened:
                # 读取帧
                success, frame = self.video_source.get_frame()
                
                if not success or frame is None:
                    retry_count += 1
                    if retry_count < max_retries:
                        logging.warning(f"⏳ 等待视频帧... ({retry_count}/{max_retries})")
                        time.sleep(0.1)
                        continue
                    else:
                        logging.error("⚠️ 无法获取视频帧,退出循环")
                        break
                
                retry_count = 0  # 重置重试计数
                
                # # 执行检测
                # result = self.detector._detect_from_array(frame)
                # output_frame = result["output_image"]
                # detections = result.get("results", [])
                
                # Rerun 可视化
                self.visualizer.log_frame(frame)
                
                # RTSP 推流
                output_frame = frame
                self.streamer.push_frame(output_frame)
                
                # 更新 FPS
                current_fps = self.fps_counter.tick()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #rr.log("video/original", rr.Image(frame_rgb))
                output_video.write(frame)

                
                # # 定期打印状态
                frame_count = self.visualizer.get_frame_count()
                # if frame_count % 50 == 0:
                #     logging.info(
                #         f"✅ Frame: {frame_count} | "
                #         f"FPS: {current_fps:.1f} | "
                #         f"Detections: {len(detections)}"
                #     )
                
                # 帧率控制
                self.fps_counter.sleep_to_maintain_fps()
        
        except KeyboardInterrupt:
            logging.info("\n🛑 用户中断,正在停止...")
        except Exception as e:
            logging.error(f"❌ 发生错误: {e}", exc_info=True)
    
    def cleanup(self):
        """清理资源"""
        logging.info("🧹 清理资源...")
        
        if self.video_source:
            self.video_source.release()
        
        if self.streamer:
            self.streamer.stop()
        
        logging.info("👋 程序退出")


def main():
    """主函数入口"""
    app = MotionTerminalApp()
    
    if not app.initialize():
        logging.error("初始化失败,程序退出")
        return
    
    try:
        app.run()
    finally:
        app.cleanup()


if __name__ == "__main__":
    main()