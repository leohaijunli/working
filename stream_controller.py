import subprocess
import cv2
import time

class StreamController:
    def __init__(self, rtsp_url, res=(1280, 720), fps=25):
        self.rtsp_url = rtsp_url
        self.res = res
        self.fps = fps
        self.pipe = None

    def start(self):
        """启动 FFmpeg 进程"""
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24', '-s', f"{self.res[0]}x{self.res[1]}", '-r', str(self.fps),
            '-i', '-', 
            '-c:v', 'libx264', '-preset', 'veryfast', '-tune', 'zerolatency',
            '-b:v', '5000k', '-bufsize', '500k', '-g', str(self.fps),
            '-pix_fmt', 'yuv420p', '-f', 'rtsp', self.rtsp_url
        ]
        try:
            self.pipe = subprocess.Popen(
                ffmpeg_cmd, stdin=subprocess.PIPE, 
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )
            print(f"🚀 [Stream] 视频流管道已建立")
        except Exception as e:
            print(f"❌ [Stream] 启动失败: {e}")

    def push_frame(self, frame):
        """非阻塞推流"""
        if not self.pipe or self.pipe.poll() is not None:
            self.start() # 如果管道断了，尝试重启
            
        try:
            # 缩放至推流分辨率
            out_frame = cv2.resize(frame, self.res)
            self.pipe.stdin.write(out_frame.tobytes())
            return True
        except Exception:
            pass # 忽略单帧推流失败，保证主循环流畅
            return False

    def stop(self):
        if self.pipe:
            try:
                self.pipe.stdin.close()
                self.pipe.wait(timeout=2)
            except:
                self.pipe.kill()