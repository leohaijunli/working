import cv2
import time
from ultralytics import YOLO
from ws_data_server import WSDataServer
from stream_controller import StreamController
from video_source import UniversalVideoSource

# --- 配置 ---
VIDEO_SRC = '2.mp4'  # 视频文件路径，或摄像头索引（如0）
RTSP_TARGET = "rtsp://127.0.0.1:8554/live/mystream"
TARGET_FPS = 25
FRAME_TIME = 1.0 / TARGET_FPS

def main():
    # 1. 初始化视频源
    video_source = UniversalVideoSource(source=VIDEO_SRC, width=640, height=480, fps=TARGET_FPS)
    
    if not video_source.open():
        print("❌ 视频源打开失败，程序退出")
        return

    ws_server = WSDataServer(port=8765)
    ws_server.start()
    
    streamer = StreamController(rtsp_url=RTSP_TARGET, res=(640, 480), fps=TARGET_FPS)
    streamer.start()

    # 2. 初始化 YOLO（可暂时注释掉测试纯推流）
    # model = YOLO('best_ncnn_model', task='detect')

    print(f"📹 视频分辨率: {video_source._width}x{video_source._height}")
    print("🟢 系统就绪，开始推流...")

    frame_count = 0

    try:
        while video_source.is_opened:
            start_time = time.time()

            success, frame = video_source.get_frame()
            
            if not success or frame is None:
                # 第一次可能还没捕获到帧，给一点缓冲时间
                if frame_count < 5:
                    print(f"⏳ 等待视频帧... ({frame_count+1}/5)")
                    time.sleep(0.1)
                    continue
                else:
                    print("⚠️ 无法获取视频帧，退出循环")
                    break

            frame_count += 1

            # 可选：显示本地预览（调试用）
            # cv2.imshow("Preview", frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            # 推送帧到 RTSP
            push_success = streamer.push_frame(frame)
            if not push_success:
                print(f"⚠️ 第 {frame_count} 帧推送失败")

            # 帧率控制
            elapsed = time.time() - start_time
            sleep_time = FRAME_TIME - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # 处理太慢时跳过一帧
                pass

            if frame_count % 50 == 0:
                print(f"✅ 已成功推送 {frame_count} 帧")

    except KeyboardInterrupt:
        print("\n用户中断，正在关闭...")
    except Exception as e:
        print(f"运行异常: {e}")
    finally:
        print("正在安全关闭系统...")
        video_source.release()
        streamer.stop()
        print("👋 程序已退出。")


if __name__ == "__main__":
    main()