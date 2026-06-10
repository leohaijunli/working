import subprocess
import threading
import queue
import time
import json
import cv2
from datetime import datetime, timezone
import signal


class StreamController:
    def __init__(self, rtsp_url, res=(1280, 720), fps=25):
        self.rtsp_url = rtsp_url
        self.res = res
        self.fps = fps
        self.pipe = None
        self.metadata = []
        self.metadata_path = "output_video.json"
        self._push_count = 0
        self._lock = threading.Lock()
        self._restarting = False
        self._stopping = False

        self._stream_queue = queue.Queue(maxsize=10)
        self._record_queue = queue.Queue(maxsize=10)

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.output_video = cv2.VideoWriter(
            'output_video.avi',
            fourcc,
            float(self.fps),
            (int(self.res[0]), int(self.res[1]))
        )

        self._stream_thread = threading.Thread(target=self._stream_worker, daemon=True)
        self._record_thread = threading.Thread(target=self._record_worker, daemon=True)

    def start(self):
        """Start FFmpeg process and worker threads."""
        self._start_pipe()
        self._stream_thread.start()
        self._record_thread.start()

    def _start_pipe(self):
        """Start (or restart) the FFmpeg subprocess."""
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
            time.sleep(0.1)
            if self.pipe.poll() is not None:
                err = self.pipe.stderr.read().decode(errors='replace')
                print(f"❌ [Stream] FFmpeg exited immediately:\n{err}")
                self.pipe = None
                return
            print(f"🚀 [Stream] RTSP pipe established")
        except Exception as e:
            print(f"❌ [Stream] Failed to start: {e}")

    def _restart_pipe(self):
        """后台线程：清理旧 pipe 再重启。"""
        try:
            if self.pipe:
                try:
                    self.pipe.stdin.close()
                except Exception:
                    pass
                self.pipe.kill()
            time.sleep(0.5)
            self._start_pipe()
        finally:
            with self._lock:
                self._restarting = False

    def _stream_worker(self):
        """推流线程：从队列取帧写入 FFmpeg stdin。"""
        while True:
            try:
                item = self._stream_queue.get(timeout=1)
            except queue.Empty:
                if self._stopping:
                    break
                continue

            if item is None:
                self._stream_queue.task_done()
                break

            pipe_dead = not self.pipe or self.pipe.poll() is not None
            if pipe_dead:
                with self._lock:
                    if not self._restarting:
                        self._restarting = True
                        threading.Thread(target=self._restart_pipe, daemon=True).start()
                self._stream_queue.task_done()
                continue

            try:
                self.pipe.stdin.write(item.tobytes())
            except OSError:
                pass
            finally:
                self._stream_queue.task_done()

    def _record_worker(self):
        """录制线程：写文件成功后才写 metadata，保证两者严格对应。"""
        record_idx = 0
        while True:
            try:
                item = self._record_queue.get(timeout=1)
            except queue.Empty:
                if self._stopping:
                    break
                continue

            if item is None:
                self._record_queue.task_done()
                break

            frame, utc_time = item
            self.output_video.write(frame)
            self.metadata.append({
                "frame_idx": record_idx,
                "utc_time": utc_time
            })
            record_idx += 1
            self._record_queue.task_done()

    def push_frame(self, frame):
        """Resize and dispatch frame to both worker queues."""
        out_frame = cv2.resize(frame, self.res)
        utc_time = datetime.now(timezone.utc).isoformat()

        try:
            self._stream_queue.put_nowait(out_frame)
        except queue.Full:
            print(f"⚠️ [Stream] Stream queue full, push #{self._push_count} dropped")

        try:
            self._record_queue.put_nowait((out_frame, utc_time))
        except queue.Full:
            print(f"⚠️ [Record] Record queue full, push #{self._push_count} dropped")

        self._push_count += 1

    def stop(self):
        """Drain queues, stop workers, release resources, save metadata."""
        print(f"[Stop] 1. setting _stopping flag")
        self._stopping = True

        print(f"[Stop] 2. draining queues and sending sentinel")
        for q in (self._stream_queue, self._record_queue):
            while True:
                try:
                    q.put_nowait(None)
                    break
                except queue.Full:
                    try:
                        q.get_nowait()
                        q.task_done()
                    except queue.Empty:
                        pass

        print(f"[Stop] 3. joining stream thread")
        self._stream_thread.join(timeout=5)
        print(f"[Stop] 4. joining record thread, stream thread alive={self._stream_thread.is_alive()}")
        self._record_thread.join(timeout=5)
        print(f"[Stop] 5. threads done, record thread alive={self._record_thread.is_alive()}")

        if self.pipe:
            try:
                self.pipe.stdin.close()
                self.pipe.wait(timeout=2)
            except Exception:
                self.pipe.kill()

        print(f"[Stop] 6. releasing VideoWriter")
        self.output_video.release()

        print(f"[Stop] 7. saving metadata, len={len(self.metadata)}")
        try:
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            print(f"💾 [Stream] Metadata saved → {self.metadata_path} ({len(self.metadata)} frames)")
        except OSError as e:
            print(f"❌ [Stream] Failed to save metadata: {e}")
        print(f"[Stop] 8. done")


if __name__ == "__main__":
    RTSP_URL = "rtsp://localhost:8554/live"
    cap = cv2.VideoCapture(0)

    controller = StreamController(rtsp_url=RTSP_URL)
    controller.start()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ [Main] Failed to read frame")
                break
            controller.push_frame(frame)

    except KeyboardInterrupt:
        print("\n[Main] Ctrl+C received")

    finally:
        cap.release()
        controller.stop()