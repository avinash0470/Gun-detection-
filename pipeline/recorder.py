import cv2
import os
import logging
import threading
from datetime import datetime

logger = logging.getLogger("GunDetectionPipeline")


class ThreadedCamera:
    """
    Reads frames from the camera in a background thread so the main loop
    always gets the latest frame instantly instead of waiting on a stale buffer.
    This eliminates the lag caused by YOLO inference blocking the read loop.
    """

    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret = False
        self.frame = None
        self.lock = threading.Lock()
        self.stopped = False

        # Read one frame to initialize
        self.ret, self.frame = self.cap.read()

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        """Continuously grabs the latest frame in a background thread."""
        while not self.stopped:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        """Returns the most recent frame (never stale)."""
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def isOpened(self):
        return self.cap.isOpened()

    def get(self, prop):
        return self.cap.get(prop)

    def release(self):
        self.stopped = True
        self.thread.join(timeout=2.0)
        self.cap.release()


class VideoRecorder:
    """Records annotated detection output frames to a timestamped video file."""

    def __init__(self, output_dir: str = "recordings"):
        self.output_dir = output_dir
        self.writer = None
        self.output_path = None
        self.frame_size = None
        self.fps = 20.0

        os.makedirs(self.output_dir, exist_ok=True)

    def start(self, frame_width: int, frame_height: int, fps: float = 20.0):
        """Initializes the video writer with a timestamped filename."""
        self.fps = fps
        self.frame_size = (frame_width, frame_height)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"detection_{timestamp}.avi"
        self.output_path = os.path.join(self.output_dir, filename)

        # XVID codec in AVI container — plays reliably on all Windows media players
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, self.frame_size)

        if self.writer.isOpened():
            logger.info(f"Recording started: {self.output_path}")
        else:
            logger.error(f"Failed to open video writer for: {self.output_path}")
            self.writer = None

    def write_frame(self, frame):
        """Writes a single annotated frame to the output video."""
        if self.writer and self.writer.isOpened():
            self.writer.write(frame)

    def stop(self):
        """Releases the video writer and finalizes the file."""
        if self.writer:
            self.writer.release()
            self.writer = None
            logger.info(f"Recording saved: {self.output_path}")

    @property
    def is_recording(self) -> bool:
        return self.writer is not None and self.writer.isOpened()
