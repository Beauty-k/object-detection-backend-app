import os
import cv2

class FrameWriter:
    def __init__(self, output_path, fps, width, height):
        self.output_path = output_path
        self._ensure_output_directory()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    def _ensure_output_directory(self):
        folder = os.path.dirname(self.output_path)
        if folder:
            os.makedirs(folder, exist_ok=True)

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()
