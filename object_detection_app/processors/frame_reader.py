import cv2

class FrameReader:
    def __init__(self, cap):
        self.cap = cap

    def read(self):
        success, frame = self.cap.read()
        return frame if success else None
