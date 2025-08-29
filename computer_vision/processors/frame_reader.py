import cv2

class FrameReader:
    def __init__(self, cap):
        self.cap = cap

    def read(self):
        success, current_frame = self.cap.read()
        return current_frame if success else None
