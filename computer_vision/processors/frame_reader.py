import cv2

class FrameReader:
    def __init__(self, cap):
        self.cap = cap

    def read(self):
        """
        Reads the next frame from the video capture.
        
        Returns:
            np.ndarray or None: The next frame if successful, else None.
        """
        success, current_frame = self.cap.read()
        return current_frame if success else None
