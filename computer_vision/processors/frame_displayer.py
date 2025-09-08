import cv2
from computer_vision.utils.logger import setup_logger

logger = setup_logger(__name__)

class FrameDisplayer:
    def __init__(self, window_name="Live Detection", wait_time=25, quit_key="q"):
        self.window_name = window_name
        self.wait_time = wait_time
        self.quit_key = quit_key

    def display(self, frame):
        """Show the frame and check for quit key."""
        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(self.wait_time) & 0xFF
        if key == ord(self.quit_key):
            logger.info("Display stopped by user.")
            raise StopIteration

    def close(self):
        """Destroy the window when done."""
        cv2.destroyAllWindows()
