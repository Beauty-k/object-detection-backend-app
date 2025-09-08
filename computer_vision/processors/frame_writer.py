import os
import cv2

class FrameWriter:
    """
    Writes video frames to a file using OpenCV.
    """
    
    def __init__(self, output_path, fps, width, height):
        """
        Initializes the video writer and ensures the output directory exists.

        Args:
            output_path (str): Path to save the video file.
            fps (float): Frames per second for the output video.
            width (int): Frame width.
            height (int): Frame height.
        """

        self.output_path = output_path
        self._ensure_output_directory()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    def _ensure_output_directory(self):
        """Creates the output directory if it does not exist."""
        folder = os.path.dirname(self.output_path)
        if folder:
            os.makedirs(folder, exist_ok=True)

    def write(self, frame):
        """
        Writes a single frame to the video file.

        Args:
            frame (np.ndarray): The image frame to write.
        """
        self.writer.write(frame)

    def release(self):
        self.writer.release()
