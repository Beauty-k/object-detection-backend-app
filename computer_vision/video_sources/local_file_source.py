from video_sources.i_video_source import IVideoSource
import os

class LocalFileSource(IVideoSource):
    def __init__(self, file_path):
        self.file_path = file_path
    
    def get_video_source(self) -> str:
        """
        Retrieve the video source path.

        Returns:
            str: Path to the video file.

        Raises:
            FileNotFoundError: If the specified file does not exist.
        """
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"File {self.file_path} not found")
        return self.file_path
    