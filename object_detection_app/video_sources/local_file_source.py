from video_sources.video_source_interface import VideoSourceInterface
import os

class LocalFileSource(VideoSourceInterface):
    def __init__(self, file_path):
        self.file_path = file_path
    
    def get_video_source(self):
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"File {self.file_path} not found")
        return self.file_path
    