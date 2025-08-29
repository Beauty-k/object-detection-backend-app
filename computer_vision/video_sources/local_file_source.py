from computer_vision.video_sources.i_video_source import IVideoSource
import os

class LocalFileSource(IVideoSource):
    def __init__(self, file_path):
        self.file_path = file_path
    
    def get_video_source(self):
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"File {self.file_path} not found")
        return self.file_path
    