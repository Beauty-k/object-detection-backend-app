from abc import ABC, abstractmethod
import os
import uuid
import yt_dlp
import tempfile

class VideoSourceInterface(ABC):
    @abstractmethod
    def get_video_source(self):
        pass



    
