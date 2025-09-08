from computer_vision.video_sources.i_video_source import IVideoSource
import yt_dlp
import os
import tempfile
import uuid
from computer_vision.utils.logger import setup_logger

logger = setup_logger(__name__)

class YouTubeSource(IVideoSource):
    """
    Video source implementation for downloading and reading YouTube videos.
    """

    def __init__(self, yt_url):
        """
        Initialize the YouTube source.

        Args:
            yt_url (str): The full URL of the YouTube video to download.
        """

        self.yt_url = yt_url

    def get_video_source(self) -> str:
       """
        Download the YouTube video and return the local file path.

        Side Effects:
            - Downloads a video file using `yt-dlp`.
            - Stores the file temporarily in the system temp directory.

        Returns:
            str: Absolute path to the downloaded `.mp4` video file.

        Raises:
            Exception: If the download fails (errors from `yt-dlp`).
        """
       temp_filename = f"youtube_video_{uuid.uuid4().hex}.mp4"
       temp_path = os.path.join(tempfile.gettempdir(), temp_filename)
       ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': temp_path,
            'quiet': True,
        }

       try:
            logger.info(f"Downloading video using yt-dlp: {self.yt_url}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([self.yt_url])
            return temp_path

       except Exception as e:
            print(f"[ERROR] yt-dlp failed: {e}")
            raise
