from object_detection_app.video_sources.video_source_interface import VideoSourceInterface
import yt_dlp
import os
import tempfile
import uuid
class YouTubeSource(VideoSourceInterface):
    def __init__(self, yt_url):
        self.yt_url = yt_url

    def get_video_source(self):
       temp_filename = f"youtube_video_{uuid.uuid4().hex}.mp4"
       temp_path = os.path.join(tempfile.gettempdir(), temp_filename)
       ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': temp_path,
            'quiet': True,
        }

       try:
            print(f"[INFO] Downloading video using yt-dlp: {self.yt_url}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([self.yt_url])
            return temp_path

       except Exception as e:
            print(f"[ERROR] yt-dlp failed: {e}")
            raise
