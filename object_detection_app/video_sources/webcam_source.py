from object_detection_app.video_sources.video_source_interface import VideoSourceInterface

class WebcamSource(VideoSourceInterface):
    def get_video_source(self):
        return 0