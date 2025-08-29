from computer_vision.video_sources.i_video_source import IVideoSource

class WebcamSource(IVideoSource):
    """
    Video source implementation for a local webcam.
    """
    def get_video_source(self) -> int:
        """
        Retrieve the webcam source.

        Returns:
            int: The device index of the webcam (default is `0`, the primary camera).
        """
        return 0
    