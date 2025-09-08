
from computer_vision.video_sources.local_file_source import LocalFileSource
from computer_vision.video_sources.webcam_source import WebcamSource
from computer_vision.video_sources.youtube_source import YouTubeSource
from computer_vision.factories.detector_factory import DetectorFactory
from computer_vision.processors.video_processor import VideoProcessor
# from calculators.test_sample_distance import GeometryDistanceCalculator

# video_source = WebcamSource()
# video_source = LocalFileSource("temp/wall_video.mp4")
video_source = LocalFileSource("temp/sample_video_004.mp4")

# youtube_url = "https://www.youtube.com/shorts/nLXBinY7BwI" 
# video_source = YouTubeSource(youtube_url)
detector = DetectorFactory.create("yolov8", model_path = "runs/detect/train14/weights/best.pt", device = "cuda")

# Create Faster R-CNN
# detector = DetectorFactory.create("rcnn",  device = "cuda")
# Create SSD
# detector = DetectorFactory.create("ssd", device="cuda")

video_processor = VideoProcessor(video_source, display=True)
output_path = "static/output.mp4"

label1 = "blessing_card"
label2 = "wallet"
target_labels = [label1, label2]
detections = video_processor.process_video(detector, target_labels)
