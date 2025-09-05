
from video_sources.local_file_source import LocalFileSource
from video_sources.webcam_source import WebcamSource
from video_sources.youtube_source import YouTubeSource
from factories.detector_factory import DetectorFactory
# from detectors.object_detector import ObjectDetector
from processors.video_processor import VideoProcessor
from calculators.distance_calculator import DistanceCalculator
from detectors.yolov8_detector import YOLOv8Detector
# from calculators.test_sample_distance import GeometryDistanceCalculator

# video_source = WebcamSource()
# video_source = LocalFileSource("temp/wall_video.mp4")
video_source = LocalFileSource("temp/sample_video_004.mp4")

# youtube_url = "https://www.youtube.com/shorts/nLXBinY7BwI" 
# video_source = YouTubeSource(youtube_url)
detector = DetectorFactory.create("yolov8", model_path = "runs/detect/train14/weights/best.pt", device = "cuda")
# detector = YOLOv8Detector("runs/detect/train14/weights/best.pt", "cuda")
# detector = ObjectDetector()
# Create Faster R-CNN
# detector = DetectorFactory.create("rcnn", "cuda")
# Create SSD
# detector = DetectorFactory.create("ssd", device="cpu")

video_processor = VideoProcessor(video_source, display=True)
output_path = "static/output.mp4"

label1 = "blessing_card"
label2 = "wallet"
target_labels = [label1, label2]
detections = video_processor.process_video(detector, target_labels)
