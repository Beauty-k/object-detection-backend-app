
from video_sources.local_file_source import LocalFileSource
from video_sources.webcam_source import WebcamSource
from video_sources.youtube_source import YouTubeSource
from detector.object_detector import ObjectDetector
from processors.video_processor import VideoProcessor
from calculators.distance_calculator import DistanceCalculator
# from calculators.test_sample_distance import GeometryDistanceCalculator

# video_source = WebcamSource()
# video_source = LocalFileSource("temp/wall_video.mp4")
video_source = LocalFileSource("temp/sample_video_002.mp4")

# youtube_url = "https://www.youtube.com/shorts/nLXBinY7BwI" 
# video_source = YouTubeSource(youtube_url)
detector = ObjectDetector("runs/detect/train14/weights/best.pt")
# detector = ObjectDetector("models/yolov8s.pt")

video_processor = VideoProcessor(video_source)
output_path = "static/output.mp4"
VideoProcessor.ensure_output_directory(output_path)
# available_objects = ["pen", "Book", "scale", "plate"]
# print("Objects available: ", available_objects)
label1 = "blessing_card"
label2 = "wallet"
target_labels = (label1, label2)
detections = video_processor.process_video(detector, output_path, True, target_labels)
