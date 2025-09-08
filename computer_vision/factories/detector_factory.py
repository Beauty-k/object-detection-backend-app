from typing import Type
from computer_vision.detectors.yolov8_detector import YOLOv8Detector
from computer_vision.detectors.rcnn_detector import RCNNDetector
from computer_vision.detectors.ssd_detector import SSDDetector
from computer_vision.utils.exceptions import DetectionError

class DetectorFactory:
    """
    Factory for creating object detector instances.

    Provides a centralized way to initialize different 
    detector implementations (YOLOv8, RCNN, SSD, etc.)
    using a simple string identifier.

    Example:
        detector = DetectorFactory.create("yolov8", model_path="yolov8n.pt", device="cuda")
    """

    _registry = {
        "yolov8": YOLOv8Detector,
        "rcnn": RCNNDetector,
        "ssd": SSDDetector,
    }

    @classmethod
    def create(cls, detector_type: str, **kwargs):
        detector_type = detector_type.lower()
        if detector_type not in cls._registry:
            raise DetectionError(f"Unknown detector type: {detector_type}")

        detector_class: Type = cls._registry[detector_type]
        return detector_class(**kwargs)
