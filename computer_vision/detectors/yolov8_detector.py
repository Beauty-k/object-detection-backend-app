from ultralytics import YOLO
import numpy as np
from typing import List, Dict, Any
from computer_vision.detectors.base_detector import BaseDetector
from computer_vision.detectors.i_object_detector import IObjectDetector
from computer_vision.utils.exceptions import DetectionError
from computer_vision.utils.logger import setup_logger
from computer_vision.models.bounding_box import BoundingBox
from computer_vision.utils.exceptions import DetectionError

logger = setup_logger(__name__)

class YOLOv8Detector(BaseDetector, IObjectDetector):
    """
    YOLOv8-based object detector.

    Wraps the Ultralytics YOLOv8 model to provide object detection
    functionality with structured outputs and error handling.
    """

    def __init__(self, model_path: str, device: str = None):
        super().__init__(device)
        try:
            self.model = YOLO(model_path)
            logger.info(f"Loaded YOLOv8 model from {model_path} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {str(e)}")

    def detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in a frame using YOLOv8.

        Args:
            frame (np.ndarray): Input image frame.

        Returns:
            List[Dict[str, Any]]: A list of detection results, where each result contains:
                - "box" (BoundingBox): The detected bounding box in (x_center, y_center, w, h).
                - "label" (str): Class label of the detected object.
                - "confidence" (float): Detection confidence score.
        """

        if frame is None or frame.size == 0:
            logger.warning("Empty frame provided to YOLOv8Detector.")
            return [] 
    
        try:
            results = self.model(frame, device=self.device)

            if not results:
                logger.warning("YOLOv8 returned no results for this frame")
                return []
            
            detections = []

            for result in results:
                for box, conf, cls_id in zip(result.boxes.xywh, result.boxes.conf, result.boxes.cls):
                    bounding_box = BoundingBox(*box.tolist())
                    conf_val = float(conf)
                    label = self.model.names[int(cls_id)]

                    detections.append(
                        self._build_detection_dict(
                            bounding_box,
                            label,
                            conf_val
                        )
                    )

            return detections
        except Exception as e:
            logger.error(f"Detection failed: {str(e)}")
            raise DetectionError("YOLOv8 detection failed") from e
        
