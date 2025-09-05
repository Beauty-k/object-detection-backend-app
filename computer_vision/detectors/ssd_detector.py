import numpy as np
from typing import List, Dict, Any
from .base_detector import BaseDetector
from .i_object_detector import IObjectDetector
from utils.logger import setup_logger
from models.bounding_box import BoundingBox
from utils.exceptions import DetectionError

import torchvision
import torch

logger = setup_logger(__name__)

class SSDDetector(BaseDetector, IObjectDetector):
    """
    Single Shot MultiBox Detector (SSD) implementation using PyTorch.

    This class wraps the pretrained `ssd300_vgg16` model from `torchvision` 
    and provides a structured interface for object detection. 
    It converts raw SSD outputs (bounding boxes, class labels, and scores) 
    into a unified format with `BoundingBox` objects so that downstream 
    components (e.g., drawers, trackers, calculators) can work consistently 
    across different detection models.

    Attributes:
        model (torch.nn.Module): The SSD model loaded from `torchvision`.
        device (str): The device on which the model is loaded 
            ("cpu" or "cuda").
    """

    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        try:
            # Load pretrained SSD (COCO classes)
            self.model = torchvision.models.detection.ssd300_vgg16(
                weights="DEFAULT"
            ).to(self.device)
            self.model.eval()
            logger.info(f"Loaded SSD model on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load SSD model: {str(e)}")
            raise DetectionError("Failed to initialize SSD model") from e

    def detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in a single frame using the SSD model.

        This method:
        1. Converts the input OpenCV frame (BGR NumPy array) into a normalized 
           PyTorch tensor.
        2. Runs inference through the SSD model.
        3. Filters predictions below a confidence threshold (default: 0.5).
        4. Converts bounding boxes into `BoundingBox` objects.
        5. Returns results in a structured dictionary format.

        Args:
            frame (np.ndarray): Input image in BGR format 
                (H x W x 3), as provided by OpenCV.

        Returns:
            List[Dict[str, Any]]: A list of detection dictionaries. 
            Each dictionary contains:
                - "box" (BoundingBox): The bounding box of the detected object.
                - "label" (str): The class label ID as a string.
                - "confidence" (float): The detection confidence score.

        Raises:
            DetectionError: If the SSD model fails during inference 
            or if preprocessing/postprocessing encounters an error.
        """

        try:
            tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0).to(self.device)

            outputs = self.model(tensor)[0]
            detections = []

            for box, label, score in zip(outputs["boxes"], outputs["labels"], outputs["scores"]):
                if float(score) < 0.5:
                    continue

                x1, y1, x2, y2 = box.tolist()
                bbox = BoundingBox.from_xyxy(x1, y1, x2, y2)

                detections.append(
                    self._build_detection_dict(
                        bbox.x_center, bbox.y_center, bbox.width, bbox.height,
                        str(label.item()), float(score)
                    )
                )

            return detections
        except Exception as e:
            logger.error(f"Detection failed (SSD): {str(e)}")
            raise DetectionError("SSD detection failed") from e
