import torch
import cv2
from typing import Dict, Any, Tuple
from utils.logger import setup_logger
from models.bounding_box import BoundingBox

logger = setup_logger(__name__)

class BaseDetector:
    """
    Base class providing shared utilities for all detectors.
    """
    def __init__(self, device: str = None):
        self.device = self._setup_device(device)
    
    def _setup_device(self, device: str = None) -> str:
        """
        Choose computation device (CPU or GPU).
        """
        if device:
            return device
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    def _build_detection_dict(self, box: BoundingBox, label: str, confidence: float) -> Dict[str, Any]:

        """
        Build a structured detection dictionary.
        Returns:
            Dict[str, Any]: Dictionary containing detection result.
        """

        return {
            "box": box,
            "label": label,
            "confidence": confidence
        }

