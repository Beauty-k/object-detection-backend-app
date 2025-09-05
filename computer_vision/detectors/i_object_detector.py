from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Any

class IObjectDetector(ABC):
    """
    Interface for all object detection models.
    """
    @abstractmethod
    def detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        pass