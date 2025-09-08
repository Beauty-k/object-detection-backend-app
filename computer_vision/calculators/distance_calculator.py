import math
import cv2
from typing import Tuple
from computer_vision.models.bounding_box import BoundingBox
from computer_vision.calculators.i_calculator import ICalculator
from computer_vision.utils.logger import setup_logger

logger = setup_logger(__name__)

CORRECTION_FACTOR = 1.079 

class DistanceCalculator(ICalculator): 
    """
    A calculator for measuring real-world distances between detected objects.

    This class uses a known reference object size (in millimeters) to compute
    a pixel-to-millimeter ratio. Distances between objects are then calculated
    in millimeters, with a correction factor applied for accuracy.
    """
        
    def __init__(self, reference_label:str, reference_mm: float = 300):
        self.reference_label = reference_label
        self.reference_mm = reference_mm
        self.pixel_per_mm = None

    def _update_pixel_mm_ratio(self, detections):
        for detection in detections:
            if detection["label"] == self.reference_label:
                box: BoundingBox = detection["box"]
                self.pixel_per_mm = box.width / self.reference_mm
        return False
    
    def ensure_initialized(self, detections):
        """
        Initialize the pixel-to-millimeter ratio using a reference object.

        Args:
            detections (list[dict]): A list of detection results containing labels and bounding boxes.

        Side Effects:
            Sets `self.pixel_per_mm` if a reference object is found.
        """

        if self.pixel_per_mm is None:
            self._update_pixel_mm_ratio(detections)

    def _get_center(self, box: BoundingBox) -> Tuple[int, int]:
        return int(box.x_center), int(box.y_center)
    
    def calculate(self, box1, box2):
        
        """
        Calculate the real-world distance (in millimeters) between the centers of two bounding boxes.

        This method computes the Euclidean distance between the centers of `box1` and `box2` 
        in pixel units, then converts it into millimeters using the previously initialized 
        pixel-to-millimeter ratio. A correction factor is applied to improve accuracy.

        Args:
            box1 (BoundingBox): The first bounding box.
            box2 (BoundingBox): The second bounding box.

        Returns:
            tuple:
                - float: The corrected distance between the two objects in millimeters.
                - tuple[int, int]: The (x, y) coordinates of the center of `box1`.
                - tuple[int, int]: The (x, y) coordinates of the center of `box2`.

        Raises:
            ValueError: If the pixel-per-mm ratio has not been initialized by calling
            `ensure_initialized()` with a reference object.
        """

        logger.debug(f"Calculating distance between {box1} and {box2}")
        x1, y1 = self._get_center(box1)
        x2, y2 = self._get_center(box2)

        pixel_distance = math.hypot(x2 - x1, y2 - y1)

        if self.pixel_per_mm is not None:
            raw_mm = pixel_distance / self.pixel_per_mm
            corrected_mm = round(raw_mm * CORRECTION_FACTOR, 2)
            return corrected_mm, (x1, y1), (x2, y2)
        else:
            raise ValueError("Pixel-per-mm ratio has not been initialized.")
    