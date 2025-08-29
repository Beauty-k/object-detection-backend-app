import math
import cv2
from typing import Tuple
from calculators.i_calculator import ICalculator
from models.bounding_box import BoundingBox
from utils.logger import setup_logger

# get module-specific logger
logger = setup_logger(__name__)

class GeometryDistanceCalculator(ICalculator):
    """
    Calculates distances between detected objects in an image frame
    based on camera field of view and frame dimensions.
    """

    def __init__(self, fov_deg: float, camera_distance_mm: float, frame_width_px: int):
        """
        Initialize the distance calculator.

        Args:
            fov_deg (float): Horizontal field of view of the camera in degrees.
            camera_distance_mm (float): Distance from camera to the wall/scene in millimeters.
            frame_width_px (int): Width of the video frame in pixels.
        """

        self.fov_deg = fov_deg
        self.camera_distance_mm = camera_distance_mm
        self.frame_width_px = frame_width_px
        self.pixel_per_mm = self.calculate_pixel_per_mm()

    def calculate_pixel_per_mm(self):
        fov_rad = math.radians(self.fov_deg)
        scene_width_mm = 2 * self.camera_distance_mm * math.tan(fov_rad / 2)
        # self.pixel_per_mm = 52.14 / 169
        self.pixel_per_mm = self.frame_width_px / scene_width_mm
        logger.info(
            "Scene width: %.2f mm, pixel/mm: %.4f",
            scene_width_mm,
            self.pixel_per_mm
        )
        return self.pixel_per_mm

    def get_center(self, box: BoundingBox) -> Tuple[int, int]:
        return int(box.x_center), int(box.y_center)

    def calculate(self, box1: BoundingBox, box2: BoundingBox):
        """
        Calculate the distance between two bounding boxes in millimeters.

        Args:
            box1 (BoundingBox): First bounding box.
            box2 (BoundingBox): Second bounding box.

        Returns:
            Tuple[float, Tuple[int, int], Tuple[int, int]]:
                - Distance in millimeters (float).
                - Center coordinates of box1 (x, y).
                - Center coordinates of box2 (x, y).

        Raises:
            ValueError: If pixel-per-mm ratio is not initialized.
        """
         
        x1, y1 = self.get_center(box1)
        x2, y2 = self.get_center(box2)
        pixel_distance = math.hypot(x2 - x1, y2 - y1)
        if self.pixel_per_mm is not None:
            distance_mm = round(pixel_distance / self.pixel_per_mm, 2)
        else:
            raise ValueError("Pixel-per-mm ratio has not been initialized.")
        return distance_mm, (x1, y1), (x2, y2)

    def annotate_distance(self, frame, box1: BoundingBox, box2: BoundingBox, label1, label2):
        """
        Draw a line and distance annotation between two bounding boxes on a frame.

        Args:
            frame (ndarray): Image frame to annotate.
            box1 (BoundingBox): First bounding box.
            box2 (BoundingBox): Second bounding box.
            label1 (str): Label for the first object.
            label2 (str): Label for the second object.

        Returns:
            ndarray: Annotated frame with line and distance text.
        """
        
        distance_mm, p1, p2 = self.calculate(box1, box2)
        cv2.line(frame, p1, p2, (255, 0, 255), 2)
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        cv2.putText(frame, f"{distance_mm} mm", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        # cv2.putText(frame, label1, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        # cv2.putText(frame, label2, p2, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
