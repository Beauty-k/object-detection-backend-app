from dataclasses import dataclass
from typing import Tuple

@dataclass
class BoundingBox:
    """
    Represents a bounding box in (x_center, y_center, width, height) format.
    
    Attributes:
        x_center (float): X-coordinate of the box center.
        y_center (float): Y-coordinate of the box center.
        width (float): Width of the bounding box.
        height (float): Height of the bounding box.
    """

    x_center: float
    y_center: float
    width: float
    height: float

    def to_xyxy(self) -> Tuple[int, int, int, int]:
        """
        Convert the bounding box to corner format.

        Returns:
            Tuple[int, int, int, int]: (x_min, y_min, x_max, y_max) coordinates
            representing the top-left and bottom-right corners.
        """

        x1 = int(self.x_center - self.width / 2)
        y1 = int(self.y_center - self.height / 2)
        x2 = int(self.x_center + self.width / 2)
        y2 = int(self.y_center + self.height / 2)
        return (x1, y1, x2, y2)
    
    def to_xywh(self) -> Tuple[int, int, int, int]:
        """
        Convert bounding box to (x_min, y_min, width, height) format.
        Useful for trackers like DeepSORT.
        """

        x = int(self.x_center - self.width / 2)
        y = int(self.y_center - self.height / 2)
        return (x, y, int(self.width), int(self.height))
    
    def to_center_format(self) -> Tuple[float, float, float, float]:

        """
        Return the bounding box in center format.

        Returns:
            Tuple[float, float, float, float]: (x_center, y_center, width, height),
            the format commonly used by YOLO models.
        """
        
        return self.x_center, self.y_center, self.width, self.height
    