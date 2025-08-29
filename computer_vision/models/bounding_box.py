from dataclasses import dataclass

@dataclass
class BoundingBox:
    x_center: float
    y_center: float
    width: float
    height: float
    