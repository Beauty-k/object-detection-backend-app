
from typing import List, Any, Dict, Optional
import numpy as np
import cv2
from computer_vision.models.bounding_box import BoundingBox

FONT_SCALE_LABEL = 0.6
FONT_SCALE_COORD = 0.5
FONT_THICKNESS = 2
BOX_COLOR = (0, 255, 255)
TEXT_COLOR = (255, 255, 255)
BG_COLOR = (0, 0, 0)

class DetectionAnnotator:
    """
    Utility class for drawing object detection results on image frames.
    """

    @staticmethod
    def draw_detections(frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        for det in detections:
            bounding_box: BoundingBox = det["box"]
            label: str = det.get("label", "")
            confidence: float = det.get("confidence", 0.0)
            track_id: Optional[int] = det.get("track_id", None)
            DetectionAnnotator._draw_coordinates(frame, bounding_box, label, confidence, track_id)
        return frame

    @staticmethod
    def _draw_coordinates(frame: np.ndarray,bounding_box: BoundingBox,label: str,confidence: float,track_id: Optional[int] = None) -> None:
        """
        Draws a single detection bounding box and delegates text drawing to a helper.
        """
        x1, y1, x2, y2 = bounding_box.to_xyxy()
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 3)

        try:
            conf_val: float = float(confidence)
        except (ValueError, TypeError):
            conf_val = 0.0

        display_label: str = f"{label} {conf_val:.2f}"
        if track_id is not None:
            display_label += f" | ID:{track_id}"

        coord_text: str = f"({x1},{y1})-({x2},{y2})"

        DetectionAnnotator._draw_text_block(frame, x1, y1, display_label, coord_text)

    @staticmethod
    def _draw_text_block(frame: np.ndarray, x: int, y: int, label_text: str, coord_text: str) -> None:
        """
        Draws label and coordinate text on the frame with a background rectangle.
        """
        (label_w, label_h), base1 = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_LABEL, FONT_THICKNESS)
        (coord_w, coord_h), base2 = cv2.getTextSize(coord_text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_COORD, 1)

        text_w: int = max(label_w, coord_w)
        total_h: int = label_h + coord_h + base1 + base2 + 6

        cv2.rectangle(frame, (x, y - total_h), (x + text_w + 6, y), BG_COLOR, -1)

        cv2.putText(frame, label_text, (x + 3, y - coord_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_LABEL, TEXT_COLOR, FONT_THICKNESS)

        cv2.putText(frame, coord_text, (x + 3, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_COORD, TEXT_COLOR, 1)
