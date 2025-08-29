from ultralytics import YOLO
import cv2
import torch
from models.bounding_box import BoundingBox
from utils.logger import setup_logger

logger = setup_logger(__name__)

class ObjectDetector:
    FONT_SCALE = 0.75
    FONT_COLOR = (0, 0, 0)
    FONT_THICKNESS = 1
    FONT_TYPE = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = YOLO(model_path)
        
        self._setup_device()

    def _setup_device(self) -> None:
        if torch.cuda.is_available():
            device = "cuda:0"
            self.model.to(device)
            logger.info("Running on GPU")
        else:
            logger.info("Running on CPU")

    def get_detection(self, frame):
        try:
                model_results = self.model(frame)
                if not model_results or len(model_results) == 0:
                    return frame, []

                results = model_results[0]
                frame_with_annotations = results.plot()

                detections = []
                if hasattr(results, "boxes") and results.boxes is not None:
                    for box in results.boxes:
                        try:
                            detections.append(self._parse_box(box, results, frame_with_annotations))
                        except Exception as e:
                            logger.warning(f"Failed to parse a box: {e}")
                
                return frame_with_annotations, detections
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return frame, []

    def _parse_box(self, box, results, frame):
        class_id = int(box.cls[0].item())
        label = results.names[class_id]
        confidence_score = round(float(box.conf[0].item()), 2)

        x_center, y_center, width, height = box.xywh[0].tolist()
        x1, y1 = int(x_center - width / 2), int(y_center - height / 2)

        self._draw_coordinates(frame, x1, y1, x_center, y_center, width, height)

        return {
            "label": label,
            "confidence": confidence_score,
            "box": BoundingBox(
                round(x_center, 2),
                round(y_center, 2),
                round(width, 2),
                round(height, 2)
            )
        }

    def _draw_coordinates(self, frame, x, y, x_center, y_center, width, height):
        coord_text = f"XYWH: {round(x_center), 2}, {round(y_center), 2}, {round(width), 2}, {round(height), 2}"
        offset = max(10, int(height * 0.1))
        cv2.putText(
            frame,
            coord_text,
            (x, y - offset),
            self.FONT_TYPE,
            self.FONT_SCALE,
            self.FONT_COLOR,
            self.FONT_THICKNESS,
            cv2.LINE_AA,
        )
