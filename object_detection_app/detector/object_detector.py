from ultralytics import YOLO
import cv2
import torch

class ObjectDetector:
    FONT_SCALE = 0.75
    FONT_COLOR = (0, 0, 0)
    FONT_THICKNESS = 1
    FONT_TYPE = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self._setup_device()

    def _setup_device(self) -> None:
        if torch.cuda.is_available():
            self.model.to("cuda:0")
            print("Running on GPU")
        else:
            print("Running on CPU")

    def get_detection(self, frame):
        results = self.model(frame)[0]
        frame_with_annotations = results.plot()
        detections = [self._parse_box(box, results, frame_with_annotations) for box in results.boxes]
        return frame_with_annotations, detections

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
            "box": [round(x, 2) for x in [x_center, y_center, width, height]],
        }

    def _draw_coordinates(self, frame, x, y, x_center, y_center, width, height):
        coord_text = f"XYWH: {round(x_center)}, {round(y_center)}, {round(width)}, {round(height)}"
        cv2.putText(
            frame,
            coord_text,
            (x, y - 10),
            self.FONT_TYPE,
            self.FONT_SCALE,
            self.FONT_COLOR,
            self.FONT_THICKNESS,
            cv2.LINE_AA,
        )



    