import cv2
from models.bounding_box import BoundingBox

class DistanceAnnotator:
      
    @staticmethod
    def annotate_distance(frame, box1: BoundingBox, box2: BoundingBox, label1: str, label2: str, calculator):
            """
            Annotate the distance between two bounding boxes on a video frame.

            Args:
                frame (ndarray): The image frame where the distance will be drawn.
                box1 (BoundingBox): The first bounding box.
                box2 (BoundingBox): The second bounding box.
                label1 (str): The label for the first object.
                label2 (str): The label for the second object.

            Side Effects:
                Draws a line with arrowheads between the two objects and overlays the distance (in mm).
            """

            result = calculator.calculate(box1, box2)
            if result:
                distance_mm, p1, p2 = result

                cv2.arrowedLine(frame, p1, p2, (0, 0, 255), 2, tipLength=0.03)
                cv2.arrowedLine(frame, p2, p1, (0, 0, 255), 2, tipLength=0.03)

                mid_x = int((p1[0] + p2[0]) / 2)
                mid_y = int((p1[1] + p2[1]) / 2) - 10

                distance_text = f"{distance_mm:.2f} mm"
                (text_w, text_h), baseline = cv2.getTextSize(distance_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

                cv2.rectangle(frame,
                            (mid_x - text_w // 2, mid_y - text_h - baseline),
                            (mid_x + text_w // 2, mid_y + baseline),
                            (0, 0, 0), -1)

                cv2.putText(frame, distance_text,
                            (mid_x - text_w // 2, mid_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                