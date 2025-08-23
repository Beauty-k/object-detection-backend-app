import math
import cv2
from typing import Tuple
from object_detection_app.calculators.i_calculator import ICalculator

class DistanceCalculator(ICalculator):
    def __init__(self, reference_label:str, reference_mm: float = 300):
        self.reference_label = reference_label
        self.reference_mm = reference_mm
        self.pixel_per_mm = None
        self.reference_widths = []

    def _update_pixel_mm_ratio(self, detections):
        for detection in detections:
            if detection["label"] == self.reference_label:
                _,_,w,_ = detection["box"]
                self.pixel_per_mm = w / self.reference_mm
                # print(f"Pixel/mm ratio: {self.pixel_per_mm:.4f}, Detected Distance: {self.distance_mm} mm")
                return True
        return False
    
    def _get_center(self, box):
        x_center, y_center, _, _ = box
        return x_center, y_center
    
    def calculate(self, box1, box2):
        x1, y1 = self.get_center(box1)
        x2, y2 = self.get_center(box2)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        pixel_distance = math.hypot(x2 - x1, y2 - y1)
        if self.pixel_per_mm:
            raw_mm = pixel_distance / self.pixel_per_mm
            corrected_mm = round(raw_mm * 1.079, 2)
            return corrected_mm, (x1, y1), (x2, y2)
    
    def annotate_distance(self, frame, box1, box2, label1, label2):
        distance_mm, p1, p2 = self.calculate(box1, box2)
        if distance_mm is not None:
            cv2.line(frame, p1, p2, (0, 0, 255), 2)
            mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            cv2.putText(frame, f"{distance_mm} mm", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    

    

    