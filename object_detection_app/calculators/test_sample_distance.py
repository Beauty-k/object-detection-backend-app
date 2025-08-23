import math
import cv2
from typing import Tuple
from calculators.i_calculator import ICalculator

class GeometryDistanceCalculator(ICalculator):
    def __init__(self, fov_deg: float, camera_distance_mm: float, frame_width_px: int):
        self.fov_deg = fov_deg
        self.camera_distance_mm = camera_distance_mm
        self.frame_width_px = frame_width_px
        self.pixel_per_mm = self.calculate_pixel_per_mm()

    def calculate_pixel_per_mm(self):
        fov_rad = math.radians(self.fov_deg)
        scene_width_mm = 2 * self.camera_distance_mm * math.tan(fov_rad / 2)
        # self.pixel_per_mm = 52.14 / 169
        self.pixel_per_mm = self.frame_width_px / scene_width_mm
        print(f"[INFO] scene width: {scene_width_mm:.2f} mm, pixel/mm: {self.pixel_per_mm:.4f}")
        return self.pixel_per_mm

    def get_center(self, box):
        x_center, y_center, _, _ = box
        return int(x_center), int(y_center)

    def calculate(self, box1, box2):
        x1, y1 = self.get_center(box1)
        x2, y2 = self.get_center(box2)
        pixel_distance = math.hypot(x2 - x1, y2 - y1)
        distance_mm = round(pixel_distance / self.pixel_per_mm, 2)
        return distance_mm, (x1, y1), (x2, y2)

    def annotate_distance(self, frame, box1, box2, label1, label2):
        distance_mm, p1, p2 = self.calculate(box1, box2)
        cv2.line(frame, p1, p2, (255, 0, 255), 2)
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        cv2.putText(frame, f"{distance_mm} mm", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        # cv2.putText(frame, label1, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        # cv2.putText(frame, label2, p2, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
