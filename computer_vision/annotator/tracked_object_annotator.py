import cv2
from typing import List, Dict

class TrackedObjectAnnotator:
    """
    Handles drawing tracked objects (ID + label) on video frames.
    """

    @staticmethod
    def annotate_tracked_objects(frame, tracked_detections: List[Dict]):
        """
        Draws labels and track IDs on the frame.
        tracked_detections: List of dicts with keys: 'id', 'label', 'box'
        """
        for det in tracked_detections:
            track_id = det["track_id"]
            label = det["label"]
            box = det["box"]
            x, y, w, h = box
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}-{track_id}", (int(x), int(y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.0, (255, 255, 255), 0)
        return frame
