import cv2
from typing import List, Dict
from computer_vision.models.bounding_box import BoundingBox
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
            
            bounding_box = BoundingBox(*det["box"]) 
            l, t, r, b = bounding_box.to_ltrb()      

            # draw rectangle
            cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)

            cv2.putText(
                frame,
                f"{label}-{track_id}",
                (int(l), int(t) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 255, 255),
                2 
            )
            return frame
