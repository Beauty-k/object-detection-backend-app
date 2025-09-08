from deep_sort_realtime.deepsort_tracker import DeepSort
from computer_vision.tracker.i_tracker import ITracker
from typing import List, Dict, Tuple
import numpy as np

class DeepSortTracker(ITracker):
    def __init__(self, max_age: int = 30):
        """
        Initialize Deep SORT tracker.

        Args:
            max_age (int, optional): Maximum number of frames to keep a track without updates. Defaults to 30.
        """
        self.tracker = DeepSort(max_age=max_age)

    def track_objects(self, detections: List, frame: np.ndarray) -> List[Dict[str, object]]:
        """
        Track objects across frames using Deep SORT.

        Args:
            detections (list): List of detections for the current frame.
            frame (numpy.ndarray): Current video frame.

        Returns:
            list[dict]: A list of confirmed tracks, each containing:
                - "id" (int): Unique track ID.
                - "label" (str): Object class label.
                - "box" (tuple[int, int, int, int]): Bounding box in (x, y, width, height).
        """
        tracks = self.tracker.update_tracks(detections, frame=frame)
        results = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, r, b = track.to_ltrb()
            results.append({
                "id": track_id,
                "label": track.det_class,
                "box": (int(l), int(t), int(r - l), int(b - t))
            })
        return results
