from deep_sort_realtime.deepsort_tracker import DeepSort
from object_detection_app.tracker.i_tracker import ITracker

class DeepSortTracker(ITracker):
    def __init__(self):
        self.tracker = DeepSort(max_age=30)

    def track(self, detections: list, frame):
        tracks = self.tracker.update_tracks(detections, frame=frame)
        results = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, r, b = track.to_ltrb()
            label = track.det_class
            results.append({
                "id": track_id,
                "label": label,
                "box": (int(l), int(t), int(r - l), int(b - t))
            })
        return results