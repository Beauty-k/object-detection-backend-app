import cv2
import os
from deep_sort_realtime.deepsort_tracker import DeepSort
from object_detection_app.calculators.distance_calculator import DistanceCalculator


class VideoProcessor:
    def __init__(self, video_source, reference_label="blessing_card", reference_mm=85, tracker_max_age=30):
        self.video_source = video_source
        self.cap = self._open_video_source()
        self.width, self.height, self.fps = self._get_video_properties()
        self.distance_calculator = DistanceCalculator(reference_label, reference_mm)
        self.tracker = DeepSort(max_age=tracker_max_age)

    def _open_video_source(self):
        source = self.video_source.get_video_source()
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise FileNotFoundError(f"[ERROR] Could not open video source: {source}")
        print("[INFO] Video source opened successfully")
        return cap

    def _get_video_properties(self):
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30  # default fps
        return width, height, fps

    def _ensure_output_directory(self, path):
        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)

    def _create_video_writer(self, output_path):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

    def _prepare_tracking_inputs(self, detections):
        tracking_inputs = []
        for d in detections:
            xc, yc, w, h = d["box"]
            x, y = xc - w / 2, yc - h / 2
            tracking_inputs.append(([x, y, w, h], d["confidence"], d["label"]))
        return tracking_inputs

    def _get_tracked_detections(self, frame, detections):
        tracking_inputs = self._prepare_tracking_inputs(detections)
        tracks = self.tracker.update_tracks(tracking_inputs, frame=frame)

        tracked_detections = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            l, t, r, b = track.to_ltrb()
            label, track_id = track.det_class, track.track_id
            box = [l, t, r - l, b - t]

            cv2.putText(frame, f"{label}-{track_id}", (int(l), int(t) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            tracked_detections.append({"id": track_id, "label": label, "box": box})
        return tracked_detections

    def _calculate_and_annotate_distance(self, frame, detections, target_labels):
        if self.distance_calculator.pixel_per_mm is None:
            self.distance_calculator.update_pixel_mm_ratio(detections)

        target_boxes = [d for d in detections if d["label"] in target_labels]
        if len(target_boxes) == 2:
            box1, label1 = target_boxes[0]["box"], target_boxes[0]["label"]
            box2, label2 = target_boxes[1]["box"], target_boxes[1]["label"]

            distance_mm, _, _ = self.distance_calculator.calculate(box1, box2)
            self.distance_calculator.annotate_distance(frame, box1, box2, label1, label2)
            return distance_mm
        return None

    def process_video(self, detector, output_path, display=False, target_labels=()):
        print("[INFO] Processing frames...")
        self._ensure_output_directory(output_path)
        writer = self._create_video_writer(output_path)
        all_detections = []
        measured_distance_mm = None
        frame_count = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            annotated_frame, detections = detector.get_detection(frame)

            tracked_detections = self._get_tracked_detections(annotated_frame, detections)

            measured_distance_mm = self._calculate_and_annotate_distance(
                annotated_frame, detections, target_labels
            )

            # save results
            writer.write(annotated_frame)
            if display:
                cv2.imshow("Live Detection", annotated_frame)
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    print("[INFO] Stream stopped by user.")
                    break

            all_detections.append({"frame": frame_count, "detections": detections})
            frame_count += 1

        self.cap.release()
        writer.release()
        print("[INFO] Video processing complete.")
        return measured_distance_mm, all_detections



