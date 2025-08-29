import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from calculators.distance_calculator import DistanceCalculator
from processors.frame_reader import FrameReader
from processors.frame_writer import FrameWriter
from processors.frame_displayer import FrameDisplayer
from utils.logger import setup_logger

logger = setup_logger(__name__)

class VideoProcessor:
    """
    Orchestrates the full video analysis pipeline: reading frames, detecting objects,
    tracking them, measuring distances, and managing video output.

    This class integrates multiple components:
    - `FrameReader` for reading frames
    - `ObjectDetector` (passed at runtime) for object detection
    - `DeepSort` for object tracking
    - `DistanceCalculator` for measuring distances between objects
    - `FrameWriter` and `FrameDisplayer` for output

    Attributes:
        cap (cv2.VideoCapture): OpenCV video capture instance.
        width (int): Width of the video frames.
        height (int): Height of the video frames.
        fps (float): Frames per second of the video.
        distance_calculator (DistanceCalculator): Utility for distance measurement.
        tracker (DeepSort): Object tracker instance.
        frame_reader (FrameReader): Frame reader wrapper around cv2.VideoCapture.
        frame_writer (FrameWriter | None): Writes processed frames to file if configured.
        frame_displayer (FrameDisplayer | None): Displays processed frames if enabled.
    """
    def __init__(
        self,
        video_source,
        output_path=None,
        display=False,
        reference_label="blessing_card",
        reference_mm=85,
        tracker_max_age=30,
    ):
        self.cap = self._init_video_capture(video_source)
        self.width, self.height, self.fps = self._get_video_properties()

        self.distance_calculator = DistanceCalculator(reference_label, reference_mm)
        self.tracker = DeepSort(max_age=tracker_max_age)

        self.frame_reader = FrameReader(self.cap)
        self.frame_writer = self._init_frame_writer(output_path)
        self.frame_displayer = FrameDisplayer() if display else None

    def _init_video_capture(self, video_source):
        cap = cv2.VideoCapture(video_source.get_video_source())
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video source: {video_source}")
        return cap

    def _get_video_properties(self):
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        return width, height, fps

    def _init_frame_writer(self, output_path):
        if output_path:
            return FrameWriter(output_path, self.fps, self.width, self.height)
        return None

    def _get_next_frame(self):
        return self.frame_reader.read()

    def _detect_objects(self, detector, frame):
        return detector.detect_objects(frame)

    def _track_objects(self, frame, detections):
        tracking_inputs = []
        for d in detections:
            box = d["box"]
            x = box.x_center - box.width / 2
            y = box.y_center - box.height / 2
            w = box.width
            h = box.height
            tracking_inputs.append(([x, y, w, h], d["confidence"], d["label"]))

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

    def _annotate_distances(self, frame, detections, target_labels):
        self.distance_calculator.ensure_initialized(detections)
        target_boxes = [d for d in detections if d["label"] in target_labels]
        if len(target_boxes) == 2:
            box1, label1 = target_boxes[0]["box"], target_boxes[0]["label"]
            box2, label2 = target_boxes[1]["box"], target_boxes[1]["label"]
            try:
                distance_mm, _, _ = self.distance_calculator.calculate(box1, box2)
                self.distance_calculator.annotate_distance(frame, box1, box2, label1, label2)
                return distance_mm
            except ValueError as e:
                logger.warning(f"Distance calculation failed: {e}")
        return None

    def _output_frame(self, frame):
        if self.frame_writer:
            self.frame_writer.write(frame)
        if self.frame_displayer:
            try:
                self.frame_displayer.display(frame)
            except StopIteration:
                # on pressing 'q', break the processing loop
                return False
        return True

    def process_video(self, detector, target_labels=()):
        """
        Process video frames to detect, track, and measure distances between objects.
        Parameters:-
        detector : ObjectDetector
            The object detector instance used for performing detections on frames.
        target_labels : list[str]
            The labels of objects between which distances should be measured.

        Returns:-
        tuple[list[float | None], list[dict]]
            measured_distances_mm : list of float | None
                List of measured distances (in millimeters) for each processed frame,
                or None if measurement could not be made for a frame.
            all_detections : list of dict
                Detection metadata for each frame, including bounding boxes, labels,
                and confidence scores.
        
        Side Effects:-
        - Opens a display window if a FrameDisplayer is provided.
        - Writes annotated frames to a video file if a FrameWriter is configured.
        - Logs progress and errors during video processing.
        - Releases video capture, writer, and display resources when finished.
        """

        logger.info("Starting video processing...")
        all_detections = []
        measured_distances_mm = []
        frame_count = 0

        try:
            while self.cap.isOpened():
                frame = self._get_next_frame()
                if frame is None:
                    break

                try:
                    annotated_frame, detections = self._detect_objects(detector, frame)
                    _ = self._track_objects(annotated_frame, detections)
                    distance_mm = self._annotate_distances(annotated_frame, detections, target_labels)
                    measured_distances_mm.append(distance_mm)

                    if not self._output_frame(annotated_frame):
                        break 

                    all_detections.append({"frame": frame_count, "detections": detections})
                    frame_count += 1
                except Exception as e:
                    logger.error(f"Error processing frame {frame_count}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Unexpected error during video processing: {e}")

        finally:
            self.cap.release()
            if self.frame_writer:
                self.frame_writer.release()
            if self.frame_displayer:
                self.frame_displayer.close()
            logger.info("Video processing complete.")

        return measured_distances_mm, all_detections
    