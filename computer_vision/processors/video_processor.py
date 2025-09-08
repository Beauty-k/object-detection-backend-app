import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from computer_vision.calculators.distance_calculator import DistanceCalculator
from computer_vision.processors.frame_reader import FrameReader
from computer_vision.processors.frame_writer import FrameWriter
from computer_vision.processors.frame_displayer import FrameDisplayer
from computer_vision.annotator.detection_annotator import DetectionAnnotator
from computer_vision.annotator.distance_annotator import DistanceAnnotator
from computer_vision.annotator.tracked_object_annotator import TrackedObjectAnnotator
from computer_vision.models.bounding_box import BoundingBox
from computer_vision.utils.logger import setup_logger

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
        """Initialize a cv2.VideoCapture from the given source object."""

        cap = cv2.VideoCapture(video_source.get_video_source())
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video source: {video_source}")
        return cap

    def _get_video_properties(self):
        """Retrieve width, height, and fps from the video capture object."""

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

    def _detect_objects(self, detector, frame)-> tuple[np.ndarray, list[dict]]:
        """
        Run object detection on a frame using the provided detector.
        Returns the annotated frame and structured detection results.
        """
        detections = detector.detect_objects(frame)
        frame = DetectionAnnotator.draw_detections(frame, detections)
        return frame, detections
       
    def _track_objects(self, frame, detections):
        """
        Apply DeepSORT tracking on current detections.
        Returns a list of tracked detections with IDs, labels, and boxes.
        """
        tracking_inputs = []
        for d in detections:
            bounding_box: BoundingBox = d["box"]
            x, y, w, h = bounding_box.to_xywh()
            class_id = int(d.get("class_id", 0))
            tracking_inputs.append(([x, y, w, h], d["confidence"], class_id, d["label"]))

        tracks = self.tracker.update_tracks(tracking_inputs, frame=frame)

        track_ids = [t.track_id for t in tracks]
        label_map = {tid: label for (_, _, _, label), tid in zip(tracking_inputs, track_ids)}

        tracked_detections = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            l, t, r, b = track.to_ltrb()
            track_id = track.track_id
            label = label_map.get(track_id, "unknown")
            setattr(track, "label", label)

            box = [l, t, r - l, b - t]

            tracked_detections.append({
                "track_id": track_id,
                "label": label,
                "box": box
            })

        frame = TrackedObjectAnnotator.annotate_tracked_objects(frame, tracked_detections)

        return tracked_detections

    def _measure_distance(self, frame, detections, target_labels):
        """
        Measure and annotate distance between two target objects on the frame.
        Returns the computed distance in mm, or None if not possible.
        """

        self.distance_calculator.ensure_initialized(detections)
        
        if not target_labels:
            if len(detections) >= 2:
                box1, label1 = detections[0]["box"], detections[0]["label"]
                box2, label2 = detections[1]["box"], detections[1]["label"]
            else:
                return None
        else:
   
            target_boxes = [d for d in detections if d["label"] in target_labels]
            if len(target_boxes) == 2:
                box1, label1 = target_boxes[0]["box"], target_boxes[0]["label"]
                box2, label2 = target_boxes[1]["box"], target_boxes[1]["label"]
            else:
                return None

        try:
            distance_mm, _, _ = self.distance_calculator.calculate(box1, box2)
            DistanceAnnotator.annotate_distance(frame, box1, box2, label1, label2, self.distance_calculator)
            return distance_mm
        except ValueError as e:
            logger.warning(f"Distance calculation failed: {e}")
            return None


    def _output_frame(self, frame):
        """Write or display the frame depending on configuration. Returns False if stopped."""

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
                    distance_mm = self._measure_distance(annotated_frame, detections, target_labels)
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
    