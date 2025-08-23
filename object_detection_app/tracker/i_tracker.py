from abc import ABC, abstractmethod

class ITracker(ABC):

    @abstractmethod
    def track(self, detections: list, frame):
        pass

