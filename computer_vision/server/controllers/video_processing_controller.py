from fastapi import UploadFile
from services.distance_service import DistanceService 

class VideoProcessingController:
    def calculate_distance(self, file: UploadFile, label1: str, label2: str) -> dict:
        model_path = "runs/detect/train14/weights/best.pt" 
        output_path = "static/output.mp4"
        video_service = DistanceService(model_path,output_path)
        distance_mm, result_video_path = video_service.process(file, label1, label2)
        return {
        "object1": label1,
        "object2": label2,
        "distance": distance_mm if distance_mm is not None else "N/A",
        "unit": "mm",
        "video_url": result_video_path
    }