from fastapi import APIRouter, UploadFile, File
from controllers.video_processing_controller import VideoProcessingController

video_router = APIRouter()
controller = VideoProcessingController()

@video_router.post("/calculate-distance")
def calculate_distance(
    file: UploadFile = File(...),
    label1: str = "blessing_card",
    label2: str = "wallet"
):
    print("Received a request from Flutter!")
    return controller.calculate_distance(file, label1, label2)