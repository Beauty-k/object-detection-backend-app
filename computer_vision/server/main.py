import sys
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles


# Add the 'Project' folder to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from fastapi import FastAPI
from routes.video_router import video_router

app = FastAPI(title="Distance Measurement API")

app.get("/video/output")
def get_video():
    file_path = "static/output.mp4"
    return StreamingResponse(open(file_path, "rb"), media_type="video/mp4")
# Make sure the folder exists
os.makedirs("static", exist_ok=True)

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")
@app.get("/ping")
def ping():
    return {"status": "ok", "message": "Server is alive!"}
app.include_router(video_router, prefix="/video", tags=["Video Distance"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
