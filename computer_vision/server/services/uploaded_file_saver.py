import shutil
from pathlib import Path
from fastapi import UploadFile


class UploadedFileSaver:
    def __init__(self, uploaded_file: UploadFile):
        self.uploaded_file = uploaded_file

    def save(self, directory: str) -> Path:
        Path(directory).mkdir(parents=True, exist_ok=True)
        file_path = Path(directory) / self.uploaded_file.filename

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(self.uploaded_file.file, buffer)

        return file_path
