from pydantic import BaseModel

class DistanceRequest(BaseModel):
    label1: str
    label2: str
