from pydantic import BaseModel

class PredictionOut(BaseModel):
    model: str
    inputs: dict
    prediction: float