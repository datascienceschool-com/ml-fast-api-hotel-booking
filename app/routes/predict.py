import joblib
import pandas as pd
from fastapi import APIRouter
from app.schemas.predict import PredictionOut
from app.schemas.hotel_booking import HotelBookingIn

# Load hotel booking model
model = joblib.load('models/hotel_booking_model.joblib')

# Instantiate predict router
predict_router = APIRouter(prefix="/predict")

@predict_router.post("/", response_model=PredictionOut)
async def generate_hotel_booking_prediction(inputs: HotelBookingIn):

    # Convert inputs to dictionary
    input_dict = inputs.dict()

    # Create prediction from input
    input_values = pd.DataFrame([input_dict])
    prediction = model.predict_proba(input_values)[:,1]

    # Return prediction
    return PredictionOut(
        **{"model": "hotel_booking", "inputs": input_dict, "prediction": prediction}
    )
