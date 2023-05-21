from pydantic import BaseModel

class HotelBookingIn(BaseModel):
    age: int 
    destination: str
    first_browser: str
    language: str
    booking: int