"""This test script checks errors on the model API.
"""

from fastapi.testclient import TestClient
from app.main import api

client = TestClient(api)

# Write unit tests for the API
def test_read_main():
    """Check if the API generates a valid API response."""
    response = client.get("/", headers={"Content-Type": "application/json"})
    assert response.status_code == 200
    assert response.json() == {'message': 'Hello and welcome to the Data Science School ML API!'}

# Write unit tests for the API
def test_generate_prediction():
    """Test model prediction API."""
    response = client.post(
        "/predict",
        headers={"Content-Type": "application/json"},
        json={
            "age": 0,
            "destination": "string",
            "first_browser": "string",
            "language": "string",
            "booking": 0
        }
    )
    assert response.status_code == 200
    assert response.json() == {
        "model": "hotel_booking",
        "inputs": {
            "age": 0,
            "destination": "string",
            "first_browser": "string",
            "language": "string",
            "booking": 0
        },
        "prediction": 0.4521548453451801
    }
