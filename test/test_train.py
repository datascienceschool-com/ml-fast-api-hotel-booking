"""This test script checks errors in the train.py script.
"""

import pytest
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from pipeline import train

# Load dataset from API
URL = "http://localhost:8000/hotel_booking"  # Replace with your API endpoint
HEADERS = {"Accept": "application/json"}

# Set model parameter seed for consistent results
SEED = 123

@pytest.fixture(name="create_model_pipeline")
def fixture_model_pipeline():
    """Create a fixture for the model pipeline to be used for test functions."""
    rf_model = RandomForestClassifier(random_state=SEED)
    model_pipeline = train.make_pipeline(rf_model)
    return model_pipeline

def test_load_data():
    """Check if the data loaded from API contains records."""
    booking_data = train.load_data(url=URL, headers=HEADERS)
    assert len(booking_data) != 0

def test_steps_in_pipeline(create_model_pipeline):
    """Check if the instantiated model pipeline contains the correct steps."""
    pipe_steps = create_model_pipeline.named_steps
    assert 'preprocess' in pipe_steps
    assert 'rf_model' in pipe_steps

def test_pipeline_prediction(create_model_pipeline):
    """Test if the fitted model pipeline generates the expected prediction values."""
    sample_x = pd.DataFrame({'age': {497: 36.0, 21: 58.0, 1710: 29.0},
                    'destination': {497: 'US', 21: 'US', 1710: 'other'},
                    'first_browser': {497: '-unknown-', 21: 'Safari', 1710: 'Chrome'},
                    'language': {497: 'en', 21: 'en', 1710: 'en'}})
    sample_y = np.array([1,1,0])
    model = create_model_pipeline.fit(sample_x, sample_y)
    assert all(model.predict(sample_x) == sample_y)
