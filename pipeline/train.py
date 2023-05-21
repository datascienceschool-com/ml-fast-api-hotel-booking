"""Trains and exports the Hotel Booking Prediction Model object.

This script fetches the hotel booking data from the DS School Dataset API.
Data preprocessing functions and model classifier are combined using Sklearn
pipeline. And, the pipeline model object is tuned using GridSearch (cv=5). 
The best model from the GridSearch is persisted in a model store folder.
"""

import requests
import joblib

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

from sklearn.metrics import roc_auc_score, make_scorer

# Fetch dataset from API
URL = "http://localhost:8000/hotel_booking"  # Replace with your API endpoint
HEADERS = {"Accept": "application/json"}

# Set model parameter
SEED = 123

# Set model export file path
MODEL_FILEPATH = "./models/hotel_booking_model.joblib"

def fetch_data(url, headers):
    """Get raw data from an API.

    This function requests hotel booking data to DS School Data API,
    which sends back raw data in JSON. The function then outputs
    the data which is in the form of a list containing hotel booking
    records.

    Args:
        url: API endpoint to fetch raw data for model training
        headers: meta-data for API request

    Returns:
        A list of hotel booking records.
    """
    try:
        response = requests.get(url, headers=headers)
        # If the response contains an HTTP error status code, this will raise a HTTPError
        response.raise_for_status()
        # Parse the JSON data from the response
        data = response.json()
        print("Success! Data loaded.")
        return data
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except requests.exceptions.RequestException as err:
        print(f"Error occurred: {err}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def make_pipeline(model):
    """Make model pipeline for preprocessing and training.

    This function combines preprocessing strategies including 
    mean imputation of numerical variables (e.g. age) and ordinal
    encoding of categorical variables (e.g. destination, language) and
    the model estimator in a single preprocessing pipeline for model 
    fitting and prediction.

    Args:
        model: a classifer to fit on hotel booking data and generate 
            booking prediction.

    Returns:
        A pipeline that combines preprocessing and model estimator.
    """

    # Define preprocessing functions
    simple_imputer = SimpleImputer(strategy='mean')
    ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-99)

    preprocess = ColumnTransformer([
        ('impute', simple_imputer, ['age']),
        ('ordinal_encoder', ordinal_encoder, ['destination', 'first_browser', 'language']),
    ], verbose_feature_names_out=False)

    # build a model pipeline
    pipeline = Pipeline(steps=[
        ('preprocess', preprocess),
        ('rf_model', model)
    ])

    return pipeline

def serializable_model_metric(label, pred):
    """Creates a wrapper for model metric function for pickling.

    When persisting GridSearchCV model using Joblib, the model metric
    function needs to be serializable. This wraps the roc_auc_score 
    function such that the make_scorer function from sklearn.metrics
    could be used to create a metric function that is serializable.

    Args:
        label: An array of labels
        pred: An array of prediction values

    Returns:
        A float value with AUC score
    """
    error = roc_auc_score(label, pred)
    return error

if __name__ == '__main__':

    # Make an API call to fetch training data
    booking_data = fetch_data(url=URL, headers=HEADERS)
    # Load into Pandas DataFrame
    colnames = ['age','destination','first_browser','language','booking']
    df = pd.DataFrame(booking_data, columns=colnames)
    # Apply 80-20 split on the train and test
    features = ['age','destination','first_browser','language']
    target = ['booking']
    X_train, X_test, y_train, y_test = (
        train_test_split(df[features], df[target], test_size=.2, random_state=SEED)
    )
    # Create a model pipeline
    rf_model = RandomForestClassifier(random_state=SEED)
    model_pipeline = make_pipeline(rf_model)

    # Create a model evaluator object
    eval_func = make_scorer(serializable_model_metric, greater_is_better=True)

    # Grid Search tuning
    param_grid = {
        'rf_model__max_depth': range(3, 10, 1),
        'rf_model__n_estimators': range(20, 80, 10)
    }
    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        scoring='roc_auc',
        n_jobs=-1,
        cv=5
    )
    grid_search.fit(X_train, y_train.values.ravel())

    # Export model
    joblib.dump(grid_search, MODEL_FILEPATH)
