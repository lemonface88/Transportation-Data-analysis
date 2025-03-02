import pandas as pd
import joblib
import numpy as np

def load_model():
    """Loads the trained subway delay prediction model and feature names."""
    model = joblib.load("subway_delay_model.pkl")
    feature_names = joblib.load("model_features.pkl")
    return model, feature_names

def make_prediction(input_data):
    """Makes a prediction using the trained model."""
    model, feature_names = load_model()
    
    # Convert input to DataFrame with correct feature order
    input_df = pd.DataFrame([input_data], columns=feature_names)
    
    # Make prediction
    predicted_delay = model.predict(input_df)[0]
    print(f"Predicted Delay Duration: {predicted_delay:.2f} minutes")
    return predicted_delay

if __name__ == "__main__":
    # Example input values for prediction
    sample_input = {
        "Time": 0.5,  # Normalized time (e.g., 12 PM)
        "Mean Temp (°C)": 0.4,  # Normalized temperature value
        "Total Rain (mm)": 0.1,  # Normalized precipitation
        "Total Snow (cm)": 0.0,  # No snow
        "Total Precip (mm)": 0.1,  # Combined precipitation
        "Snow on Grnd (cm)": 0.0,  # No snow accumulation
        "Spd of Max Gust (km/h)": 0.2,  # Normalized wind speed
        "Min Gap": 15.0,  # Minutes since the last train
        "Vehicle": 5023.0,  # Example vehicle ID
        "Latitude": 43.7,  # Example latitude of station
        "Longitude": -79.4,  # Example longitude of station
        "Station_MUSEUM STATION": 1,  # One-hot encoded station
        "Incident_Priority One - Train in Contact With Person": 0,  # Incident type
        "Line_YU": 1,  # Subway line indicator
        "Direction_N": 1,  # Northbound train
        "Time of Day_Evening": 0,  # One-hot encoded time category
        "Time of Day_Morning": 1,  # Morning category active
        "Time of Day_Night": 0   # Night category inactive
    }
    
    make_prediction(sample_input)