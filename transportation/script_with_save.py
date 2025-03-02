import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_data(file_path):
    """Loads the subway dataset."""
    return pd.read_csv(file_path)

def clean_data(df):
    """Handles missing values and cleans the dataset."""
    df['Direction'].fillna(df['Direction'].mode()[0], inplace=True)
    df['Incident'].fillna("Unknown", inplace=True)
    
    # Fill weather-related missing values with median
    weather_cols = ["Mean Temp (°C)", "Total Rain (mm)", "Total Snow (cm)", "Spd of Max Gust (km/h)"]
    for col in weather_cols:
        df[col].fillna(df[col].median(), inplace=True)
    
    # Assume no snow if "Snow on Grnd (cm)" is missing
    df.fillna({"Snow on Grnd (cm)": 0}, inplace=True)
    
    return df

def feature_engineering(df):
    """Creates new features and encodes categorical variables."""
    df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.hour
    
    def categorize_time(hour):
        if 6 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 18:
            return "Afternoon"
        elif 18 <= hour < 24:
            return "Evening"
        else:
            return "Night"
    
    df["Time of Day"] = df["Time"].apply(categorize_time)
    
    # One-hot encoding
    categorical_cols = ["Station", "Incident", "Line", "Direction", "Time of Day"]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Normalize numerical features
    scaler = MinMaxScaler()
    weather_features = ["Mean Temp (°C)", "Total Rain (mm)", "Total Snow (cm)", "Total Precip (mm)", "Snow on Grnd (cm)", "Spd of Max Gust (km/h)", "Time"]
    df[weather_features] = scaler.fit_transform(df[weather_features])
    
    return df, scaler

def train_model(df):
    """Trains a Random Forest regression model, evaluates it, and saves the model."""
    X = df.drop(columns=["Min Delay", "Date", "Day", "Row", "Delay Code"])
    y = df["Min Delay"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the trained model
    joblib.dump(model, "subway_delay_model.pkl") 
    joblib.dump(X.columns, "model_features.pkl")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluate performance
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("Model Performance:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return model, X.columns

def predict_delay(input_data):
    """Loads the trained model and makes a prediction based on input features."""
    model = joblib.load("subway_delay_model.pkl")
    feature_names = joblib.load("model_features.pkl")
    
    # Convert input to DataFrame with correct feature order
    input_df = pd.DataFrame([input_data], columns=feature_names)
    
    prediction = model.predict(input_df)[0]
    print(f"Predicted Delay Duration: {prediction:.2f} minutes")
    return prediction

def main():
    file_path = "cleaned_data/subway_data.csv"  # Update path if needed
    
    # Load and process data
    df = load_data(file_path)
    df = clean_data(df)
    df, scaler = feature_engineering(df)
    
    # Train model and evaluate performance
    model, feature_names = train_model(df)
    
    # # Example prediction
    # sample_input = dict(zip(feature_names, np.zeros(len(feature_names))))  # Dummy input
    # feature_names = joblib.load("model_features.pkl")
    # sample_input = dict(zip(feature_names, np.zeros(len(feature_names))))  # Initialize all features with 0
    # sample_input["Time"] = scaler.transform(pd.DataFrame([[12] * len(weather_features)], columns=weather_features))[0][0]  # Example: 12 PM (normalized)
    # predict_delay(sample_input)

if __name__ == "__main__":
    main()
