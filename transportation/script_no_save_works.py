import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

'''
Data Loading & Cleaning
Feature Engineering (Encoding & Normalization)
Random Forest Model Training & Evaluation
Feature Importance Analysis
'''


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
    df["Snow on Grnd (cm)"].fillna(0, inplace=True)
    
    return df

def feature_engineering(df):
    """Creates new features and encodes categorical variables."""
    # Convert 'Time' column to hour
    df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.hour
    
    # Define time of day categories
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
    
    # One-hot encode categorical features
    categorical_cols = ["Station", "Incident", "Line", "Direction", "Time of Day"]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Normalize numerical features
    scaler = MinMaxScaler()
    weather_features = ["Mean Temp (°C)", "Total Rain (mm)", "Total Snow (cm)", "Total Precip (mm)", "Snow on Grnd (cm)", "Spd of Max Gust (km/h)", "Time"]
    df[weather_features] = scaler.fit_transform(df[weather_features])
    
    return df

def train_model(df):
    """Trains a Random Forest regression model and evaluates it."""
    # Define features and target variable
    X = df.drop(columns=["Min Delay", "Date", "Day", "Row", "Delay Code"])
    y = df["Min Delay"]
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
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
    
    return model, X

def feature_importance(model, X):
    """Displays feature importance from the trained model."""
    feature_importances = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False).head(15)
    
    # Plot feature importance
    plt.figure(figsize=(10, 5))
    sns.barplot(y=feature_importances["Feature"], x=feature_importances["Importance"])
    plt.xlabel("Feature Importance Score")
    plt.ylabel("Feature")
    plt.title("Top 15 Most Important Features in Delay Prediction")
    plt.show()
    
    return feature_importances

def main():
    file_path = "cleaned_data/subway_data.csv"  
    
    # Load and process data
    df = load_data(file_path)
    df = clean_data(df)
    df = feature_engineering(df)
    
    # Train model and evaluate performance
    model, X = train_model(df)
    
    # Analyze feature importance
    feature_importance(model, X)

if __name__ == "__main__":
    main()
