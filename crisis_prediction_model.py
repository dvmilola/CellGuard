import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
import joblib

# Function to generate artificial sensor data
def generate_artificial_data(n_samples=1000):
    """
    Generate artificial sensor data for crisis prediction.
    
    Args:
        n_samples: Number of data points to generate
        
    Returns:
        DataFrame with GSR, temperature, SpO2, and crisis labels
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate normal ranges for each sensor
    gsr_normal = np.random.normal(loc=2.5, scale=0.5, size=n_samples)
    temp_normal = np.random.normal(loc=36.6, scale=0.3, size=n_samples)  # in Celsius
    spo2_normal = np.random.normal(loc=97, scale=1, size=n_samples)
    
    # Ensure values are in realistic ranges
    gsr = np.clip(gsr_normal, 0.5, 5.0)  # GSR in microSiemens
    temp = np.clip(temp_normal, 35.0, 39.0)  # Temperature in Celsius
    spo2 = np.clip(spo2_normal, 85, 100)  # SpO2 in percentage
    
    # Create crisis probability based on sensor values
    # Crisis more likely when GSR is high, temp is high, and SpO2 is low
    crisis_prob = (gsr / 5.0) * 0.4 + ((temp - 35) / 4.0) * 0.3 + ((100 - spo2) / 15.0) * 0.3
    
    # Add some noise to make the relationship less perfect
    crisis_prob = crisis_prob + np.random.normal(0, 0.1, n_samples)
    crisis_prob = np.clip(crisis_prob, 0, 1)
    
    # Set crisis label based on probability threshold
    crisis = (crisis_prob > 0.65).astype(int)
    
    # Create DataFrame
    data = pd.DataFrame({
        'gsr': gsr,
        'temperature': temp,
        'spo2': spo2,
        'crisis': crisis,
        'crisis_probability': crisis_prob
    })
    
    return data

# Generate artificial data
data = generate_artificial_data(1000)

# Split data into features and target
X = data[['gsr', 'temperature', 'spo2']]
y = data['crisis']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create pipeline with preprocessing and model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

# Evaluate the model
if __name__ == "__main__":
    # Only run model evaluation if this file is run directly
    print("Model Evaluation:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save the trained model
    joblib.dump(pipeline, 'crisis_model.joblib')
    print("\nModel saved to crisis_model.joblib")

# Function to predict crisis likelihood from new sensor readings
def predict_crisis(gsr, temperature, spo2, model=pipeline):
    """
    Predict crisis likelihood from sensor readings.
    
    Args:
        gsr: Galvanic Skin Response value
        temperature: Body temperature in Celsius
        spo2: Blood oxygen saturation percentage
        model: Trained model to use for prediction (default: pipeline)
        
    Returns:
        Tuple of (crisis_predicted, crisis_probability)
    """
    # Create input array
    X_new = np.array([[gsr, temperature, spo2]])
    
    # Make prediction
    crisis_prob = model.predict_proba(X_new)[0, 1]
    crisis_predicted = crisis_prob > 0.65
    
    return crisis_predicted, crisis_prob