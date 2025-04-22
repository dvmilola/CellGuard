import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime

class CrisisPredictionModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.threshold = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model, scaler, and threshold"""
        try:
            self.model = joblib.load('crisis_model_20250422_024149.joblib')
            self.scaler = joblib.load('crisis_scaler_20250422_024149.joblib')
            with open('crisis_threshold_20250422_024149.txt', 'r') as f:
                self.threshold = float(f.read().strip())
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess_features(self, features):
        """Preprocess features using the same pipeline as training"""
        # Convert features to numpy array
        features_array = np.array(features).reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features_array)
        
        return features_scaled
    
    def predict(self, features):
        """Make prediction using the loaded model"""
        try:
            # Preprocess features
            features_scaled = self.preprocess_features(features)
            
            # Get probability prediction
            proba = self.model.predict_proba(features_scaled)[:, 1]
            
            # Apply threshold
            prediction = (proba >= self.threshold).astype(int)
            
            return {
                'prediction': int(prediction[0]),
                'probability': float(proba[0]),
                'threshold': self.threshold
            }
        except Exception as e:
            print(f"Error making prediction: {e}")
            raise

# Create a global instance
prediction_model = CrisisPredictionModel()

def load_model():
    """Load the saved model and scaler, or create new ones if they don't exist"""
    try:
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Construct paths to model and scaler files
        model_path = os.path.join(script_dir, 'crisis_model.joblib')
        scaler_path = os.path.join(script_dir, 'crisis_scaler.joblib')
        
        # If files don't exist, create and train new model
        if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
            print("[INFO] No existing model found. Creating new model...")
            model = CrisisPredictionModel()
            return model.model
        
        # Load existing model and scaler
        model = joblib.load(model_path)
        print("[INFO] Loaded existing model and scaler")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load model: {str(e)}")
        raise

def predict_crisis(gsr, temperature, spo2):
    """Make a prediction using the loaded model"""
    try:
        # Get the model instance
        model = load_model()
        
        # Make prediction
        result = prediction_model.predict([gsr, temperature, spo2])
        
        return {
            'prediction': int(result['prediction']),
            'probability': result['probability'],
            'threshold': result['threshold'],
            'timestamp': datetime.utcnow().isoformat()
        }
    except Exception as e:
        print(f"[ERROR] Prediction failed: {str(e)}")
        raise 