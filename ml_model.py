import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from train_model import train_model  # Import the training function

class CrisisPredictionModel:
    def __init__(self):
        self.global_model = None
        self.user_models = {}  # Dictionary to store user-specific models
        self.scaler = None
        self.threshold = None
        self.load_global_model()
    
    def load_global_model(self):
        """Load the global trained model, scaler, and threshold"""
        try:
            self.global_model = joblib.load('crisis_model_20250422_024149.joblib')
            self.scaler = joblib.load('crisis_scaler_20250422_024149.joblib')
            with open('crisis_threshold_20250422_024149.txt', 'r') as f:
                self.threshold = float(f.read().strip())
            print("Global model loaded successfully")
        except Exception as e:
            print(f"Error loading global model: {e}")
            raise
    
    def load_user_model(self, user_id):
        """Load a user-specific model if it exists and has enough data"""
        model_path = f'models/user_{user_id}_model.joblib'
        scaler_path = f'models/user_{user_id}_scaler.joblib'
        threshold_path = f'models/user_{user_id}_threshold.txt'
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                with open(threshold_path, 'r') as f:
                    threshold = float(f.read().strip())
                self.user_models[user_id] = {
                    'model': model,
                    'scaler': scaler,
                    'threshold': threshold
                }
                print(f"User {user_id} model loaded successfully")
                return True
            except Exception as e:
                print(f"Error loading user {user_id} model: {e}")
                return False
        return False
    
    def train_user_model(self, user_id, sensor_readings, symptoms):
        """Train a user-specific model using their historical data"""
        try:
            # Check if we have enough data
            if len(sensor_readings) < 50 or len(symptoms) < 50:
                print(f"Not enough data to train user-specific model for user {user_id}")
                return False
                
            # Convert sensor readings and symptoms to DataFrame
            df = pd.DataFrame(sensor_readings)
            df['crisis_occurred'] = [1 if s['pain_level'] >= 7 else 0 for s in symptoms]
            
            # Add symptom features
            df['pain_level'] = [s['pain_level'] for s in symptoms]
            df['symptom_count'] = [len(s['symptoms'].split(',')) for s in symptoms]
            
            # Split data
            X = df.drop('crisis_occurred', axis=1)
            y = df['crisis_occurred']
            
            # Train model
            model, scaler, threshold = train_model(X, y)
            
            # Save user model
            os.makedirs('models', exist_ok=True)
            joblib.dump(model, f'models/user_{user_id}_model.joblib')
            joblib.dump(scaler, f'models/user_{user_id}_scaler.joblib')
            with open(f'models/user_{user_id}_threshold.txt', 'w') as f:
                f.write(str(threshold))
            
            # Update user models dictionary
            self.user_models[user_id] = {
                'model': model,
                'scaler': scaler,
                'threshold': threshold
            }
            
            print(f"User {user_id} model trained and saved successfully")
            return True
            
        except Exception as e:
            print(f"Error training user {user_id} model: {e}")
            return False
    
    def preprocess_features(self, features, user_id=None):
        """Preprocess features using the appropriate scaler"""
        # Convert features to numpy array
        features_array = np.array(features).reshape(1, -1)
        
        # Use user-specific scaler if available and has enough data, otherwise use global scaler
        if user_id and user_id in self.user_models:
            features_scaled = self.user_models[user_id]['scaler'].transform(features_array)
        else:
            features_scaled = self.scaler.transform(features_array)
        
        return features_scaled
    
    def predict(self, features, user_id=None):
        """Make prediction using the appropriate model"""
        try:
            # Preprocess features
            features_scaled = self.preprocess_features(features, user_id)
            
            # Use user-specific model if available and has enough data, otherwise use global model
            if user_id and user_id in self.user_models:
                model = self.user_models[user_id]['model']
                threshold = self.user_models[user_id]['threshold']
                model_type = 'user_specific'
            else:
                model = self.global_model
                threshold = self.threshold
                model_type = 'global'
            
            # Get probability prediction
            proba = model.predict_proba(features_scaled)[:, 1]
            
            # Apply threshold
            prediction = (proba >= threshold).astype(int)
            
            return {
                'prediction': int(prediction[0]),
                'probability': float(proba[0]),
                'threshold': threshold,
                'model_type': model_type
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
            return model.global_model
        
        # Load existing model and scaler
        model = joblib.load(model_path)
        print("[INFO] Loaded existing model and scaler")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load model: {str(e)}")
        raise

def predict_crisis(gsr, temperature, spo2, user_id=None):
    """Make a prediction using the appropriate model"""
    try:
        # Make prediction
        result = prediction_model.predict([gsr, temperature, spo2], user_id)
        
        return {
            'prediction': int(result['prediction']),
            'probability': result['probability'],
            'threshold': result['threshold'],
            'model_type': result['model_type'],
            'timestamp': datetime.utcnow().isoformat()
        }
    except Exception as e:
        print(f"[ERROR] Prediction failed: {str(e)}")
        raise 