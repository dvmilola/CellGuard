import time
import platform
from datetime import datetime
import random
import logging
import numpy as np
from ml_model import CrisisPredictionModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SensorInterface:
    def __init__(self):
        self.is_macos = platform.system() == 'Darwin'
        self.sensor_current = 0.001  # 1 mA assumed sensor current
        self.model = CrisisPredictionModel()
        self.feature_names = [
            'SpO2 (%)', 'Temperature (°C)', 'Dehydration_Label',
            'Age', 'Gender', 'age_group', 'is_child', 'is_elderly', 'age_risk',
            'spo2_critical', 'spo2_severe', 'spo2_moderate', 'spo2_mild',
            'temp_critical', 'temp_severe', 'temp_moderate', 'temp_mild', 'temp_low',
            'dehydration_critical', 'dehydration_moderate',
            'exp_temp', 'exp_spo2',
            'temp_spo2_critical', 'temp_spo2_interaction',
            'age_temp_risk', 'age_spo2_risk',
            'clinical_risk_score', 'medical_risk_score'
        ]
        
        if not self.is_macos:
            try:
                # Import Raspberry Pi specific modules
                import board
                import busio
                import adafruit_ads1x15.ads1115 as ADS
                from adafruit_ads1x15.analog_in import AnalogIn
                
                # Initialize I2C bus
                self.i2c = busio.I2C(board.SCL, board.SDA)
                
                # Initialize ADS1115 for GSR
                self.ads = ADS.ADS1115(self.i2c)
                self.gsr_channel = AnalogIn(self.ads, ADS.P0)
                
                logger.info("Successfully initialized GSR sensor")
            except ImportError as e:
                logger.error(f"Failed to import Raspberry Pi modules: {str(e)}")
                logger.error("Running in simulation mode")
                self.is_macos = True
            except Exception as e:
                logger.error(f"Failed to initialize GSR sensor: {str(e)}")
                logger.error("Please check if I2C is enabled and ADS1115 is connected")
                raise
        else:
            logger.info("Running in simulation mode on macOS")
            self.baseline_gsr = 300  # Simulated baseline GSR value
            
    def read_gsr(self):
        """Read GSR value from ADS1115 or simulate reading on macOS"""
        if self.is_macos:
            # Simulate GSR readings around baseline with some noise
            noise = random.uniform(-50, 50)
            return max(0, self.baseline_gsr + noise)
            
        try:
            # Read voltage from ADS1115
            voltage = self.gsr_channel.voltage
            
            # Convert to conductance (microsiemens)
            conductance = (self.sensor_current / voltage) * 1e6 if voltage > 0 else 0
            
            logger.debug(f"GSR reading: {conductance:.2f} µS")
            return conductance
            
        except Exception as e:
            logger.error(f"Error reading GSR: {str(e)}")
            return 0

    def read_temperature(self):
        """Simulate temperature reading"""
        # Simulate temperature readings around normal body temperature
        return 36.5 + random.uniform(-0.5, 0.5)  # Normal range: 36-37°C

    def read_spo2(self):
        """Simulate SpO2 reading"""
        # Simulate SpO2 readings around normal range
        return 98 + random.uniform(-2, 0)  # Normal range: 96-100%
            
    def calibrate_gsr(self, duration=10):
        """Calibrate GSR sensor by taking average readings over duration seconds"""
        if self.is_macos:
            logger.info("Simulating GSR calibration...")
            time.sleep(2)
            logger.info(f"GSR baseline: {self.baseline_gsr:.2f} µS")
            return self.baseline_gsr
            
        logger.info(f"Collecting GSR data for {duration} seconds...")
        start_time = time.time()
        total_conductance = 0
        readings = 0
        
        try:
            while time.time() - start_time < duration:
                conductance = self.read_gsr()
                if conductance > 0:
                    total_conductance += conductance
                    readings += 1
                time.sleep(0.1)
                
            if readings > 0:
                baseline = total_conductance / readings
                logger.info(f"GSR baseline: {baseline:.2f} µS")
                return baseline
            else:
                logger.error("No valid GSR readings during calibration")
                return 0
        except Exception as e:
            logger.error(f"Error during GSR calibration: {str(e)}")
            return 0
        
    def continuous_monitoring(self, callback, interval=1.0):
        """Continuously monitor sensors and call callback with readings"""
        while True:
            try:
                # Get GSR reading
                gsr = self.read_gsr()
                
                # Simulate temperature and SpO2 readings
                temperature = self.read_temperature()
                spo2 = self.read_spo2()
                
                readings = {
                    'gsr': gsr,
                    'temperature': temperature,
                    'spo2': spo2,
                    'age': 30,  # Default age
                    'gender': 0,  # Default gender (0 for male)
                    'dehydration': 0,  # Default dehydration level
                    'timestamp': datetime.now().isoformat()
                }
                
                logger.debug(f"Sensor readings: {readings}")
                callback(readings)
                time.sleep(interval)
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in continuous monitoring: {str(e)}")
                time.sleep(interval)

    def preprocess_sensor_data(self, raw_data):
        """Preprocess raw sensor data into model features"""
        try:
            # Extract basic measurements
            spo2 = raw_data.get('spo2', 0)
            temp = raw_data.get('temperature', 0)
            age = raw_data.get('age', 30)
            gender = raw_data.get('gender', 0)  # 0 for male, 1 for female
            dehydration = raw_data.get('dehydration', 0)
            
            # Calculate derived features
            features = {
                'SpO2 (%)': spo2,
                'Temperature (°C)': temp,
                'Dehydration_Label': dehydration,
                'Age': age,
                'Gender': gender,
                
                # Age-related features
                'is_child': 1 if age < 12 else 0,
                'is_elderly': 1 if age > 65 else 0,
                'age_risk': (1 if age < 12 else 0) + (1 if age > 65 else 0),
                'age_group': self._get_age_group(age),
                
                # SpO2 risk levels
                'spo2_critical': 1 if spo2 < 80 else 0,
                'spo2_severe': 1 if 80 <= spo2 < 85 else 0,
                'spo2_moderate': 1 if 85 <= spo2 < 90 else 0,
                'spo2_mild': 1 if 90 <= spo2 < 95 else 0,
                
                # Temperature risk levels
                'temp_critical': 1 if temp > 39.5 else 0,
                'temp_severe': 1 if 38.5 < temp <= 39.5 else 0,
                'temp_moderate': 1 if 38.0 < temp <= 38.5 else 0,
                'temp_mild': 1 if 37.5 < temp <= 38.0 else 0,
                'temp_low': 1 if temp < 36.0 else 0,
                
                # Dehydration risk
                'dehydration_critical': 1 if dehydration >= 2 else 0,
                'dehydration_moderate': 1 if dehydration == 1 else 0,
                
                # Transformations
                'exp_temp': np.exp(temp - 37.0),
                'exp_spo2': np.exp((100 - spo2) / 10),
                
                # Interactions
                'temp_spo2_critical': (1 if temp > 39.5 else 0) * (1 if spo2 < 80 else 0),
                'temp_spo2_interaction': temp * (100 - spo2),
                
                # Age-weighted risks
                'age_temp_risk': (1 + 0.5 * ((1 if age < 12 else 0) + (1 if age > 65 else 0))) * 
                                (1 + (1 if temp > 39.5 else 0) + 0.5 * (1 if 38.5 < temp <= 39.5 else 0)),
                'age_spo2_risk': (1 + 0.5 * ((1 if age < 12 else 0) + (1 if age > 65 else 0))) * 
                                (1 + (1 if spo2 < 80 else 0) + 0.5 * (1 if 80 <= spo2 < 85 else 0)),
                
                # Clinical scores
                'clinical_risk_score': self._calculate_clinical_risk_score(spo2, temp, dehydration, age),
                'medical_risk_score': self._calculate_medical_risk_score(spo2, temp, dehydration, age)
            }
            
            # Convert to array in correct order
            feature_array = [features[name] for name in self.feature_names]
            
            return feature_array
            
        except Exception as e:
            print(f"Error preprocessing sensor data: {e}")
            raise
    
    def _get_age_group(self, age):
        """Convert age to age group index"""
        age_bins = [0, 2, 5, 12, 18, 40, 65, 75, 85, 100]
        for i in range(len(age_bins)-1):
            if age_bins[i] <= age < age_bins[i+1]:
                return i
        return len(age_bins)-2
    
    def _calculate_clinical_risk_score(self, spo2, temp, dehydration, age):
        """Calculate clinical risk score"""
        score = 0
        
        # SpO2 contribution
        if spo2 < 80:
            score += 4
        elif spo2 < 85:
            score += 3
        elif spo2 < 90:
            score += 2
        elif spo2 < 95:
            score += 1
        
        # Temperature contribution
        if temp > 39.5:
            score += 4
        elif temp > 38.5:
            score += 3
        elif temp > 38.0:
            score += 2
        elif temp > 37.5:
            score += 1
        elif temp < 36.0:
            score += 2
        
        # Dehydration contribution
        if dehydration >= 2:
            score += 3
        elif dehydration == 1:
            score += 1.5
        
        # Age risk contribution
        if age < 12 or age > 65:
            score += 2
        
        return score
    
    def _calculate_medical_risk_score(self, spo2, temp, dehydration, age):
        """Calculate medical risk score"""
        score = (
            ((100 - spo2) / 5) +              # Oxygen deficit
            ((temp - 37.0) * 3) +             # Temperature deviation
            (dehydration * 2) +                # Dehydration factor
            (2 if age < 12 or age > 65 else 0)  # Age risk
        )
        return score
    
    def process_sensor_data(self, raw_data):
        """Process raw sensor data and return prediction"""
        try:
            # Preprocess data
            features = self.preprocess_sensor_data(raw_data)
            
            # Make prediction
            result = self.model.predict(features)
            
            return {
                'prediction': result['prediction'],
                'probability': result['probability'],
                'threshold': result['threshold'],
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            print(f"Error processing sensor data: {e}")
            raise

# Example usage
if __name__ == "__main__":
    def print_readings(readings):
        print(f"\nSensor Readings at {readings['timestamp']}:")
        print(f"GSR: {readings['gsr']:.2f} µS")
        print(f"Temperature: {readings['temperature']:.1f}°C")
        print(f"SpO2: {readings['spo2']:.1f}%")
    
    # Create sensor interface
    sensors = SensorInterface()
    
    # Calibrate GSR sensor
    sensors.calibrate_gsr()
    
    # Start continuous monitoring
    print("Starting continuous monitoring (press Ctrl+C to stop)...")
    sensors.continuous_monitoring(print_readings) 