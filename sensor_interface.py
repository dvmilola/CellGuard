import time
import platform
from datetime import datetime
import random
import logging
import numpy as np
from ml_model import CrisisPredictionModel
import math
from periphery import I2C
import smbus

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

### ---------- GSR SENSOR (PCF8591) SETUP ---------- ###

class SensorInterface:
    def __init__(self):
        self.is_macos = platform.system() == 'Darwin'
        self.model = CrisisPredictionModel()
        
        # Feature names for model preprocessing
        self.feature_names = [
            # Raw vital signs (3)
            'SpO2 (%)', 'Temperature (°C)', 'Dehydration_Label',
            
            # Demographics (6)
            'Age', 'Gender', 'age_group', 'is_child', 'is_elderly', 'age_risk',
            
            # Risk tiers (10)
            'spo2_critical', 'spo2_severe', 'spo2_moderate',
            'temp_critical', 'temp_severe', 'temp_moderate', 'temp_low',
            'dehydration_critical', 'dehydration_moderate',
            
            # Transformations (2)
            'exp_temp', 'exp_spo2',
            
            # Interactions (4)
            'temp_spo2_critical', 'temp_spo2_interaction',
            'age_temp_risk', 'age_spo2_risk',
            
            # Clinical scores (2)
            'clinical_risk_score', 'medical_risk_score'
        ]
        
        if not self.is_macos:
            try:
                # Initialize I2C bus for GSR
                self.bus = smbus.SMBus(1)
                self.GSR_ADDRESS = 0x48
                self.sensor_current = 0.001  # 1 mA assumed
                
                # Initialize I2C for MAX30102
                self.i2c = I2C("/dev/i2c-1")
                self.MAX30102_ADDR = 0x57
                
                # MAX30102 registers
                self.REG_FIFO_DATA = 0x07
                self.REG_MODE_CONFIG = 0x09
                self.REG_SPO2_CONFIG = 0x0A
                self.REG_LED1_PA = 0x0C
                self.REG_LED2_PA = 0x0D
                self.REG_INT_ENABLE = 0x02
                self.REG_PART_ID = 0xFF
                self.REG_FIFO_WR_PTR = 0x04
                self.REG_OVF_COUNTER = 0x05
                self.REG_FIFO_RD_PTR = 0x06
                
                # Setup MAX30102
                self._setup_sensor()
                
                logger.info("Successfully initialized sensors")
            except Exception as e:
                logger.error(f"Failed to initialize sensors: {str(e)}")
                logger.error("Running in simulation mode")
                self.is_macos = True
        else:
            logger.info("Running in simulation mode on macOS")

    def _write_register(self, reg, value):
        self.i2c.transfer(self.MAX30102_ADDR, [I2C.Message([reg, value])])

    def _read_register(self, reg):
        messages = [I2C.Message([reg]), I2C.Message(bytearray(1), read=True)]
        self.i2c.transfer(self.MAX30102_ADDR, messages)
        return messages[1].data[0]

    def _setup_sensor(self):
        self._write_register(self.REG_MODE_CONFIG, 0x40)
        time.sleep(0.1)
        while self._read_register(self.REG_MODE_CONFIG) & 0x40:
            time.sleep(0.1)
        self._reset_sensor()
        self._write_register(self.REG_MODE_CONFIG, 0x03)
        self._write_register(self.REG_SPO2_CONFIG, 0x27)
        self._write_register(self.REG_LED1_PA, 0x24)
        self._write_register(self.REG_LED2_PA, 0x24)
        self._write_register(self.REG_INT_ENABLE, 0x10)

    def _reset_sensor(self):
        self._write_register(self.REG_FIFO_WR_PTR, 0x00)
        self._write_register(self.REG_OVF_COUNTER, 0x00)
        self._write_register(self.REG_FIFO_RD_PTR, 0x00)

    def _read_fifo(self):
        messages = [I2C.Message([self.REG_FIFO_DATA]), I2C.Message(bytearray(6), read=True)]
        self.i2c.transfer(self.MAX30102_ADDR, messages)
        data = messages[1].data
        red = (data[0] << 16 | data[1] << 8 | data[2]) & 0x3FFFF
        ir = (data[3] << 16 | data[4] << 8 | data[5]) & 0x3FFFF
        return red, ir

    def read_gsr(self):
        """Read GSR value from sensor"""
        if self.is_macos:
            logger.error("GSR sensor not available in simulation mode")
            return 0, 0, 0
            
        try:
            self.bus.write_byte(self.GSR_ADDRESS, 0x40)
            self.bus.read_byte(self.GSR_ADDRESS)
            time.sleep(0.01)
            adc_value = self.bus.read_byte(self.GSR_ADDRESS)
        except IOError as e:
            logger.error(f"GSR I2C read error: {e}")
            return 0, 0, 0

        voltage = (adc_value / 255.0) * 3.3
        conductance = (self.sensor_current / voltage) * 1e6 if voltage > 0 else 0
        return adc_value, voltage, conductance

    def get_average_gsr(self, duration=10):
        """Get average GSR readings over specified duration"""
        total_adc = total_voltage = total_conductance = 0
        readings = 0
        logger.info(f"[GSR] Collecting data for {duration} seconds...")
        start_time = time.time()

        while time.time() - start_time < duration:
            adc, volt, cond = self.read_gsr()
            total_adc += adc
            total_voltage += volt
            total_conductance += cond
            readings += 1
            time.sleep(0.1)

        if readings == 0:
            return 0, 0, 0

        return total_adc / readings, total_voltage / readings, total_conductance / readings

    def read_temperature(self):
        """Read temperature from sensor"""
        if self.is_macos:
            logger.error("Temperature sensor not available in simulation mode")
            return 0
            
        try:
            # Read temperature from sensor
            # TODO: Implement actual temperature sensor reading
            logger.error("Temperature sensor not implemented")
            return 0
        except Exception as e:
            logger.error(f"Error reading temperature: {str(e)}")
            return 0

    def read_spo2(self):
        """Read SpO2 from MAX30102 sensor"""
        if self.is_macos:
            logger.error("SpO2 sensor not available in simulation mode")
            return 0
            
        try:
            red_readings = []
            ir_readings = []
            logger.info("[MAX30102] Reading in 3 seconds...")
            time.sleep(3)
            self._reset_sensor()
            start = time.time()
            while time.time() - start < 10:  # Collect data for 10 seconds
                try:
                    red, ir = self._read_fifo()
                    if ir > 1000:
                        red_readings.append(red)
                        ir_readings.append(ir)
                except:
                    continue
                time.sleep(0.01)  # 100Hz sampling rate
                
            if len(red_readings) >= 100:
                spo2 = self._calculate_spo2(red_readings, ir_readings)
                logger.info(f"SpO₂: {spo2}%")
                return spo2
            else:
                logger.error(f"Not enough data from MAX30102. Got {len(red_readings)} readings")
                return 0
                
        except Exception as e:
            logger.error(f"Error reading SpO2: {str(e)}")
            return 0

    def _calculate_spo2(self, red, ir):
        """Calculate SpO2 from red and IR readings"""
        if len(red) < 100:
            return 0
            
        r_mean = sum(red) / len(red)
        ir_mean = sum(ir) / len(ir)
        
        r_rms = math.sqrt(sum([(x - r_mean)**2 for x in red]) / len(red))
        ir_rms = math.sqrt(sum([(x - ir_mean)**2 for x in ir]) / len(ir))
        
        r = (r_rms / r_mean) / (ir_rms / ir_mean)
        spo2 = 110 - 25 * r
        
        return round(spo2, 1) if 70 <= spo2 <= 100 else 0

    def calibrate_gsr(self, duration=10):
        """Calibrate GSR sensor by taking average readings over duration seconds"""
        if self.is_macos:
            logger.info("Simulating GSR calibration...")
            time.sleep(2)
            self.baseline_gsr = 300  # Simulated baseline GSR value
            logger.info(f"GSR baseline: {self.baseline_gsr:.2f} µS")
            return self.baseline_gsr
            
        logger.info(f"Collecting GSR data for {duration} seconds...")
        start_time = time.time()
        total_conductance = 0
        readings = 0
        
        try:
            while time.time() - start_time < duration:
                _, _, conductance = self.read_gsr()
                if conductance > 0:
                    total_conductance += conductance
                    readings += 1
                time.sleep(0.1)
                
            if readings > 0:
                self.baseline_gsr = total_conductance / readings
                logger.info(f"GSR baseline: {self.baseline_gsr:.2f} µS")
                return self.baseline_gsr
            else:
                logger.error("No valid GSR readings during calibration")
                self.baseline_gsr = 0
                return 0
        except Exception as e:
            logger.error(f"Error during GSR calibration: {str(e)}")
            self.baseline_gsr = 0
            return 0
        
    def continuous_monitoring(self, callback, interval=1.0):
        """Continuously monitor sensors and call callback with readings"""
        while True:
            try:
                # Get GSR reading
                _, _, gsr = self.read_gsr()
                
                # Get temperature and SpO2 readings
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
                # Raw vital signs
                'SpO2 (%)': spo2,
                'Temperature (°C)': temp,
                'Dehydration_Label': dehydration,
                
                # Demographics
                'Age': age,
                'Gender': gender,
                'age_group': self._get_age_group(age),
                'is_child': int(age < 12),
                'is_elderly': int(age > 65),
                'age_risk': int(age < 12) + int(age > 65),
                
                # Risk tiers
                'spo2_critical': int(spo2 < 80) * 4,
                'spo2_severe': int((spo2 >= 80) & (spo2 < 85)) * 3,
                'spo2_moderate': int((spo2 >= 85) & (spo2 < 90)) * 2,
                'temp_critical': int(temp > 39.5) * 4,
                'temp_severe': int((temp > 38.5) & (temp <= 39.5)) * 3,
                'temp_moderate': int((temp > 38.0) & (temp <= 38.5)) * 2,
                'temp_low': int(temp < 37.5) * 1,
                'dehydration_critical': int(dehydration >= 2) * 3,
                'dehydration_moderate': int(dehydration == 1) * 1.5,
                
                # Transformations
                'exp_temp': np.exp(temp - 37.0),
                'exp_spo2': np.exp((100 - spo2) / 10),
                
                # Interactions
                'temp_spo2_critical': int(temp > 39.5) * int(spo2 < 80),
                'temp_spo2_interaction': temp * (100 - spo2),
                
                # Age-weighted risks
                'age_temp_risk': (1 + 0.5 * (int(age < 12) + int(age > 65))) * 
                                (1 + int(temp > 39.5) + 0.5 * int((temp > 38.5) & (temp <= 39.5))),
                'age_spo2_risk': (1 + 0.5 * (int(age < 12) + int(age > 65))) * 
                                (1 + int(spo2 < 80) + 0.5 * int((spo2 >= 80) & (spo2 < 85))),
                
                # Clinical scores
                'clinical_risk_score': self._calculate_clinical_risk_score(spo2, temp, dehydration, age),
                'medical_risk_score': self._calculate_medical_risk_score(spo2, temp, dehydration, age)
            }
            
            # Convert to array in correct order
            feature_array = [features[name] for name in self.feature_names]
            
            # Ensure we have exactly 26 features
            if len(feature_array) != 26:
                logger.error(f"Expected 26 features but got {len(feature_array)}")
                raise ValueError(f"Expected 26 features but got {len(feature_array)}")
            
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
        score = (
            int(spo2 < 80) * 2 + int((spo2 >= 80) & (spo2 < 85)) * 1.5 + 
            int((spo2 >= 85) & (spo2 < 90)) + int((spo2 >= 90) & (spo2 < 95)) * 0.5 +
            int(temp > 39.5) * 2 + int((temp > 38.5) & (temp <= 39.5)) * 1.5 + 
            int((temp > 38.0) & (temp <= 38.5)) + int((temp > 37.5) & (temp <= 38.0)) * 0.5 +
            int(dehydration >= 2) * 1.5 + int(dehydration == 1) +
            (int(age < 12) + int(age > 65))
        )
        return score
    
    def _calculate_medical_risk_score(self, spo2, temp, dehydration, age):
        """Calculate medical risk score"""
        score = (
            ((100 - spo2) / 5) +              # Oxygen deficit
            ((temp - 37.0) * 3) +             # Temperature deviation
            (dehydration * 2) +               # Dehydration factor
            (int(age < 12) + int(age > 65))   # Age risk
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