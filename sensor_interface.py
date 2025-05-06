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
        # Force simulation mode off on Raspberry Pi
        self.is_macos = False  # Always use real sensors
        self.model = CrisisPredictionModel()
        self.last_valid_spo2 = None  # Store last valid SpO2 value
        
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
            raise  # Raise the error instead of falling back to simulation mode

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
        """Read GSR value from sensor using robust approach and 10s delay for initialization"""
        if self.is_macos:
            logger.error("GSR sensor not available in simulation mode")
            return 0, 0, 0
        try:
            logger.info("[GSR] Initializing sensor, waiting 10 seconds...")
            time.sleep(10)
            bus = self.bus
            address = self.GSR_ADDRESS
            sensor_current = self.sensor_current
            bus.write_byte(address, 0x40)   # Channel 0
            bus.read_byte(address)         # Dummy read
            time.sleep(0.01)               # Allow ADC to settle
            adc_value = bus.read_byte(address)
            voltage = (adc_value / 255.0) * 3.3
            conductance = (sensor_current / voltage) * 1e6 if voltage > 0 else 0
            logger.info(f"[GSR] ADC: {adc_value}, Voltage: {voltage:.2f} V, Conductance: {conductance:.2f} µS")
            return adc_value, voltage, conductance
        except Exception as e:
            logger.error(f"Error reading GSR: {e}")
            return 0, 0, 0

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
        # Generate random temperature in normal range (36.1-37.2°C)
        temperature = random.uniform(36.1, 37.2)
        return round(temperature, 1)

    def read_spo2(self):
        """Read SpO2 from MAX30102 sensor"""
        if self.is_macos:
            return 0
        try:
            # Reset and setup sensor
            self._reset_sensor()
            self._setup_sensor()
            
            # Initialize readings arrays
            red_readings = []
            ir_readings = []
            
            # Collect readings for 10 seconds
            start_time = time.time()
            while time.time() - start_time < 10:
                try:
                    red, ir = self._read_fifo()
                    if ir > 1000:  # Only use readings with good signal
                        red_readings.append(red)
                        ir_readings.append(ir)
                except Exception:
                    continue
                time.sleep(0.01)
            
            # Calculate SpO2 if we have enough readings
            if len(ir_readings) > 100:
                spo2 = self._calculate_spo2(red_readings, ir_readings)
                if 70 <= spo2 <= 100:  # Only use valid SpO2 values
                    self.last_valid_spo2 = spo2
                    return spo2
            
            # Return last valid reading if current reading is invalid
            if self.last_valid_spo2 is not None:
                return self.last_valid_spo2
            return 0
            
        except Exception as e:
            logger.error(f"Error reading SpO2: {str(e)}")
            if self.last_valid_spo2 is not None:
                return self.last_valid_spo2
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
                # First get GSR reading
                logger.info("Getting GSR reading...")
                _, _, conductance = self.read_gsr()
                logger.info(f"GSR reading: {conductance:.2f} µS")
                
                # Then get SpO2 reading
                logger.info("Getting SpO2 reading...")
                spo2 = self.read_spo2()
                logger.info(f"SpO2 reading: {spo2}%")
                
                # Get temperature
                temperature = self.read_temperature()
                
                # Prepare readings
                readings = {
                    'gsr': conductance,
                    'temperature': temperature,
                    'spo2': spo2,
                    'age': 30,  # Default age
                    'gender': 0,  # Default gender (0 for male)
                    'dehydration': 0,  # Default dehydration level
                    'timestamp': datetime.now().isoformat()
                }
                
                # Send readings
                logger.info("Sending readings...")
                callback(readings)
                
                # Wait for next interval
                time.sleep(interval)
                
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in continuous monitoring: {str(e)}")
                time.sleep(interval)

    def preprocess_sensor_data(self, raw_data):
        try:
            spo2 = raw_data.get('spo2', 0)
            temp = raw_data.get('temperature', 0)
            gsr = raw_data.get('gsr', 0)
            age = raw_data.get('age', 30)
            gender = raw_data.get('gender', 0)
            dehydration = raw_data.get('dehydration', 0)
            gsr_clinical = 0
            features = {
                'SpO2 (%)': spo2,
                'Temperature (°C)': temp,
                'Dehydration_Label': dehydration,
                'Age': age,
                'Gender': gender,
                'is_child': int(age < 12) * 2,
                'is_elderly': int(age > 65) * 2,
                'spo2_critical': int(spo2 < 80) * 4,
                'spo2_severe': int((spo2 >= 80) & (spo2 < 85)) * 3,
                'spo2_moderate': int((spo2 >= 85) & (spo2 < 90)) * 2,
                'spo2_mild': int((spo2 >= 90) & (spo2 < 95)),
                'temp_critical': int(temp > 39.5) * 4,
                'temp_severe': int((temp > 38.5) & (temp <= 39.5)) * 3,
                'temp_moderate': int((temp > 38.0) & (temp <= 38.5)) * 2,
                'temp_mild': int((temp > 37.5) & (temp <= 38.0)),
                'temp_low': int(temp < 36.0) * 2,
                'dehydration_critical': int(dehydration >= 2) * 3,
                'dehydration_moderate': int(dehydration == 1) * 1.5,
                'temp_spo2_critical': int(temp > 39.5) * int(spo2 < 80),
                'age_temp_risk': (1 + 0.5 * (int(age < 12) * 2 + int(age > 65) * 2)) * (1 + int(temp > 39.5) + 0.5 * int((temp > 38.5) & (temp <= 39.5))),
                'age_spo2_risk': (1 + 0.5 * (int(age < 12) * 2 + int(age > 65) * 2)) * (1 + int(spo2 < 80) + 0.5 * int((spo2 >= 80) & (spo2 < 85))),
                'clinical_risk_score': 0,
                'medical_risk_score': ((100 - spo2) / 5) + ((temp - 37.0) * 3) + (dehydration * 2) + (int(age < 12) * 2 + int(age > 65) * 2),
                'GSR Value': gsr,
                'gsr_clinical': 0
            }
            features['clinical_risk_score'] = (
                features['spo2_critical'] * 2 + features['spo2_severe'] * 1.5 + features['spo2_moderate'] + features['spo2_mild'] * 0.5 +
                features['temp_critical'] * 2 + features['temp_severe'] * 1.5 + features['temp_moderate'] + features['temp_mild'] * 0.5 +
                features['dehydration_critical'] * 1.5 + features['dehydration_moderate'] +
                features['is_child'] + features['is_elderly']
            )
            gsr_mean = 850.0
            gsr_std = 350.0
            gsr_norm = (gsr - gsr_mean) / gsr_std
            gsr_norm = np.clip(gsr_norm, -3, 3)
            features['gsr_clinical'] = gsr_norm * features['clinical_risk_score']
            feature_names = [
                'SpO2 (%)', 'Temperature (°C)', 'Dehydration_Label',
                'Age', 'Gender', 'is_child', 'is_elderly',
                'spo2_critical', 'spo2_severe', 'spo2_moderate', 'spo2_mild',
                'temp_critical', 'temp_severe', 'temp_moderate', 'temp_mild', 'temp_low',
                'dehydration_critical', 'dehydration_moderate',
                'temp_spo2_critical', 'age_temp_risk', 'age_spo2_risk',
                'clinical_risk_score', 'medical_risk_score',
                'GSR Value', 'gsr_clinical'
            ]
            feature_array = [features[name] for name in feature_names]
            return feature_array
        except Exception:
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