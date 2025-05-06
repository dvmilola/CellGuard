from sensor_interface import SensorInterface
import time
import numpy as np

def test_prediction():
    # Create sensor interface
    sensors = SensorInterface()
    
    # Test cases with different scenarios
    test_cases = [
        {
            'name': 'Normal readings',
            'readings': {
                'gsr': 300,  # Normal GSR
                'temperature': 36.5,  # Normal temperature
                'spo2': 98,  # Normal SpO2
                'age': 30,
                'gender': 0,
                'dehydration': 0,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        },
        {
            'name': 'Mild warning',
            'readings': {
                'gsr': 350,  # Slightly elevated GSR
                'temperature': 37.8,  # Slightly elevated temperature
                'spo2': 92,  # Slightly low SpO2
                'age': 30,
                'gender': 0,
                'dehydration': 1,  # Some dehydration
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        },
        {
            'name': 'Moderate warning',
            'readings': {
                'gsr': 400,  # Elevated GSR
                'temperature': 38.2,  # Elevated temperature
                'spo2': 88,  # Low SpO2
                'age': 30,
                'gender': 0,
                'dehydration': 1,  # Some dehydration
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        },
        {
            'name': 'Severe warning',
            'readings': {
                'gsr': 450,  # High GSR
                'temperature': 38.8,  # High temperature
                'spo2': 85,  # Very low SpO2
                'age': 30,
                'gender': 0,
                'dehydration': 2,  # Severe dehydration
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        },
        {
            'name': 'Extreme Crisis',
            'readings': {
                'gsr': 1200,  # Extremely high GSR
                'temperature': 40.5,  # Dangerous fever
                'spo2': 75,  # Critically low SpO2
                'age': 30,
                'gender': 0,
                'dehydration': 3,  # Severe dehydration
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
    ]
    
    # Run tests
    for case in test_cases:
        print(f"\n{'='*50}")
        print(f"Testing {case['name']}:")
        print(f"{'='*50}")
        print("Input readings:", case['readings'])
        result = sensors.process_sensor_data(case['readings'])
        print(f"\nPrediction results:")
        print(f"Prediction: {result['prediction']}")
        print(f"Probability: {result['probability']:.2f}")
        print(f"Threshold: {result['threshold']:.2f}")
        print(f"Model type: {result.get('model_type', 'unknown')}")

if __name__ == "__main__":
    test_prediction() 