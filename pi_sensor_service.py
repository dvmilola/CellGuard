from flask import Flask, request, jsonify
import threading
import time
import requests
import os
from sensor_interface import SensorInterface

# Configuration
MAIN_SERVER_URL = os.environ.get('MAIN_SERVER_URL', 'http://172.20.10.2:5001/api/sensor-readings')
SHARED_SECRET = os.environ.get('SENSOR_API_SECRET', 'cellguard_secret_2024')
USER_ID = os.environ.get('SENSOR_USER_ID', '1')  # Set this to the correct user ID

app = Flask(__name__)

monitoring = False
monitoring_thread = None

sensor_interface = SensorInterface()

age = 30
gender = 'male'

def collect_and_send_sensor_data():
    global monitoring, age, gender
    gender_map = {'male': 0, 'female': 1, 'other': 2}
    gender_code = gender_map.get(gender, 0)
    while monitoring:
        # Get real sensor readings
        _, _, gsr = sensor_interface.read_gsr()
        temperature = sensor_interface.read_temperature()
        spo2 = sensor_interface.read_spo2()
        # Prepare raw data for model
        raw_data = {
            'gsr': gsr,
            'temperature': temperature,
            'spo2': spo2,
            'age': age,
            'gender': gender_code,
            'dehydration': 0,  # or get from config
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')
        }
        result = sensor_interface.process_sensor_data(raw_data)
        probability = result['probability']
        prediction = result['prediction']
        sensor_data = {
            'gsr': gsr,
            'temperature': temperature,
            'spo2': spo2,
            'dehydration': raw_data.get('dehydration', 0),
            'age': raw_data.get('age', 30),
            'gender': gender_code,
            'probability': probability,
            'prediction': prediction,
            'timestamp': raw_data['timestamp'],
            'user_id': USER_ID
        }
        try:
            headers = {'Authorization': f'Bearer {SHARED_SECRET}'}
            resp = requests.post(MAIN_SERVER_URL, json=sensor_data, headers=headers, timeout=5)
            print(f"[PI] Sent sensor data, response: {resp.status_code}")
        except Exception as e:
            print(f"[PI] Error sending sensor data: {e}")
        time.sleep(2)  # Adjust interval as needed

@app.route('/start-monitoring', methods=['POST'])
def start_monitoring():
    global monitoring, monitoring_thread, age, gender
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    if token != SHARED_SECRET:
        return jsonify({'error': 'Unauthorized'}), 401
    if monitoring:
        return jsonify({'status': 'already_running'})
    # Get age and gender from request body if present
    data = request.get_json(silent=True) or {}
    age = int(data.get('age', 30))
    gender = data.get('gender', 'male')
    monitoring = True
    monitoring_thread = threading.Thread(target=collect_and_send_sensor_data, daemon=True)
    monitoring_thread.start()
    return jsonify({'status': 'started'})

@app.route('/stop-monitoring', methods=['POST'])
def stop_monitoring():
    global monitoring
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    if token != SHARED_SECRET:
        return jsonify({'error': 'Unauthorized'}), 401
    monitoring = False
    return jsonify({'status': 'stopped'})

@app.route('/status', methods=['GET'])
def status():
    return jsonify({'monitoring': monitoring})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True) 