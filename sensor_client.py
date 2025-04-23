import time
import requests
from sensor_interface import SensorInterface
import json
from datetime import datetime
import urllib3
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class SensorClient:
    def __init__(self, server_url="http://localhost:5001", username="adamilola311@gmail.com", password="Adebayo2004."):
        self.server_url = server_url
        self.sensors = SensorInterface()
        self.retry_count = 0
        self.max_retries = 3
        self.session = requests.Session()
        self.session.verify = False  # Disable SSL verification for local development
        
        # First try to create the user if it doesn't exist
        try:
            logger.info(f"Attempting to create user {username}")
            response = self.session.post(
                f"{self.server_url}/api/signup",
                json={
                    "fullName": "Test User",
                    "email": username,
                    "password": password,
                    "userType": "patient"
                },
                timeout=5,
                headers={
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                }
            )
            logger.debug(f"Signup response: {response.status_code} - {response.text}")
            if response.status_code == 200:
                logger.info("User created successfully")
            elif response.status_code == 400 and "already registered" in response.text:
                logger.info("User already exists")
            else:
                logger.error(f"Failed to create user: {response.text}")
        except requests.exceptions.ConnectionError:
            logger.error(f"\nCould not connect to {self.server_url}. Please check:")
            logger.error("1. The server URL is correct")
            logger.error("2. The server is running")
            logger.error("3. The network connection is working")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error creating user: {str(e)}")
            sys.exit(1)
        
        # Then try to login
        try:
            logger.info(f"Attempting to connect to {self.server_url}")
            response = self.session.post(
                f"{self.server_url}/api/login",
                json={"email": username, "password": password},
                timeout=5,
                headers={
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                }
            )
            logger.debug(f"Login response: {response.status_code} - {response.text}")
            logger.debug(f"Response headers: {response.headers}")
            logger.debug(f"Session cookies: {self.session.cookies.get_dict()}")
            
            if response.status_code == 200:
                logger.info("Successfully logged in")
            else:
                logger.error(f"Login failed with status {response.status_code}: {response.text}")
                sys.exit(1)
        except requests.exceptions.ConnectionError:
            logger.error(f"\nCould not connect to {self.server_url}. Please check:")
            logger.error("1. The server URL is correct")
            logger.error("2. The server is running")
            logger.error("3. The network connection is working")
            sys.exit(1)
        except requests.exceptions.Timeout:
            logger.error("Connection timed out. Is the server responding?")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            sys.exit(1)
        
    def send_readings(self, readings):
        """Send sensor readings and predictions to the server."""
        try:
            print("\n=== Sending Sensor Readings ===")
            print(f"Readings to send: {readings}")
            
            # Make prediction using the ML model
            prediction = self.sensors.process_sensor_data(readings)
            
            # Combine readings and prediction
            data = {
                **readings,
                'prediction': prediction['prediction'],
                'probability': prediction['probability'],
                'threshold': prediction['threshold'],
                'timestamp': prediction['timestamp']
            }
            
            response = self.session.post(
                f"{self.server_url}/api/sensor-readings",
                json=data,
                timeout=5,
                headers={
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                }
            )
            print(f"Response status: {response.status_code}")
            print(f"Response content: {response.text}")
            
            if response.status_code in [200, 201]:
                print("Successfully sent readings and prediction")
                self.retry_count = 0  # Reset retry count on success
            else:
                print(f"Failed to send readings (status {response.status_code})")
        except requests.exceptions.ConnectionError:
            self.retry_count += 1
            if self.retry_count >= self.max_retries:
                print("\nCould not connect to the server. Please check:")
                print("1. The server URL is correct")
                print("2. The server is running")
                print("3. The network connection is working")
                print("\nRetrying in 5 seconds...")
                time.sleep(5)
                self.retry_count = 0
            else:
                print("Connection error, retrying...")
        except requests.exceptions.Timeout:
            print("Connection timed out. Is the server responding?")
        except Exception as e:
            print(f"Error sending readings: {str(e)}")
    
    def start_monitoring(self, interval=1.0):
        """Start continuous monitoring and sending data to server."""
        logger.info("\nStarting sensor client...")
        logger.info("Make sure the Flask server is running")
        
        # First calibrate the GSR sensor
        logger.info("\nCalibrating GSR sensor...")
        self.sensors.calibrate_gsr()
        
        logger.info("\nStarting continuous monitoring...")
        logger.info("Press Ctrl+C to stop")
        try:
            self.sensors.continuous_monitoring(
                callback=self.send_readings,
                interval=interval
            )
        except KeyboardInterrupt:
            logger.info("\nMonitoring stopped by user")
        except Exception as e:
            logger.error(f"Error during monitoring: {str(e)}")

if __name__ == "__main__":
    # Update these values to match your server
    client = SensorClient(
        server_url="http://172.20.10.2:5001/",  # Using your computer's local IP address
        username="adamilola311@gmail.com",
        password="Adebayo2004."
    )
    
    try:
        client.start_monitoring(interval=1.0)  # Send readings every second
    except KeyboardInterrupt:
        logger.info("\nMonitoring stopped by user") 