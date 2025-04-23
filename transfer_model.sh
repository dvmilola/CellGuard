#!/bin/bash

# Raspberry Pi connection details
PI_USER="pi"  # Change this to your Pi's username
PI_HOST="raspberrypi.local"  # Change this to your Pi's hostname or IP
PI_PATH="/home/pi/crisis_prediction/models"  # Change this to your desired path on the Pi

# Create models directory on Pi if it doesn't exist
ssh $PI_USER@$PI_HOST "mkdir -p $PI_PATH"

# Transfer the model files
scp models/crisis_model_20250423_011731.joblib $PI_USER@$PI_HOST:$PI_PATH/
scp models/scaler_20250423_011731.joblib $PI_USER@$PI_HOST:$PI_PATH/
scp models/threshold_20250423_011731.txt $PI_USER@$PI_HOST:$PI_PATH/

echo "Model files transferred successfully!" 