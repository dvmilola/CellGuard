import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import joblib
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from ml_model import predict_crisis, load_model  # Import load_model function
from flask_cors import CORS
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_migrate import Migrate
from models import (
    db, User, SensorReading, Prediction, EmergencyContact, 
    HealthcareProvider, Medication, MedicationSchedule, 
    MedicationRefill, Symptom, CrisisPrediction
)
from flask_socketio import SocketIO, emit
from sensor_interface import SensorInterface
import threading
import time


# Configure the application
app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app, supports_credentials=True, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes with credentials

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///crisis_predictions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this to a secure secret key in production
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # Change from 'None' to 'Lax'
app.config['SESSION_COOKIE_SECURE'] = False  # Change to False for local development
app.config['REMEMBER_COOKIE_SAMESITE'] = 'Lax'  # Change from 'None' to 'Lax'
app.config['REMEMBER_COOKIE_SECURE'] = False  # Change to False for local development
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour
db.init_app(app)
migrate = Migrate(app, db)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login_page'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'
login_manager.session_protection = 'basic'  # Change from 'strong' to 'basic'

# Initialize Flask-SocketIO
socketio = SocketIO(app, 
    cors_allowed_origins="*",
    async_mode='threading',
    logger=True,
    engineio_logger=True
)

# Add global variables for sensor control
sensor_interface = None
monitoring_thread = None
is_monitoring = False

# Global variable to track monitoring state
is_monitoring = False
monitoring_thread = None

# Initialize the model
model = load_model()

def start_sensor_monitoring(user_id):
    """Background thread for continuous sensor monitoring"""
    global is_monitoring
    
    try:
        # Initialize sensor interface
        sensor_interface = SensorInterface()
        
        # Calibrate GSR sensor
        baseline = sensor_interface.calibrate_gsr()
        logger.info(f"GSR baseline: {baseline:.2f} µS")
        
        def process_readings(readings):
            if not is_monitoring:
                return
                
            try:
                with app.app_context():
                    # Create sensor reading
                    reading = SensorReading(
                        user_id=user_id,
                        gsr=readings['gsr'],
                        temperature=readings['temperature'],
                        spo2=readings['spo2'],
                        timestamp=datetime.fromisoformat(readings['timestamp'])
                    )
                    db.session.add(reading)
                    
                    # Make prediction
                    features = np.array([[readings['gsr'], readings['temperature'], readings['spo2']]])
                    prediction = model.predict(features)[0]
                    probability = model.predict_proba(features)[0][1]
                    
                    # Create prediction record
                    prediction_record = Prediction(
                        user_id=user_id,
                        reading_id=reading.id,
                        prediction=prediction,
                        probability=probability,
                        timestamp=datetime.now()
                    )
                    db.session.add(prediction_record)
                    db.session.commit()
                    
                    # Emit real-time update
                    socketio.emit('sensor_update', {
                        'gsr': readings['gsr'],
                        'temperature': readings['temperature'],
                        'spo2': readings['spo2'],
                        'prediction': prediction,
                        'probability': probability,
                        'timestamp': readings['timestamp']
                    })
                    
            except Exception as e:
                logger.error(f"Error processing readings: {str(e)}")
        
        # Start continuous monitoring
        sensor_interface.continuous_monitoring(process_readings)
        
    except Exception as e:
        logger.error(f"Error in monitoring thread: {str(e)}")
        is_monitoring = False

@login_manager.user_loader
def load_user(user_id):
    print(f"[DEBUG] Loading user with ID: {user_id}")
    user = User.query.get(int(user_id))
    print(f"[DEBUG] Found user: {user}")
    return user

# Create database tables
with app.app_context():
    print("\n[INFO] Initializing database...")
    # Drop all tables and recreate them
    db.drop_all()
    db.create_all()
    print("[INFO] Database tables recreated")
    
    # Print database info for verification
    print("\nDatabase initialized at:", app.config['SQLALCHEMY_DATABASE_URI'])
    print("Tables created:", db.metadata.tables.keys())
    
    # Create a test user if none exists
    test_user = User.query.filter_by(email="adamilola311@gmail.com").first()
    if not test_user:
        print("[INFO] Creating test user...")
        test_user = User(
            name="Test User",
            email="adamilola311@gmail.com",
            user_type="patient"
        )
        test_user.set_password("Adebayo2004.")
        db.session.add(test_user)
        db.session.commit()
        print("[INFO] Test user created")
    else:
        print("[INFO] Test user already exists")
    
    # Count existing records
    prediction_count = Prediction.query.count()
    print(f"Existing prediction records: {prediction_count}\n")

# Add debug logging for all requests
@app.before_request
def log_request_info():
    print(f"\n[DEBUG] Request: {request.method} {request.url}")
    print(f"[DEBUG] Headers: {dict(request.headers)}")
    print(f"[DEBUG] Data: {request.get_data()}")
    print(f"[DEBUG] JSON: {request.get_json(silent=True)}")

@app.after_request
def log_response_info(response):
    print(f"[DEBUG] Response: {response.status_code}")
    print(f"[DEBUG] Response Headers: {dict(response.headers)}")
    return response

# No longer need to store predictions in memory as we're using a database

# Authentication routes
@app.route('/login', methods=['GET'])
def login_page():
    return render_template('login.html')

@app.route('/signup', methods=['GET'])
def signup_page():
    return render_template('signup.html')

@app.route('/api/login', methods=['POST'])
def login():
    try:
        print("\n[DEBUG] Login attempt received")
        data = request.get_json()
        print(f"[DEBUG] Login data: {data}")
        
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            print("[DEBUG] Missing email or password")
            return jsonify({'error': 'Email and password are required'}), 400
            
        user = User.query.filter_by(email=email).first()
        if not user or not user.check_password(password):
            print("[DEBUG] Invalid credentials")
            return jsonify({'error': 'Invalid email or password'}), 401
            
        login_user(user, remember=True)
        print(f"[DEBUG] User {user.email} logged in successfully")
        
        return jsonify({
            'message': 'Login successful',
            'user': {
                'id': user.id,
                'name': user.name,
                'email': user.email,
                'user_type': user.user_type
            }
        })
        
    except Exception as e:
        print(f"[ERROR] Login error: {str(e)}")
        return jsonify({'error': 'An error occurred during login'}), 500

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/api/signup', methods=['POST'])
def api_signup():
    data = request.get_json()
    name = data.get('fullName')
    email = data.get('email')
    password = data.get('password')
    user_type = data.get('userType')
    
    # Check if user already exists
    existing_user = User.query.filter_by(email=email).first()
    if existing_user:
        return jsonify({
            'success': False,
            'message': 'Email already registered'
        }), 400
    
    # Create new user
    new_user = User(name=name, email=email, user_type=user_type)
    new_user.set_password(password)
    
    try:
        db.session.add(new_user)
        db.session.commit()
        return jsonify({
            'success': True,
            'message': 'Account created successfully!'
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'message': 'Error creating account. Please try again.'
        }), 500

# Main routes
@app.route('/')
def home():
    try:
        print("Attempting to render index.html")
        return render_template('index.html')
    except Exception as e:
        print(f"Error rendering index.html: {str(e)}")
        print(f"Template path: {app.template_folder}")
        return "Error loading page", 500

@app.route('/predictions')
@login_required
def predictions():
    print("\n=== Loading Predictions ===")
    predictions = Prediction.query.filter_by(user_id=current_user.id)\
        .order_by(Prediction.timestamp.desc())\
        .all()
    
    print(f"Found {len(predictions)} predictions")
    return render_template('predictions.html', 
                         predictions=predictions,
                         is_monitoring=is_monitoring)

@app.route('/emergency')
@login_required
def emergency():
    contacts = EmergencyContact.query.filter_by(user_id=current_user.id).all()
    return render_template('emergency.html', contacts=contacts)

@app.route('/profile')
@login_required
def profile():
    user = current_user
    providers = HealthcareProvider.query.filter_by(user_id=current_user.id).all()
    return render_template('profile.html', user=user, providers=providers)

@app.route('/provider')
@login_required
def provider():
    if current_user.user_type != 'healthcare-provider':
        flash('Access denied. This page is for healthcare providers only.', 'error')
        return redirect(url_for('home'))
    
    # Get recent predictions for display
    recent_predictions = Prediction.query.order_by(Prediction.timestamp.desc()).limit(10).all()
    return render_template('provider.html', predictions=recent_predictions)

@app.route('/settings')
@login_required
def settings():
    user = current_user
    return render_template('settings.html', user=user)

@app.route('/medication')
@login_required
def medications():
    try:
        print("Attempting to render medications.html")
        # In a real implementation, this would fetch the user's medications from the database
        # For now, we'll use sample data
        sample_medications = [
            {
                'name': 'Hydroxyurea',
                'dosage': '500mg',
                'frequency': 'Once daily',
                'start_date': '2024-01-15',
                'is_active': True
            },
            {
                'name': 'Folic Acid',
                'dosage': '1mg',
                'frequency': 'Once daily',
                'start_date': '2024-01-15',
                'is_active': True
            }
        ]
        
        return render_template('medications.html', medications=sample_medications)
    except Exception as e:
        print(f"Error rendering medications.html: {str(e)}")
        print(f"Template path: {app.template_folder}")
        return "Error loading page", 500

@app.route('/api/medications', methods=['GET'])
@login_required
def get_medications():
    # In a real implementation, this would fetch medications from the database
    # For now, we'll return sample data
    return jsonify([
        {
            'name': 'Hydroxyurea',
            'dosage': '500mg',
            'frequency': 'Once daily',
            'start_date': '2024-01-15',
            'is_active': True
        },
        {
            'name': 'Folic Acid',
            'dosage': '1mg',
            'frequency': 'Once daily',
            'start_date': '2024-01-15',
            'is_active': True
        }
    ])

@app.route('/api/medications', methods=['POST'])
@login_required
def save_medication():
    try:
        data = request.get_json()
        
        # Create new medication
        new_medication = Medication(
            user_id=current_user.id,
            name=data['name'],
            dosage=data['dosage'],
            frequency=data['frequency'],
            start_date=datetime.utcnow().date(),
            notes=data.get('notes', '')
        )
        
        db.session.add(new_medication)
        db.session.commit()
        
        # Create initial schedule
        new_schedule = MedicationSchedule(
            medication_id=new_medication.id,
            time=datetime.strptime(data['time'], '%H:%M').time(),
            is_taken=False
        )
        
        db.session.add(new_schedule)
        db.session.commit()
        
        return jsonify({
            'id': new_medication.id,
            'name': new_medication.name,
            'dosage': new_medication.dosage,
            'frequency': new_medication.frequency,
            'start_date': new_medication.start_date.isoformat(),
            'notes': new_medication.notes
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400

@app.route('/api/medications/schedule', methods=['GET'])
@login_required
def get_medication_schedule():
    # In a real implementation, this would fetch the schedule from the database
    # For now, we'll return sample data
    return jsonify([
        {
            'medication': 'Hydroxyurea',
            'dosage': '500mg',
            'time': '08:00',
            'is_taken': False
        },
        {
            'medication': 'Folic Acid',
            'dosage': '1mg',
            'time': '08:00',
            'is_taken': False
        }
    ])

@app.route('/api/medications/refills', methods=['GET'])
def get_medication_refills():
    # In a real implementation, this would fetch refill data from the database
    # For now, we'll return sample data
    return jsonify([
        {
            'medication': 'Hydroxyurea',
            'dosage': '500mg',
            'days_remaining': 15,
            'progress': 60
        },
        {
            'medication': 'Folic Acid',
            'dosage': '1mg',
            'days_remaining': 20,
            'progress': 80
        }
    ])

@app.route('/api/medications/history', methods=['GET'])
@login_required
def get_medication_history():
    try:
        # Get the time range from query parameters (default to 7 days)
        days = int(request.args.get('days', 7))
        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=days)

        # Get all medications for the current user
        medications = Medication.query.filter_by(user_id=current_user.id).all()
        
        # Generate labels for the chart (dates)
        labels = []
        for i in range(days):
            date = start_date + timedelta(days=i)
            labels.append(date.strftime('%a'))

        # Prepare datasets for each medication
        datasets = []
        colors = [
            'rgb(255, 99, 132)',
            'rgb(54, 162, 235)',
            'rgb(255, 206, 86)',
            'rgb(75, 192, 192)',
            'rgb(153, 102, 255)'
        ]

        for i, medication in enumerate(medications):
            # Get schedule entries for this medication within the date range
            schedules = MedicationSchedule.query.filter(
                MedicationSchedule.medication_id == medication.id,
                MedicationSchedule.taken_at >= start_date,
                MedicationSchedule.taken_at <= end_date
            ).all()

            # Create a map of dates to taken status
            date_status = {date: 0 for date in labels}
            for schedule in schedules:
                date = schedule.taken_at.strftime('%a')
                date_status[date] = 1 if schedule.is_taken else 0

            # Add dataset for this medication
            datasets.append({
                'label': medication.name,
                'data': [date_status[date] for date in labels],
                'borderColor': colors[i % len(colors)],
                'backgroundColor': colors[i % len(colors)].replace('rgb', 'rgba').replace(')', ', 0.1)'),
                'tension': 0.4,
                'fill': True
            })

        return jsonify({
            'labels': labels,
            'datasets': datasets
        })
    except Exception as e:
        print(f"Error getting medication history: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/medications/schedule/<int:medication_id>', methods=['POST'])
@login_required
def mark_medication_taken(medication_id):
    try:
        # Get the medication to verify ownership
        medication = Medication.query.filter_by(
            id=medication_id,
            user_id=current_user.id
        ).first_or_404()

        # Get today's date
        today = datetime.utcnow().date()

        # Find or create a schedule entry for today
        schedule = MedicationSchedule.query.filter_by(
            medication_id=medication_id,
            scheduled_date=today
        ).first()

        if not schedule:
            schedule = MedicationSchedule(
                medication_id=medication_id,
                scheduled_date=today,
                is_taken=True
            )
            db.session.add(schedule)
        else:
            schedule.is_taken = True

        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        print(f"Error marking medication as taken: {str(e)}")
        return jsonify({'error': str(e)}), 400

# API routes
@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    try:
        # Get data from request
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['spo2', 'temperature', 'age', 'gender', 'dehydration']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Process sensor data
        sensor_interface = SensorInterface()
        result = sensor_interface.process_sensor_data(data)
        
        # Store prediction in database
        prediction = CrisisPrediction(
            spo2=data['spo2'],
            temperature=data['temperature'],
            age=data['age'],
            gender=data['gender'],
            dehydration=data['dehydration'],
            prediction=result['prediction'],
            probability=result['probability'],
            threshold=result['threshold'],
            timestamp=result['timestamp']
        )
        db.session.add(prediction)
        db.session.commit()
        
        # Prepare response
        response = {
            'prediction': result['prediction'],
            'probability': result['probability'],
            'threshold': result['threshold'],
            'timestamp': result['timestamp'],
            'risk_level': 'High' if result['prediction'] == 1 else 'Low',
            'confidence': 'High' if abs(result['probability'] - result['threshold']) > 0.2 else 'Medium',
            'recommendations': get_recommendations(result, data)
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_recommendations(result, data):
    """Generate recommendations based on prediction and vital signs"""
    recommendations = []
    
    # Check SpO2
    if data['spo2'] < 95:
        recommendations.append("Monitor oxygen levels closely")
    if data['spo2'] < 90:
        recommendations.append("Consider supplemental oxygen")
    
    # Check temperature
    if data['temperature'] > 37.5:
        recommendations.append("Monitor temperature regularly")
    if data['temperature'] > 38.5:
        recommendations.append("Consider antipyretics")
    
    # Check dehydration
    if data['dehydration'] > 0:
        recommendations.append("Encourage fluid intake")
    if data['dehydration'] >= 2:
        recommendations.append("Consider IV fluids")
    
    # Age-specific recommendations
    if data['age'] < 12 or data['age'] > 65:
        recommendations.append("Monitor more frequently due to age risk")
    
    # Add prediction-specific recommendations
    if result['prediction'] == 1:
        recommendations.append("High risk detected - increase monitoring frequency")
        recommendations.append("Prepare emergency response plan")
    
    return recommendations

@app.route('/api/recent', methods=['GET'])
def get_recent():
    """Return recent predictions from the database"""
    # Get the 100 most recent predictions
    predictions = Prediction.query.order_by(Prediction.timestamp.desc()).limit(100).all()
    return jsonify([prediction.to_dict() for prediction in predictions])

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Return statistics about the predictions"""
    total_count = Prediction.query.count()
    crisis_count = Prediction.query.filter_by(crisis_predicted=True).count()
    
    if total_count > 0:
        crisis_percentage = (crisis_count / total_count) * 100
    else:
        crisis_percentage = 0
    
    # Get average readings
    result = db.session.query(
        db.func.avg(Prediction.gsr).label('avg_gsr'),
        db.func.avg(Prediction.temperature).label('avg_temperature'),
        db.func.avg(Prediction.spo2).label('avg_spo2'),
        db.func.avg(Prediction.crisis_probability).label('avg_probability')
    ).first()
    
    return jsonify({
        'total_predictions': total_count,
        'crisis_predictions': crisis_count,
        'crisis_percentage': crisis_percentage,
        'average_readings': {
            'gsr': float(result.avg_gsr) if result.avg_gsr else 0,
            'temperature': float(result.avg_temperature) if result.avg_temperature else 0,
            'spo2': float(result.avg_spo2) if result.avg_spo2 else 0,
            'crisis_probability': float(result.avg_probability) if result.avg_probability else 0
        }
    })

# Emergency contact API routes
@app.route('/api/contacts', methods=['GET'])
@login_required
def get_contacts():
    contacts = EmergencyContact.query.filter_by(user_id=current_user.id).all()
    return jsonify([{
        'id': contact.id,
        'name': contact.name,
        'type': contact.contact_type,
        'phone': contact.phone,
        'email': contact.email,
        'notes': contact.notes
    } for contact in contacts])

@app.route('/api/contacts', methods=['POST'])
@login_required
def add_contact():
    data = request.get_json()
    
    new_contact = EmergencyContact(
        user_id=current_user.id,
        name=data.get('name'),
        contact_type=data.get('type'),
        phone=data.get('phone'),
        email=data.get('email'),
        notes=data.get('notes')
    )
    
    db.session.add(new_contact)
    db.session.commit()
    
    return jsonify({
        'id': new_contact.id,
        'name': new_contact.name,
        'type': new_contact.contact_type,
        'phone': new_contact.phone,
        'email': new_contact.email,
        'notes': new_contact.notes
    })

@app.route('/symptom-tracker')
@login_required
def symptom_tracker():
    try:
        print("Attempting to render symptom-tracker.html")
        return render_template('symptom-tracker.html')
    except Exception as e:
        print(f"Error rendering symptom-tracker.html: {str(e)}")
        print(f"Template path: {app.template_folder}")
        return "Error loading page", 500

@app.route('/api/symptoms', methods=['POST'])
@login_required
def save_symptoms():
    data = request.json
    try:
        new_symptom = Symptom(
            user_id=current_user.id,
            date=datetime.strptime(data['date'], '%Y-%m-%d').date(),
            pain_level=data['pain_level'],
            symptoms=json.dumps(data['symptoms']),
            notes=data.get('notes', '')
        )
        db.session.add(new_symptom)
        db.session.commit()
        return jsonify({'message': 'Symptoms saved successfully'}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/symptoms', methods=['GET'])
@login_required
def get_symptoms():
    try:
        symptoms = Symptom.query.filter_by(user_id=current_user.id).order_by(Symptom.date.desc()).all()
        return jsonify([{
            'id': s.id,
            'date': s.date.isoformat(),
            'pain_level': s.pain_level,
            'symptoms': json.loads(s.symptoms),
            'notes': s.notes
        } for s in symptoms])
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/knowledge-library')
@login_required
def knowledge_library():
    try:
        print("Attempting to render knowledge-library.html")
        return render_template('knowledge-library.html')
    except Exception as e:
        print(f"Error rendering knowledge-library.html: {str(e)}")
        print(f"Template path: {app.template_folder}")
        return "Error loading page", 500

@app.route('/api/sensor-readings', methods=['POST'])
@login_required
def receive_sensor_readings():
    try:
        data = request.json
        user_id = current_user.id
        
        # Store sensor reading
        reading = SensorReading(
            user_id=user_id,
            gsr=data['gsr'],
            temperature=data['temperature'],
            spo2=data['spo2'],
            crisis_probability=0.0  # Will be updated after prediction
        )
        db.session.add(reading)
        db.session.commit()
        
        # Make prediction using user-specific model if available
        result = predict_crisis(
            data['gsr'],
            data['temperature'],
            data['spo2'],
            user_id=user_id
        )
        
        # Update reading with prediction
        reading.crisis_probability = result['probability']
        db.session.commit()
        
        # Store prediction
        prediction = CrisisPrediction(
            user_id=user_id,
            gsr=data['gsr'],
            temperature=data['temperature'],
            spo2=data['spo2'],
            crisis_predicted=bool(result['prediction']),
            crisis_probability=result['probability'],
            features=data,
            recommendations=get_recommendations(result, data)
        )
        db.session.add(prediction)
        db.session.commit()
        
        # Check if we should retrain user model
        if should_retrain_user_model(user_id):
            retrain_user_model(user_id)
        
        # Emit prediction to connected clients
        socketio.emit('prediction', {
            'user_id': user_id,
            'prediction': result['prediction'],
            'probability': result['probability'],
            'model_type': result['model_type'],
            'timestamp': result['timestamp']
        })
        
        return jsonify({
            'status': 'success',
            'prediction': result
        })
        
    except Exception as e:
        logger.error(f"Error processing sensor readings: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

def should_retrain_user_model(user_id):
    """Check if we should retrain the user's model"""
    try:
        # Get count of new readings since last training
        last_training = db.session.query(CrisisPrediction).filter_by(
            user_id=user_id
        ).order_by(CrisisPrediction.timestamp.desc()).first()
        
        if not last_training:
            return False
            
        new_readings = db.session.query(SensorReading).filter(
            SensorReading.user_id == user_id,
            SensorReading.timestamp > last_training.timestamp
        ).count()
        
        # Retrain if we have at least 50 new readings
        return new_readings >= 50
        
    except Exception as e:
        logger.error(f"Error checking retrain condition: {str(e)}")
        return False

def retrain_user_model(user_id):
    """Retrain the user's model with their historical data"""
    try:
        # Get user's historical data
        readings = db.session.query(SensorReading).filter_by(
            user_id=user_id
        ).all()
        
        symptoms = db.session.query(Symptom).filter_by(
            user_id=user_id
        ).all()
        
        if not readings or not symptoms:
            logger.warning(f"Not enough data to retrain model for user {user_id}")
            return False
            
        # Convert to lists of dictionaries
        readings_data = [r.to_dict() for r in readings]
        symptoms_data = [s.to_dict() for s in symptoms]
        
        # Train new model
        success = prediction_model.train_user_model(user_id, readings_data, symptoms_data)
        
        if success:
            logger.info(f"Successfully retrained model for user {user_id}")
            return True
        else:
            logger.error(f"Failed to retrain model for user {user_id}")
            return False
            
    except Exception as e:
        logger.error(f"Error retraining user model: {str(e)}")
        return False

@app.route('/api/sensor-readings', methods=['GET'])
@login_required
def get_sensor_readings():
    try:
        time_range = request.args.get('time_range', '1')  # Default to last 24 hours
        hours = int(time_range) * 24
        
        # Get readings within the specified time range
        readings = SensorReading.query.filter(
            SensorReading.user_id == current_user.id,
            SensorReading.timestamp >= datetime.utcnow() - timedelta(hours=hours)
        ).order_by(SensorReading.timestamp.asc()).all()
        
        return jsonify([reading.to_dict() for reading in readings])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    print('\n[DEBUG] Client connected to Socket.IO')
    print(f'[DEBUG] Client IP: {request.remote_addr}')
    emit('connected', {'data': 'Connected to Socket.IO server'})

@socketio.on('disconnect')
def handle_disconnect():
    print('\n[DEBUG] Client disconnected from Socket.IO')
    print(f'[DEBUG] Client IP: {request.remote_addr}')

@socketio.on('message')
def handle_message(message):
    print('\n[DEBUG] Received WebSocket message:', message)
    try:
        if isinstance(message, dict) and message.get('type') == 'connection_test':
            print('[DEBUG] Received connection test message')
            emit('message', {'status': 'connection_verified'})
    except Exception as e:
        print(f'[ERROR] Error handling WebSocket message: {str(e)}')

@app.route('/api/start-monitoring', methods=['POST'])
@login_required
def start_monitoring():
    global is_monitoring, monitoring_thread
    
    try:
        if not is_monitoring:
            is_monitoring = True
            monitoring_thread = threading.Thread(
                target=start_sensor_monitoring,
                args=(current_user.id,)
            )
            monitoring_thread.daemon = True
            monitoring_thread.start()
            logger.info("Started sensor monitoring")
            return jsonify({'status': 'success', 'message': 'Monitoring started'})
        else:
            return jsonify({'status': 'info', 'message': 'Monitoring already running'})
    except Exception as e:
        logger.error(f"Error starting monitoring: {str(e)}")
        is_monitoring = False
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/stop-monitoring', methods=['POST'])
@login_required
def stop_monitoring():
    global is_monitoring, monitoring_thread
    
    if is_monitoring:
        is_monitoring = False
        if monitoring_thread:
            monitoring_thread.join(timeout=1.0)
        logger.info("Stopped sensor monitoring")
        return jsonify({'status': 'success', 'message': 'Monitoring stopped'})
    else:
        return jsonify({'status': 'info', 'message': 'Monitoring not running'})

# Make sure templates directory exists
os.makedirs('templates', exist_ok=True)

if __name__ == '__main__':
    print("\n[INFO] Starting Flask server with Socket.IO...")
    print("[INFO] Server will be available at http://localhost:5001")
    socketio.run(app, host='0.0.0.0', port=5001, debug=True)
