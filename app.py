import logging
import requests
import click
from flask.cli import with_appcontext

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_migrate import Migrate
from models import (
    db, User, SensorReading, Prediction, EmergencyContact, 
    HealthcareProvider, Medication, MedicationSchedule, 
    MedicationRefill, Symptom, CrisisPrediction
)
from flask_socketio import SocketIO, emit
import threading
import time
from sqlalchemy.orm import joinedload
from collections import Counter # For most_common_symptom


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

# Global monitoring state
monitoring_state = {
    'is_monitoring': False,
    'sensor_client': None
}

# Pi configuration
PI_API_URL = os.environ.get('PI_API_URL', 'http://172.20.10.4:5002')
SENSOR_API_SECRET = os.environ.get('SENSOR_API_SECRET', 'cellguard_secret_2024')

@app.route('/api/sensor-readings', methods=['POST'])
def receive_sensor_readings():
    # Check for Authorization header (for Pi)
    auth_header = request.headers.get('Authorization', '')
    if auth_header == f'Bearer {SENSOR_API_SECRET}':
        data = request.json
        user_id = int(data.get('user_id', 1))  # Use user_id from payload or default
    elif hasattr(current_user, 'is_authenticated') and current_user.is_authenticated:
        data = request.json
        user_id = current_user.id
    else:
        return jsonify({'error': 'Unauthorized'}), 401

    if 'dehydration' not in data:
        data['dehydration'] = 0
    if 'age' not in data:
        data['age'] = 30
    if 'gender' not in data:
        data['gender'] = 0
    if 'spo2' not in data:
        data['spo2'] = 98.0
    if 'temperature' not in data:
        data['temperature'] = 36.7

    # Store sensor reading
    reading = SensorReading(
        user_id=user_id,
        gsr=data['gsr'],
        temperature=data['temperature'],
        spo2=data['spo2'],
        crisis_probability=data['probability']
    )
    db.session.add(reading)
    db.session.commit()

    # Store prediction
    prediction = CrisisPrediction(
        user_id=user_id,
        gsr=data['gsr'],
        temperature=data['temperature'],
        spo2=data['spo2'],
        crisis_predicted=bool(data['prediction']),
        crisis_probability=data['probability'],
        features=data,
        recommendations=get_recommendations({
            'prediction': data['prediction'],
            'probability': data['probability']
        }, data)
    )
    db.session.add(prediction)
    db.session.commit()

    # Emit prediction to connected clients
    socketio.emit('sensor_update', {
        'user_id': user_id,
        'prediction': data['prediction'],
        'probability': data['probability'],
        'timestamp': data['timestamp'],
        'gsr': data['gsr'],
        'temperature': data['temperature'],
        'spo2': data['spo2']
    })

    return jsonify({
        'status': 'success',
        'message': 'Readings and prediction stored successfully'
    })

@login_manager.user_loader
def load_user(user_id):
    print(f"[DEBUG] Loading user with ID: {user_id}")
    user = User.query.get(int(user_id))
    print(f"[DEBUG] Found user: {user}")
    return user

# Create database tables
with app.app_context():
    print("\n[INFO] Initializing database...")
    # Create tables if they don't exist
    db.create_all()
    print("[INFO] Database tables created/verified")
    
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
    age = data.get('age')
    gender = data.get('gender')
    
    # Check if user already exists
    existing_user = User.query.filter_by(email=email).first()
    if existing_user:
        return jsonify({
            'success': False,
            'message': 'Email already registered'
        }), 400
    
    # Create new user
    new_user = User(name=name, email=email, user_type=user_type, age=age, gender=gender)
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
    predictions = CrisisPrediction.query.filter_by(user_id=current_user.id)\
        .order_by(CrisisPrediction.timestamp.desc())\
        .all()
    print(f"Found {len(predictions)} predictions")
    return render_template('predictions.html', 
                         predictions=predictions,
                         is_monitoring=monitoring_state['is_monitoring'])  # Use global monitoring state

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
    try:
        medications = Medication.query.filter_by(user_id=current_user.id).all()
        return jsonify([{
            'id': med.id,
            'name': med.name,
            'dosage': med.dosage,
            'frequency': med.frequency,
            'start_date': med.start_date.isoformat() if med.start_date else None,
            'is_active': med.end_date is None # Assuming active if no end date
        } for med in medications])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/medications', methods=['POST'])
@login_required
def save_medication():
    try:
        data = request.get_json()
        med_start_date = datetime.utcnow().date() # Default to today
        # Potentially allow start_date from form in future if needed
        # if data.get('start_date'):
        #     med_start_date = datetime.strptime(data.get('start_date'), '%Y-%m-%d').date()

        days_supply_from_form = data.get('days_supply')
        if days_supply_from_form is None:
            # Fallback or error if not provided, for now let's make it optional and default if you wish
            # For this implementation, let's assume it might be provided, or we handle it being None later in logic.
            # Or, make it required from frontend.
            # For now, if None, current_supply_days might remain None in DB unless a default is set here.
            pass # Keep days_supply_from_form as None if not sent

        # Create new medication
        new_medication = Medication(
            user_id=current_user.id,
            name=data['name'],
            dosage=data['dosage'],
            frequency=data['frequency'],
            start_date=med_start_date,
            notes=data.get('notes', ''),
            current_supply_days=days_supply_from_form, # Set from form
            current_supply_start_date=med_start_date    # Set to medication start date initially
        )
        
        db.session.add(new_medication)
        db.session.flush()  # Flush to get the new_medication.id for the MedicationRefill record

        # Create an initial MedicationRefill record if days_supply was provided
        if days_supply_from_form is not None:
            initial_refill = MedicationRefill(
                medication_id=new_medication.id,
                refill_date=med_start_date,
                days_supply=days_supply_from_form,
                quantity=1 # Assuming initial batch is 1 unit/pack for now
                # next_refill_date could be calculated here: med_start_date + timedelta(days=days_supply_from_form)
            )
            db.session.add(initial_refill)
        
        # Create initial schedule for today (or med_start_date if different)
        today_date = datetime.utcnow().date()
        # If a specific start_date is provided from form and is in future, use that
        # For now, we'll assume the first schedule is for today or medication's start_date if available
        schedule_date_to_use = today_date 
        if new_medication.start_date and new_medication.start_date > today_date:
            schedule_date_to_use = new_medication.start_date
        elif new_medication.start_date:
             schedule_date_to_use = new_medication.start_date

        new_schedule = MedicationSchedule(
            medication_id=new_medication.id,
            scheduled_date=schedule_date_to_use, # Set the scheduled_date
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
    try:
        today = datetime.utcnow().date()
        # Fetch schedules for today for the current user
        # Join with Medication to get medication name and dosage
        schedules = db.session.query(
            MedicationSchedule, Medication.name, Medication.dosage
        ).join(Medication, Medication.id == MedicationSchedule.medication_id)\
        .filter(\
            Medication.user_id == current_user.id,\
            MedicationSchedule.scheduled_date == today\
        ).order_by(MedicationSchedule.time).all()

        schedule_list = []
        for schedule_entry, med_name, med_dosage in schedules:
            schedule_list.append({
                'medication_id': schedule_entry.medication_id,
                'medication_name': med_name,
                'dosage': med_dosage,
                'time': schedule_entry.time.strftime('%H:%M'), # Format time as HH:MM
                'is_taken': schedule_entry.is_taken,
                'schedule_id': schedule_entry.id 
            })
        return jsonify(schedule_list)
    except Exception as e:
        print(f"Error fetching medication schedule: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/medications/refills', methods=['GET'])
@login_required
def get_medication_refills():
    try:
        user_medications = Medication.query.filter_by(user_id=current_user.id).all()
        refill_info_list = []
        today = datetime.utcnow().date()

        for med in user_medications:
            days_remaining = None
            progress_percentage = 0

            if med.current_supply_start_date and med.current_supply_days is not None and med.current_supply_days > 0:
                supply_end_date = med.current_supply_start_date + timedelta(days=med.current_supply_days)
                days_remaining = (supply_end_date - today).days
                
                if days_remaining < 0:
                    days_remaining = 0 # Show 0 if past due, or could be negative to indicate overdue
                
                # Calculate progress: (days remaining / total supply days)
                # Ensure days_remaining is not greater than total supply days (can happen if clock issues or future start date)
                # and current_supply_days is not zero to avoid division by zero error.
                if med.current_supply_days > 0:
                    progress_percentage = (max(0, days_remaining) / med.current_supply_days) * 100
                else:
                    progress_percentage = 0 # Or 100 if no supply days means it's always full/never depletes?
                                            # For now, 0 if no supply_days defined.
                progress_percentage = min(100, max(0, progress_percentage)) # Cap between 0 and 100
            else:
                # If no supply info, default to 0 days remaining or handle as needed
                days_remaining = 0 

            refill_info_list.append({
                'medication_id': med.id,
                'medication_name': med.name,
                'dosage': med.dosage, # Added for display consistency
                'days_remaining': days_remaining,
                'progress': round(progress_percentage) # Send as whole number for progress bar
            })
        
        return jsonify(refill_info_list)

    except Exception as e:
        logger.error(f"Error fetching medication refills: {str(e)}", exc_info=True)
        return jsonify({'error': 'Failed to fetch refill data'}), 500

@app.route('/api/medications/history', methods=['GET'])
@login_required
def get_medication_history():
    try:
        # Get the time range from query parameters (default to 7 days)
        days = int(request.args.get('days', 7))
        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=days)

        # Corrected start_date to be inclusive for the period
        actual_start_date_for_period = end_date - timedelta(days=days - 1)

        # Get all active medications for the current user
        medications = Medication.query.filter(
            Medication.user_id == current_user.id,
            db.or_(Medication.end_date == None, Medication.end_date >= actual_start_date_for_period)
        ).all()
        medication_ids = [m.id for m in medications]
        
        # Generate labels for the chart (dates)
        # Iterate 'days' times, starting from actual_start_date_for_period
        labels_for_chart = [(actual_start_date_for_period + timedelta(days=i)).strftime('%a') for i in range(days)]
        query_dates = [(actual_start_date_for_period + timedelta(days=i)) for i in range(days)]

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
            schedules = MedicationSchedule.query.filter(
                MedicationSchedule.medication_id == medication.id,
                MedicationSchedule.scheduled_date >= actual_start_date_for_period,
                MedicationSchedule.scheduled_date <= end_date
            ).all()

            today_for_debug = datetime.utcnow().date()
            
            logger.info(f"-- Debug for Med ID {medication.id} ('{medication.name}') --")
            logger.info(f"  Target Date Range: {actual_start_date_for_period} to {end_date}")
            logger.info(f"  Query Dates (date objects) for Chart Logic: {query_dates}")
            logger.info(f"  Labels for Chart Display (strings): {labels_for_chart}")
            logger.info(f"  Schedules Found in DB ({len(schedules)}):")
            for s_debug in schedules:
                logger.info(f"    - Date: {s_debug.scheduled_date}, Time: {s_debug.time}, Taken: {s_debug.is_taken}")
            
            # taken_status_map will store the actual status (1 for taken, 0 for not taken) 
            # for dates where a schedule entry exists.
            taken_status_map = {sch.scheduled_date: (1 if sch.is_taken else 0) for sch in schedules}
            logger.info(f"  Taken Status Map Created (keys are date objects): {taken_status_map}")
            
            daily_data = []
            actually_scheduled_days_for_medication = 0 # New counter for this medication

            for q_idx, q_date in enumerate(query_dates):
                status = taken_status_map.get(q_date, 0) # Reverted: Default to 0 for chart visual
                daily_data.append(status)
                
                if q_date in taken_status_map: # Check if a schedule actually existed for this day
                    actually_scheduled_days_for_medication += 1

                if q_date == today_for_debug:
                    logger.info(f"    For today ({q_date}), status from map: {status}. Corresponding display label: {labels_for_chart[q_idx]}")
            
            logger.info(f"  Resulting daily_data for chart: {daily_data}")
            logger.info(f"  Actually scheduled days for this medication in view: {actually_scheduled_days_for_medication}") # Log new count

            datasets.append({
                'label': medication.name,
                'data': daily_data,
                'borderColor': colors[i % len(colors)],
                'backgroundColor': colors[i % len(colors)].replace('rgb', 'rgba').replace(')', ', 0.1)'),
                'tension': 0.4,
                'fill': True,
                'scheduled_days_count': actually_scheduled_days_for_medication # Add new field
            })

        return jsonify({
            'labels': labels_for_chart,
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
        ).first_or_404("Medication not found or access denied")

        # Get today's date
        today = datetime.utcnow().date()

        # Find the specific schedule entry for today
        schedule = MedicationSchedule.query.filter_by(
            medication_id=medication_id,
            scheduled_date=today
            # We might need to filter by time as well if multiple doses per day
            # time=specific_time_if_known 
        ).first()

        if not schedule:
            # If no specific schedule entry exists for today, maybe log an error
            # or decide if we should create one retroactively (depends on requirements)
            return jsonify({'error': 'No scheduled dose found for today'}), 404 
            
            # Alternatively, if you want to allow marking even if not explicitly scheduled:
            # schedule = MedicationSchedule(
            #     medication_id=medication_id,
            #     scheduled_date=today,
            #     time=datetime.utcnow().time(), # Use current time? Or fetch scheduled time?
            #     is_taken=True,
            #     taken_at=datetime.utcnow()
            # )
            # db.session.add(schedule)
        else:
            schedule.is_taken = True
            schedule.taken_at = datetime.utcnow() 

        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        print(f"Error marking medication as taken: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/medications/<int:medication_id>', methods=['DELETE'])
@login_required
def delete_medication(medication_id):
    try:
        logger.info(f"Attempting to delete medication ID: {medication_id} for user ID: {current_user.id}")
        # Fetch the medication ensuring it belongs to the current user
        medication = Medication.query.filter_by(id=medication_id, user_id=current_user.id).first()

        if not medication:
            logger.warning(f"Medication ID: {medication_id} not found for user ID: {current_user.id}")
            return jsonify({'error': 'Medication not found or you do not have permission to delete it'}), 404

        logger.info(f"Found medication: {medication.name}. Preparing to delete its associations.")

        # Explicitly delete associated schedules
        schedules_to_delete = MedicationSchedule.query.filter_by(medication_id=medication.id).all()
        for schedule in schedules_to_delete:
            logger.info(f"Deleting schedule ID: {schedule.id} for medication ID: {medication.id}")
            db.session.delete(schedule)

        # Explicitly delete associated refills
        refills_to_delete = MedicationRefill.query.filter_by(medication_id=medication.id).all()
        for refill in refills_to_delete:
            logger.info(f"Deleting refill ID: {refill.id} for medication ID: {medication.id}")
            db.session.delete(refill)
            
        logger.info(f"Deleting medication ID: {medication.id} itself ({medication.name}).")
        db.session.delete(medication)
        
        db.session.commit()
        logger.info(f"Successfully committed deletion of medication ID: {medication_id} and its associations.")
        
        return jsonify({'success': True, 'message': 'Medication deleted successfully'})

    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting medication ID: {medication_id}. Error: {str(e)}", exc_info=True)
        return jsonify({'error': 'An error occurred while deleting the medication'}), 500

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
            'symptoms': json.loads(s.symptoms) if s.symptoms else [], # Handle case where symptoms might be None/empty string before json.loads
            'notes': s.notes
        } for s in symptoms])
    except Exception as e:
        logger.error(f"Error getting symptoms: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 400

@app.route('/api/symptoms/stats', methods=['GET'])
@login_required
def get_symptom_stats():
    try:
        user_id = current_user.id
        thirty_days_ago = datetime.utcnow().date() - timedelta(days=30)

        recent_symptoms = Symptom.query.filter(
            Symptom.user_id == user_id,
            Symptom.date >= thirty_days_ago
        ).all()

        avg_pain_level = 0
        most_common_symptom_str = "N/A"
        symptom_free_days_count = 0

        if recent_symptoms:
            # Calculate average pain level
            total_pain = sum(s.pain_level for s in recent_symptoms if s.pain_level is not None)
            num_pain_entries = sum(1 for s in recent_symptoms if s.pain_level is not None)
            if num_pain_entries > 0:
                avg_pain_level = round(total_pain / num_pain_entries, 1)
            else:
                avg_pain_level = 0 # Or None, or "N/A" depending on preference

            # Calculate most common symptom
            all_symptoms_list = []
            for s_entry in recent_symptoms:
                try:
                    symptoms_list = json.loads(s_entry.symptoms)
                    if isinstance(symptoms_list, list):
                        all_symptoms_list.extend(symptoms_list)
                except (TypeError, json.JSONDecodeError):
                    # Handle cases where s_entry.symptoms is None or not valid JSON
                    pass 
            
            if all_symptoms_list:
                symptom_counts = Counter(all_symptoms_list)
                if symptom_counts:
                    # Get all symptoms with the max count
                    max_count = 0
                    top_symptoms = []
                    for symptom, count in symptom_counts.items():
                        if count > max_count:
                            max_count = count
                            top_symptoms = [symptom]
                        elif count == max_count:
                            top_symptoms.append(symptom)
                    most_common_symptom_str = ", ".join(top_symptoms) if top_symptoms else "N/A"

            # Calculate symptom-free days (days with pain_level 0)
            symptom_free_dates = set()
            for s_entry in recent_symptoms:
                if s_entry.pain_level == 0:
                    symptom_free_dates.add(s_entry.date)
            symptom_free_days_count = len(symptom_free_dates)
        else:
            avg_pain_level = 0 # Default if no recent symptoms

        return jsonify({
            'average_pain_level': avg_pain_level,
            'most_common_symptom': most_common_symptom_str,
            'symptom_free_days': symptom_free_days_count
        })

    except Exception as e:
        logger.error(f"Error calculating symptom stats: {str(e)}", exc_info=True)
        return jsonify({'error': 'Failed to calculate symptom stats'}), 500

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
    try:
        # Fetch age and gender for the current user
        age = getattr(current_user, 'age', 30) or 30
        gender = getattr(current_user, 'gender', 'male') or 'male'
        headers = {'Authorization': f'Bearer {SENSOR_API_SECRET}', 'Content-Type': 'application/json'}
        payload = {'age': age, 'gender': gender}
        resp = requests.post(f"{PI_API_URL}/start-monitoring", headers=headers, json=payload, timeout=5)
        if resp.status_code == 200:
            return jsonify({'status': 'success', 'message': 'Monitoring started on Pi'})
        else:
            return jsonify({'status': 'error', 'message': resp.json().get('error', 'Failed to start monitoring on Pi')}), 500
    except Exception as e:
        logger.error(f"Error starting monitoring: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/stop-monitoring', methods=['POST'])
@login_required
def stop_monitoring():
    try:
        headers = {'Authorization': f'Bearer {SENSOR_API_SECRET}'}
        resp = requests.post(f"{PI_API_URL}/stop-monitoring", headers=headers, timeout=5)
        if resp.status_code == 200:
            return jsonify({'status': 'success', 'message': 'Monitoring stopped on Pi'})
        else:
            return jsonify({'status': 'error', 'message': resp.json().get('error', 'Failed to stop monitoring on Pi')}), 500
    except Exception as e:
        logger.error(f"Error stopping monitoring: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/monitoring-status', methods=['GET'])
@login_required
def get_monitoring_status():
    return jsonify({
        'is_monitoring': monitoring_state['is_monitoring']
    })

# CLI command to create medication schedules
@app.cli.command("create-schedules")
@with_appcontext
@click.option('--date', default='today', help='Target date for schedule creation (YYYY-MM-DD or "today").')
def create_medication_schedules(date):
    """Creates medication schedule entries for a given date for all active medications."""
    target_date_str = date
    if date == 'today':
        target_date = datetime.utcnow().date()
    else:
        try:
            target_date = datetime.strptime(date, '%Y-%m-%d').date()
        except ValueError:
            print(f"Error: Invalid date format for {date}. Please use YYYY-MM-DD or 'today'.")
            return

    print(f"Creating medication schedules for: {target_date.isoformat()}")

    users = User.query.all()
    schedules_created_count = 0

    for user in users:
        active_medications = Medication.query.filter(
            Medication.user_id == user.id,
            Medication.start_date <= target_date,
            db.or_(Medication.end_date == None, Medication.end_date >= target_date)
        ).all()

        for med in active_medications:
            # Determine scheduled times based on frequency
            # This is a simplified example; you'll need more robust parsing for frequency
            scheduled_times = []
            if med.frequency.lower() == 'once daily':
                # Try to get the time from an existing schedule or default to a common time like 8 AM
                existing_schedule_time = MedicationSchedule.query.filter_by(medication_id=med.id).first()
                if existing_schedule_time and existing_schedule_time.time:
                    scheduled_times.append(existing_schedule_time.time)
                else:
                    scheduled_times.append(datetime.strptime('08:00', '%H:%M').time()) 
            elif med.frequency.lower() == 'twice daily':
                scheduled_times.append(datetime.strptime('08:00', '%H:%M').time())
                scheduled_times.append(datetime.strptime('20:00', '%H:%M').time())
            # Add more frequency logic here (e.g., "Three times daily", "As needed" might be skipped or handled differently)
            else:
                print(f"Warning: Unsupported frequency '{med.frequency}' for medication '{med.name}' (ID: {med.id}). Skipping.")
                continue

            for sched_time in scheduled_times:
                exists = MedicationSchedule.query.filter_by(
                    medication_id=med.id,
                    scheduled_date=target_date,
                    time=sched_time
                ).first()

                if not exists:
                    new_schedule = MedicationSchedule(
                        medication_id=med.id,
                        scheduled_date=target_date,
                        time=sched_time,
                        is_taken=False
                    )
                    db.session.add(new_schedule)
                    schedules_created_count += 1
                    print(f"  Created schedule for {med.name} at {sched_time.strftime('%H:%M')} for user {user.id}")
                else:
                    print(f"  Schedule already exists for {med.name} at {sched_time.strftime('%H:%M')} for user {user.id}")
    
    if schedules_created_count > 0:
        db.session.commit()
        print(f"Successfully created {schedules_created_count} new schedule(s).")
    else:
        print("No new schedules needed to be created.")

# Make sure templates directory exists
os.makedirs('templates', exist_ok=True)

if __name__ == '__main__':
    print("\n[INFO] Starting Flask server with Socket.IO...")
    print("[INFO] Server will be available at http://localhost:5001")
    socketio.run(app, host='0.0.0.0', port=5001, debug=True)
