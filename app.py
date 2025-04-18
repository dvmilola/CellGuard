from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
import json
import pandas as pd
import numpy as np
from datetime import datetime
import os
import joblib
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from crisis_prediction_model import predict_crisis  # Only import the prediction function
from flask_cors import CORS
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user


# Configure the application
app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)  # Enable CORS for all routes

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///crisis_predictions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this to a secure secret key in production
db = SQLAlchemy(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login_page'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Define the database models
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    gsr = db.Column(db.Float, nullable=False)
    temperature = db.Column(db.Float, nullable=False)
    spo2 = db.Column(db.Float, nullable=False)
    crisis_predicted = db.Column(db.Boolean, nullable=False)
    crisis_probability = db.Column(db.Float, nullable=False)
    
    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'gsr': self.gsr,
            'temperature': self.temperature,
            'spo2': self.spo2,
            'crisis_predicted': self.crisis_predicted,
            'crisis_probability': self.crisis_probability
        }

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    user_type = db.Column(db.String(20), nullable=False)  # patient, caregiver, healthcare-provider
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class EmergencyContact(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    contact_type = db.Column(db.String(50), nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(120))
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class HealthcareProvider(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    specialty = db.Column(db.String(100), nullable=False)
    hospital = db.Column(db.String(200))
    phone = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Medication(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    dosage = db.Column(db.String(50), nullable=False)
    frequency = db.Column(db.String(50), nullable=False)
    start_date = db.Column(db.Date, nullable=False)
    end_date = db.Column(db.Date)
    is_active = db.Column(db.Boolean, default=True)
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class MedicationSchedule(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    medication_id = db.Column(db.Integer, db.ForeignKey('medication.id'), nullable=False)
    time = db.Column(db.Time, nullable=False)
    is_taken = db.Column(db.Boolean, default=False)
    taken_at = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class MedicationRefill(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    medication_id = db.Column(db.Integer, db.ForeignKey('medication.id'), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    days_supply = db.Column(db.Integer, nullable=False)
    refill_date = db.Column(db.Date, nullable=False)
    next_refill_date = db.Column(db.Date, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Symptom(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.Date, nullable=False)
    pain_level = db.Column(db.Integer, nullable=False)
    symptoms = db.Column(db.String(500), nullable=False)  # JSON string of symptoms
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Import the prediction function from your model script
# In a real implementation, you would import from your model file
# from crisis_prediction_model import predict_crisis, pipeline
try:
    model = joblib.load('crisis_model.joblib')
    print("Model loaded successfully")
except FileNotFoundError:
    print("Model file not found. Using default prediction function.")
    model = None

# Create the database tables
with app.app_context():
    db.create_all()
    # Print database info for verification
    print("\nDatabase initialized at:", app.config['SQLALCHEMY_DATABASE_URI'])
    print("Tables created:", db.metadata.tables.keys())
    # Count existing records
    prediction_count = Prediction.query.count()
    print(f"Existing prediction records: {prediction_count}\n")

# For this example, we'll redefine a simplified version
def predict_crisis(gsr, temperature, spo2):
    """Simplified prediction function for the web app example"""
    # This is a placeholder - in production, use your trained model
    crisis_prob = (gsr / 5.0) * 0.4 + ((temperature - 35) / 4.0) * 0.3 + ((100 - spo2) / 15.0) * 0.3
    crisis_prob = min(max(crisis_prob, 0), 1)  # Ensure between 0 and 1
    prediction = 1 if crisis_prob > 0.65 else 0
    
    return {
        'crisis_predicted': prediction,
        'crisis_probability': float(crisis_prob),
        'timestamp': datetime.now().isoformat()
    }

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
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({'error': 'Email and password are required'}), 400
            
        user = User.query.filter_by(email=email).first()
        if not user or not user.check_password(password):
            return jsonify({'error': 'Invalid email or password'}), 401
            
        login_user(user)
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
        app.logger.error(f"Login error: {str(e)}")
        return jsonify({'error': 'An error occurred during login'}), 500

@app.route('/api/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    return jsonify({'message': 'Logout successful'})

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
    try:
        print("Attempting to render predictions.html")
        return render_template('predictions.html')
    except Exception as e:
        print(f"Error rendering predictions.html: {str(e)}")
        print(f"Template path: {app.template_folder}")
        return "Error loading page", 500

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
def get_medication_history():
    # In a real implementation, this would fetch history from the database
    # For now, we'll return sample data
    return jsonify({
        'labels': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        'datasets': [
            {
                'label': 'Hydroxyurea',
                'data': [1, 1, 1, 1, 1, 1, 1]
            },
            {
                'label': 'Folic Acid',
                'data': [1, 1, 1, 1, 1, 1, 1]
            }
        ]
    })

# API routes
@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    try:
        # Get data from request
        data = request.get_json()
        
        # Extract sensor readings
        gsr = float(data.get('gsr', 0))
        temperature = float(data.get('temperature', 0))
        spo2 = float(data.get('spo2', 0))
        
        # Make prediction
        result = predict_crisis(gsr, temperature, spo2)
        
        # Store prediction in database
        prediction = Prediction(
            user_id=current_user.id,
            timestamp=datetime.fromisoformat(result['timestamp']),
            gsr=gsr,
            temperature=temperature,
            spo2=spo2,
            crisis_predicted=bool(result['crisis_predicted']),
            crisis_probability=result['crisis_probability']
        )
        db.session.add(prediction)
        db.session.commit()
        
        return jsonify(result)
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400

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

# Make sure templates directory exists
os.makedirs('templates', exist_ok=True)

if __name__ == '__main__':
    app.run(debug=True)
