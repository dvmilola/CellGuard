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


# Configure the application
app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)  # Enable CORS for all routes

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///crisis_predictions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this to a secure secret key in production
db = SQLAlchemy(app)

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

class User(db.Model):
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
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        remember = request.form.get('remember') == 'on'
        
        user = User.query.filter_by(email=email).first()
        
        if user and user.check_password(password):
            session['user_id'] = user.id
            session['user_name'] = user.name
            session['user_type'] = user.user_type
            
            if remember:
                session.permanent = True
                
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password', 'error')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('full-name')
        email = request.form.get('email')
        password = request.form.get('password')
        user_type = request.form.get('user-type')
        
        # Check if user already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Email already registered', 'error')
            return redirect(url_for('signup'))
        
        # Create new user
        new_user = User(name=name, email=email, user_type=user_type)
        new_user.set_password(password)
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Account created successfully! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('home'))

# Main routes
@app.route('/')
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        print(f"Error rendering template: {str(e)}")
        return "Error loading page", 500

@app.route('/predictions')
def predictions():
    try:
        return render_template('predictions.html')
    except Exception as e:
        print(f"Error rendering template: {str(e)}")
        return "Error loading page", 500

@app.route('/emergency')
def emergency():
    if 'user_id' not in session:
        flash('Please login to access this page', 'warning')
        return redirect(url_for('login'))
    
    contacts = EmergencyContact.query.filter_by(user_id=session['user_id']).all()
    return render_template('emergency.html', contacts=contacts)

@app.route('/profile')
def profile():
    if 'user_id' not in session:
        flash('Please login to access this page', 'warning')
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])
    providers = HealthcareProvider.query.filter_by(user_id=session['user_id']).all()
    return render_template('profile.html', user=user, providers=providers)

@app.route('/provider')
def provider():
    if 'user_id' not in session:
        flash('Please login to access this page', 'warning')
        return redirect(url_for('login'))
    
    if session['user_type'] != 'healthcare-provider':
        flash('Access denied. This page is for healthcare providers only.', 'error')
        return redirect(url_for('home'))
    
    # Get recent predictions for display
    recent_predictions = Prediction.query.order_by(Prediction.timestamp.desc()).limit(10).all()
    return render_template('provider.html', predictions=recent_predictions)

@app.route('/settings')
def settings():
    if 'user_id' not in session:
        flash('Please login to access this page', 'warning')
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])
    return render_template('settings.html', user=user)

# API routes
@app.route('/api/predict', methods=['POST'])
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
def get_contacts():
    if 'user_id' not in session:
        return jsonify({'error': 'Authentication required'}), 401
    
    contacts = EmergencyContact.query.filter_by(user_id=session['user_id']).all()
    return jsonify([{
        'id': contact.id,
        'name': contact.name,
        'type': contact.contact_type,
        'phone': contact.phone,
        'email': contact.email,
        'notes': contact.notes
    } for contact in contacts])

@app.route('/api/contacts', methods=['POST'])
def add_contact():
    if 'user_id' not in session:
        return jsonify({'error': 'Authentication required'}), 401
    
    data = request.get_json()
    
    new_contact = EmergencyContact(
        user_id=session['user_id'],
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

# Make sure templates directory exists
os.makedirs('templates', exist_ok=True)

if __name__ == '__main__':
    app.run(debug=True)
