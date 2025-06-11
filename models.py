from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    user_type = db.Column(db.String(20), nullable=False)  # patient, caregiver, healthcare-provider
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    age = db.Column(db.Integer, nullable=True)
    gender = db.Column(db.String(10), nullable=True)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class SensorReading(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    gsr = db.Column(db.Float, nullable=False)
    temperature = db.Column(db.Float, nullable=False)
    spo2 = db.Column(db.Float, nullable=False)
    crisis_probability = db.Column(db.Float, nullable=False)
    
    user = db.relationship('User', backref=db.backref('sensor_readings', lazy=True))

    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'gsr': self.gsr,
            'temperature': self.temperature,
            'spo2': self.spo2,
            'crisis_probability': self.crisis_probability
        }

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    gsr = db.Column(db.Float, nullable=False)
    temperature = db.Column(db.Float, nullable=False)
    spo2 = db.Column(db.Float, nullable=False)
    crisis_predicted = db.Column(db.Boolean, nullable=False)
    crisis_probability = db.Column(db.Float, nullable=False)
    
    user = db.relationship('User', backref=db.backref('predictions', lazy=True))

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

class EmergencyContact(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    contact_type = db.Column(db.String(50), nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(120))
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    user = db.relationship('User', backref=db.backref('emergency_contacts', lazy=True))

class HealthcareProvider(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    specialty = db.Column(db.String(100), nullable=False)
    hospital = db.Column(db.String(200))
    phone = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    user = db.relationship('User', backref=db.backref('healthcare_providers', lazy=True))

class Medication(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    dosage = db.Column(db.String(50), nullable=False)
    frequency = db.Column(db.String(50), nullable=False)
    start_date = db.Column(db.Date, nullable=False)
    end_date = db.Column(db.Date)
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Fields for current supply tracking
    current_supply_days = db.Column(db.Integer, nullable=True) 
    current_supply_start_date = db.Column(db.Date, nullable=True)

    user = db.relationship('User', backref=db.backref('medications', lazy=True))

class MedicationSchedule(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    medication_id = db.Column(db.Integer, db.ForeignKey('medication.id'), nullable=False)
    scheduled_date = db.Column(db.Date, nullable=False)
    time = db.Column(db.Time, nullable=False)
    is_taken = db.Column(db.Boolean, default=False)
    taken_at = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    medication = db.relationship('Medication', backref=db.backref('schedules', lazy=True))

class MedicationRefill(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    medication_id = db.Column(db.Integer, db.ForeignKey('medication.id'), nullable=False)
    quantity = db.Column(db.Integer, nullable=True) # Made nullable, might represent number of packs/pills
    days_supply = db.Column(db.Integer, nullable=False) # Days this specific refill provides
    refill_date = db.Column(db.Date, nullable=False)
    next_refill_date = db.Column(db.Date, nullable=True) # Made nullable
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    medication = db.relationship('Medication', backref=db.backref('refills', lazy=True))

class Symptom(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.Date, nullable=False)
    pain_level = db.Column(db.Integer, nullable=False)
    symptoms = db.Column(db.String(500), nullable=False)  # JSON string of symptoms
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user = db.relationship('User', backref=db.backref('symptoms', lazy=True))

class CrisisPrediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    gsr = db.Column(db.Float, nullable=False)
    temperature = db.Column(db.Float, nullable=False)
    spo2 = db.Column(db.Float, nullable=False)
    crisis_predicted = db.Column(db.Boolean, nullable=False)
    crisis_probability = db.Column(db.Float, nullable=False)
    features = db.Column(db.JSON, nullable=False)  # Store all features used for prediction
    recommendations = db.Column(db.JSON)  # Store any recommendations generated
    
    user = db.relationship('User', backref=db.backref('crisis_predictions', lazy=True))

    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'gsr': self.gsr,
            'temperature': self.temperature,
            'spo2': self.spo2,
            'crisis_predicted': self.crisis_predicted,
            'crisis_probability': self.crisis_probability,
            'features': self.features,
            'recommendations': self.recommendations
        }

class CaregiverPatientLink(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    caregiver_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    patient_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    relationship = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Define relationships
    caregiver = db.relationship('User', foreign_keys=[caregiver_id], backref='patient_links')
    patient = db.relationship('User', foreign_keys=[patient_id], backref='caregiver_links')

    # Add unique constraint to prevent duplicate links
    __table_args__ = (db.UniqueConstraint('caregiver_id', 'patient_id', name='_caregiver_patient_uc'),)

class CaregiverLinkToken(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    token = db.Column(db.String(100), unique=True, nullable=False)
    caregiver_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    patient_email = db.Column(db.String(120), nullable=False)
    relationship = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime, nullable=False)

    caregiver = db.relationship('User', backref='pending_links')

class PatientOTP(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    otp_code = db.Column(db.String(8), unique=True, nullable=False)
    patient_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime)
    used = db.Column(db.Boolean, default=False)
    
    patient = db.relationship('User', backref=db.backref('otps', lazy=True)) 