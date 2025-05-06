from app import app, db
from models import User, Caregiver, SensorReading, CrisisPrediction
from datetime import datetime, timedelta
import random

with app.app_context():
    # Get the caregiver
    caregiver = Caregiver.query.filter_by(user_id=User.query.filter_by(email='caregiver@example.com').first().id).first()
    
    if caregiver:
        # Create a sample patient if they don't exist
        patient = User.query.filter_by(email='patient@example.com').first()
        if not patient:
            patient = User(
                name='John Doe',
                email='patient@example.com',
                user_type='patient',
                password_hash='dummy_hash'  # We'll set a real password later
            )
            db.session.add(patient)
            db.session.commit()
            print('Created sample patient')
        
        # Link patient to caregiver if not already linked
        if patient not in caregiver.patients:
            caregiver.patients.append(patient)
            db.session.commit()
            print('Linked patient to caregiver')
        
        # Add some sample sensor readings
        for i in range(5):
            reading = SensorReading(
                user_id=patient.id,
                gsr=random.uniform(0.5, 2.0),
                temperature=random.uniform(36.5, 37.5),
                spo2=random.uniform(95, 100),
                crisis_probability=random.uniform(0, 0.3),
                timestamp=datetime.utcnow() - timedelta(hours=i)
            )
            db.session.add(reading)
            
            # Add corresponding crisis prediction
            prediction = CrisisPrediction(
                user_id=patient.id,
                gsr=reading.gsr,
                temperature=reading.temperature,
                spo2=reading.spo2,
                crisis_predicted=False,
                crisis_probability=reading.crisis_probability,
                features={'age': 45, 'gender': 1},
                recommendations=['Monitor vitals', 'Stay hydrated']
            )
            db.session.add(prediction)
        
        db.session.commit()
        print('Added sample sensor readings and predictions')
    else:
        print('Caregiver not found') 