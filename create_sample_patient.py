from app import app, db
from models import User, Caregiver
from werkzeug.security import generate_password_hash

with app.app_context():
    # Create patient user
    patient_user = User(
        name='John Doe',
        email='patient@example.com',
        user_type='patient',
        password_hash=generate_password_hash('patient123')
    )
    
    # Check if patient already exists
    existing_patient = User.query.filter_by(email='patient@example.com').first()
    if not existing_patient:
        db.session.add(patient_user)
        db.session.commit()
        print('Created patient user account')
        
        # Get the caregiver
        caregiver = Caregiver.query.filter_by(user_id=User.query.filter_by(email='caregiver@example.com').first().id).first()
        
        # Link patient to caregiver
        if caregiver:
            caregiver.patients.append(patient_user)
            db.session.commit()
            print('Linked patient to caregiver')
    else:
        print('Patient account already exists') 