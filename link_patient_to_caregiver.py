from app import app, db
from models import User, Caregiver

with app.app_context():
    # Get your caregiver account
    caregiver_user = User.query.filter_by(email='adamilola311@gmail.com').first()
    caregiver = Caregiver.query.filter_by(user_id=caregiver_user.id).first()
    
    # Get John Doe
    patient = User.query.filter_by(email='patient@example.com').first()
    
    # Link them if not already linked
    if patient and caregiver:
        if patient not in caregiver.patients:
            caregiver.patients.append(patient)
            db.session.commit()
            print('Linked John Doe to your caregiver account')
        else:
            print('John Doe is already linked to your account')
    else:
        print('Could not find caregiver or patient') 