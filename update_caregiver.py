from app import app, db
from models import User, Caregiver

with app.app_context():
    # Get your user account
    user = User.query.filter_by(email='adamilola311@gmail.com').first()
    
    if user:
        # Update user type to caregiver if not already
        if user.user_type != 'caregiver':
            user.user_type = 'caregiver'
            db.session.commit()
            print('Updated user type to caregiver')
        
        # Get or create caregiver profile
        caregiver = Caregiver.query.filter_by(user_id=user.id).first()
        if not caregiver:
            caregiver = Caregiver(
                user_id=user.id,
                relationship='primary',
                phone='1234567890',  # You can update this later
                emergency_contact=True
            )
            db.session.add(caregiver)
            db.session.commit()
            print('Created caregiver profile')
        else:
            print('Caregiver profile already exists')
            
        # Print current status
        print(f'\nCurrent Status:')
        print(f'User Type: {user.user_type}')
        print(f'Caregiver ID: {caregiver.id}')
        print(f'Patients: {[p.name for p in caregiver.patients]}')
    else:
        print('User not found') 