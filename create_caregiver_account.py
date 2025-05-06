from app import app, db
from models import User, Caregiver
from werkzeug.security import generate_password_hash

with app.app_context():
    # Create caregiver user
    caregiver_user = User(
        name='Sarah Johnson',
        email='caregiver@example.com',
        user_type='caregiver',
        password_hash=generate_password_hash('caregiver123')
    )
    
    # Check if caregiver already exists
    existing_caregiver = User.query.filter_by(email='caregiver@example.com').first()
    if not existing_caregiver:
        db.session.add(caregiver_user)
        db.session.commit()
        print('Created caregiver user account')
        
        # Create caregiver profile
        caregiver_profile = Caregiver(
            user_id=caregiver_user.id,
            relationship='family_member',  # Could be parent, spouse, sibling, etc.
            phone='123-456-7890',
            emergency_contact=True
        )
        db.session.add(caregiver_profile)
        db.session.commit()
        print('Created caregiver profile')
    else:
        print('Caregiver account already exists') 