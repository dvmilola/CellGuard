from app import app, db
from models import User, Caregiver

with app.app_context():
    # Find the user
    user = User.query.filter_by(email='adamilola311@gmail.com').first()
    if user:
        # Check if caregiver profile already exists
        caregiver = Caregiver.query.filter_by(user_id=user.id).first()
        if not caregiver:
            # Create new caregiver profile
            caregiver = Caregiver(
                user_id=user.id,
                relationship='primary',  # You can change this as needed
                phone='1234567890',     # You can update this later
                emergency_contact=True
            )
            db.session.add(caregiver)
            db.session.commit()
            print('Created new caregiver profile')
        else:
            print('Caregiver profile already exists')
    else:
        print('User not found') 