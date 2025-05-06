from app import app, db
from models import User, HealthcareProvider
from werkzeug.security import generate_password_hash

with app.app_context():
    # Create provider user
    provider_user = User(
        name='Dr. Smith',
        email='provider@example.com',
        user_type='healthcare-provider',
        password_hash=generate_password_hash('provider123')
    )
    
    # Check if provider already exists
    existing_provider = User.query.filter_by(email='provider@example.com').first()
    if not existing_provider:
        db.session.add(provider_user)
        db.session.commit()
        print('Created provider user account')
        
        # Create healthcare provider profile
        provider_profile = HealthcareProvider(
            user_id=provider_user.id,
            name='Dr. Smith',
            specialty='Hematology',
            hospital='General Hospital',
            phone='123-456-7890',
            email='provider@example.com'
        )
        db.session.add(provider_profile)
        db.session.commit()
        print('Created healthcare provider profile')
    else:
        print('Provider account already exists') 