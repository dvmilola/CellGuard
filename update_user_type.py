from app import app, db
from models import User

with app.app_context():
    # Find the user by email
    user = User.query.filter_by(email='briggs@gmail.com').first()
    if user:
        print(f'Before: {user.name}, user_type: {user.user_type}')
        user.user_type = 'caregiver'
        db.session.commit()
        print(f'After: {user.name}, user_type: {user.user_type}')
    else:
        print('User not found') 