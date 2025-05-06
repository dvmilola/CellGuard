from app import app, db
from models import User

with app.app_context():
    # Find the user
    user = User.query.filter_by(email='adamilola311@gmail.com').first()
    if user:
        # Update user type
        user.user_type = 'caregiver'
        db.session.commit()
        print(f'Updated user type to: {user.user_type}')
    else:
        print('User not found') 