from app import app, db
from models import User, Caregiver

with app.app_context():
    # Check all users
    users = User.query.all()
    print("\nAll Users:")
    for user in users:
        print(f"ID: {user.id}, Name: {user.name}, Email: {user.email}, Type: {user.user_type}")
    
    # Check all caregivers
    caregivers = Caregiver.query.all()
    print("\nAll Caregivers:")
    for caregiver in caregivers:
        print(f"ID: {caregiver.id}, User ID: {caregiver.user_id}, Relationship: {caregiver.relationship}")
        print(f"Patients: {[p.name for p in caregiver.patients]}") 