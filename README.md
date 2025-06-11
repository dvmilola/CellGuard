# Crisis Prediction System

A real-time health monitoring and crisis prediction system that uses machine learning to analyze sensor data and predict potential health crises. The system provides early warnings and recommendations for patients and caregivers.

## Features

- Real-time health monitoring using multiple sensors (GSR, temperature, SpO2)
- Machine learning-based crisis prediction
- User authentication and role-based access (patients, caregivers, healthcare providers)
- Medication management and scheduling
- Symptom tracking and analysis
- Emergency contact management
- Real-time alerts and notifications
- Knowledge library for health information
- Web-based dashboard for monitoring and management

## Technology Stack

- **Backend**: Python/Flask
- **Database**: SQLite (SQLAlchemy ORM)
- **Machine Learning**: scikit-learn, pandas, numpy
- **Real-time Communication**: Flask-SocketIO
- **Frontend**: HTML, CSS, JavaScript
- **Data Visualization**: Matplotlib, Seaborn

## Prerequisites

- Python 3.8+
- pip (Python package manager)
- Virtual environment (recommended)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd crisis_prediction
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Initialize the database:
```bash
flask db init
flask db migrate
flask db upgrade
```

## Configuration

1. Set up environment variables:
```bash
export PI_API_URL='http://your-pi-ip:5002'  # For sensor integration
export SENSOR_API_SECRET='your-secret-key'   # For sensor authentication
```

2. Update the secret key in `app.py` for production use.

## Usage

1. Start the application:
```bash
python app.py
```

2. Access the web interface at `http://localhost:5000`

3. Create accounts for:
   - Patients
   - Caregivers
   - Healthcare providers

4. Set up sensor monitoring:
   - Configure sensor devices
   - Start monitoring through the dashboard
   - View real-time predictions and alerts

## Project Structure

```
crisis_prediction/
├── app.py                 # Main application file
├── ml_model.py           # Machine learning model implementation
├── models.py             # Database models
├── sensor_interface.py   # Sensor data handling
├── templates/            # HTML templates
├── static/              # Static files (CSS, JS, images)
├── models/              # Trained ML models
├── data/                # Data files
├── tests/               # Test files
└── requirements.txt     # Project dependencies
```

## Machine Learning

The system uses various machine learning models for crisis prediction:
- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)

Model performance is evaluated using:
- ROC curves
- Precision-Recall curves
- Confusion matrices
- Feature importance analysis

## API Endpoints

### Authentication
- `POST /api/login` - User login
- `POST /api/signup` - User registration
- `GET /logout` - User logout

### Sensor Data
- `POST /api/sensor-readings` - Receive sensor data
- `GET /api/sensor-readings` - Retrieve sensor readings

### Predictions
- `POST /api/predict` - Generate crisis predictions
- `GET /api/recent` - Get recent predictions
- `GET /api/stats` - Get prediction statistics

### Medication Management
- `GET /api/medications` - Get medication list
- `POST /api/medications` - Add new medication
- `GET /api/medications/schedule` - Get medication schedule
- `POST /api/medications/schedule/<id>` - Update medication status

### Emergency Contacts
- `GET /api/contacts` - Get emergency contacts
- `POST /api/contacts` - Add emergency contact

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Specify your license here]

## Support

For support, please contact [your contact information] 