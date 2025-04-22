import numpy as np
from datetime import datetime

def predict_crisis(gsr, temperature, spo2):
    """
    Make a crisis prediction based on sensor readings.
    
    Args:
        gsr: Galvanic skin response value (normal range: 1-20 µS)
        temperature: Body temperature (normal range: 36-37.5°C)
        spo2: Blood oxygen saturation (normal range: 95-100%)
        
    Returns:
        Dictionary with prediction results
    """
    # Normalize values to 0-1 range
    gsr_norm = min(max((gsr - 1) / 19, 0), 1)  # 1-20 µS -> 0-1
    temp_norm = min(max(abs(temperature - 36.75) / 1.25, 0), 1)  # 36-37.5°C -> 0-1
    spo2_norm = min(max((100 - spo2) / 5, 0), 1)  # 95-100% -> 0-1
    
    # Calculate crisis probability using weighted formula
    crisis_prob = (gsr_norm * 0.4) + (temp_norm * 0.3) + (spo2_norm * 0.3)
    crisis_prob = min(max(crisis_prob, 0), 1)  # Ensure between 0 and 1
    
    # Only predict crisis if probability is above 0.7 (70%)
    crisis_predicted = crisis_prob > 0.7
    
    return {
        'crisis_predicted': bool(crisis_predicted),
        'crisis_probability': float(crisis_prob),
        'timestamp': datetime.utcnow().isoformat()
    }
