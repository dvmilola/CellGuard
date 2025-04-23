from train_model import main
import os
from datetime import datetime

if __name__ == "__main__":
    # Path to the rescaled dataset
    data_path = "final_dataset_rescaled.csv"
    
    # Create backup directory with timestamp
    backup_dir = f"backup_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(backup_dir, exist_ok=True)
    
    # Backup existing model files
    for file in os.listdir('models'):
        if file.startswith('crisis_model_') or file.startswith('scaler_') or file.startswith('threshold_'):
            os.rename(os.path.join('models', file), os.path.join(backup_dir, file))
    
    try:
        # Train the model with rescaled data
        print("\nStarting model retraining with rescaled dataset...")
        main(data_path)
        print("\nModel retraining completed successfully!")
    except Exception as e:
        print(f"\nError during model retraining: {str(e)}")
        print("Restoring from backup...")
        # Restore from backup if training fails
        for file in os.listdir(backup_dir):
            os.rename(os.path.join(backup_dir, file), os.path.join('models', file))
        raise 