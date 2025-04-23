import sys
import os

# Print current working directory
print(f"Current working directory: {os.getcwd()}")

# Print Python path
print("\nPython path:")
for path in sys.path:
    print(f"- {path}")

# Try to import ml_model
try:
    from ml_model import CrisisPredictionModel
    print("\nSuccessfully imported ml_model!")
except ImportError as e:
    print(f"\nFailed to import ml_model: {e}")
    print("\nFiles in current directory:")
    for file in os.listdir('.'):
        print(f"- {file}") 