import sys
import os

# Add the project root to the Python path
# The entry point for Netlify Functions is /var/task/api/index.py
# The project root is one level up.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app as application

# This file is the entry point for Netlify's serverless functions.
# It simply imports the Flask app instance. 