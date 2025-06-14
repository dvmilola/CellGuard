# All the necessary code is now inside the 'api' directory,
# so we can import the app object directly.
from app import app as application

# This file is the entry point for Netlify's serverless functions.
# It simply imports the Flask app instance. 