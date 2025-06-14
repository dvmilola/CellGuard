#!/bin/bash
set -ex # Exit on error and print commands

# Tell mise to install the required python version
echo "Installing Python 3.11..."
mise install python@3.11

# Tell mise to use the version for this shell
echo "Activating Python 3.11..."
eval "$(mise use python@3.11)"

# Check the python version now
echo "Python version is now:"
python --version

# Upgrade pip and setuptools first
echo "Upgrading pip and setuptools..."
python -m pip install --upgrade pip setuptools

# Install dependencies
echo "Installing dependencies from requirements.txt..."
python -m pip install -r requirements.txt

echo "Build script finished successfully." 