#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# Explicitly activate the correct python version using mise
mise use python@3.11

# Upgrade pip and setuptools first
python -m pip install --upgrade pip setuptools

# Install dependencies
python -m pip install -r requirements.txt 