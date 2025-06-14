#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# Tell mise to install the required python version
mise install

# Upgrade pip and setuptools first
python -m pip install --upgrade pip setuptools

# Install dependencies
python -m pip install -r requirements.txt 