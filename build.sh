#!/bin/bash
set -ex # Exit on error and print commands

# This command will read the .mise.toml file and install the correct tools.
mise install

# Upgrade pip and setuptools first
python -m pip install --upgrade pip setuptools

# Install dependencies
python -m pip install -r requirements.txt

echo "Build script finished successfully." 