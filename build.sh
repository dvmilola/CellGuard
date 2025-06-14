#!/bin/bash
set -ex # Exit on error and print commands

# This command will read the .mise.toml file and install the correct tools.
echo "Installing tools defined in .mise.toml..."
mise install

# Activate the mise environment for the current shell
echo "Activating mise environment..."
eval "$(mise activate bash)"

# Now, the 'python' and 'pip' commands should be the correct versions.
echo "Checking Python version after activation:"
python --version

# Upgrade pip and setuptools first
echo "Upgrading pip and setuptools..."
python -m pip install --upgrade pip setuptools

# Install dependencies
echo "Installing dependencies from requirements.txt..."
python -m pip install -r requirements.txt

echo "Build script finished successfully." 