#!/bin/bash
set -ex # Exit on error and print commands

# This will install both python@3.11 and node@20 as defined in .mise.toml
mise install

# The environment should now have the correct versions of python and pip
python -m pip install --upgrade pip setuptools
python -m pip install -r requirements.txt

echo "Build script finished successfully." 