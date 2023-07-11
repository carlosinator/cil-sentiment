#!/bin/bash

# Create virtual environment
python3 -m venv venv-cil

# Activate the virtual environment
source venv-cil/bin/activate

# Install requirements
pip install -r requirements.txt

# Output activation command
echo "Virtual environment 'venv-cil' created and activated."
echo "To activate it again in the future, use the command: source venv-cil/bin/activate"
