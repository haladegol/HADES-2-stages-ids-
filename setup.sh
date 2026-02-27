#!/bin/bash
# Setup script for HADES on WSL

echo "Setting up HADES - Hierarchical Intrusion Detection System"
echo "=========================================================="

# Update system
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
echo "Installing Python and build tools..."
sudo apt install -y python3 python3-pip python3-venv build-essential

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python packages
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create directory structure
echo "Creating project directories..."
mkdir -p data models logs dashboard/utils

# Initialize database
echo "Initializing database..."
python main.py init

echo ""
echo "Setup complete!"
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To train models:"
echo "  python main.py train"
echo ""
echo "To start dashboard:"
echo "  python main.py dashboard"
echo ""
echo "To run inference:"
echo "  python main.py infer --file <path_to_csv>"
