#!/bin/bash
# Universal Launcher for Linux/Mac
# Run this file to start the application: bash start.sh

echo "========================================"
echo " Federated Learning Framework"
echo " Starting Universal Launcher..."
echo "========================================"
echo ""

# Set environment variables to prevent errors
export TF_ENABLE_ONEDNN_OPTS=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=2

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8+ using your package manager:"
    echo "  Ubuntu/Debian: sudo apt install python3 python3-pip"
    echo "  macOS: brew install python3"
    exit 1
fi

echo "Python found:"
python3 --version
echo ""

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "ERROR: pip3 is not installed"
    echo "Please install: sudo apt install python3-pip"
    exit 1
fi

# Check if dependencies are installed
python3 -c "import tensorflow, numpy, streamlit" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing dependencies..."
    echo "This may take a few minutes..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo ""
        echo "ERROR: Failed to install dependencies"
        echo "Please run manually: pip3 install -r requirements.txt"
        exit 1
    fi
fi

echo "Dependencies OK"
echo ""
echo "Launching application..."
echo ""

# Run the universal launcher
python3 run.py

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Application failed to start"
    echo "Please check TROUBLESHOOTING.md for help"
    exit 1
fi

echo ""
echo "Application closed successfully"
