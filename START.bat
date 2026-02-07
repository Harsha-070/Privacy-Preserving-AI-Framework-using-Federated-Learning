@echo off
REM Universal Launcher for Windows
REM Double-click this file to start the application

echo ========================================
echo  Federated Learning Framework
echo  Starting Universal Launcher...
echo ========================================
echo.

REM Set environment variables to prevent errors
set TF_ENABLE_ONEDNN_OPTS=0
set OMP_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set TF_CPP_MIN_LOG_LEVEL=2

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo Python found:
python --version
echo.

REM Check if dependencies are installed
python -c "import tensorflow, numpy, streamlit" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    echo This may take a few minutes...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to install dependencies
        echo Please run manually: pip install -r requirements.txt
        pause
        exit /b 1
    )
)

echo Dependencies OK
echo.
echo Launching application...
echo.

REM Run the universal launcher
python run.py

if errorlevel 1 (
    echo.
    echo ERROR: Application failed to start
    echo Please check TROUBLESHOOTING.md for help
    pause
    exit /b 1
)

pause
