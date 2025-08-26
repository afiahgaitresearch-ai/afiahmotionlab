@echo off
REM ============================================================================
REM  Batch Script to Setup and Run the AI-Powered Gait Analyzer (v4)
REM ============================================================================
REM  This script will:
REM  1. Check if Python is installed.
REM  2. Check if the Python installation is 64-bit.
REM  3. Check if the Python version is compatible (3.8-3.11).
REM  4. Create a virtual environment named 'venv'.
REM  5. Activate the virtual environment.
REM  6. Upgrade pip, setuptools, and wheel.
REM  7. Install all required packages from requirements.txt.
REM  8. Launch the Streamlit application (main.py).
REM  9. Keep the window open after the app closes to show any errors.
REM ============================================================================

title Gait Analyzer Setup & Launcher

echo =================================================
echo  Welcome to the AI-Powered Gait Analyzer Setup
echo =================================================
echo.
echo This script will prepare the environment and run the application.
echo Please do not close this window manually.
echo.

REM --- Step 1: Check for Python installation ---
echo [STEP 1/8] Checking for Python...
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not found in your system's PATH.
    echo Please install Python 3.8-3.11 from python.org and ensure you check "Add Python to PATH" during installation.
    echo.
    goto :end
)
echo Python found!
echo.


REM --- Step 7: Install dependencies ---
echo [STEP 7/8] Installing packages...
if not exist requirements.txt (
    echo [ERROR] requirements.txt file not found!
    echo Please make sure requirements.txt is in the same directory as this script.
    goto :end
)
echo Installing required packages from requirements.txt. This may take several minutes...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install required packages.
    echo Please check your internet connection and the contents of requirements.txt.
    echo Look for any error messages above.
    goto :end
)
echo All packages installed successfully.
echo.

REM --- Step 8: Run the Streamlit application ---
echo [STEP 8/8] Launching the Gait Analyzer Application...
echo.
echo ===================================================================
echo  Your web browser should open with the application shortly.
echo  If it doesn't, please open your browser and go to the 'Network URL'
echo  shown below (e.g., http://192.168.x.x:8501).
echo.
echo  CLOSE THIS WINDOW to stop the application.
echo ===================================================================
echo.

streamlit run main.py

:end
echo.
echo ===================================================================
echo  The application has been closed or an error occurred.
echo  If the app did not run, please review any error messages above.
echo.
echo  This window will remain open. Press any key to exit.
echo ===================================================================
pause
