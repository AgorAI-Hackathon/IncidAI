@echo off
echo ========================================
echo ITSM AI - Quick Setup Script (Windows)
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3.8+
    pause
    exit /b 1
)

REM Check Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js not found! Please install Node.js 18+
    pause
    exit /b 1
)

echo All prerequisites found!
echo.

echo Step 1: Setting up Backend...
cd backend

echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate

echo Installing Python dependencies...
pip install -r requirements.txt

echo Copying environment file...
copy .env.example .env

echo.
echo Step 2: Setting up Frontend...
cd ..\frontend

echo Installing Node.js dependencies...
call npm install

echo Copying environment file...
copy .env.example .env

cd ..

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Setup PostgreSQL database (see README.md)
echo 2. Update backend/.env with database credentials
echo 3. Run: backend\venv\Scripts\activate
echo 4. Run: python backend\manage.py migrate
echo 5. Run: python backend\manage.py createsuperuser
echo 6. Start backend: python backend\manage.py runserver
echo 7. Start frontend: cd frontend ^&^& npm run dev
echo.
pause
