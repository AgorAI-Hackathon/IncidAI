#!/bin/bash

echo "========================================"
echo "ITSM AI - Quick Setup Script (Linux/Mac)"
echo "========================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python not found! Please install Python 3.8+"
    exit 1
fi

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "ERROR: Node.js not found! Please install Node.js 18+"
    exit 1
fi

echo "All prerequisites found!"
echo ""

echo "Step 1: Setting up Backend..."
cd backend

echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Copying environment file..."
cp .env.example .env

echo ""
echo "Step 2: Setting up Frontend..."
cd ../frontend

echo "Installing Node.js dependencies..."
npm install

echo "Copying environment file..."
cp .env.example .env

cd ..

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Setup PostgreSQL database (see README.md)"
echo "2. Update backend/.env with database credentials"
echo "3. Run: source backend/venv/bin/activate"
echo "4. Run: python backend/manage.py migrate"
echo "5. Run: python backend/manage.py createsuperuser"
echo "6. Start backend: python backend/manage.py runserver"
echo "7. Start frontend: cd frontend && npm run dev"
echo ""
