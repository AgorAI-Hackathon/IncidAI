# ITSM AI - Full-Stack Hackathon Project

🚀 **AI-Powered IT Service Management Ticket Classification System**

Complete full-stack web application combining Machine Learning, Deep Learning, LLM/RAG with React frontend and Django backend.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Tech Stack](#tech-stack)
3. [Features](#features)
4. [Prerequisites](#prerequisites)
5. [Quick Start Guide](#quick-start-guide)
6. [Detailed Setup Instructions](#detailed-setup-instructions)
7. [Running the Application](#running-the-application)
8. [Deployment Guide](#deployment-guide)
9. [API Documentation](#api-documentation)
10. [GitHub Setup](#github-setup)
11. [Troubleshooting](#troubleshooting)
12. [Project Structure](#project-structure)

---

## 🎯 Project Overview

This hackathon project demonstrates a production-ready ITSM system that uses AI to:
- **Automatically classify** support tickets into categories
- **Find similar tickets** using semantic search
- **Generate resolutions** using RAG (Retrieval-Augmented Generation)
- **Visualize insights** through an interactive dashboard

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FRONTEND (React + Vite)              │
│  Dashboard | Ticket List | New Ticket | AI Analyze     │
└─────────────────────┬───────────────────────────────────┘
                      │ HTTP/REST API
┌─────────────────────▼───────────────────────────────────┐
│              BACKEND (Django + PostgreSQL)              │
│  REST API | Authentication | Database Models            │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                   AI/ML LAYER                           │
│  ML Models | Embeddings | FAISS | RAG | LLM            │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠 Tech Stack

### Frontend
- **React 18** - UI framework
- **Vite** - Build tool & dev server
- **React Router** - Navigation
- **Tailwind CSS** - Styling
- **Recharts** - Data visualization
- **Axios** - HTTP client

### Backend
- **Django 4.2** - Web framework
- **Django REST Framework** - API
- **PostgreSQL** - Database
- **CORS Headers** - Cross-origin support

### AI/ML
- **scikit-learn** - ML models (Logistic Regression, Random Forest, XGBoost)
- **Transformers** - BERT/DistilBERT
- **Sentence-Transformers** - Semantic embeddings
- **FAISS** - Vector similarity search
- **OpenAI** - Optional LLM integration

---

## ✨ Features

### 🤖 AI-Powered Classification
- Automatic ticket categorization using ML models
- Real-time prediction with confidence scores
- Multiple model comparison (Logistic Regression, RF, XGBoost)

### 🔍 Semantic Search
- Find similar historical tickets
- Sentence-BERT embeddings
- FAISS vector database for fast search

### 💡 Smart Resolutions
- RAG-based resolution generation
- Context-aware suggestions
- Similar ticket analysis

### 📊 Interactive Dashboard
- Real-time statistics
- Status and priority distributions
- Category analytics
- Performance metrics

### 🎨 Modern UI
- Responsive design
- Clean, intuitive interface
- Real-time updates
- Professional styling with Tailwind CSS

---

## 📦 Prerequisites

Before you begin, ensure you have:

1. **Python 3.8+** - [Download](https://www.python.org/downloads/)
2. **Node.js 18+** - [Download](https://nodejs.org/)
3. **PostgreSQL 12+** - [Download](https://www.postgresql.org/download/)
4. **Git** - [Download](https://git-scm.com/)

### Verify installations:
```bash
python --version    # Should show 3.8+
node --version      # Should show 18+
npm --version       # Should show 9+
psql --version      # Should show 12+
git --version       # Should show 2.0+
```

---

## 🚀 Quick Start Guide

### Step 1: Clone or Extract Project
```bash
# If you have the zip file:
unzip itsm-hackathon-fullstack.zip
cd itsm-hackathon-fullstack

# Or clone from GitHub (after pushing):
git clone <your-github-repo-url>
cd itsm-hackathon-fullstack
```

### Step 2: Database Setup
```bash
# Start PostgreSQL service
# On Ubuntu/Debian:
sudo service postgresql start

# On macOS (with Homebrew):
brew services start postgresql

# On Windows:
# Start PostgreSQL from Services or pg_ctl

# Create database and user
psql -U postgres
```

In PostgreSQL shell:
```sql
CREATE DATABASE itsm_db;
CREATE USER itsm_user WITH PASSWORD 'itsm_password';
ALTER ROLE itsm_user SET client_encoding TO 'utf8';
ALTER ROLE itsm_user SET default_transaction_isolation TO 'read committed';
ALTER ROLE itsm_user SET timezone TO 'UTC';
GRANT ALL PRIVILEGES ON DATABASE itsm_db TO itsm_user;
\q
```

### Step 3: Backend Setup
```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env

# Edit .env file with your settings (use any text editor)
# Make sure database credentials match what you created

# Run migrations
python manage.py makemigrations
python manage.py migrate

# Create superuser (for admin panel)
python manage.py createsuperuser
# Follow prompts to create admin user

# Load initial data (optional - creates categories)
python manage.py shell
```

In Django shell:
```python
from tickets.models import TicketCategory
categories = ['Password Reset', 'Folder Access', 'Slow Computer', 'VPN Issues', 'Email Issues', 'Hardware Issue', 'Software Installation']
for cat in categories:
    TicketCategory.objects.get_or_create(name=cat)
exit()
```

### Step 4: Frontend Setup
```bash
# Open new terminal
cd frontend

# Install dependencies
npm install

# Copy environment file
cp .env.example .env

# Environment should point to backend (default: http://localhost:8000/api)
```

### Step 5: Start Services

**Terminal 1 - Backend:**
```bash
cd backend
source venv/bin/activate  # or venv\Scripts\activate on Windows
python manage.py runserver
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

### Step 6: Access Application

- **Frontend:** http://localhost:5173
- **Backend API:** http://localhost:8000/api
- **Admin Panel:** http://localhost:8000/admin
- **API Docs:** http://localhost:8000/api/docs/

---

## 📝 Detailed Setup Instructions

### Backend Configuration

1. **Environment Variables** (`backend/.env`):
```env
SECRET_KEY=your-super-secret-key-change-this
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1

DB_NAME=itsm_db
DB_USER=itsm_user
DB_PASSWORD=itsm_password
DB_HOST=localhost
DB_PORT=5432

CORS_ALLOWED_ORIGINS=http://localhost:5173,http://localhost:3000

# Optional: Add OpenAI API key for LLM features
OPENAI_API_KEY=sk-...
```

2. **Install AI Models**:

The AI models are already in the `ai-models` directory. If you need to retrain:

```bash
cd ai-models
python src/main_pipeline.py
```

This will:
- Clean and preprocess data
- Train ML models
- Generate embeddings
- Build FAISS index

3. **Collect Static Files** (for production):
```bash
python manage.py collectstatic --noinput
```

### Frontend Configuration

1. **Environment Variables** (`frontend/.env`):
```env
VITE_API_URL=http://localhost:8000/api
```

2. **Build for Production**:
```bash
npm run build
```

This creates optimized production build in `dist/` folder.

---

## 🎬 Running the Application

### Development Mode

**Start Backend:**
```bash
cd backend
source venv/bin/activate
python manage.py runserver
```

**Start Frontend:**
```bash
cd frontend
npm run dev
```

### Production Mode

See [Deployment Guide](#deployment-guide) below.

---

## 🌐 Deployment Guide

### Option 1: Deploy to Render (Free Tier)

#### Backend Deployment

1. **Create `render.yaml`** (already included):
```yaml
services:
  - type: web
    name: itsm-backend
    env: python
    buildCommand: "pip install -r requirements.txt && python manage.py collectstatic --noinput && python manage.py migrate"
    startCommand: "gunicorn itsm_backend.wsgi:application"
    envVars:
      - key: DATABASE_URL
        fromDatabase:
          name: itsm-db
          property: connectionString
      - key: SECRET_KEY
        generateValue: true
      - key: DEBUG
        value: "False"

databases:
  - name: itsm-db
    plan: free
```

2. **Push to GitHub** (see GitHub section below)

3. **Deploy on Render:**
   - Go to https://render.com
   - Click "New +" → "Blueprint"
   - Connect your GitHub repo
   - Render will auto-deploy using render.yaml

#### Frontend Deployment

1. **Build frontend:**
```bash
cd frontend
npm run build
```

2. **Deploy to Vercel/Netlify:**

**Vercel:**
```bash
npm install -g vercel
vercel --prod
```

**Netlify:**
```bash
npm install -g netlify-cli
netlify deploy --prod --dir=dist
```

3. **Update API URL:**
   - Set `VITE_API_URL` to your backend URL
   - Rebuild: `npm run build`

### Option 2: Deploy to Heroku

#### Backend:
```bash
cd backend
heroku create itsm-backend
heroku addons:create heroku-postgresql:mini
heroku config:set SECRET_KEY=your-secret-key
heroku config:set DEBUG=False
git push heroku main
heroku run python manage.py migrate
```

#### Frontend:
```bash
cd frontend
# Add to package.json:
"scripts": {
  "start": "vite preview --port $PORT"
}

heroku create itsm-frontend
git push heroku main
```

### Option 3: Docker Deployment

**Backend Dockerfile** (create `backend/Dockerfile`):
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python manage.py collectstatic --noinput

CMD ["gunicorn", "itsm_backend.wsgi:application", "--bind", "0.0.0.0:8000"]
```

**Frontend Dockerfile** (create `frontend/Dockerfile`):
```dockerfile
FROM node:18-alpine as builder

WORKDIR /app
COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

**Docker Compose** (create `docker-compose.yml`):
```yaml
version: '3.8'

services:
  db:
    image: postgres:15
    environment:
      POSTGRES_DB: itsm_db
      POSTGRES_USER: itsm_user
      POSTGRES_PASSWORD: itsm_password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  backend:
    build: ./backend
    command: gunicorn itsm_backend.wsgi:application --bind 0.0.0.0:8000
    volumes:
      - ./backend:/app
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://itsm_user:itsm_password@db:5432/itsm_db
    depends_on:
      - db

  frontend:
    build: ./frontend
    ports:
      - "80:80"
    depends_on:
      - backend

volumes:
  postgres_data:
```

**Run:**
```bash
docker-compose up -d
```

---

## 📚 API Documentation

### Base URL
```
http://localhost:8000/api
```

### Authentication
Currently using `AllowAny` permissions for hackathon. In production, add JWT/Token authentication.

### Endpoints

#### Health Check
```http
GET /api/health/
```

**Response:**
```json
{
  "status": "healthy",
  "services": {
    "api": true,
    "ml_model": true,
    "semantic_search": true,
    "rag_service": true
  }
}
```

#### Create Ticket
```http
POST /api/tickets/
Content-Type: application/json

{
  "title": "Cannot access shared folder",
  "description": "Getting permission denied error",
  "email": "user@example.com",
  "priority": "medium"
}
```

**Response:**
```json
{
  "id": 1,
  "title": "Cannot access shared folder",
  "description": "Getting permission denied error",
  "predicted_category": "Folder Access",
  "confidence_score": 0.87,
  "priority": "medium",
  "status": "open",
  "created_at": "2024-02-12T10:30:00Z"
}
```

#### Get All Tickets
```http
GET /api/tickets/
```

#### Predict Category
```http
POST /api/predict/
Content-Type: application/json

{
  "title": "Email not syncing",
  "description": "Outlook won't sync with server"
}
```

**Response:**
```json
{
  "category": "Email Issues",
  "confidence": 0.92,
  "method": "machine_learning",
  "all_probabilities": {
    "Email Issues": 0.92,
    "VPN Issues": 0.04,
    "Hardware Issue": 0.02,
    ...
  }
}
```

#### Semantic Search
```http
POST /api/search/?k=5
Content-Type: application/json

{
  "title": "Database connection timeout",
  "description": "Cannot connect to Oracle database"
}
```

#### Generate Resolution
```http
POST /api/resolve/
Content-Type: application/json

{
  "title": "Printer not working",
  "description": "Print queue stuck",
  "use_llm": false
}
```

#### Complete Analysis
```http
POST /api/analyze/
Content-Type: application/json

{
  "title": "Server down",
  "description": "Production server not responding"
}
```

**Response:**
```json
{
  "prediction": {
    "category": "Hardware Issue",
    "confidence": 0.85,
    "method": "machine_learning"
  },
  "similar_tickets": [...],
  "resolution": {
    "text": "**Recommended Resolution Steps:**\n\n1. Check server status...",
    "method": "rule_based"
  }
}
```

#### Dashboard Statistics
```http
GET /api/stats/
```

**Full API documentation available at:**
- Swagger UI: http://localhost:8000/api/docs/
- ReDoc: http://localhost:8000/api/schema/

---

## 🐙 GitHub Setup

### Initial Setup

1. **Create GitHub Repository:**
   - Go to https://github.com
   - Click "New repository"
   - Name: `itsm-ai-hackathon`
   - Don't initialize with README (we have one)
   - Click "Create repository"

2. **Initialize Git:**
```bash
cd itsm-hackathon-fullstack
git init
git add .
git commit -m "Initial commit: Full-stack ITSM AI application"
```

3. **Connect to GitHub:**
```bash
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/itsm-ai-hackathon.git
git push -u origin main
```

### Important: Don't Commit Large Files

The `.gitignore` is configured to exclude:
- Model files (*.joblib, *.bin, *.npy) - too large
- Virtual environments
- Node modules
- Environment files (.env)
- Database files

### If Models are Already Tracked:
```bash
# Remove large files from git (keeps local copy)
git rm --cached ai-models/models/**/*.joblib
git rm --cached ai-models/models/**/*.bin
git rm --cached ai-models/models/**/*.npy

git commit -m "Remove large model files"
git push
```

### Sharing Models with Team

Use Git LFS for large files:
```bash
git lfs install
git lfs track "*.joblib"
git lfs track "*.bin"
git lfs track "*.npy"
git add .gitattributes
git commit -m "Add Git LFS tracking"
git push
```

Or upload models to cloud storage and download separately.

### Collaboration Workflow

```bash
# Create feature branch
git checkout -b feature/new-feature

# Make changes and commit
git add .
git commit -m "Add new feature"

# Push branch
git push origin feature/new-feature

# Create Pull Request on GitHub
# Merge when approved
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Database Connection Error

**Error:** `django.db.utils.OperationalError: could not connect to server`

**Solution:**
```bash
# Check PostgreSQL is running
sudo service postgresql status

# Start if not running
sudo service postgresql start

# Verify credentials in .env match database
# Try connecting manually:
psql -U itsm_user -d itsm_db
```

#### 2. CORS Error in Frontend

**Error:** `Access to XMLHttpRequest blocked by CORS policy`

**Solution:**
- Check `CORS_ALLOWED_ORIGINS` in `backend/.env`
- Should include `http://localhost:5173`
- Restart Django server after changing .env

#### 3. Models Not Loading

**Error:** `AI services not available`

**Solution:**
```bash
# Check if models exist
ls -la ai-models/models/baseline/
ls -la ai-models/models/embeddings/

# If missing, retrain:
cd ai-models
python src/main_pipeline.py
```

#### 4. Port Already in Use

**Error:** `Error: listen EADDRINUSE: address already in use :::5173`

**Solution:**
```bash
# Find process using port
# On Linux/Mac:
lsof -ti:5173 | xargs kill -9

# On Windows:
netstat -ano | findstr :5173
taskkill /PID <PID> /F

# Or use different port:
npm run dev -- --port 3000
```

#### 5. Module Not Found

**Error:** `ModuleNotFoundError: No module named 'rest_framework'`

**Solution:**
```bash
# Activate virtual environment
source venv/bin/activate  # or venv\Scripts\activate

# Reinstall dependencies
pip install -r requirements.txt
```

#### 6. Frontend Build Fails

**Error:** `npm ERR! Build failed`

**Solution:**
```bash
# Clear cache
rm -rf node_modules package-lock.json
npm install

# Or use different package manager
npm install --legacy-peer-deps
```

### Performance Issues

**Slow predictions:**
- Reduce model complexity in `ai-models/configs/config.py`
- Use smaller embedding models
- Enable caching for predictions

**High memory usage:**
- Reduce batch size for embeddings
- Use CPU instead of GPU if memory limited
- Close unused applications

---

## 📁 Project Structure

```
itsm-hackathon-fullstack/
│
├── backend/                      # Django Backend
│   ├── itsm_backend/             # Django project settings
│   │   ├── __init__.py
│   │   ├── settings.py           # Main settings
│   │   ├── urls.py               # URL routing
│   │   ├── wsgi.py               # WSGI config
│   │   └── asgi.py               # ASGI config
│   │
│   ├── tickets/                  # Tickets app
│   │   ├── models.py             # Database models
│   │   ├── admin.py              # Admin interface
│   │   ├── apps.py               # App configuration
│   │   └── __init__.py
│   │
│   ├── api/                      # API app
│   │   ├── views.py              # API views/endpoints
│   │   ├── serializers.py        # DRF serializers
│   │   ├── urls.py               # API routing
│   │   ├── ai_service.py         # ML model integration
│   │   ├── apps.py               # App configuration
│   │   └── __init__.py
│   │
│   ├── manage.py                 # Django management
│   ├── requirements.txt          # Python dependencies
│   ├── .env.example              # Environment template
│   └── .env                      # Environment variables (create this)
│
├── frontend/                     # React Frontend
│   ├── src/
│   │   ├── components/           # Reusable components
│   │   ├── pages/                # Page components
│   │   │   ├── Dashboard.jsx     # Dashboard page
│   │   │   ├── TicketList.jsx    # Ticket list page
│   │   │   ├── NewTicket.jsx     # Create ticket page
│   │   │   └── AnalyzeTicket.jsx # AI analysis page
│   │   ├── services/
│   │   │   └── api.js            # API client
│   │   ├── utils/                # Helper functions
│   │   ├── App.jsx               # Main app component
│   │   ├── main.jsx              # Entry point
│   │   └── index.css             # Global styles
│   │
│   ├── public/                   # Static assets
│   ├── index.html                # HTML template
│   ├── package.json              # Node dependencies
│   ├── vite.config.js            # Vite configuration
│   ├── tailwind.config.js        # Tailwind CSS config
│   ├── postcss.config.js         # PostCSS config
│   ├── .env.example              # Environment template
│   └── .env                      # Environment variables (create this)
│
├── ai-models/                    # AI/ML Models
│   ├── src/                      # ML source code
│   │   ├── data_preprocessing.py
│   │   ├── train_ml_models.py
│   │   ├── embeddings_and_search.py
│   │   ├── llm_rag.py
│   │   └── main_pipeline.py
│   │
│   ├── models/                   # Trained models
│   │   ├── baseline/             # ML models
│   │   │   ├── model.joblib
│   │   │   ├── tfidf.joblib
│   │   │   └── label_encoder.joblib
│   │   └── embeddings/           # Embeddings & FAISS
│   │       ├── sentence_embeddings.npy
│   │       └── faiss_index.bin
│   │
│   ├── data/                     # Dataset
│   │   ├── raw/                  # Original data
│   │   └── processed/            # Cleaned data
│   │
│   ├── configs/                  # Configuration
│   │   └── config.py
│   │
│   └── requirements.txt          # Python dependencies
│
├── docs/                         # Documentation
│   └── CODE_EXPLANATION.md       # Detailed code explanation
│
├── deployment/                   # Deployment configs
│   ├── render.yaml               # Render deployment
│   ├── docker-compose.yml        # Docker setup
│   └── nginx.conf                # Nginx config
│
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

---

## 🎓 Understanding the Code

Each file has detailed comments explaining:
- What it does
- How it works
- Why it's important

### Key Files to Understand:

1. **Backend:**
   - `backend/api/ai_service.py` - ML model integration
   - `backend/api/views.py` - API endpoints
   - `backend/tickets/models.py` - Database schema

2. **Frontend:**
   - `frontend/src/App.jsx` - Main app structure
   - `frontend/src/services/api.js` - API communication
   - `frontend/src/pages/*.jsx` - UI pages

3. **AI/ML:**
   - `ai-models/src/train_ml_models.py` - Model training
   - `ai-models/src/llm_rag.py` - RAG implementation

---

## 📊 Demo Data

### Sample Ticket Categories
- Password Reset
- Folder Access
- Slow Computer
- VPN Issues
- Email Issues
- Hardware Issue
- Software Installation

### Test Tickets to Try

1. **Password Reset:**
   - Title: "Cannot log into system"
   - Description: "Forgot my password and cannot access my account"

2. **Folder Access:**
   - Title: "Permission denied on shared drive"
   - Description: "Getting access denied when trying to open HR folder"

3. **Email Issues:**
   - Title: "Outlook not syncing"
   - Description: "Emails not downloading from server, stuck in outbox"

---

## 🏆 Hackathon Presentation Tips

1. **Start with Dashboard:**
   - Shows professional UI
   - Demonstrates analytics
   - Real-time data visualization

2. **Demo AI Features:**
   - Create new ticket with auto-classification
   - Show confidence scores
   - Demonstrate semantic search
   - Generate AI resolution

3. **Highlight Tech Stack:**
   - Full-stack application
   - Modern technologies
   - Production-ready architecture
   - Scalable design

4. **Show API Documentation:**
   - Swagger UI at `/api/docs/`
   - Well-documented endpoints
   - RESTful design

5. **Explain AI Pipeline:**
   - ML classification
   - Semantic search with FAISS
   - RAG for resolutions
   - Multiple model approach

---

## 📞 Support & Contact

**For Issues:**
- Check Troubleshooting section
- Review API documentation
- Check console logs (browser & server)
- Verify environment variables

**Hackathon Judges:**
- Live demo available at: [Your deployment URL]
- GitHub repository: [Your GitHub URL]
- API docs: [Your API URL]/docs/

---

## 📄 License

MIT License - Free to use for hackathon and beyond.

---

## 🙏 Acknowledgments

- **Anthropic** - Claude AI assistance
- **Lightning AI** - Hackathon organization
- **Open Source Community** - All the amazing libraries

---

**Built with ❤️ for Lightning AI Hackathon 2024**

**Good luck with your presentation! 🚀**
#   I n c i d A I  
 