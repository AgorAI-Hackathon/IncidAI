# GitHub Push Instructions

## Step 1: Create GitHub Repository

1. Go to https://github.com
2. Click "New" button (green button, top right)
3. Repository name: `itsm-ai-hackathon`
4. Description: `AI-Powered ITSM Ticket Classification - Full-Stack Hackathon Project`
5. Select: **Public** (so judges can see it)
6. **DO NOT** check "Initialize with README" (we already have one)
7. Click "Create repository"

## Step 2: Initialize Git (if not done)

Open terminal in project root directory:

```bash
cd itsm-hackathon-fullstack
git init
```

## Step 3: Add All Files

```bash
git add .
```

This will stage all files except those in `.gitignore`

## Step 4: Make First Commit

```bash
git commit -m "Initial commit: Full-stack ITSM AI application

Features:
- Django REST API backend
- React + Vite frontend
- PostgreSQL database
- ML/DL/LLM integration
- Semantic search with FAISS
- RAG-based resolutions
- Interactive dashboard
"
```

## Step 5: Connect to GitHub

Replace `YOUR_USERNAME` with your GitHub username:

```bash
git remote add origin https://github.com/YOUR_USERNAME/itsm-ai-hackathon.git
```

Verify:
```bash
git remote -v
```

Should show:
```
origin  https://github.com/YOUR_USERNAME/itsm-ai-hackathon.git (fetch)
origin  https://github.com/YOUR_USERNAME/itsm-ai-hackathon.git (push)
```

## Step 6: Set Default Branch

```bash
git branch -M main
```

## Step 7: Push to GitHub

```bash
git push -u origin main
```

You'll be prompted for GitHub credentials.

### If using personal access token:
1. Go to GitHub.com → Settings → Developer settings → Personal access tokens
2. Generate new token (classic)
3. Select scopes: `repo` (all)
4. Copy token
5. Use token as password when pushing

### If using SSH:
```bash
# Generate SSH key if you don't have one
ssh-keygen -t ed25519 -C "your_email@example.com"

# Copy public key
cat ~/.ssh/id_ed25519.pub

# Add to GitHub: Settings → SSH and GPG keys → New SSH key

# Change remote to SSH
git remote set-url origin git@github.com:YOUR_USERNAME/itsm-ai-hackathon.git

# Push
git push -u origin main
```

## Step 8: Verify on GitHub

1. Go to https://github.com/YOUR_USERNAME/itsm-ai-hackathon
2. You should see all files
3. README.md should display automatically

## Handling Large Files

If you get error about large files:

```bash
# Remove large model files from git history
git rm --cached ai-models/models/**/*.joblib
git rm --cached ai-models/models/**/*.bin
git rm --cached ai-models/models/**/*.npy
git rm --cached ai-models/data/raw/*.csv
git rm --cached ai-models/data/processed/*.csv

# Commit removal
git commit -m "Remove large files from tracking"

# Push again
git push -u origin main
```

### Using Git LFS for Large Files

If you want to include model files:

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.joblib"
git lfs track "*.bin"
git lfs track "*.npy"
git lfs track "*.csv"

# Add gitattributes
git add .gitattributes

# Add files
git add ai-models/models/

# Commit and push
git commit -m "Add ML models via Git LFS"
git push
```

## Making Updates

After making changes:

```bash
# Check what changed
git status

# Stage changes
git add .

# Commit with message
git commit -m "Description of changes"

# Push to GitHub
git push
```

## Creating Release for Judges

```bash
# Tag this version
git tag -a v1.0-hackathon -m "Hackathon submission version"

# Push tag
git push origin v1.0-hackathon
```

On GitHub:
1. Go to repository
2. Click "Releases" → "Create a new release"
3. Choose tag: v1.0-hackathon
4. Title: "ITSM AI - Hackathon Submission"
5. Description: Copy features from README
6. Attach files: Optionally attach pre-trained models as zip
7. Click "Publish release"

## Share with Judges

Your repository URL:
```
https://github.com/YOUR_USERNAME/itsm-ai-hackathon
```

Live demo (after deployment):
```
Frontend: https://your-app.vercel.app
Backend API: https://your-api.render.com
API Docs: https://your-api.render.com/api/docs/
```

## Collaboration

Add collaborators:
1. Repository → Settings → Collaborators
2. Add team members
3. They can clone with:
```bash
git clone https://github.com/YOUR_USERNAME/itsm-ai-hackathon.git
cd itsm-ai-hackathon
# Follow setup instructions in README.md
```

## Troubleshooting

**Authentication failed:**
- Use personal access token instead of password
- Or set up SSH keys

**File too large:**
- Remove from git: `git rm --cached filename`
- Add to .gitignore
- Consider Git LFS

**Merge conflicts:**
```bash
# Pull latest changes
git pull origin main

# Fix conflicts in files
# Then:
git add .
git commit -m "Resolve merge conflicts"
git push
```

**Reset if needed:**
```bash
# Careful! This erases uncommitted changes
git reset --hard HEAD
```

## Best Practices

1. **Commit often** with clear messages
2. **Don't commit:**
   - Sensitive data (.env files)
   - Large binary files (unless using LFS)
   - Generated files (node_modules, venv)
3. **Use branches** for new features
4. **Write good README** (already done!)
5. **Add LICENSE** file
6. **Include .gitignore** (already done!)

## Example Workflow

```bash
# 1. Create feature branch
git checkout -b feature/analytics-dashboard

# 2. Make changes
# ... edit files ...

# 3. Commit changes
git add .
git commit -m "Add analytics dashboard with charts"

# 4. Push branch
git push origin feature/analytics-dashboard

# 5. Create Pull Request on GitHub
# 6. Review and merge
# 7. Switch back to main
git checkout main
git pull origin main
```

---

**Your code is now safely on GitHub and ready to share with judges! 🎉**
