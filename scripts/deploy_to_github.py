#!/usr/bin/env python
"""
🔗 DEPLOY TO GITHUB SCRIPT
Automated GitHub deployment for African Agriculture Revolution
"""

import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def check_git_installed():
    """Check if git is installed"""
    try:
        subprocess.run(['git', '--version'], check=True, capture_output=True)
        return True
    except:
        return False

def initialize_git_repo():
    """Initialize Git repository"""
    print("📦 Initializing Git repository...")
    
    # Check if already a git repo
    if os.path.exists('.git'):
        print("✅ Git repository already exists")
        return True
    
    # Initialize git
    try:
        subprocess.run(['git', 'init'], check=True)
        print("✅ Git repository initialized")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize git: {e}")
        return False

def create_gitignore():
    """Create .gitignore file"""
    gitignore_content = """
# Data
data/processed/
data/raw/*.xlsx
data/raw/*.xls

# Results
results/
!results/README.md

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Jupyter
.ipynb_checkpoints

# OS
.DS_Store
Thumbs.db

# Models
production_models/
*.pkl
*.joblib

# Logs
*.log
logs/
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    
    print("✅ .gitignore file created")

def setup_git_config():
    """Setup git configuration"""
    print("⚙️  Setting up git configuration...")
    
    # Set user info
    subprocess.run(['git', 'config', 'user.email', 'erwin@example.com'], check=True)
    subprocess.run(['git', 'config', 'user.name', 'Erwin Nanyaro'], check=True)
    
    # Set default branch to main
    subprocess.run(['git', 'config', 'init.defaultBranch', 'main'], check=True)
    
    print("✅ Git configuration set")

def add_files_to_git():
    """Add all files to git"""
    print("📁 Adding files to git...")
    
    # Add all files
    subprocess.run(['git', 'add', '.'], check=True)
    
    # Check what's being added
    result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
    files_count = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
    
    print(f"✅ {files_count} files staged for commit")
    return files_count

def commit_changes(message=None):
    """Commit changes to git"""
    if not message:
        message = f"African Agriculture Revolution update - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    print(f"💾 Committing changes: {message}")
    
    try:
        subprocess.run(['git', 'commit', '-m', message], check=True)
        print("✅ Changes committed")
        return True
    except Exception as e:
        print(f"❌ Commit failed: {e}")
        return False

def setup_github_remote(repo_url):
    """Setup GitHub remote repository"""
    print(f"🔗 Setting up GitHub remote: {repo_url}")
    
    # Check if remote already exists
    result = subprocess.run(['git', 'remote', '-v'], capture_output=True, text=True)
    if 'origin' in result.stdout:
        print("✅ Remote 'origin' already exists")
        return True
    
    # Add remote
    try:
        subprocess.run(['git', 'remote', 'add', 'origin', repo_url], check=True)
        print("✅ GitHub remote added")
        return True
    except Exception as e:
        print(f"❌ Failed to add remote: {e}")
        return False

def push_to_github(branch='main'):
    """Push to GitHub"""
    print(f"🚀 Pushing to GitHub (branch: {branch})...")
    
    try:
        # Push to GitHub
        subprocess.run(['git', 'push', '-u', 'origin', branch], check=True)
        print("✅ Successfully pushed to GitHub!")
        return True
    except Exception as e:
        print(f"❌ Push failed: {e}")
        print("\n💡 TROUBLESHOOTING:")
        print("   1. Check your internet connection")
        print("   2. Verify GitHub repository URL")
        print("   3. Check GitHub authentication")
        print("   4. Make sure you have write permissions")
        return False

def create_readme():
    """Create README.md file"""
    readme_content = """# 🌍 African Agriculture Data Science Revolution

## Transforming African Agriculture Through Data Science

### 🎯 Mission
To revolutionize African agriculture by replacing outdated SPSS/Stata methods with modern data science pipelines that truly listen to farmers' voices through their data.

### 📊 What This Project Does
1. **Analyzes** Tanzanian farmer survey data (Morogoro strawberry + Arusha multi-crop)
2. **Quantifies** social capital and its impact on agricultural outcomes
3. **Predicts** which interventions will be most effective
4. **Generates** revenue through data-driven consulting services

### 🚀 Quick Start

#### Option 1: Google Colab (Recommended)
```python
# Run this in Colab
!git clone https://github.com/ErwinNanyaro/African-Agriculture-Revolution.git
%cd African-Agriculture-Revolution
!python scripts/setup_colab.py