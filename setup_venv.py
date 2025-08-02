#!/usr/bin/env python3
"""
Virtual Environment Setup Script for YouTube Video Intelligence App
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    
    if sys.version_info < (3, 8):
        print(f"Python {sys.version} is too old. Please use Python 3.8 or higher.")
        return False
    
    print(f"Python {sys.version_info.major}.{sys.version_info.minor} is compatible")
    return True

def create_venv():
    """Create virtual environment if it doesn't exist."""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("Virtual environment already exists")
        return True
    
    print("Creating virtual environment...")
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("Virtual environment created successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to create virtual environment: {e}")
        return False

def get_activate_script():
    """Get the appropriate activation script for the current platform."""
    if platform.system() == "Windows":
        return "venv\\Scripts\\activate"
    else:
        return "venv/bin/activate"

def install_dependencies():
    """Install dependencies in the virtual environment."""
    print("Installing dependencies...")
    
    # Get the pip path in the virtual environment
    if platform.system() == "Windows":
        pip_path = "venv\\Scripts\\pip"
    else:
        pip_path = "venv/bin/pip"
    
    try:
        # Upgrade pip first
        subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)
        print("Upgraded pip")
        
        # Install requirements
        subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
        print("Installed all dependencies")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        return False

def create_activation_scripts():
    """Create activation scripts for different shells."""
    print("Creating activation scripts...")
    
    # Create a simple activation script
    activate_script = """#!/bin/bash
# YouTube Video Intelligence App - Virtual Environment Activation
echo "Activating YouTube Video Intelligence App virtual environment..."
source venv/bin/activate
echo "Virtual environment activated!"
echo "To deactivate, run: deactivate"
"""
    
    with open("activate_env.sh", "w") as f:
        f.write(activate_script)
    
    # Make it executable
    os.chmod("activate_env.sh", 0o755)
    print("Created activate_env.sh")
    
    # Create Windows batch file
    windows_script = """@echo off
REM YouTube Video Intelligence App - Virtual Environment Activation
echo Activating YouTube Video Intelligence App virtual environment...
call venv\\Scripts\\activate.bat
echo Virtual environment activated!
echo To deactivate, run: deactivate
"""
    
    with open("activate_env.bat", "w") as f:
        f.write(windows_script)
    
    print("Created activate_env.bat")

def create_run_scripts():
    """Create run scripts that automatically activate the environment."""
    
    # Unix/Linux run script
    unix_run_script = """#!/bin/bash
# YouTube Video Intelligence App - Run Script
echo "Starting YouTube Video Intelligence App..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Please run setup_venv.py first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if dependencies are installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "Dependencies not installed. Please run setup_venv.py first."
    exit 1
fi

# Run the app
echo "Starting Streamlit app..."
streamlit run app/app.py --server.port=8501 --server.address=localhost
"""
    
    with open("run_app_venv.sh", "w") as f:
        f.write(unix_run_script)
    
    os.chmod("run_app_venv.sh", 0o755)
    print("Created run_app_venv.sh")
    
    # Windows run script
    windows_run_script = """@echo off
REM YouTube Video Intelligence App - Run Script
echo Starting YouTube Video Intelligence App...

REM Check if virtual environment exists
if not exist "venv" (
    echo Virtual environment not found. Please run setup_venv.py first.
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\\Scripts\\activate.bat

REM Check if dependencies are installed
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo Dependencies not installed. Please run setup_venv.py first.
    pause
    exit /b 1
)

REM Run the app
echo Starting Streamlit app...
streamlit run app/app.py --server.port=8501 --server.address=localhost
"""
    
    with open("run_app_venv.bat", "w") as f:
        f.write(windows_run_script)
    
    print("Created run_app_venv.bat")

def update_gitignore():
    """Update .gitignore to properly handle virtual environment."""
    gitignore_content = """# Environment variables
.env
.env.local
.env.production

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
MANIFEST

# Virtual environments
venv/
env/
ENV/
env.bak/
venv.bak/

# Keep venv activation scripts but exclude the actual environment
!venv/bin/activate
!venv/bin/activate.csh
!venv/bin/activate.fish

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Model files (large files)
models/*.pkl
models/*.joblib
models/t5_summary_model/

# Data files
data/*.csv
data/*.json
data/*.parquet

# Logs
*.log
logs/

# MLflow
mlruns/

# Jupyter
.ipynb_checkpoints

# Streamlit
.streamlit/secrets.toml

# Archive files
archive/
archive.zip
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    
    print("Updated .gitignore")

def main():
    """Main setup function."""
    print("YouTube Video Intelligence App - Virtual Environment Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create virtual environment
    if not create_venv():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Create activation scripts
    create_activation_scripts()
    
    # Create run scripts
    create_run_scripts()
    
    # Update gitignore
    update_gitignore()
    
    print("\n" + "=" * 60)
    print("Virtual environment setup completed successfully!")
    print("\nNext steps:")
    print("1. Activate the environment:")
    if platform.system() == "Windows":
        print("   activate_env.bat")
    else:
        print("   source activate_env.sh")
    print("\n2. Run the app:")
    if platform.system() == "Windows":
        print("   run_app_venv.bat")
    else:
        print("   ./run_app_venv.sh")
    print("\n3. Or manually:")
    print("   source venv/bin/activate  # (Unix/Linux)")
    print("   venv\\Scripts\\activate.bat  # (Windows)")
    print("   streamlit run app/app.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 