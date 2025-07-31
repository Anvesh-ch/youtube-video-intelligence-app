#!/usr/bin/env python3
"""
Setup script for YouTube Video Intelligence App
"""

import os
import sys
import subprocess
import shutil

def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")
    
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'scikit-learn', 'transformers',
        'torch', 'sentence-transformers', 'google-api-python-client', 'yt-dlp'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} (missing)")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install missing packages with: pip install -r requirements.txt")
        return False
    
    print("All dependencies are installed!")
    return True

def create_directories():
    """Create necessary directories."""
    print("Creating directories...")
    
    directories = ['models', 'data', 'logs']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created {directory}/")
    
    return True

def train_models():
    """Train the ML models."""
    print("Training ML models...")
    
    try:
        # Run the training script
        result = subprocess.run([
            sys.executable, 'scripts/train_models.py'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Models trained successfully!")
            return True
        else:
            print(f"✗ Model training failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ Error training models: {e}")
        return False

def create_env_file():
    """Create .env file if it doesn't exist."""
    env_file = '.env'
    env_example = 'env.example'
    
    if not os.path.exists(env_file) and os.path.exists(env_example):
        print("Creating .env file from template...")
        shutil.copy(env_example, env_file)
        print("✓ Created .env file")
        print("⚠️  Please edit .env file and add your YouTube API key")
    elif os.path.exists(env_file):
        print("✓ .env file already exists")
    else:
        print("⚠️  No .env template found")

def main():
    """Main setup function."""
    print("YouTube Video Intelligence App - Setup")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        print("\nSetup failed: Missing dependencies")
        return False
    
    # Create directories
    if not create_directories():
        print("\nSetup failed: Could not create directories")
        return False
    
    # Create .env file
    create_env_file()
    
    # Train models
    if not train_models():
        print("\nSetup failed: Could not train models")
        return False
    
    print("\n" + "=" * 40)
    print("Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit .env file and add your YouTube API key")
    print("2. Run the app: streamlit run app/app.py")
    print("3. Open your browser and go to http://localhost:8501")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 