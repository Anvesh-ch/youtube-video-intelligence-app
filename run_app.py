#!/usr/bin/env python3
"""
Quick start script for YouTube Video Intelligence App
"""

import os
import sys
import subprocess
import webbrowser
import time

def check_setup():
    """Check if the app is properly set up."""
    print("Checking setup...")
    
    # Check if models directory exists and has files
    models_dir = 'models'
    if not os.path.exists(models_dir):
        print("Models directory not found. Run setup.py first.")
        return False
    
    # Check if .env file exists
    env_file = '.env'
    if not os.path.exists(env_file):
        print(".env file not found. Run setup.py first.")
        return False
    
    print("Setup looks good!")
    return True

def run_streamlit():
    """Run the Streamlit app."""
    print("Starting YouTube Video Intelligence App...")
    
    try:
        # Run streamlit
        process = subprocess.Popen([
            sys.executable, '-m', 'streamlit', 'run', 'app/app.py',
            '--server.port=8501',
            '--server.address=localhost'
        ])
        
        # Wait a moment for the server to start
        time.sleep(3)
        
        # Open browser
        print("Opening browser...")
        webbrowser.open('http://localhost:8501')
        
        print("App is running!")
        print("Open your browser and go to: http://localhost:8501")
        print("Press Ctrl+C to stop the app")
        
        # Wait for the process to complete
        process.wait()
        
    except KeyboardInterrupt:
        print("\nStopping app...")
        if 'process' in locals():
            process.terminate()
        print("App stopped.")
    except Exception as e:
        print(f"Error running app: {e}")
        return False
    
    return True

def main():
    """Main function."""
    print("YouTube Video Intelligence App - Quick Start")
    print("=" * 50)
    
    # Check setup
    if not check_setup():
        print("\nPlease run setup first:")
        print("python setup.py")
        return False
    
    # Run the app
    return run_streamlit()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 