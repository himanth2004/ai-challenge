#!/usr/bin/env python3
"""
AI Sentinel - Behavioral Analysis Application
Run script for the Flask web application
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from app import app
    print("✅ Successfully imported Flask app")
except ImportError as e:
    print(f"❌ Error importing Flask app: {e}")
    print("Make sure you're running this from the correct directory (ai-challenge/src)")
    sys.exit(1)

if __name__ == "__main__":
    print("🚀 Starting AI Sentinel Behavioral Analysis Application...")
    print("📍 Application will be available at: http://localhost:5000")
    print("📊 Command Center: http://localhost:5000/main")
    print("ℹ️  About: http://localhost:5000/about")
    print("🧪 Test endpoint: http://localhost:5000/test")
    print("=" * 60)
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"❌ Error starting Flask app: {e}")
        sys.exit(1)
