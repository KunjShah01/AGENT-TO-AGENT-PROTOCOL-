#!/usr/bin/env python3
"""
RL-A2A Setup Script
Automated installation and configuration for MVP deployment
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_python_version():
    """Ensure Python 3.8+ is being used"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current:", sys.version)
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def install_dependencies():
    """Install all required dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        sys.exit(1)

def setup_environment():
    """Create .env file if it doesn't exist"""
    env_file = Path(".env")
    if not env_file.exists():
        print("🔧 Creating .env configuration file...")
        env_content = """# RL-A2A Configuration
# Add your API keys here

# AI Provider API Keys
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
GOOGLE_API_KEY=your-google-api-key-here

# Ollama Configuration (for local models)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2

# Server Configuration
A2A_HOST=localhost
A2A_PORT=8000
DASHBOARD_PORT=8501

# Security Configuration
SECRET_KEY=your-secret-key-for-jwt-signing
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8501
RATE_LIMIT_PER_MINUTE=60

# System Configuration
DEBUG=false
MAX_AGENTS=100
DEFAULT_AI_PROVIDER=openai
"""
        env_file.write_text(env_content)
        print("✅ .env file created")
    else:
        print("✅ .env file already exists")

def run_tests():
    """Run basic functionality tests"""
    print("🧪 Running tests...")
    try:
        subprocess.check_call([sys.executable, "test_rla2a.py"])
        print("✅ All tests passed")
    except subprocess.CalledProcessError:
        print("⚠️ Some tests failed, but setup continues")

def main():
    """Main setup process"""
    print("🚀 RL-A2A MVP Setup")
    print("=" * 50)
    
    check_python_version()
    install_dependencies()
    setup_environment()
    run_tests()
    
    print("\n🎉 Setup Complete!")
    print("\nNext steps:")
    print("1. Edit .env file with your API keys")
    print("2. Run: python rla2a.py server --demo-agents 3")
    print("3. Run: python rla2a.py dashboard")
    print("4. Visit: http://localhost:8501")

if __name__ == "__main__":
    main()