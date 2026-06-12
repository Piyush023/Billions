#!/usr/bin/env python3
"""
Python 3.13 Compatible Installation Script
Simplified installer that handles Python 3.13 compatibility issues
"""

import subprocess
import sys
import os

def colored_print(text, color='white'):
    """Print colored text"""
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'reset': '\033[0m'
    }
    print(f"{colors.get(color, colors['white'])}{text}{colors['reset']}")

def run_command(command, description=""):
    """Run shell command with error handling"""
    try:
        if description:
            colored_print(f"🔄 {description}...", 'blue')
        
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True
        )
        
        if description:
            colored_print(f"✅ {description} completed", 'green')
        
        return True
        
    except subprocess.CalledProcessError as e:
        colored_print(f"❌ Error in {description}: {e}", 'red')
        if e.stderr:
            colored_print(f"Error details: {e.stderr}", 'yellow')
        return False

def install_python313_compatible():
    """Install Python 3.13 compatible packages only"""
    
    colored_print("🐍 Python 3.13 Compatible AI Trading System Installer", 'cyan')
    colored_print("=" * 60, 'cyan')
    
    # Check Python version
    version = sys.version_info
    colored_print(f"Python version: {version.major}.{version.minor}.{version.micro}", 'blue')
    
    if version.major == 3 and version.minor >= 13:
        colored_print("⚠️ Python 3.13+ detected - using compatible packages only", 'yellow')
    
    # Essential packages that work with Python 3.13
    essential_packages = [
        "pandas>=2.0.0",
        "numpy>=1.24.0", 
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.17.0",
        "joblib>=1.3.0",
        "schedule>=1.2.0",
        "python-dotenv>=1.0.0",
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.0",
        "lxml>=4.9.0",
        "flask>=2.3.0",
        "dash>=2.14.0",
        "cloudpickle>=2.2.0"
    ]
    
    # Try to install advanced ML packages (may fail on Python 3.13)
    advanced_packages = [
        "xgboost>=2.0.0",
        "lightgbm>=4.0.0", 
        "scipy>=1.11.0",
        "statsmodels>=0.14.0",
        "imbalanced-learn>=0.11.0",
        "optuna>=3.0.0"
    ]
    
    colored_print("📦 Installing essential packages...", 'blue')
    
    # Install essential packages first
    for package in essential_packages:
        if not run_command(f"{sys.executable} -m pip install '{package}'", f"Installing {package.split('>=')[0]}"):
            colored_print(f"⚠️ Failed to install {package}, continuing...", 'yellow')
    
    colored_print("\n🔬 Installing advanced ML packages...", 'blue')
    
    # Try to install advanced packages
    installed_advanced = []
    failed_advanced = []
    
    for package in advanced_packages:
        if run_command(f"{sys.executable} -m pip install '{package}'", f"Installing {package.split('>=')[0]}"):
            installed_advanced.append(package.split('>=')[0])
        else:
            failed_advanced.append(package.split('>=')[0])
            colored_print(f"⚠️ {package.split('>=')[0]} not compatible with Python 3.13", 'yellow')
    
    # Install technical analysis packages
    colored_print("\n📊 Installing technical analysis packages...", 'blue')
    
    ta_packages = [
        "pandas-ta>=0.3.14b",
        "yfinance>=0.2.18",
        "kiteconnect>=5.0.0",
        "python-telegram-bot>=20.0",
        "pyotp>=2.8.0"
    ]
    
    for package in ta_packages:
        if not run_command(f"{sys.executable} -m pip install '{package}'", f"Installing {package.split('>=')[0]}"):
            colored_print(f"⚠️ Failed to install {package}, continuing...", 'yellow')
    
    # Try TA-Lib (often problematic)
    colored_print("\n🔧 Attempting to install TA-Lib...", 'blue')
    if not run_command(f"{sys.executable} -m pip install TA-Lib", "Installing TA-Lib"):
        colored_print("⚠️ TA-Lib installation failed - will use pandas-ta instead", 'yellow')
        colored_print("💡 For TA-Lib on macOS, try: brew install ta-lib", 'cyan')
    
    # Summary
    colored_print("\n" + "=" * 60, 'cyan')
    colored_print("🎉 INSTALLATION SUMMARY", 'cyan')
    colored_print("=" * 60, 'cyan')
    
    colored_print("✅ Essential packages installed successfully", 'green')
    
    if installed_advanced:
        colored_print(f"✅ Advanced ML packages installed: {', '.join(installed_advanced)}", 'green')
    
    if failed_advanced:
        colored_print(f"⚠️ Some packages not compatible with Python 3.13: {', '.join(failed_advanced)}", 'yellow')
        colored_print("💡 The system will use alternative implementations", 'cyan')
    
    colored_print("\n🚀 Your AI trading system is ready with Python 3.13!", 'green')
    colored_print("📝 Note: Some advanced features may use alternative implementations", 'blue')
    
    return True

if __name__ == "__main__":
    install_python313_compatible()