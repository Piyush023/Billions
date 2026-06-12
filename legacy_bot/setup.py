#!/usr/bin/env python3
"""
Trading Bot Setup Assistant
Helps you configure and start your Zerodha trading bot
"""

import json
import os
import sys
import subprocess
from datetime import datetime

def colored_print(text, color="white"):
    """Print colored text"""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "purple": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m"
    }
    print(f"{colors.get(color, colors['white'])}{text}{colors['reset']}")

def print_header():
    """Print welcome header"""
    colored_print("="*60, "cyan")
    colored_print("🤖 ZERODHA TRADING BOT SETUP ASSISTANT", "cyan")
    colored_print("="*60, "cyan")
    print()

def check_dependencies():
    """Check if required packages are installed"""
    colored_print("📦 Checking dependencies...", "blue")
    
    # Map package names to their actual import names
    package_imports = {
        'kiteconnect': 'kiteconnect',
        'yfinance': 'yfinance', 
        'nsepy': 'nsepy',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'TA-Lib': 'talib',  # TA-Lib imports as 'talib'
        'python-telegram-bot': 'telegram',  # python-telegram-bot imports as 'telegram'
        'schedule': 'schedule',
        'pyotp': 'pyotp'
    }
    
    missing_packages = []
    
    for package, import_name in package_imports.items():
        try:
            __import__(import_name)
            colored_print(f"  ✅ {package}", "green")
        except ImportError:
            colored_print(f"  ❌ {package}", "red")
            missing_packages.append(package)
    
    if missing_packages:
        colored_print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}", "yellow")
        colored_print("💡 Run: pip install -r requirements.txt", "yellow")
        return False
    
    colored_print("\n✅ All dependencies installed!", "green")
    return True

def check_config():
    """Check if config.json exists and is valid"""
    colored_print("⚙️  Checking configuration...", "blue")
    
    if not os.path.exists('config.json'):
        colored_print("  ❌ config.json not found", "red")
        return False
    
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        # Check required fields
        required_fields = [
            'zerodha.api_key',
            'zerodha.api_secret', 
            'capital.total_capital',
            'trading_symbols'
        ]
        
        for field in required_fields:
            keys = field.split('.')
            value = config
            for key in keys:
                value = value.get(key, {})
            
            if not value:
                colored_print(f"  ❌ Missing: {field}", "red")
                return False
            else:
                colored_print(f"  ✅ {field}", "green")
        
        colored_print("\n✅ Configuration valid!", "green")
        return True
        
    except json.JSONDecodeError:
        colored_print("  ❌ Invalid JSON format", "red")
        return False

def setup_telegram():
    """Help setup Telegram bot"""
    colored_print("📱 Setting up Telegram notifications...", "blue")
    print()
    
    print("To get Telegram notifications:")
    print("1. Open Telegram and search for @BotFather")
    print("2. Send /newbot and follow instructions")
    print("3. Get your bot token")
    print("4. Message your bot, then visit:")
    print("   https://api.telegram.org/bot<TOKEN>/getUpdates")
    print("5. Find your chat_id in the response")
    print()
    
    token = input("Enter your Telegram bot token (or press Enter to skip): ").strip()
    if not token:
        colored_print("⏭️  Skipping Telegram setup", "yellow")
        return None, None
    
    chat_id = input("Enter your chat ID: ").strip()
    if not chat_id:
        colored_print("⏭️  Skipping Telegram setup", "yellow")
        return None, None
    
    colored_print("✅ Telegram configured!", "green")
    return token, chat_id

def update_config():
    """Update configuration with user input"""
    colored_print("⚙️  Updating configuration...", "blue")
    
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except:
        config = {}
    
    print("\nEnter your Zerodha API credentials:")
    api_key = input("API Key: ").strip()
    api_secret = input("API Secret: ").strip()
    user_id = input("User ID: ").strip()
    
    print("\nTrade setup:")
    try:
        capital = float(input("Total capital (₹): ").strip() or "10000")
        max_positions = int(input("Max positions (default 3): ").strip() or "3")
    except ValueError:
        capital = 10000
        max_positions = 3
    
    # Update config
    config.update({
        "zerodha": {
            "api_key": api_key,
            "api_secret": api_secret,
            "user_id": user_id,
            "password": "",
            "totp_key": "",
            "access_token": ""
        },
        "capital": {
            "total_capital": capital,
            "max_position_size": 0.20,
            "daily_loss_limit": 0.03,
            "stop_loss_pct": 0.05,
            "take_profit_pct": 0.08,
            "max_positions": max_positions,
            "min_trade_amount": 1000
        },
        "strategies": {
            "mean_reversion": {"active": True, "allocation": 0.5, "rsi_oversold": 30, "rsi_overbought": 70, "bb_period": 20},
            "momentum": {"active": True, "allocation": 0.5, "sma_fast": 20, "sma_slow": 50, "min_volume_ratio": 1.2}
        },
        "notifications": {
            "telegram_token": "",
            "chat_id": "",
            "send_daily_report": True,
            "send_trade_alerts": True
        },
        "trading_symbols": ["RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR", "ICICIBANK", "KOTAKBANK", "SBIN", "BHARTIARTL", "ITC"],
        "market_hours": {"start_time": "09:15", "end_time": "15:30", "trading_days": [0, 1, 2, 3, 4]},
        "risk_management": {"max_correlation": 0.7, "max_sector_exposure": 0.4, "trailing_stop_loss": False, "position_timeout_days": 5}
    })
    
    # Telegram setup
    token, chat_id = setup_telegram()
    if token and chat_id:
        config["notifications"]["telegram_token"] = token
        config["notifications"]["chat_id"] = chat_id
    
    # Save config
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    colored_print("✅ Configuration saved!", "green")

def run_auth_setup():
    """Run Zerodha authentication setup"""
    colored_print("🔐 Setting up Zerodha authentication...", "blue")
    
    if not os.path.exists('zerodha_auth.py'):
        colored_print("❌ zerodha_auth.py not found", "red")
        return False
    
    try:
        result = subprocess.run([sys.executable, 'zerodha_auth.py'], 
                              capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        colored_print(f"❌ Authentication setup failed: {e}", "red")
        return False

def test_connection():
    """Test bot connection"""
    colored_print("🧪 Testing bot connection...", "blue")
    
    if not os.path.exists('billions.py'):
        colored_print("❌ billions.py not found", "red")
        return False
    
    try:
        result = subprocess.run([sys.executable, 'billions.py', 'test'], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            colored_print("✅ Connection test passed!", "green")
            return True
        else:
            colored_print("❌ Connection test failed", "red")
            colored_print(result.stderr, "red")
            return False
            
    except subprocess.TimeoutExpired:
        colored_print("⏱️  Test timed out", "yellow")
        return False
    except Exception as e:
        colored_print(f"❌ Test failed: {e}", "red")
        return False

def start_bot():
    """Start the trading bot"""
    colored_print("🚀 Starting trading bot...", "blue")
    
    try:
        subprocess.run([sys.executable, 'billions.py'])
    except KeyboardInterrupt:
        colored_print("\n🛑 Bot stopped by user", "yellow")
    except Exception as e:
        colored_print(f"❌ Bot error: {e}", "red")

def main():
    """Main setup flow"""
    print_header()
    
    # Step 1: Check dependencies
    if not check_dependencies():
        colored_print("\n❌ Please install missing dependencies first", "red")
        colored_print("Run: pip install -r requirements.txt", "yellow")
        return
    
    print()
    
    # Step 2: Check/update configuration
    if not check_config():
        colored_print("⚙️  Setting up configuration...", "yellow")
        update_config()
    
    print()
    
    # Step 3: Zerodha authentication
    colored_print("🔐 Zerodha API authentication required", "yellow")
    choice = input("Run authentication setup? (y/n): ").lower().strip()
    
    if choice == 'y':
        if not run_auth_setup():
            colored_print("❌ Authentication failed. Please try again.", "red")
            return
    
    print()
    
    # Step 4: Test connection
    colored_print("🧪 Testing connection...", "yellow")
    choice = input("Run connection test? (y/n): ").lower().strip()
    
    if choice == 'y':
        if not test_connection():
            colored_print("❌ Connection test failed. Check your configuration.", "red")
            return
    
    print()
    
    # Step 5: Start bot
    colored_print("🚀 Ready to start trading bot!", "green")
    print("\nOptions:")
    print("1. Start bot now")
    print("2. Run test session only")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == '1':
        colored_print("\n🚀 Starting automated trading bot...", "green")
        colored_print("Press Ctrl+C to stop", "yellow")
        start_bot()
    elif choice == '2':
        colored_print("\n🧪 Running test session...", "blue")
        subprocess.run([sys.executable, 'billions.py', 'test'])
    else:
        colored_print("\n👋 Setup complete! Run 'python billions.py' to start trading.", "green")

if __name__ == "__main__":
    main()