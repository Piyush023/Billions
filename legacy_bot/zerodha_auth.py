#!/usr/bin/env python3
"""
Zerodha Authentication Module
Handles login, 2FA, and access token generation for Zerodha API
"""

import json
import webbrowser
import pyotp
from kiteconnect import KiteConnect
import logging
import sys
from urllib.parse import parse_qs, urlparse
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def load_config():
    """Load configuration from config.json"""
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ config.json not found. Please run setup.py first.")
        sys.exit(1)
    except json.JSONDecodeError:
        print("❌ Invalid config.json format")
        sys.exit(1)

def save_config(config):
    """Save updated configuration"""
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=4)

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

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/auth.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def get_access_token_interactive(config):
    """Get access token through interactive web login"""
    # zerodha_config = config['zerodha']
    api_key = os.getenv('ZERODHA_API_KEY')
    api_secret = os.getenv('ZERODHA_SECRET_KEY')
    
    if not api_key or not api_secret:
        colored_print("❌ API key or secret not found in config.json", "red")
        colored_print("Please run setup.py to configure your credentials", "yellow")
        return None
    
    kite = KiteConnect(api_key=api_key)
    
    # Generate login URL
    login_url = kite.login_url()
    
    colored_print("🌐 Opening Zerodha login page...", "blue")
    colored_print(f"If browser doesn't open, visit: {login_url}", "yellow")
    
    # Open browser
    webbrowser.open(login_url)
    
    print()
    colored_print("📋 Instructions:", "cyan")
    print("1. Login with your Zerodha credentials")
    print("2. Complete 2FA if required")
    print("3. After login, you'll be redirected to a URL with 'request_token'")
    print("4. Copy the complete redirected URL and paste it here")
    print()
    
    # Get redirect URL from user
    redirect_url = input("🔗 Paste the redirected URL here: ").strip()
    
    if not redirect_url:
        colored_print("❌ No URL provided", "red")
        return None
    
    try:
        # Parse request token from URL
        parsed_url = urlparse(redirect_url)
        query_params = parse_qs(parsed_url.query)
        request_token = query_params.get('request_token', [None])[0]
        
        if not request_token:
            colored_print("❌ No request_token found in URL", "red")
            return None
        
        colored_print(f"✅ Request token extracted: {request_token[:20]}...", "green")
        
        # Generate access token
        data = kite.generate_session(request_token, api_secret=api_secret)
        access_token = data["access_token"]
        
        colored_print("✅ Access token generated successfully!", "green")
        
        # Save access token to config
        config['zerodha']['access_token'] = access_token
        save_config(config)
        
        colored_print("💾 Access token saved to config.json", "green")
        
        # Test the connection
        kite.set_access_token(access_token)
        profile = kite.profile()
        
        colored_print(f"🎉 Successfully authenticated as: {profile['user_name']}", "green")
        colored_print(f"📧 Email: {profile['email']}", "blue")
        colored_print(f"🏢 Broker: {profile['broker']}", "blue")
        
        return access_token
        
    except Exception as e:
        colored_print(f"❌ Authentication failed: {str(e)}", "red")
        return None

def get_access_token_automated(config):
    """Get access token using saved credentials (if TOTP is configured)"""
    # zerodha_config = config['zerodha']
    api_key = os.getenv('ZERODHA_API_KEY')
    api_secret = os.getenv('ZERODHA_API_SECRET')
    user_id = os.getenv('ZERODHA_USER_ID')
    password = os.getenv('ZERODHA_PASSWORD')
    totp_key = os.getenv('ZERODHA_TOTP_KEY')
    
    if not all([api_key, api_secret, user_id, password, totp_key]):
        colored_print("⚠️  Automated login not configured. Using interactive mode.", "yellow")
        return get_access_token_interactive(config)
    
    try:
        kite = KiteConnect(api_key=api_key)
        
        # Generate TOTP
        totp = pyotp.TOTP(totp_key)
        current_totp = totp.now()
        
        colored_print("🔐 Performing automated login...", "blue")
        
        # This is a simplified example - actual implementation would need to handle
        # the complete login flow including session management
        colored_print("⚠️  Automated login requires advanced session handling.", "yellow")
        colored_print("💡 For now, using interactive mode.", "yellow")
        
        return get_access_token_interactive(config)
        
    except Exception as e:
        colored_print(f"❌ Automated login failed: {str(e)}", "red")
        colored_print("🔄 Falling back to interactive mode...", "yellow")
        return get_access_token_interactive(config)

def validate_existing_token(config):
    """Validate existing access token"""
    access_token = os.getenv('ZERODHA_ACCESS_TOKEN')
    api_key = os.getenv('ZERODHA_API_KEY')
    
    if not access_token:
        return False
    
    try:
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        
        # Test API call
        profile = kite.profile()
        colored_print(f"✅ Existing token valid for: {profile['user_name']}", "green")
        return True
        
    except Exception as e:
        colored_print(f"⚠️  Existing token invalid: {str(e)}", "yellow")
        return False

def main():
    """Main authentication flow"""
    colored_print("=" * 50, "cyan")
    colored_print("🔐 ZERODHA AUTHENTICATION SETUP", "cyan")
    colored_print("=" * 50, "cyan")
    print()
    
    # Setup logging
    logger = setup_logging()
    
    # Load configuration
    config = load_config()
    
    # Check if existing token is valid
    if validate_existing_token(config):
        colored_print("✅ You're already authenticated!", "green")
        
        choice = input("\n🔄 Generate new token anyway? (y/n): ").lower().strip()
        if choice != 'y':
            colored_print("👍 Using existing token", "green")
            return
    
    # Get new access token
    colored_print("🔑 Generating new access token...", "blue")
    
    # Check if automated login is configured
    if os.getenv('ZERODHA_TOTP_KEY'):
        choice = input("🤖 Use automated login? (y/n): ").lower().strip()
        if choice == 'y':
            access_token = get_access_token_automated(config)
        else:
            access_token = get_access_token_interactive(config)
    else:
        access_token = get_access_token_interactive(config)
    
    if access_token:
        colored_print("\n🎉 Authentication completed successfully!", "green")
        colored_print("💡 You can now start the trading bot with: python billions.py", "blue")
    else:
        colored_print("\n❌ Authentication failed", "red")
        colored_print("💡 Please check your credentials and try again", "yellow")

if __name__ == "__main__":
    main()
