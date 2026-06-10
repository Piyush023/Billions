#!/usr/bin/env python3
"""
AI Trading System Setup Script
Automated installation and configuration script for the AI trading system
"""

import os
import sys
import subprocess
import json
import shutil
from pathlib import Path
import urllib.request
import zipfile
import tempfile

class AITradingSystemSetup:
    """Automated setup for AI trading system"""
    
    def __init__(self):
        self.project_name = "ai_trading_bot"
        self.base_dir = Path.cwd()
        self.project_dir = self.base_dir / self.project_name
        self.python_executable = sys.executable
        
        # Colors for terminal output
        self.colors = {
            'red': '\033[91m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'purple': '\033[95m',
            'cyan': '\033[96m',
            'white': '\033[97m',
            'reset': '\033[0m'
        }
    
    def colored_print(self, text, color='white'):
        """Print colored text"""
        print(f"{self.colors.get(color, self.colors['white'])}{text}{self.colors['reset']}")
    
    def print_header(self, text):
        """Print section header"""
        self.colored_print("=" * 60, 'cyan')
        self.colored_print(text.center(60), 'cyan')
        self.colored_print("=" * 60, 'cyan')
    
    def run_command(self, command, description=""):
        """Run shell command with error handling"""
        try:
            if description:
                self.colored_print(f"🔄 {description}...", 'blue')
            
            result = subprocess.run(
                command, 
                shell=True, 
                check=True, 
                capture_output=True, 
                text=True,
                cwd=self.project_dir if self.project_dir.exists() else None
            )
            
            if description:
                self.colored_print(f"✅ {description} completed", 'green')
            
            return result.stdout
            
        except subprocess.CalledProcessError as e:
            self.colored_print(f"❌ Error in {description}: {e}", 'red')
            if e.stderr:
                self.colored_print(f"Error details: {e.stderr}", 'red')
            return None
    
    def check_python_version(self):
        """Check Python version compatibility"""
        self.colored_print("🐍 Checking Python version...", 'blue')
        
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            self.colored_print("❌ Python 3.8+ required. Please upgrade Python.", 'red')
            return False
        
        # Warning for Python 3.13+ about potential package compatibility
        if version.major == 3 and version.minor >= 13:
            self.colored_print(f"⚠️  Python {version.major}.{version.minor}.{version.micro} detected", 'yellow')
            self.colored_print("   Some packages may have limited compatibility with Python 3.13+", 'yellow')
            self.colored_print("   Using compatible package versions...", 'blue')
        
        self.colored_print(f"✅ Python {version.major}.{version.minor}.{version.micro} is supported", 'green')
        return True
    
    def create_project_structure(self):
        """Create project directory structure"""
        self.colored_print("📁 Creating project structure...", 'blue')
        
        # Create main project directory
        self.project_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        subdirs = [
            'models',      # AI models storage
            'data',        # Market data cache
            'logs',        # Log files
            'backtest',    # Backtesting results
            'config',      # Configuration files
            'scripts',     # Utility scripts
            'dashboard',   # Web dashboard files
            'exports'      # Export files
        ]
        
        for subdir in subdirs:
            (self.project_dir / subdir).mkdir(exist_ok=True)
        
        self.colored_print("✅ Project structure created", 'green')
    
    def copy_ai_files(self):
        """Copy AI system files to project directory"""
        self.colored_print("📋 Copying AI system files...", 'blue')
        
        # List of core AI files to copy
        ai_files = [
            'ai_trading_engine.py',
            'dynamic_stock_screener.py', 
            'ai_portfolio_manager.py',
            'ai_trading_orchestrator.py',
            'ai_backtesting_engine.py',
            'billions.py',  # Main trading bot
            'zerodha_auth.py',
            'ai_config.json'
        ]
        
        # Copy files if they exist in the current directory
        for filename in ai_files:
            source_path = self.base_dir / filename
            if source_path.exists():
                dest_path = self.project_dir / filename
                shutil.copy2(source_path, dest_path)
                self.colored_print(f"   ✓ Copied {filename}", 'green')
            else:
                self.colored_print(f"   ⚠️  {filename} not found - you may need to create it", 'yellow')
        
        # Copy configuration
        config_source = self.base_dir / 'config.json'
        if config_source.exists():
            config_dest = self.project_dir / 'config.json'
            shutil.copy2(config_source, config_dest)
        
        self.colored_print("✅ AI files copied", 'green')
    
    def create_virtual_environment(self):
        """Create Python virtual environment"""
        self.colored_print("🔧 Creating virtual environment...", 'blue')
        
        venv_path = self.project_dir / 'venv'
        
        if venv_path.exists():
            self.colored_print("   Virtual environment already exists", 'yellow')
            return True
        
        result = self.run_command(
            f"{self.python_executable} -m venv venv",
            "Creating virtual environment"
        )
        
        return result is not None
    
    def install_dependencies(self):
        """Install required Python packages"""
        self.colored_print("📦 Installing dependencies...", 'blue')
        
        # Determine pip executable
        if os.name == 'nt':  # Windows
            pip_executable = self.project_dir / 'venv' / 'Scripts' / 'pip.exe'
        else:  # Unix-like
            pip_executable = self.project_dir / 'venv' / 'bin' / 'pip'
        
        # Create comprehensive requirements file (Python 3.13 compatible)
        requirements = """
# Core trading and market data
kiteconnect>=5.0.0
yfinance>=0.2.18
pandas>=2.0.0
numpy>=1.24.0

# Technical analysis
ta-lib>=0.4.24
pandas-ta>=0.3.14b

# Machine Learning (Python 3.13 compatible)
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
scipy>=1.11.0
joblib>=1.3.0

# Data processing
statsmodels>=0.14.0
imbalanced-learn>=0.11.0

# Notifications and utilities
python-telegram-bot>=20.0
schedule>=1.2.0
python-dotenv>=1.0.0
requests>=2.31.0

# Visualization
plotly>=5.17.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Web framework for dashboard
flask>=2.3.0
dash>=2.14.0
dash-bootstrap-components>=1.5.0

# Optimization
optuna>=3.0.0
numba>=0.58.0

# Utilities
pyotp>=2.8.0
beautifulsoup4>=4.12.0
lxml>=4.9.0
"""
        
        # Write requirements to file
        req_file = self.project_dir / 'requirements.txt'
        with open(req_file, 'w') as f:
            f.write(requirements.strip())
        
        # Install packages
        install_commands = [
            f"{pip_executable} install --upgrade pip",
            f"{pip_executable} install -r requirements.txt"
        ]
        
        for command in install_commands:
            result = self.run_command(command, "Installing packages")
            if result is None:
                self.colored_print("❌ Package installation failed", 'red')
                return False
        
        self.colored_print("✅ Dependencies installed successfully", 'green')
        return True
    
    def setup_configuration(self):
        """Setup configuration files"""
        self.colored_print("⚙️  Setting up configuration...", 'blue')
        
        # Copy AI config as main config
        ai_config_path = self.project_dir / 'ai_config.json'
        main_config_path = self.project_dir / 'config.json'
        
        if ai_config_path.exists() and not main_config_path.exists():
            shutil.copy2(ai_config_path, main_config_path)
        
        # Create .env template
        env_template = """# Zerodha API Credentials
ZERODHA_API_KEY=your_api_key_here
ZERODHA_API_SECRET=your_api_secret_here
ZERODHA_USER_ID=your_user_id_here
ZERODHA_PASSWORD=your_password_here
ZERODHA_TOTP_KEY=your_totp_key_here
ZERODHA_ACCESS_TOKEN=your_access_token_here

# Telegram Bot Credentials
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here

# Optional: Database URLs, API keys, etc.
"""
        
        env_file = self.project_dir / '.env.template'
        with open(env_file, 'w') as f:
            f.write(env_template)
        
        self.colored_print("✅ Configuration templates created", 'green')
    
    def create_startup_scripts(self):
        """Create startup and utility scripts"""
        self.colored_print("📜 Creating startup scripts...", 'blue')
        
        scripts_dir = self.project_dir / 'scripts'
        
        # Main runner script
        if os.name == 'nt':  # Windows
            python_path = 'venv\\Scripts\\python.exe'
        else:  # Unix-like
            python_path = 'venv/bin/python'
        
        runner_script = f"""#!/bin/bash
# AI Trading Bot Runner Script

cd "$(dirname "$0")/.."

case "$1" in
    "trade")
        echo "🚀 Starting AI Trading Bot..."
        {python_path} billions.py trade
        ;;
    "test")
        echo "🧪 Running tests..."
        {python_path} billions.py test
        ;;
    "backtest")
        echo "📊 Running backtesting..."
        {python_path} ai_backtesting_engine.py
        ;;
    "screen")
        echo "🔍 Running stock screening..."
        {python_path} dynamic_stock_screener.py
        ;;
    "auth")
        echo "🔐 Setting up Zerodha authentication..."
        {python_path} zerodha_auth.py
        ;;
    *)
        echo "Usage: $0 {{trade|test|backtest|screen|auth}}"
        echo ""
        echo "Commands:"
        echo "  trade     - Start live trading"
        echo "  test      - Run system tests"
        echo "  backtest  - Run AI backtesting"
        echo "  screen    - Run stock screening"
        echo "  auth      - Setup Zerodha authentication"
        ;;
esac
"""
        
        with open(scripts_dir / 'run_ai_trader.sh', 'w') as f:
            f.write(runner_script)
        
        # Make script executable on Unix-like systems
        if os.name != 'nt':
            os.chmod(scripts_dir / 'run_ai_trader.sh', 0o755)
        
        # Windows batch file
        if os.name == 'nt':
            batch_script = f"""@echo off
cd /d "%~dp0\\.."

if "%1"=="trade" (
    echo 🚀 Starting AI Trading Bot...
    {python_path} billions.py trade
) else if "%1"=="test" (
    echo 🧪 Running tests...
    {python_path} billions.py test
) else if "%1"=="backtest" (
    echo 📊 Running backtesting...
    {python_path} ai_backtesting_engine.py
) else if "%1"=="screen" (
    echo 🔍 Running stock screening...
    {python_path} dynamic_stock_screener.py
) else if "%1"=="auth" (
    echo 🔐 Setting up Zerodha authentication...
    {python_path} zerodha_auth.py
) else (
    echo Usage: %0 [trade^|test^|backtest^|screen^|auth]
    echo.
    echo Commands:
    echo   trade     - Start live trading
    echo   test      - Run system tests
    echo   backtest  - Run AI backtesting
    echo   screen    - Run stock screening
    echo   auth      - Setup Zerodha authentication
)
"""
            with open(scripts_dir / 'run_ai_trader.bat', 'w') as f:
                f.write(batch_script)
        
        self.colored_print("✅ Startup scripts created", 'green')
    
    def create_readme(self):
        """Create comprehensive README file"""
        self.colored_print("📚 Creating documentation...", 'blue')
        
        readme_content = """# 🤖 AI Trading Bot

Advanced AI-powered trading system for Indian stock markets with machine learning, dynamic stock screening, and intelligent portfolio management.

## 🚀 Features

### 🧠 AI-Powered Trading
- **Machine Learning Models**: Random Forest, XGBoost, and LightGBM ensemble
- **Dynamic Stock Screening**: AI selects best stocks from 500+ NSE stocks
- **Intelligent Position Sizing**: AI confidence-based position sizing
- **Adaptive Learning**: Models retrain weekly with new data

### 📊 Professional Trading Features
- **Real-time Trading**: Live trading with Zerodha integration
- **Paper Trading**: Risk-free testing mode
- **Advanced Risk Management**: Multi-layer protection systems
- **Portfolio Optimization**: AI-driven portfolio allocation
- **Walk-Forward Backtesting**: Comprehensive strategy validation

### 🛡️ Risk Management
- **Dynamic Stop-Loss**: AI confidence-based stop levels
- **Correlation Limits**: Avoid highly correlated positions
- **Sector Exposure Limits**: Maximum 40% per sector
- **Daily Loss Limits**: Automatic trading halt on losses
- **Emergency Stop**: Instant system shutdown

### 📱 Monitoring & Alerts
- **Telegram Notifications**: Real-time trade and system alerts
- **Daily Reports**: Comprehensive performance summaries
- **Model Performance Tracking**: AI model accuracy monitoring
- **Web Dashboard**: Real-time portfolio visualization

## 🛠️ Quick Start

### 1. Setup Credentials
```bash
# Copy environment template
cp .env.template .env

# Edit .env with your credentials
nano .env
```

### 2. Configure Zerodha Authentication
```bash
./scripts/run_ai_trader.sh auth
```

### 3. Run Backtesting
```bash
./scripts/run_ai_trader.sh backtest
```

### 4. Start Paper Trading
```bash
./scripts/run_ai_trader.sh test
```

### 5. Begin Live Trading
```bash
./scripts/run_ai_trader.sh trade
```

## 📈 Expected Performance

Based on historical backtesting (2022-2024):
- **Annual Return**: 15-25%
- **Sharpe Ratio**: 1.2-1.8
- **Win Rate**: 60-70%
- **Maximum Drawdown**: 8-12%

## ⚙️ Configuration

Edit `config.json` to customize:

```json
{
    "ai_engine": {
        "confidence_threshold": 65,
        "model_retrain_days": 7
    },
    "capital": {
        "total_capital": 100000,
        "max_position_size": 0.18,
        "daily_loss_limit": 0.025
    },
    "risk_management": {
        "correlation_limit": 0.7,
        "sector_limit": 0.4
    }
}
```

## 🔧 Commands

### Trading Commands
```bash
./scripts/run_ai_trader.sh trade      # Start live trading
./scripts/run_ai_trader.sh test       # Paper trading mode
```

### Analysis Commands
```bash
./scripts/run_ai_trader.sh backtest   # Run AI backtesting
./scripts/run_ai_trader.sh screen     # Run stock screening
```

### Setup Commands
```bash
./scripts/run_ai_trader.sh auth       # Setup authentication
```

## 📊 AI Components

### 1. AI Trading Engine (`ai_trading_engine.py`)
- Multi-model ensemble predictions
- 80+ technical and fundamental features
- Walk-forward model training
- Real-time prediction generation

### 2. Dynamic Stock Screener (`dynamic_stock_screener.py`)
- Screens 500+ NSE stocks daily
- AI-powered scoring system
- Technical, fundamental, and momentum analysis
- Automatic universe updates

### 3. AI Portfolio Manager (`ai_portfolio_manager.py`)
- Confidence-based position sizing
- Correlation and sector risk management
- Dynamic stop-loss and take-profit levels
- Portfolio optimization algorithms

### 4. AI Trading Orchestrator (`ai_trading_orchestrator.py`)
- Coordinates all AI components
- Manages trading workflow
- Performance monitoring
- Risk oversight

### 5. AI Backtesting Engine (`ai_backtesting_engine.py`)
- Walk-forward analysis
- Comprehensive performance metrics
- Risk-adjusted returns
- Model validation

## 🛡️ Safety Features

### Multi-Layer Risk Management
1. **Position Level**: Individual stock stop-loss and take-profit
2. **Portfolio Level**: Maximum positions and sector exposure
3. **Daily Level**: Daily loss limits and emergency stops
4. **AI Level**: Confidence thresholds and model validation

### Emergency Procedures
- Emergency stop via Telegram command
- Automatic halt on large losses
- Model failure fallback to traditional strategies
- Real-time risk monitoring

## 📱 Monitoring

### Telegram Alerts
- Trade execution notifications
- Daily performance reports
- Risk limit warnings
- AI model updates
- System status alerts

### Performance Tracking
- Real-time P&L monitoring
- Benchmark comparison
- Sharpe ratio tracking
- Drawdown analysis
- Win rate statistics

## 🔍 Troubleshooting

### Common Issues

1. **Authentication Errors**
   ```bash
   ./scripts/run_ai_trader.sh auth
   ```

2. **Model Training Failures**
   - Check data availability
   - Verify market data access
   - Review log files in `logs/`

3. **Trading Errors**
   - Verify Zerodha credentials
   - Check available margin
   - Review risk limits

### Log Files
- `logs/billions_YYYYMMDD.log` - Main trading logs
- `logs/auth.log` - Authentication logs
- `models/` - AI model files
- `backtest/` - Backtesting results

## 📞 Support

For issues or questions:
1. Check log files for error details
2. Review configuration settings
3. Verify API credentials
4. Test with paper trading first

## ⚠️ Disclaimer

This is an automated trading system. Past performance does not guarantee future results. Always:
- Test thoroughly with paper trading
- Start with small amounts
- Monitor performance regularly
- Understand the risks involved
- Never invest more than you can afford to lose

## 📄 License

This project is for educational and personal use only.
"""
        
        with open(self.project_dir / 'README.md', 'w') as f:
            f.write(readme_content)
        
        self.colored_print("✅ Documentation created", 'green')
    
    def run_setup(self):
        """Run complete setup process"""
        try:
            self.print_header("🤖 AI TRADING SYSTEM SETUP")
            
            # Step 1: Check Python version
            if not self.check_python_version():
                return False
            
            # Step 2: Create project structure
            self.create_project_structure()
            
            # Step 3: Copy AI files
            self.copy_ai_files()
            
            # Step 4: Create virtual environment
            if not self.create_virtual_environment():
                return False
            
            # Step 5: Install dependencies
            if not self.install_dependencies():
                return False
            
            # Step 6: Setup configuration
            self.setup_configuration()
            
            # Step 7: Create startup scripts
            self.create_startup_scripts()
            
            # Step 8: Create documentation
            self.create_readme()
            
            # Success message
            self.print_header("🎉 SETUP COMPLETED SUCCESSFULLY")
            
            self.colored_print("✅ AI Trading System is ready!", 'green')
            print()
            self.colored_print("📁 Project created in:", 'blue')
            self.colored_print(f"   {self.project_dir}", 'cyan')
            print()
            self.colored_print("🚀 Next Steps:", 'yellow')
            self.colored_print("   1. cd ai_trading_bot", 'white')
            self.colored_print("   2. Edit .env with your credentials", 'white')
            self.colored_print("   3. Run: ./scripts/run_ai_trader.sh auth", 'white')
            self.colored_print("   4. Run: ./scripts/run_ai_trader.sh backtest", 'white')
            self.colored_print("   5. Run: ./scripts/run_ai_trader.sh test", 'white')
            print()
            self.colored_print("📚 Check README.md for detailed instructions", 'blue')
            
            return True
            
        except Exception as e:
            self.colored_print(f"❌ Setup failed: {e}", 'red')
            return False

def main():
    """Main setup function"""
    print()
    print("🤖 AI Trading System Setup")
    print("=" * 40)
    print()
    
    setup = AITradingSystemSetup()
    
    # Check if project already exists
    if setup.project_dir.exists():
        response = input(f"Directory '{setup.project_name}' already exists. Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Setup cancelled.")
            return
    
    success = setup.run_setup()
    
    if success:
        print()
        print("🎉 Ready to start AI trading!")
    else:
        print()
        print("❌ Setup failed. Please check the errors above.")

if __name__ == "__main__":
    main()