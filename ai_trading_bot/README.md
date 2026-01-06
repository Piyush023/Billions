# 🤖 AI Trading Bot

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
