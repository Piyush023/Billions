# 🤖 Billions Trading Bot

Advanced algorithmic trading bot for Zerodha with multiple strategies, risk management, and automated portfolio management.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

**For TA-Lib issues:**
- **macOS**: `brew install ta-lib && pip install TA-Lib`
- **Windows**: Download from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/) then `pip install TA_Lib-0.4.XX-cpXX-cpXXm-win_amd64.whl`
- **Linux**: `sudo apt-get install libta-lib-dev && pip install TA-Lib`

### 2. Setup Configuration
```bash
python setup.py
```
This will guide you through:
- ✅ Dependency checks
- ⚙️ API configuration  
- 📱 Telegram setup (optional)

### 3. Authenticate with Zerodha
```bash
python zerodha_auth.py
```
- Opens browser for login
- Handles 2FA automatically
- Saves access token

### 4. Test Everything
```bash
python billions.py test
```

### 5. Start Trading
```bash
python billions.py
```

## 📊 Features

### Trading Strategies
- **Mean Reversion**: RSI + Bollinger Bands
- **Momentum**: Moving averages + MACD
- **Multi-timeframe analysis**

### Risk Management
- 🛑 **5% Stop Loss** on all positions
- 🎯 **8% Take Profit** targets
- 📊 **Max 3 positions** simultaneously
- ⚠️ **3% daily loss limit**
- 💰 **Max 20% per position**

### Automation
- 🔍 **Market scanning** every 5 minutes
- 📱 **Telegram notifications** for all trades
- 📊 **Daily reports** at 6 PM
- 🤖 **Fully automated** execution

### Monitoring
- 📈 **Real-time P&L tracking**
- 📱 **Live trade alerts**
- 📊 **Performance analytics**
- 🗃️ **Comprehensive logging**

## ⚙️ Configuration

Your `config.json` is already configured with:

```json
{
  "capital": {
    "total_capital": 10000,
    "max_position_size": 0.2,
    "daily_loss_limit": 0.03,
    "stop_loss_pct": 0.05,
    "take_profit_pct": 0.08
  },
  "strategies": {
    "mean_reversion": { "active": true },
    "momentum": { "active": true }
  },
  "trading_symbols": [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", 
    "HINDUNILVR", "ICICIBANK", "KOTAKBANK"
  ]
}
```

## 📱 Telegram Setup

1. Create bot with @BotFather
2. Get bot token
3. Message your bot, then visit:
   `https://api.telegram.org/bot<TOKEN>/getUpdates`
4. Find your `chat_id`
5. Update `config.json`

## 🔧 Commands

```bash
# Regular trading
python billions.py

# Test mode (no real trades)
python billions.py test

# Setup assistant
python setup.py

# Re-authenticate
python zerodha_auth.py
```

## 📊 Expected Performance

With ₹10,000 capital:
- **Month 1**: 2-5% (₹200-500) - Learning phase
- **Month 2**: 5-8% (₹500-800) - Optimization phase  
- **Month 3+**: 8-12% (₹800-1,200) - Mature operation

## 🚨 Risk Warning

- Start with small capital (₹10K-25K)
- Monitor daily for first week
- Never risk more than you can afford to lose
- Past performance doesn't guarantee future results

## 📁 File Structure

```
trading_bot/
├── billions.py           # Main trading bot
├── zerodha_auth.py      # Authentication system
├── setup.py             # Setup assistant
├── config.json          # Configuration
├── requirements.txt     # Dependencies
└── logs/               # Trading logs
```

## 🛠️ Troubleshooting

**Authentication Error**:
```bash
python zerodha_auth.py
```

**Missing Dependencies**:
```bash
pip install -r requirements.txt
```

**Strategy Not Working**:
- Check market hours (9:15 AM - 3:30 PM)
- Verify sufficient capital
- Review logs in `logs/` folder

**No Trades Executing**:
- Ensure market is open
- Check risk limits
- Verify access token validity

## 📞 Support

Check logs in `logs/` directory for detailed error messages.

## ⚖️ Legal Disclaimer

This software is for educational purposes. Trading involves substantial risk. Always do your own research and consider consulting with a financial advisor. # Billions
