#!/bin/bash
# AI Trading Bot Runner Script

cd "$(dirname "$0")/.."

case "$1" in
    "trade")
        echo "🚀 Starting AI Trading Bot..."
        venv/bin/python billions.py trade
        ;;
    "test")
        echo "🧪 Running tests..."
        venv/bin/python billions.py test
        ;;
    "backtest")
        echo "📊 Running backtesting..."
        venv/bin/python ai_backtesting_engine.py
        ;;
    "screen")
        echo "🔍 Running stock screening..."
        venv/bin/python dynamic_stock_screener.py
        ;;
    "auth")
        echo "🔐 Setting up Zerodha authentication..."
        venv/bin/python zerodha_auth.py
        ;;
    *)
        echo "Usage: $0 {trade|test|backtest|screen|auth}"
        echo ""
        echo "Commands:"
        echo "  trade     - Start live trading"
        echo "  test      - Run system tests"
        echo "  backtest  - Run AI backtesting"
        echo "  screen    - Run stock screening"
        echo "  auth      - Setup Zerodha authentication"
        ;;
esac
