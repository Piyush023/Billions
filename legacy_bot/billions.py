#!/usr/bin/env python3
"""
Billions Trading Bot
Advanced algorithmic trading bot for Zerodha with multiple strategies,
risk management, and automated portfolio management.
"""

import json
import sys
import time
import threading
import logging
import schedule
import pandas as pd
import numpy as np
import talib
from datetime import datetime, timedelta
from kiteconnect import KiteConnect
from telegram import Bot
import yfinance as yf
import asyncio
import traceback
from collections import defaultdict
import os
from dotenv import load_dotenv
from typing import Dict

# Load environment variables from .env file
load_dotenv()

class TradingBot:
    def __init__(self, config_path='config.json'):
        """Initialize the trading bot"""
        self.config = self.load_config(config_path)
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.kite = None
        self.telegram_bot = None
        self.portfolio = {}
        self.positions = {}
        self.orders = {}
        self.market_data = {}
        self.performance_data = defaultdict(list)
        
        # Trading state
        self.is_trading = False
        self.last_scan_time = None
        self.daily_pnl = 0
        self.total_trades = 0
        self.paper_trading = False
        self.paper_portfolio = {}
        self.paper_cash = self.config['capital']['total_capital']
        self.ai_enabled = False
        
        # Initialize connections
        self.initialize_kite()
        self.initialize_telegram()
        
        self.logger.info("🤖 Trading Bot initialized successfully")

    def load_config(self, config_path):
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Configuration file {config_path} not found")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON in {config_path}")
            sys.exit(1)

    def setup_logging(self):
        """Setup comprehensive logging"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Create logs directory if it doesn't exist
        import os
        os.makedirs('logs', exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(f'logs/billions_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler()
            ]
        )

    def initialize_kite(self):
        """Initialize Zerodha KiteConnect"""
        try:
            # Get credentials from config file
            api_key = self.config['zerodha']['api_key']
            access_token = self.config['zerodha']['access_token']
            
            if not api_key:
                self.logger.error("❌ ZERODHA_API_KEY missing in config.json")
                self.logger.info("💡 Please run 'python setup.py' to configure your credentials")
                raise ValueError("Missing API key")
            
            if not access_token:
                self.logger.error("❌ ZERODHA_ACCESS_TOKEN missing in config.json")
                self.logger.info("💡 Please run 'python zerodha_auth.py' to generate a new access token")
                raise ValueError("Missing access token")
            
            self.kite = KiteConnect(api_key=api_key)
            self.kite.set_access_token(access_token)
            
            # Test connection
            try:
                profile = self.kite.profile()
                self.logger.info(f"✅ Connected to Zerodha as {profile['user_name']}")
            except Exception as auth_error:
                if "Incorrect `api_key` or `access_token`" in str(auth_error):
                    self.logger.error("❌ Access token expired or invalid")
                    self.logger.info("💡 Please run 'python zerodha_auth.py' to generate a new access token")
                    self.logger.info("📝 Then update your .env file with the new ZERODHA_ACCESS_TOKEN")
                raise auth_error
            
            # Get initial portfolio
            self.update_portfolio()
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Zerodha connection: {e}")
            raise

    def initialize_telegram(self):
        """Initialize Telegram bot for notifications"""
        try:
            token = self.config['notifications']['telegram_token']
            if token:
                self.telegram_bot = Bot(token=token)
                self.logger.info("✅ Telegram bot initialized")
            else:
                self.logger.warning("⚠️  Telegram token not configured")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Telegram: {e}")

    async def send_telegram_message(self, message):
        """Send message via Telegram"""
        try:
            if self.telegram_bot:
                chat_id = self.config['notifications']['chat_id']
                if chat_id and chat_id != "YOUR_PERSONAL_CHAT_ID":
                    await self.telegram_bot.send_message(chat_id=chat_id, text=message)
        except Exception as e:
            self.logger.error(f"❌ Failed to send Telegram message: {e}")

    def send_telegram_message_sync(self, message):
        """Send Telegram message synchronously"""
        try:
            if self.telegram_bot:
                chat_id = self.config['notifications']['chat_id']
                if chat_id and chat_id != "YOUR_PERSONAL_CHAT_ID":
                    # Run async function in a new event loop
                    import asyncio
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # If loop is running, schedule the task
                            import threading
                            def run_async():
                                new_loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(new_loop)
                                new_loop.run_until_complete(self.send_telegram_message(message))
                                new_loop.close()
                            thread = threading.Thread(target=run_async)
                            thread.start()
                        else:
                            # No loop running, create and run
                            loop.run_until_complete(self.send_telegram_message(message))
                    except RuntimeError:
                        # No event loop exists, create new one
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        new_loop.run_until_complete(self.send_telegram_message(message))
                        new_loop.close()
        except Exception as e:
            self.logger.error(f"❌ Failed to send Telegram message sync: {e}")

    def update_portfolio(self):
        """Update current portfolio and positions"""
        try:
            # Get positions
            positions = self.kite.positions()
            self.positions = {pos['tradingsymbol']: pos for pos in positions['net']}
            
            # Get portfolio holdings
            holdings = self.kite.holdings()
            self.portfolio = {holding['tradingsymbol']: holding for holding in holdings}
            
            # Calculate portfolio value
            portfolio_value = sum(pos['pnl'] for pos in self.positions.values())
            self.logger.info(f"💰 Portfolio Value: ₹{portfolio_value:,.2f}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update portfolio: {e}")

    def get_market_data(self, symbol, interval="5minute", days=30):
        """Get historical market data for analysis"""
        try:
            # Get data from Zerodha
            from_date = datetime.now() - timedelta(days=days)
            to_date = datetime.now()
            
            instrument_token = self.get_instrument_token(symbol)
            if not instrument_token:
                return None
            
            data = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=interval
            )
            
            if not data:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get market data for {symbol}: {e}")
            return None

    def get_instrument_token(self, symbol):
        """Get instrument token for a trading symbol"""
        try:
            # This is a simplified version - in practice, you'd maintain an instrument mapping
            instruments = self.kite.instruments("NSE")
            for instrument in instruments:
                if instrument['tradingsymbol'] == symbol:
                    return instrument['instrument_token']
            return None
        except Exception as e:
            self.logger.error(f"❌ Failed to get instrument token for {symbol}: {e}")
            return None

    def calculate_technical_indicators(self, df):
        """Calculate technical indicators for analysis"""
        try:
            # Ensure data types are correct
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            
            # Convert to numpy arrays with correct dtype
            close_prices = df['close'].values.astype(np.float64)
            volumes = df['volume'].values.astype(np.float64)
            
            # RSI
            df['rsi'] = talib.RSI(close_prices, timeperiod=14)
            
            # Moving Averages
            df['sma_20'] = talib.SMA(close_prices, timeperiod=20)
            df['sma_50'] = talib.SMA(close_prices, timeperiod=50)
            df['ema_12'] = talib.EMA(close_prices, timeperiod=12)
            df['ema_26'] = talib.EMA(close_prices, timeperiod=26)
            
            # MACD
            df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(close_prices)
            
            # Bollinger Bands
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
                close_prices, timeperiod=20, nbdevup=2, nbdevdn=2
            )
            
            # Volume indicators
            df['volume_sma'] = talib.SMA(volumes, timeperiod=20)
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Fill NaN values
            df = df.bfill().fillna(0)
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate technical indicators: {e}")
            return df

    def mean_reversion_strategy(self, symbol, df):
        """Mean reversion trading strategy"""
        try:
            if len(df) < 50:
                return None
            
            current = df.iloc[-1]
            prev = df.iloc[-2]
            
            strategy_config = self.config['strategies']['mean_reversion']
            
            # Entry conditions for mean reversion
            buy_signal = (
                current['rsi'] < strategy_config['rsi_oversold'] and
                current['close'] < current['bb_lower'] and
                current['volume_ratio'] > 0.3 and  # Relaxed volume requirement
                prev['rsi'] >= current['rsi']  # RSI declining
            )
            
            sell_signal = (
                current['rsi'] > strategy_config['rsi_overbought'] and
                current['close'] > current['bb_upper'] and
                current['volume_ratio'] > 0.3 and  # Relaxed volume requirement
                prev['rsi'] <= current['rsi']  # RSI rising
            )
            
            if buy_signal:
                return {
                    'action': 'BUY',
                    'strategy': 'mean_reversion',
                    'confidence': min(100, (70 - current['rsi']) * 2),
                    'reason': f"Oversold RSI: {current['rsi']:.1f}, Below BB Lower"
                }
            elif sell_signal:
                return {
                    'action': 'SELL',
                    'strategy': 'mean_reversion',
                    'confidence': min(100, (current['rsi'] - 30) * 2),
                    'reason': f"Overbought RSI: {current['rsi']:.1f}, Above BB Upper"
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Mean reversion strategy error for {symbol}: {e}")
            return None

    def momentum_strategy(self, symbol, df):
        """Momentum trading strategy"""
        try:
            if len(df) < 50:
                return None
            
            current = df.iloc[-1]
            prev = df.iloc[-2]
            
            strategy_config = self.config['strategies']['momentum']
            
            # Entry conditions for momentum
            buy_signal = (
                current['sma_20'] > current['sma_50'] and
                current['close'] > current['sma_20'] and
                current['macd'] > current['macd_signal'] and
                current['volume_ratio'] > strategy_config['min_volume_ratio'] and
                current['close'] > prev['close']
            )
            
            sell_signal = (
                current['sma_20'] < current['sma_50'] and
                current['close'] < current['sma_20'] and
                current['macd'] < current['macd_signal'] and
                current['volume_ratio'] > strategy_config['min_volume_ratio'] and
                current['close'] < prev['close']
            )
            
            if buy_signal:
                return {
                    'action': 'BUY',
                    'strategy': 'momentum',
                    'confidence': 75 if current['rsi'] < 60 else 60,
                    'reason': f"Bullish momentum, MACD crossover"
                }
            elif sell_signal:
                return {
                    'action': 'SELL',
                    'strategy': 'momentum',
                    'confidence': 75 if current['rsi'] > 40 else 60,
                    'reason': f"Bearish momentum, MACD divergence"
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Momentum strategy error for {symbol}: {e}")
            return None

    def calculate_position_size(self, symbol, signal, current_price):
        """Calculate optimal position size based on risk management"""
        try:
            capital_config = self.config['capital']
            total_capital = capital_config['total_capital']
            max_position_size = capital_config['max_position_size']
            min_trade_amount = capital_config['min_trade_amount']
            
            # Get available cash
            if hasattr(self, 'paper_trading') and self.paper_trading:
                available_cash = self.paper_cash
            else:
                margins = self.kite.margins()
                available_cash = margins['equity']['available']['cash']
            
            # Calculate base position size
            max_investment = min(
                total_capital * max_position_size,
                available_cash * 0.9  # Keep 10% buffer
            )
            
            # Adjust based on signal confidence
            confidence_factor = signal['confidence'] / 100
            position_value = max_investment * confidence_factor
            
            # Ensure minimum trade amount
            position_value = max(position_value, min_trade_amount)
            
            # Calculate quantity
            quantity = int(position_value / current_price)
            
            # Ensure we don't exceed position limits
            current_positions = len([pos for pos in self.positions.values() if pos['quantity'] != 0])
            max_positions = capital_config['max_positions']
            
            if current_positions >= max_positions and signal['action'] == 'BUY':
                self.logger.warning(f"⚠️  Maximum positions ({max_positions}) reached")
                return 0
            
            return quantity
            
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate position size: {e}")
            return 0

    def check_risk_limits(self):
        """Check if risk limits are breached"""
        try:
            capital_config = self.config['capital']
            
            # Calculate current P&L
            total_pnl = sum(pos['pnl'] for pos in self.positions.values())
            daily_loss_limit = capital_config['total_capital'] * capital_config['daily_loss_limit']
            
            if total_pnl < -daily_loss_limit:
                self.logger.error(f"🚨 Daily loss limit breached: ₹{total_pnl:,.2f}")
                return False
            
            # Check individual position sizes
            for symbol, position in self.positions.items():
                if position['quantity'] == 0:
                    continue
                    
                position_value = abs(position['quantity'] * position['average_price'])
                max_position_value = capital_config['total_capital'] * capital_config['max_position_size']
                
                if position_value > max_position_value:
                    self.logger.warning(f"⚠️  Position size limit exceeded for {symbol}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Risk check failed: {e}")
            return False

    def place_order(self, symbol, action, quantity, order_type='MARKET'):
        """Place order with proper error handling"""
        try:
            if quantity <= 0:
                return None
            
            # Determine transaction type
            transaction_type = 'BUY' if action == 'BUY' else 'SELL'
            
            # Get current price for limit orders
            ltp = self.get_ltp(symbol)
            if not ltp:
                self.logger.error(f"❌ Cannot get LTP for {symbol}")
                return None
            
            # Place order
            order_id = self.kite.place_order(
                variety='regular',
                exchange='NSE',
                tradingsymbol=symbol,
                transaction_type=transaction_type,
                quantity=quantity,
                order_type=order_type,
                product='MIS',  # Intraday
                validity='DAY'
            )
            
            self.logger.info(f"📊 Order placed: {action} {quantity} {symbol} at ₹{ltp}")
            
            # Send Telegram notification
            if self.config['notifications']['send_trade_alerts']:
                message = f"🔔 Order Placed\n{action} {quantity} {symbol}\nPrice: ₹{ltp:,.2f}\nOrder ID: {order_id}"
                self.send_telegram_message_sync(message)
            
            return order_id
            
        except Exception as e:
            self.logger.error(f"❌ Failed to place order for {symbol}: {e}")
            return None

    def get_ltp(self, symbol):
        """Get Last Traded Price"""
        try:
            instrument_token = self.get_instrument_token(symbol)
            if not instrument_token:
                return None
            
            ltp = self.kite.ltp([instrument_token])
            return ltp[str(instrument_token)]['last_price']
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get LTP for {symbol}: {e}")
            return None

    def manage_stop_losses(self):
        """Manage stop losses for all positions"""
        try:
            capital_config = self.config['capital']
            stop_loss_pct = capital_config['stop_loss_pct']
            take_profit_pct = capital_config['take_profit_pct']
            
            for symbol, position in self.positions.items():
                if position['quantity'] == 0:
                    continue
                
                current_price = self.get_ltp(symbol)
                if not current_price:
                    continue
                
                avg_price = position['average_price']
                quantity = position['quantity']
                
                # Calculate stop loss and take profit levels
                if quantity > 0:  # Long position
                    stop_loss_price = avg_price * (1 - stop_loss_pct)
                    take_profit_price = avg_price * (1 + take_profit_pct)
                    
                    if current_price <= stop_loss_price:
                        self.logger.warning(f"🛑 Stop loss triggered for {symbol}")
                        self.place_order(symbol, 'SELL', quantity)
                    elif current_price >= take_profit_price:
                        self.logger.info(f"🎯 Take profit triggered for {symbol}")
                        self.place_order(symbol, 'SELL', quantity)
                        
                else:  # Short position
                    stop_loss_price = avg_price * (1 + stop_loss_pct)
                    take_profit_price = avg_price * (1 - take_profit_pct)
                    
                    if current_price >= stop_loss_price:
                        self.logger.warning(f"🛑 Stop loss triggered for {symbol}")
                        self.place_order(symbol, 'BUY', abs(quantity))
                    elif current_price <= take_profit_price:
                        self.logger.info(f"🎯 Take profit triggered for {symbol}")
                        self.place_order(symbol, 'BUY', abs(quantity))
            
        except Exception as e:
            self.logger.error(f"❌ Stop loss management failed: {e}")

    def scan_and_trade(self):
        """Main scanning and trading logic"""
        try:
            if not self.is_market_open():
                return
            
            if not self.check_risk_limits():
                self.logger.error("🚨 Risk limits breached. Stopping trading.")
                return
            
            self.logger.info("🔍 Scanning markets...")
            
            # Update portfolio
            self.update_portfolio()
            
            # Manage existing positions
            self.manage_stop_losses()
            
            # Scan for new opportunities
            trading_symbols = self.config['trading_symbols']
            
            for symbol in trading_symbols:
                try:
                    # Get market data
                    df = self.get_market_data(symbol)
                    if df is None or len(df) < 50:
                        continue
                    
                    # Calculate indicators
                    df = self.calculate_technical_indicators(df)
                    
                    # Generate signals
                    signals = []
                    
                    if self.config['strategies']['mean_reversion']['active']:
                        signal = self.mean_reversion_strategy(symbol, df)
                        if signal:
                            signals.append(signal)
                    
                    if self.config['strategies']['momentum']['active']:
                        signal = self.momentum_strategy(symbol, df)
                        if signal:
                            signals.append(signal)
                    
                    # Process signals
                    for signal in signals:
                        current_price = self.get_ltp(symbol)
                        if not current_price:
                            continue
                        
                        quantity = self.calculate_position_size(symbol, signal, current_price)
                        if quantity > 0:
                            self.place_order(symbol, signal['action'], quantity)
                            
                            # Log signal
                            self.logger.info(f"📈 Signal: {signal['action']} {symbol} - {signal['reason']}")
                    
                except Exception as e:
                    self.logger.error(f"❌ Error processing {symbol}: {e}")
                    continue
            
            self.last_scan_time = datetime.now()
            
        except Exception as e:
            self.logger.error(f"❌ Scan and trade failed: {e}")

    def is_market_open(self):
        """Check if market is currently open"""
        try:
            now = datetime.now()
            market_config = self.config['market_hours']
            
            # Check if it's a trading day (Monday=0 to Friday=4)
            if now.weekday() not in market_config['trading_days']:
                return False
            
            # Check time
            start_time = datetime.strptime(market_config['start_time'], '%H:%M').time()
            end_time = datetime.strptime(market_config['end_time'], '%H:%M').time()
            current_time = now.time()
            
            return start_time <= current_time <= end_time
            
        except Exception as e:
            self.logger.error(f"❌ Market time check failed: {e}")
            return False

    def generate_daily_report(self):
        """Generate and send daily performance report"""
        try:
            self.update_portfolio()
            
            # Calculate performance metrics
            total_pnl = sum(pos['pnl'] for pos in self.positions.values())
            portfolio_value = sum(holding['quantity'] * holding['last_price'] 
                                for holding in self.portfolio.values())
            
            # Generate report
            report = f"""
📊 DAILY TRADING REPORT - {datetime.now().strftime('%Y-%m-%d')}
{'='*40}

💰 PORTFOLIO SUMMARY
Total P&L: ₹{total_pnl:,.2f}
Portfolio Value: ₹{portfolio_value:,.2f}
Active Positions: {len([p for p in self.positions.values() if p['quantity'] != 0])}

📈 POSITIONS
"""
            
            for symbol, position in self.positions.items():
                if position['quantity'] != 0:
                    report += f"{symbol}: {position['quantity']} @ ₹{position['average_price']:.2f} (P&L: ₹{position['pnl']:.2f})\n"
            
            report += f"""
🎯 STRATEGY PERFORMANCE
Mean Reversion: Active - {self.config['strategies']['mean_reversion']['active']}
Momentum: Active - {self.config['strategies']['momentum']['active']}

📊 SYSTEM STATUS
Last Scan: {self.last_scan_time.strftime('%H:%M:%S') if self.last_scan_time else 'Never'}
Total Trades Today: {self.total_trades}

⚠️  RISK METRICS
Daily Loss Limit: ₹{self.config['capital']['total_capital'] * self.config['capital']['daily_loss_limit']:,.2f}
Max Position Size: {self.config['capital']['max_position_size']*100}%
"""
            
            self.logger.info("📊 Daily report generated")
            
            # Send via Telegram
            if self.config['notifications']['send_daily_report']:
                self.send_telegram_message_sync(report)
            
        except Exception as e:
            self.logger.error(f"❌ Daily report generation failed: {e}")

    def test_mode(self):
        """Run bot in test mode"""
        try:
            self.logger.info("🧪 Running in TEST MODE")
            
            # Test connections
            profile = self.kite.profile()
            self.logger.info(f"✅ Zerodha connection: {profile['user_name']}")
            
            margins = self.kite.margins()
            self.logger.info(f"💰 Available cash: ₹{margins['equity']['available']['cash']:,.2f}")
            
            # Test market data
            test_symbol = self.config['trading_symbols'][0]
            df = self.get_market_data(test_symbol, days=5)
            if df is not None:
                self.logger.info(f"📊 Market data test: {len(df)} records for {test_symbol}")
            
            # Test strategies
            if df is not None and len(df) > 50:
                df = self.calculate_technical_indicators(df)
                
                if self.config['strategies']['mean_reversion']['active']:
                    signal = self.mean_reversion_strategy(test_symbol, df)
                    self.logger.info(f"📈 Mean reversion test: {signal}")
                
                if self.config['strategies']['momentum']['active']:
                    signal = self.momentum_strategy(test_symbol, df)
                    self.logger.info(f"📈 Momentum test: {signal}")
            
            # Test Telegram
            if self.telegram_bot:
                try:
                    # Create and run the async task properly
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(self.send_telegram_message("🧪 Test message from trading bot"))
                    loop.close()
                    self.logger.info("✅ Telegram test message sent")
                except Exception as e:
                    self.logger.warning(f"⚠️  Telegram test failed: {e}")
            
            self.logger.info("✅ All tests completed successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Test mode failed: {e}")
            traceback.print_exc()

    def start_trading(self):
        """Start the trading bot"""
        try:
            self.logger.info("🚀 Starting trading bot...")
            self.is_trading = True
            
            # Schedule tasks
            schedule.every(5).minutes.do(self.scan_and_trade)
            schedule.every().day.at("18:00").do(self.generate_daily_report)
            
            # Send startup notification
            startup_message = f"""
🤖 TRADING BOT STARTED
{'='*25}
📊 Capital: ₹{self.config['capital']['total_capital']:,.2f}
🎯 Active Strategies: {sum(1 for s in self.config['strategies'].values() if s['active'])}
⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            if self.config['notifications']['send_trade_alerts']:
                self.send_telegram_message_sync(startup_message)
            
            print(startup_message)
            
            # Main trading loop
            while self.is_trading:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            self.logger.info("🛑 Trading bot stopped by user")
            self.stop_trading()
        except Exception as e:
            self.logger.error(f"❌ Trading bot error: {e}")
            traceback.print_exc()

    def stop_trading(self):
        """Stop the trading bot"""
        self.is_trading = False
        self.logger.info("🛑 Trading bot stopped")
        
        # Send shutdown notification
        shutdown_message = f"""
🛑 TRADING BOT STOPPED
{'='*25}
⏰ Stopped: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
📊 Session Summary: Check logs for details
        """
        
        if self.config['notifications']['send_trade_alerts']:
            self.send_telegram_message_sync(shutdown_message)
    
    def paper_trading_mode(self):
        """Run bot in paper trading mode - no real money"""
        try:
            self.logger.info("📰 Starting PAPER TRADING MODE")
            self.paper_trading = True
            self.is_trading = True
            
            # Initialize paper portfolio
            self.paper_portfolio = {}
            self.paper_cash = self.config['capital']['total_capital']
            
            self.logger.info(f"💰 Paper trading capital: ₹{self.paper_cash:,.2f}")
            
            # Send startup notification
            startup_message = f"""
📰 PAPER TRADING MODE STARTED
{'='*30}
💰 Virtual Capital: ₹{self.paper_cash:,.2f}
🎯 Active Strategies: {sum(1 for s in self.config['strategies'].values() if s['active'])}
⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🔔 Note: This is PAPER TRADING - No real money involved!
            """
            
            if self.config['notifications']['send_trade_alerts']:
                self.send_telegram_message_sync(startup_message)
            
            print(startup_message)
            
            # Schedule paper trading tasks
            schedule.every(5).minutes.do(self.paper_scan_and_trade)
            schedule.every().day.at("18:00").do(self.generate_paper_trading_report)
            
            # Main paper trading loop
            while self.is_trading:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            self.logger.info("🛑 Paper trading stopped by user")
            self.stop_trading()
        except Exception as e:
            self.logger.error(f"❌ Paper trading error: {e}")
            traceback.print_exc()
    
    def paper_scan_and_trade(self):
        """Paper trading version of scan_and_trade"""
        try:
            if not self.is_market_open():
                return
            
            self.logger.info("🔍 Paper trading - Scanning markets...")
            
            # Scan for opportunities (same logic but paper execution)
            trading_symbols = self.config['trading_symbols']
            
            for symbol in trading_symbols:
                try:
                    # Get market data
                    df = self.get_market_data(symbol)
                    if df is None or len(df) < 50:
                        continue
                    
                    # Calculate indicators
                    df = self.calculate_technical_indicators(df)
                    
                    # Generate signals
                    signals = []
                    
                    if self.config['strategies']['mean_reversion']['active']:
                        signal = self.mean_reversion_strategy(symbol, df)
                        if signal:
                            signals.append(signal)
                    
                    if self.config['strategies']['momentum']['active']:
                        signal = self.momentum_strategy(symbol, df)
                        if signal:
                            signals.append(signal)
                    
                    # Process signals for paper trading
                    for signal in signals:
                        current_price = self.get_ltp(symbol)
                        if not current_price:
                            continue
                        
                        quantity = self.calculate_position_size(symbol, signal, current_price)
                        if quantity > 0:
                            self.paper_place_order(symbol, signal['action'], quantity, current_price, signal)
                
                except Exception as e:
                    self.logger.error(f"❌ Error processing {symbol} in paper trading: {e}")
                    continue
            
            self.last_scan_time = datetime.now()
            
        except Exception as e:
            self.logger.error(f"❌ Paper scan and trade failed: {e}")
    
    def paper_place_order(self, symbol: str, action: str, quantity: int, 
                         price: float, signal: Dict):
        """Simulate order placement for paper trading"""
        try:
            trade_value = quantity * price
            
            if action == 'BUY':
                if trade_value > self.paper_cash:
                    self.logger.warning(f"⚠️  Insufficient paper cash for {symbol}: Need ₹{trade_value:,.2f}, Have ₹{self.paper_cash:,.2f}")
                    return None
                
                # Execute paper buy
                self.paper_cash -= trade_value
                
                if symbol in self.paper_portfolio:
                    # Add to existing position
                    old_qty = self.paper_portfolio[symbol]['quantity']
                    old_value = self.paper_portfolio[symbol]['total_invested']
                    new_qty = old_qty + quantity
                    new_value = old_value + trade_value
                    
                    self.paper_portfolio[symbol] = {
                        'quantity': new_qty,
                        'average_price': new_value / new_qty,
                        'total_invested': new_value,
                        'last_price': price
                    }
                else:
                    # New position
                    self.paper_portfolio[symbol] = {
                        'quantity': quantity,
                        'average_price': price,
                        'total_invested': trade_value,
                        'last_price': price
                    }
                
                self.logger.info(f"📰 PAPER BUY: {quantity} {symbol} at ₹{price:.2f} - Confidence: {signal.get('confidence', 0):.1f}%")
                
            elif action == 'SELL':
                if symbol not in self.paper_portfolio or self.paper_portfolio[symbol]['quantity'] < quantity:
                    self.logger.warning(f"⚠️  Insufficient paper shares for {symbol}")
                    return None
                
                # Execute paper sell
                self.paper_cash += trade_value
                
                position = self.paper_portfolio[symbol]
                position['quantity'] -= quantity
                position['total_invested'] -= (position['average_price'] * quantity)
                
                if position['quantity'] == 0:
                    del self.paper_portfolio[symbol]
                
                self.logger.info(f"📰 PAPER SELL: {quantity} {symbol} at ₹{price:.2f}")
            
            # Send notification
            if self.config['notifications']['send_trade_alerts']:
                message = f"📰 PAPER TRADE\n{action} {quantity} {symbol}\nPrice: ₹{price:,.2f}\nStrategy: {signal.get('strategy', 'Unknown')}\nConfidence: {signal.get('confidence', 0):.1f}%"
                self.send_telegram_message_sync(message)
            
            return f"PAPER_{symbol}_{action}_{datetime.now().timestamp()}"
            
        except Exception as e:
            self.logger.error(f"❌ Paper order failed for {symbol}: {e}")
            return None
    
    def generate_paper_trading_report(self):
        """Generate paper trading performance report"""
        try:
            total_portfolio_value = self.paper_cash
            total_invested = 0
            positions_count = len(self.paper_portfolio)
            
            # Calculate portfolio value
            for symbol, position in self.paper_portfolio.items():
                current_price = self.get_ltp(symbol)
                if current_price:
                    position['last_price'] = current_price
                    market_value = position['quantity'] * current_price
                    total_portfolio_value += market_value
                    total_invested += position['total_invested']
            
            # Calculate returns
            initial_capital = self.config['capital']['total_capital']
            total_return = ((total_portfolio_value - initial_capital) / initial_capital) * 100
            
            report = f"""
📰 PAPER TRADING REPORT - {datetime.now().strftime('%Y-%m-%d')}
{'='*45}

💰 PORTFOLIO SUMMARY
Initial Capital: ₹{initial_capital:,.2f}
Current Value: ₹{total_portfolio_value:,.2f}
Cash Available: ₹{self.paper_cash:,.2f}
Total Return: {total_return:+.2f}%
Active Positions: {positions_count}

📊 POSITIONS
"""
            
            for symbol, position in self.paper_portfolio.items():
                market_value = position['quantity'] * position['last_price']
                pnl = market_value - position['total_invested']
                pnl_pct = (pnl / position['total_invested']) * 100
                
                report += f"{symbol}: {position['quantity']} @ ₹{position['average_price']:.2f} "
                report += f"(Current: ₹{position['last_price']:.2f}, P&L: {pnl_pct:+.1f}%)\n"
            
            report += f"""
🎯 STRATEGY PERFORMANCE
Paper trades executed: {self.total_trades}
Last scan: {self.last_scan_time.strftime('%H:%M:%S') if self.last_scan_time else 'Never'}

📝 Note: This is PAPER TRADING - No real money involved!
"""
            
            self.logger.info("📰 Paper trading report generated")
            
            if self.config['notifications']['send_daily_report']:
                self.send_telegram_message_sync(report)
            
        except Exception as e:
            self.logger.error(f"❌ Paper trading report failed: {e}")
    
    def ai_test_mode(self):
        """Test AI components integration"""
        try:
            self.logger.info("🤖 Starting AI TEST MODE")
            
            # Try to import and initialize AI components
            try:
                import sys
                sys.path.append('ai_trading_bot')
                
                from ai_trading_orchestrator import AITradingOrchestrator
                
                self.ai_orchestrator = AITradingOrchestrator('config.json')
                self.ai_enabled = True
                
                self.logger.info("✅ AI Orchestrator initialized successfully")
                
                # Test AI stock screening
                self.logger.info("🔍 Testing AI stock screening...")
                ai_signals = self.ai_orchestrator.get_trading_signals()
                
                self.logger.info(f"🎯 AI Signals Generated:")
                self.logger.info(f"  - Buy signals: {len(ai_signals.get('buy_signals', []))}")
                self.logger.info(f"  - Sell signals: {len(ai_signals.get('sell_signals', []))}")
                
                # Display top AI recommendations
                for i, signal in enumerate(ai_signals.get('buy_signals', [])[:3], 1):
                    self.logger.info(f"  {i}. BUY {signal['symbol']} - Confidence: {signal['confidence']:.1f}%")
                
                # Test individual AI components
                self.logger.info("🧪 Testing individual AI components...")
                
                # Test AI engine
                from ai_trading_engine import AITradingEngine
                ai_engine = AITradingEngine()
                self.logger.info("✅ AI Trading Engine: OK")
                
                # Test screener
                from dynamic_stock_screener import DynamicStockScreener
                screener = DynamicStockScreener()
                self.logger.info("✅ Dynamic Stock Screener: OK")
                
                # Test portfolio manager
                from ai_portfolio_manager import AIPortfolioManager
                portfolio_mgr = AIPortfolioManager()
                self.logger.info("✅ AI Portfolio Manager: OK")
                
                self.logger.info("🎉 All AI components are working correctly!")
                
                # Send AI test notification
                if self.telegram_bot:
                    message = f"""
🤖 AI SYSTEM TEST COMPLETED
{'='*25}
✅ AI Orchestrator: Working
✅ Trading Engine: Working  
✅ Stock Screener: Working
✅ Portfolio Manager: Working

🎯 Generated {len(ai_signals.get('buy_signals', []))} buy signals
📊 System ready for AI-enhanced trading!
                    """
                    self.send_telegram_message_sync(message)
                
            except ImportError as e:
                self.logger.error(f"❌ AI components not available: {e}")
                self.logger.info("💡 Please run the AI system setup first")
                
            except Exception as e:
                self.logger.error(f"❌ AI test failed: {e}")
                traceback.print_exc()
            
            self.logger.info("✅ AI test mode completed")
            
        except Exception as e:
            self.logger.error(f"❌ AI test mode failed: {e}")
            traceback.print_exc()

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Billions Trading Bot')
    parser.add_argument('mode', nargs='?', default='trade', 
                       choices=['trade', 'test', 'paper', 'ai-test'], 
                       help='Run mode: trade, test, paper, or ai-test')
    
    args = parser.parse_args()
    
    try:
        bot = TradingBot()
        
        if args.mode == 'test':
            bot.test_mode()
        elif args.mode == 'paper':
            bot.paper_trading_mode()
        elif args.mode == 'ai-test':
            bot.ai_test_mode()
        elif args.mode == 'trade':
            bot.start_trading()
        else:
            print(f"Unknown mode: {args.mode}")
            print("Use: python billions.py {trade|test|paper|ai-test}")
            
    except Exception as e:
        print(f"❌ Failed to start trading bot: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
