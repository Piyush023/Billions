#!/usr/bin/env python3
"""
AI Backtesting Engine
Comprehensive backtesting system for AI trading strategies with
walk-forward analysis, performance metrics, and risk assessment
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import yfinance as yf
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import AI components for backtesting
from ai_trading_engine import AITradingEngine
from dynamic_stock_screener import DynamicStockScreener
from ai_portfolio_manager import AIPortfolioManager

class AIBacktestingEngine:
    """Comprehensive backtesting engine for AI trading strategies"""
    
    def __init__(self, config_path='config.json'):
        self.config = self.load_config(config_path)
        self.logger = logging.getLogger(__name__)
        
        # Backtesting parameters
        self.initial_capital = self.config['capital']['total_capital']
        self.commission_rate = 0.0015  # 0.15% per trade
        self.slippage = 0.001         # 0.1% slippage
        
        # Initialize AI components for backtesting
        self.ai_engine = AITradingEngine(config_path)
        self.portfolio_manager = AIPortfolioManager(config_path)
        
        # Backtesting state
        self.current_capital = self.initial_capital
        self.positions = {}
        self.trade_history = []
        self.portfolio_history = []
        self.performance_metrics = {}
        
        # Walk-forward parameters
        self.train_period_days = 180  # 6 months training
        self.test_period_days = 30    # 1 month testing
        self.step_size_days = 15      # 2-week steps
        
    def load_config(self, config_path):
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def get_historical_data(self, symbol: str, start_date: datetime, 
                          end_date: datetime) -> pd.DataFrame:
        """Get historical data for backtesting"""
        try:
            # Add .NS suffix for NSE stocks
            ticker = f"{symbol}.NS"
            stock = yf.Ticker(ticker)
            
            # Get historical data
            data = stock.history(start=start_date, end=end_date)
            
            if data.empty:
                return pd.DataFrame()
            
            # Rename columns to match expected format
            data = data.rename(columns={
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Reset index to get date as column
            data = data.reset_index()
            data['date'] = data['Date']
            
            return data
            
        except Exception as e:
            self.logger.warning(f"Failed to get historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def simulate_trade_execution(self, symbol: str, action: str, quantity: int, 
                               price: float, date: datetime) -> Dict:
        """Simulate trade execution with costs"""
        try:
            # Apply slippage
            if action == 'BUY':
                execution_price = price * (1 + self.slippage)
            else:
                execution_price = price * (1 - self.slippage)
            
            # Calculate trade value
            trade_value = quantity * execution_price
            
            # Calculate commission
            commission = trade_value * self.commission_rate
            
            # Total cost including commission
            total_cost = trade_value + commission
            
            trade_record = {
                'date': date,
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'price': price,
                'execution_price': execution_price,
                'trade_value': trade_value,
                'commission': commission,
                'total_cost': total_cost,
                'slippage': abs(execution_price - price)
            }
            
            return trade_record
            
        except Exception as e:
            self.logger.error(f"Trade execution simulation failed: {e}")
            return {}
    
    def update_position(self, symbol: str, action: str, quantity: int, 
                       price: float, date: datetime):
        """Update position based on trade"""
        try:
            if symbol not in self.positions:
                self.positions[symbol] = {
                    'quantity': 0,
                    'average_price': 0,
                    'total_invested': 0,
                    'realized_pnl': 0,
                    'entry_date': None
                }
            
            position = self.positions[symbol]
            
            if action == 'BUY':
                # Add to position
                old_quantity = position['quantity']
                old_invested = position['total_invested']
                
                new_invested = quantity * price
                total_quantity = old_quantity + quantity
                total_invested = old_invested + new_invested
                
                position['quantity'] = total_quantity
                position['total_invested'] = total_invested
                position['average_price'] = total_invested / total_quantity if total_quantity > 0 else 0
                
                if old_quantity == 0:
                    position['entry_date'] = date
                
                # Deduct cost from capital
                trade_cost = new_invested * (1 + self.commission_rate + self.slippage)
                self.current_capital -= trade_cost
                
            elif action == 'SELL':
                # Reduce or close position
                if position['quantity'] >= quantity:
                    # Calculate realized P&L
                    avg_price = position['average_price']
                    sale_proceeds = quantity * price * (1 - self.commission_rate - self.slippage)
                    cost_basis = quantity * avg_price
                    realized_pnl = sale_proceeds - cost_basis
                    
                    position['quantity'] -= quantity
                    position['total_invested'] -= cost_basis
                    position['realized_pnl'] += realized_pnl
                    
                    # Add proceeds to capital
                    self.current_capital += sale_proceeds
                    
                    # If position is closed
                    if position['quantity'] == 0:
                        position['average_price'] = 0
                        position['total_invested'] = 0
                        position['entry_date'] = None
                
        except Exception as e:
            self.logger.error(f"Position update failed for {symbol}: {e}")
    
    def calculate_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate current portfolio value"""
        try:
            portfolio_value = self.current_capital
            
            for symbol, position in self.positions.items():
                if position['quantity'] > 0 and symbol in current_prices:
                    market_value = position['quantity'] * current_prices[symbol]
                    portfolio_value += market_value
            
            return portfolio_value
            
        except Exception as e:
            self.logger.error(f"Portfolio value calculation failed: {e}")
            return self.current_capital
    
    def walk_forward_backtest(self, symbols: List[str], start_date: datetime, 
                            end_date: datetime) -> Dict:
        """Perform walk-forward analysis"""
        try:
            self.logger.info(f"🔄 Starting walk-forward backtest from {start_date} to {end_date}")
            
            # Initialize results
            results = {
                'periods': [],
                'overall_performance': {},
                'trade_history': [],
                'portfolio_history': []
            }
            
            current_date = start_date
            period_number = 1
            
            while current_date < end_date:
                # Define training and testing periods
                train_start = current_date
                train_end = current_date + timedelta(days=self.train_period_days)
                test_start = train_end
                test_end = min(train_end + timedelta(days=self.test_period_days), end_date)
                
                if test_end <= test_start:
                    break
                
                self.logger.info(f"📅 Period {period_number}: Train {train_start.date()} to {train_end.date()}, "
                               f"Test {test_start.date()} to {test_end.date()}")
                
                # Collect training data
                training_data = {}
                for symbol in symbols:
                    df = self.get_historical_data(symbol, train_start, train_end)
                    if not df.empty and len(df) > 50:
                        training_data[symbol] = df
                
                if not training_data:
                    current_date += timedelta(days=self.step_size_days)
                    continue
                
                # Train AI models
                training_results = {}
                for symbol, df in training_data.items():
                    X, y = self.ai_engine.prepare_training_data(symbol, df)
                    if len(X) > 100:
                        result = self.ai_engine.train_ensemble_model(symbol, X, y)
                        if result:
                            training_results[symbol] = result
                
                # Test period simulation
                test_results = self.simulate_period(
                    list(training_results.keys()), 
                    test_start, 
                    test_end
                )
                
                # Record period results
                period_result = {
                    'period': period_number,
                    'train_start': train_start.isoformat(),
                    'train_end': train_end.isoformat(),
                    'test_start': test_start.isoformat(),
                    'test_end': test_end.isoformat(),
                    'symbols_trained': len(training_results),
                    'trades_executed': len(test_results['trades']),
                    'period_return': test_results['period_return'],
                    'period_pnl': test_results['period_pnl']
                }
                
                results['periods'].append(period_result)
                results['trade_history'].extend(test_results['trades'])
                results['portfolio_history'].extend(test_results['portfolio_values'])
                
                # Move to next period
                current_date += timedelta(days=self.step_size_days)
                period_number += 1
            
            # Calculate overall performance
            results['overall_performance'] = self.calculate_overall_performance(results)
            
            self.logger.info(f"✅ Walk-forward backtest completed: {period_number-1} periods")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Walk-forward backtest failed: {e}")
            return {}
    
    def simulate_period(self, symbols: List[str], start_date: datetime, 
                       end_date: datetime) -> Dict:
        """Simulate trading for a specific period"""
        try:
            period_trades = []
            portfolio_values = []
            start_capital = self.current_capital
            
            # Get data for the period
            period_data = {}
            for symbol in symbols:
                df = self.get_historical_data(symbol, start_date, end_date)
                if not df.empty:
                    period_data[symbol] = df
            
            if not period_data:
                return {'trades': [], 'portfolio_values': [], 'period_return': 0, 'period_pnl': 0}
            
            # Simulate day by day
            trading_dates = pd.bdate_range(start_date, end_date)
            
            for current_date in trading_dates:
                try:
                    daily_prices = {}
                    
                    # Get current prices
                    for symbol, df in period_data.items():
                        # Find closest date
                        df['date'] = pd.to_datetime(df['date'])
                        available_dates = df[df['date'] <= current_date]
                        
                        if not available_dates.empty:
                            latest_data = available_dates.iloc[-1]
                            daily_prices[symbol] = latest_data['close']
                    
                    if not daily_prices:
                        continue
                    
                    # Generate AI predictions
                    for symbol in daily_prices:
                        if symbol in period_data:
                            # Get historical data up to current date
                            historical_df = period_data[symbol]
                            historical_df = historical_df[
                                pd.to_datetime(historical_df['date']) <= current_date
                            ]
                            
                            if len(historical_df) < 50:
                                continue
                            
                            # Generate AI prediction
                            prediction = self.ai_engine.predict(symbol, historical_df)
                            
                            if not prediction or prediction.get('confidence', 0) < 65:
                                continue
                            
                            current_price = daily_prices[symbol]
                            
                            # Check if we should enter a position
                            if (prediction.get('prediction', 0) == 1 and 
                                symbol not in self.positions or 
                                self.positions.get(symbol, {}).get('quantity', 0) == 0):
                                
                                # Calculate position size
                                position_size = self.portfolio_manager.calculate_ai_position_size(
                                    symbol, prediction, current_price
                                )
                                
                                if position_size > 0:
                                    # Execute buy trade
                                    trade = self.simulate_trade_execution(
                                        symbol, 'BUY', position_size, current_price, current_date
                                    )
                                    
                                    if trade:
                                        self.update_position(symbol, 'BUY', position_size, 
                                                           current_price, current_date)
                                        period_trades.append(trade)
                                        
                                        self.logger.debug(f"📈 Backtest BUY: {position_size} {symbol} "
                                                        f"at ₹{current_price:.2f}")
                            
                            # Check exit conditions for existing positions
                            elif symbol in self.positions and self.positions[symbol]['quantity'] > 0:
                                exit_decision = self.portfolio_manager.should_exit_position(
                                    symbol, current_price, prediction
                                )
                                
                                if exit_decision['should_exit']:
                                    quantity = self.positions[symbol]['quantity']
                                    
                                    # Execute sell trade
                                    trade = self.simulate_trade_execution(
                                        symbol, 'SELL', quantity, current_price, current_date
                                    )
                                    
                                    if trade:
                                        self.update_position(symbol, 'SELL', quantity, 
                                                           current_price, current_date)
                                        trade['exit_reason'] = exit_decision['reason']
                                        period_trades.append(trade)
                                        
                                        self.logger.debug(f"📉 Backtest SELL: {quantity} {symbol} "
                                                        f"at ₹{current_price:.2f} "
                                                        f"({exit_decision['reason']})")
                    
                    # Record portfolio value
                    portfolio_value = self.calculate_portfolio_value(daily_prices)
                    portfolio_values.append({
                        'date': current_date.isoformat(),
                        'portfolio_value': portfolio_value,
                        'cash': self.current_capital,
                        'positions_value': portfolio_value - self.current_capital
                    })
                
                except Exception as e:
                    self.logger.warning(f"Daily simulation failed for {current_date}: {e}")
                    continue
            
            # Calculate period performance
            end_capital = self.current_capital
            if portfolio_values:
                end_portfolio_value = portfolio_values[-1]['portfolio_value']
            else:
                end_portfolio_value = end_capital
            
            period_return = (end_portfolio_value - start_capital) / start_capital * 100
            period_pnl = end_portfolio_value - start_capital
            
            return {
                'trades': period_trades,
                'portfolio_values': portfolio_values,
                'period_return': period_return,
                'period_pnl': period_pnl
            }
            
        except Exception as e:
            self.logger.error(f"Period simulation failed: {e}")
            return {'trades': [], 'portfolio_values': [], 'period_return': 0, 'period_pnl': 0}
    
    def calculate_overall_performance(self, results: Dict) -> Dict:
        """Calculate comprehensive performance metrics"""
        try:
            if not results['portfolio_history']:
                return {}
            
            # Convert portfolio history to DataFrame
            portfolio_df = pd.DataFrame(results['portfolio_history'])
            portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
            portfolio_df = portfolio_df.sort_values('date')
            
            # Calculate returns
            portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change()
            portfolio_df = portfolio_df.dropna()
            
            if len(portfolio_df) == 0:
                return {}
            
            # Basic metrics
            total_return = (portfolio_df['portfolio_value'].iloc[-1] / self.initial_capital - 1) * 100
            annualized_return = ((portfolio_df['portfolio_value'].iloc[-1] / self.initial_capital) ** 
                               (252 / len(portfolio_df)) - 1) * 100
            
            # Risk metrics
            daily_returns = portfolio_df['daily_return']
            volatility = daily_returns.std() * np.sqrt(252) * 100  # Annualized volatility
            
            # Sharpe ratio (assuming 6% risk-free rate)
            risk_free_rate = 0.06
            excess_returns = daily_returns - risk_free_rate/252
            sharpe_ratio = excess_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
            
            # Maximum drawdown
            rolling_max = portfolio_df['portfolio_value'].cummax()
            drawdown = (portfolio_df['portfolio_value'] - rolling_max) / rolling_max * 100
            max_drawdown = drawdown.min()
            
            # Win rate and trade statistics
            trades_df = pd.DataFrame(results['trade_history'])
            
            if not trades_df.empty:
                buy_trades = trades_df[trades_df['action'] == 'BUY']
                sell_trades = trades_df[trades_df['action'] == 'SELL']
                
                total_trades = len(buy_trades)
                total_commission = trades_df['commission'].sum()
                
                # Calculate trade P&Ls (simplified)
                winning_trades = 0
                losing_trades = 0
                
                # This is a simplified calculation - in practice, you'd match buy/sell pairs
                for symbol in trades_df['symbol'].unique():
                    symbol_trades = trades_df[trades_df['symbol'] == symbol]
                    symbol_buys = symbol_trades[symbol_trades['action'] == 'BUY']
                    symbol_sells = symbol_trades[symbol_trades['action'] == 'SELL']
                    
                    if len(symbol_sells) > 0 and len(symbol_buys) > 0:
                        avg_buy_price = symbol_buys['execution_price'].mean()
                        avg_sell_price = symbol_sells['execution_price'].mean()
                        
                        if avg_sell_price > avg_buy_price:
                            winning_trades += 1
                        else:
                            losing_trades += 1
                
                win_rate = (winning_trades / (winning_trades + losing_trades) * 100) if (winning_trades + losing_trades) > 0 else 0
                
            else:
                total_trades = 0
                total_commission = 0
                win_rate = 0
            
            # Benchmark comparison (simplified - using flat 12% annual return)
            benchmark_return = 12  # Assume 12% annual benchmark
            alpha = annualized_return - benchmark_return
            
            performance_metrics = {
                'total_return_pct': round(total_return, 2),
                'annualized_return_pct': round(annualized_return, 2),
                'volatility_pct': round(volatility, 2),
                'sharpe_ratio': round(sharpe_ratio, 2),
                'max_drawdown_pct': round(max_drawdown, 2),
                'win_rate_pct': round(win_rate, 2),
                'total_trades': total_trades,
                'total_commission': round(total_commission, 2),
                'alpha_vs_benchmark': round(alpha, 2),
                'final_portfolio_value': round(portfolio_df['portfolio_value'].iloc[-1], 2),
                'initial_capital': self.initial_capital,
                'trading_periods': len(results['periods']),
                'average_period_return': round(np.mean([p['period_return'] for p in results['periods']]), 2),
                'best_period_return': round(max([p['period_return'] for p in results['periods']]), 2),
                'worst_period_return': round(min([p['period_return'] for p in results['periods']]), 2)
            }
            
            return performance_metrics
            
        except Exception as e:
            self.logger.error(f"Performance calculation failed: {e}")
            return {}
    
    def export_backtest_results(self, results: Dict, filename: str = None):
        """Export backtest results to file"""
        try:
            if not filename:
                filename = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Prepare export data
            export_data = {
                'backtest_config': {
                    'initial_capital': self.initial_capital,
                    'commission_rate': self.commission_rate,
                    'slippage': self.slippage,
                    'train_period_days': self.train_period_days,
                    'test_period_days': self.test_period_days
                },
                'results': results,
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"📊 Backtest results exported to {filename}")
            
        except Exception as e:
            self.logger.error(f"Export failed: {e}")
    
    def run_full_backtest(self, symbols: List[str] = None, 
                         start_date: str = "2022-01-01", 
                         end_date: str = None) -> Dict:
        """Run complete AI strategy backtest"""
        try:
            if not symbols:
                symbols = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 
                          'SBIN', 'BHARTIARTL', 'ITC', 'LT', 'MARUTI']
            
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            self.logger.info(f"🚀 Starting full AI backtest:")
            self.logger.info(f"   Symbols: {len(symbols)} stocks")
            self.logger.info(f"   Period: {start_date} to {end_date}")
            self.logger.info(f"   Initial Capital: ₹{self.initial_capital:,.2f}")
            
            # Reset state
            self.current_capital = self.initial_capital
            self.positions = {}
            self.trade_history = []
            self.portfolio_history = []
            
            # Run walk-forward backtest
            results = self.walk_forward_backtest(symbols, start_dt, end_dt)
            
            if results:
                performance = results.get('overall_performance', {})
                
                self.logger.info("🎯 BACKTEST RESULTS:")
                self.logger.info(f"   Total Return: {performance.get('total_return_pct', 0):.2f}%")
                self.logger.info(f"   Annualized Return: {performance.get('annualized_return_pct', 0):.2f}%")
                self.logger.info(f"   Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}")
                self.logger.info(f"   Max Drawdown: {performance.get('max_drawdown_pct', 0):.2f}%")
                self.logger.info(f"   Win Rate: {performance.get('win_rate_pct', 0):.2f}%")
                self.logger.info(f"   Total Trades: {performance.get('total_trades', 0)}")
                
                # Export results
                self.export_backtest_results(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Full backtest failed: {e}")
            return {}

def main():
    """Run AI backtesting"""
    print("🔄 Starting AI Strategy Backtesting...")
    
    backtester = AIBacktestingEngine()
    
    # Run 2-year backtest
    results = backtester.run_full_backtest(
        start_date="2022-01-01",
        end_date="2024-01-01"
    )
    
    if results:
        print("\n✅ Backtesting completed successfully!")
        print("📊 Check the exported JSON file for detailed results.")
    else:
        print("\n❌ Backtesting failed. Check logs for details.")

if __name__ == "__main__":
    main()